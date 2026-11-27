"""Tier 2 tests for TrainingPipeline (credit_risk_batch): PIT joins, ONNX parity, gates, and CLI."""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest

from banking_fti.adapters.feast_store import FeastFeatureStore
from banking_fti.adapters.in_memory import InMemoryFeatureStore
from banking_fti.adapters.in_memory_model_registry import InMemoryModelRegistry
from banking_fti.core.contracts import (
    CARD_WINDOW_AGGS_CONTRACT,
    CREDIT_APPLICATIONS_CONTRACT,
    CUSTOMER_PROFILES_CONTRACT,
)
from banking_fti.features.pipeline import FeaturePipeline
from banking_fti.features.simulator import BankingEventSimulator, SimulatorConfig
from banking_fti.ports.feature_store import FeatureStore
from banking_fti.ports.model_registry import RegisteredModel
from banking_fti.training import pipeline as training_pipeline_mod
from banking_fti.training.gates import CreditRiskGateConfig
from banking_fti.training.pipeline import TrainingPipeline
from banking_fti.training.preprocessor import TabularPreprocessor

pytest.importorskip("lightgbm")


def _populate_separable_credit_store(
    store: FeatureStore,
    *,
    num_customers: int = 160,
    num_apps: int = 400,
    noisy_labels: bool = False,
    biased_low_income: bool = False,
    inject_holdout_skew: bool = False,
    seed: int = 11,
) -> pd.DataFrame:
    """Populate a FeatureStore with clean separable credit applications across cohorts."""
    rng = np.random.default_rng(seed)
    base_time = datetime(2026, 1, 1, tzinfo=UTC)

    # Ensure balanced coverage across income tiers (<50k, 50k-100k, >=100k) & home ownership
    tiers = np.array([35_000.0, 75_000.0, 130_000.0])
    incomes = np.array([tiers[i % 3] + rng.uniform(-5000.0, 5000.0) for i in range(num_customers)])
    home_choices = np.array(["RENT", "MORTGAGE", "OWN", "OTHER"])
    homes = np.array([home_choices[i % 4] for i in range(num_customers)])
    cust_ids = [f"CUST-{i:04d}" for i in range(num_customers)]

    profiles = pd.DataFrame(
        {
            "customer_id": cust_ids,
            "event_timestamp": [base_time] * num_customers,
            "age_years": rng.integers(23, 68, size=num_customers),
            "income_annual": incomes,
            "employment_years": rng.uniform(1.0, 25.0, size=num_customers),
            "home_ownership": homes,
            "tenure_months": rng.integers(6, 120, size=num_customers),
            "num_active_cards": rng.integers(1, 4, size=num_customers),
        }
    )

    aggs = pd.DataFrame(
        {
            "cc_num": [f"40000000{i:08d}" for i in range(num_customers)],
            "customer_id": cust_ids,
            "event_timestamp": [base_time] * num_customers,
            "tx_amount_mavg_7d": rng.uniform(40.0, 350.0, size=num_customers),
            "tx_amount_mstd_7d": rng.uniform(5.0, 90.0, size=num_customers),
            "tx_count_24h": rng.integers(0, 8, size=num_customers),
            "tx_count_7d": rng.integers(1, 35, size=num_customers),
            "loc_delta_mavg_7d": rng.uniform(0.1, 15.0, size=num_customers),
        }
    )

    app_cust_indices = np.arange(num_apps) % num_customers
    app_timestamps = [base_time + timedelta(hours=6 * (i + 1)) for i in range(num_apps)]
    dti = rng.uniform(0.05, 0.75, size=num_apps)
    util = rng.uniform(0.05, 0.90, size=num_apps)
    delinq = rng.integers(0, 4, size=num_apps)
    pub_rec = (rng.random(num_apps) < 0.25).astype(np.int64)
    loan_amount = rng.uniform(5_000.0, 60_000.0, size=num_apps)

    if inject_holdout_skew:
        holdout_start = int(num_apps * 0.75)
        dti[holdout_start:] = dti[holdout_start:] + 12.0
        loan_amount[holdout_start:] = np.inf

    if noisy_labels:
        labels = rng.integers(0, 2, size=num_apps, dtype=np.int64)
    else:
        risk_score = 2.6 * dti + 1.8 * util + 0.65 * delinq + 0.9 * pub_rec - 2.35
        labels = (risk_score > 0.0).astype(np.int64)
        # Flip 4% noise so probabilities are non-degenerate and calibration stays healthy
        flip_mask = rng.random(num_apps) < 0.04
        labels = np.where(flip_mask, 1 - labels, labels).astype(np.int64)

    if biased_low_income:
        # In the holdout split, invert labels for low-income customers so a clean model
        # achieves high overall metrics while failing recall specifically on low-income cohort.
        holdout_start = int(num_apps * 0.75)
        app_incomes = incomes[app_cust_indices]
        low_holdout = (np.arange(num_apps) >= holdout_start) & (app_incomes < 50_000.0)
        labels = np.where(low_holdout, 1 - labels, labels).astype(np.int64)

    apps = pd.DataFrame(
        {
            "application_id": [f"APP-{i:05d}" for i in range(num_apps)],
            "customer_id": [cust_ids[idx] for idx in app_cust_indices],
            "event_timestamp": app_timestamps,
            "loan_amount": loan_amount,
            "loan_term_months": rng.choice([12, 24, 36, 48, 60], size=num_apps),
            "dti_ratio": dti,
            "existing_delinquencies": delinq,
            "utilization_ratio": util,
            "pub_rec_flag": pub_rec,
            "default_label": labels,
        }
    )

    store.write_feature_group(
        "customer_profiles", profiles, schema_hash=CUSTOMER_PROFILES_CONTRACT.schema_hash()
    )
    store.write_feature_group(
        "card_window_aggs", aggs, schema_hash=CARD_WINDOW_AGGS_CONTRACT.schema_hash()
    )
    store.write_feature_group(
        "credit_applications", apps, schema_hash=CREDIT_APPLICATIONS_CONTRACT.schema_hash()
    )
    return apps


def test_training_pipeline_uses_point_in_time_join_without_future_leakage(tmp_path: Path) -> None:
    store = InMemoryFeatureStore()
    apps = _populate_separable_credit_store(store, num_customers=80, num_apps=200, seed=19)

    # Append a future profile snapshot AFTER all applications with extreme income shift.
    # A naive merge on customer_id would duplicate rows; PIT join must ignore future values.
    future_ts = apps["event_timestamp"].max() + timedelta(days=30)
    orig_profiles = store.read_feature_group("customer_profiles")
    future_profiles = orig_profiles.copy()
    future_profiles["event_timestamp"] = future_ts
    future_profiles["income_annual"] = 9_999_999.0
    store.write_feature_group(
        "customer_profiles",
        future_profiles,
        schema_hash=CUSTOMER_PROFILES_CONTRACT.schema_hash(),
    )

    registry = InMemoryModelRegistry()
    pipeline = TrainingPipeline(store, registry, artifact_dir=tmp_path / "artifacts")
    result = pipeline.train(num_boost_round=60, random_state=42)

    assert result.challenger_registered is True
    assert result.rows_used == len(apps)
    assert result.gate_passed is True
    assert result.promoted is True


def test_training_pipeline_passes_gate_persists_onnx_artifacts_and_lineage(tmp_path: Path) -> None:
    store = InMemoryFeatureStore()
    _populate_separable_credit_store(store, seed=23)
    registry = InMemoryModelRegistry()

    dvc_lock = tmp_path / "dvc.lock"
    dvc_lock.write_text("schema: '2.0'\nstages: {}\n", encoding="utf-8")

    pipeline = TrainingPipeline(
        store,
        registry,
        artifact_dir=tmp_path / "models",
        compute_backend="cpu",
        executor="pytest_local",
        pipeline_version="v0.1.0",
        dvc_lock_path=dvc_lock,
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = pipeline.train(num_boost_round=80, random_state=42)
    assert not any("eval_set" in str(w.message) for w in caught_warnings)

    assert result.convergence_passed is True
    assert result.nan_inf_passed is True
    assert result.value_skew_passed is True
    assert result.fairness_passed is True
    assert result.serving_smoke_passed is True
    assert result.champion_comparison_passed is True
    assert result.parity_pass is True
    assert result.onnx_max_abs_diff is not None and result.onnx_max_abs_diff < 1e-5
    assert result.holdout_auc >= 0.75
    assert result.holdout_recall >= 0.65
    assert result.brier < 0.25
    assert result.gate_passed is True
    assert result.promoted is True
    assert result.champion_version == 1

    champion = registry.get_champion("credit_risk_batch")
    assert champion is not None
    assert champion.version == 1
    assert champion.status == "champion"
    assert champion.feature_view_name == "credit_applications"
    assert champion.feature_view_version == "v1"
    assert len(champion.feature_schema_hash) == 64
    assert len(champion.dataset_sha256) == 64
    assert champion.git_sha
    assert champion.dvc_lock_hash and len(champion.dvc_lock_hash) == 64
    assert champion.pipeline_version == "v0.1.0"
    assert champion.compute_backend == "cpu"
    assert champion.executor == "pytest_local"
    assert champion.timestamp_start is not None
    assert champion.timestamp_end is not None
    assert champion.metrics["roc_auc"] == pytest.approx(result.holdout_auc)

    native_path = Path(champion.native_model_path)
    onnx_path = Path(champion.onnx_model_path)
    preproc_path = Path(champion.preprocessor_path)
    assert native_path.exists()
    assert onnx_path.exists()
    assert preproc_path.exists()

    # Reload bundled preprocessor + ONNX artifact independently and score sample rows
    reloaded_pp = TabularPreprocessor.load(preproc_path)
    apps = store.read_feature_group("credit_applications")
    joined = store.get_historical_features(
        apps.loc[:4, ["application_id", "customer_id", "event_timestamp"]],
        ["customer_profiles", "card_window_aggs", "credit_applications"],
        exclude_labels=True,
    )
    X_sample = reloaded_pp.transform_features(joined)
    sess = ort.InferenceSession(onnx_path.read_bytes())
    assert sess.get_outputs()[0].shape[0] == "N"
    outputs = sess.run(None, {"input": X_sample})
    probs = np.asarray(outputs[1], dtype=np.float64)[:, 1]
    assert probs.shape == (5,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_training_pipeline_blocks_low_metric_and_biased_cohort_without_altering_champion(
    tmp_path: Path,
) -> None:
    # Seed an existing healthy champion v1
    store_ok = InMemoryFeatureStore()
    _populate_separable_credit_store(store_ok, seed=31)
    registry = InMemoryModelRegistry()
    ok_res = TrainingPipeline(store_ok, registry, artifact_dir=tmp_path / "m1").train(
        num_boost_round=80, random_state=42
    )
    assert ok_res.promoted is True
    assert ok_res.champion_version == 1

    # 1. Noisy / low-AUC candidate must be blocked and leave champion at v1
    store_noisy = InMemoryFeatureStore()
    _populate_separable_credit_store(store_noisy, noisy_labels=True, seed=32)
    noisy_res = TrainingPipeline(store_noisy, registry, artifact_dir=tmp_path / "m2").train(
        num_boost_round=40, random_state=42
    )
    assert noisy_res.gate_passed is False
    assert noisy_res.promoted is False
    assert noisy_res.champion_version == 1
    assert registry.get_champion("credit_risk_batch") is not None
    assert registry.get_champion("credit_risk_batch").version == 1  # type: ignore[union-attr]
    assert registry.get_challenger("credit_risk_batch") is not None
    assert registry.get_challenger("credit_risk_batch").version == 2  # type: ignore[union-attr]

    # 2. Cohort fairness violation (low-income tier recall degraded) must block promotion
    store_biased = InMemoryFeatureStore()
    _populate_separable_credit_store(store_biased, biased_low_income=True, seed=33)
    biased_res = TrainingPipeline(store_biased, registry, artifact_dir=tmp_path / "m3").train(
        num_boost_round=80, random_state=42
    )
    assert biased_res.fairness_passed is False
    assert biased_res.gate_passed is False
    assert biased_res.promoted is False
    assert registry.get_champion("credit_risk_batch").version == 1  # type: ignore[union-attr]


def test_training_pipeline_blocks_value_skew_and_enforces_champion_tie_tolerance(
    tmp_path: Path,
) -> None:
    # 1. Value skew / Inf injection in holdout split must fail value_skew_passed
    store_skew = InMemoryFeatureStore()
    _populate_separable_credit_store(store_skew, inject_holdout_skew=True, seed=41)
    registry_skew = InMemoryModelRegistry()
    skew_res = TrainingPipeline(store_skew, registry_skew, artifact_dir=tmp_path / "ms").train(
        num_boost_round=40, random_state=42
    )
    assert skew_res.value_skew_passed is False
    assert skew_res.gate_passed is False
    assert skew_res.promoted is False
    assert skew_res.challenger_registered is False

    # 2. Champion vs. Challenger tie tolerance (0.005):
    # Pre-seed champion v1 with slightly higher metrics than challenger can beat -> blocked
    store = InMemoryFeatureStore()
    _populate_separable_credit_store(store, seed=43)
    registry = InMemoryModelRegistry()
    strong_champ = RegisteredModel(
        model_name="credit_risk_batch",
        version=0,
        status="challenger",
        run_id="prev-champ",
        created_at="2026-01-01T00:00:00+00:00",
        feature_view_name="credit_applications",
        feature_view_version="v1",
        feature_schema_hash="hash-1",
        dataset_sha256="sha-prev",
        native_model_path="model.txt",
        onnx_model_path="model.onnx",
        preprocessor_path="preprocessor.joblib",
        metrics={
            "roc_auc": 0.999,
            "default_recall": 0.99,
            "min_cohort_recall": 0.99,
            "max_cohort_recall_gap": 0.0,
        },
    )
    reg_champ = registry.register_challenger(strong_champ)
    registry.promote_to_champion(
        "credit_risk_batch",
        challenger_version=reg_champ.version,
        validation_passed=True,
    )

    blocked_res = TrainingPipeline(
        store,
        registry,
        artifact_dir=tmp_path / "m_blocked",
        gate_config=CreditRiskGateConfig(tie_tolerance=0.005),
    ).train(num_boost_round=60, random_state=42)
    assert blocked_res.champion_comparison_passed is False
    assert blocked_res.gate_passed is False
    assert blocked_res.promoted is False
    assert registry.get_champion("credit_risk_batch").version == 1  # type: ignore[union-attr]

    # 3. Comparable champion within tie tolerance -> promotes challenger and rotates @previous
    registry_rot = InMemoryModelRegistry()
    comparable_champ = RegisteredModel(
        model_name="credit_risk_batch",
        version=0,
        status="challenger",
        run_id="champ-1",
        created_at="2026-01-01T00:00:00+00:00",
        feature_view_name="credit_applications",
        feature_view_version="v1",
        feature_schema_hash="hash-1",
        dataset_sha256="sha-1",
        native_model_path="model.txt",
        onnx_model_path="model.onnx",
        preprocessor_path="preprocessor.joblib",
        metrics={"roc_auc": 0.80, "default_recall": 0.65, "min_cohort_recall": 0.50},
    )
    reg_c1 = registry_rot.register_challenger(comparable_champ)
    registry_rot.promote_to_champion(
        "credit_risk_batch",
        challenger_version=reg_c1.version,
        validation_passed=True,
    )

    promoted_res = TrainingPipeline(
        store,
        registry_rot,
        artifact_dir=tmp_path / "m_rot",
        gate_config=CreditRiskGateConfig(tie_tolerance=0.005),
    ).train(num_boost_round=80, random_state=42)
    assert promoted_res.champion_comparison_passed is True
    assert promoted_res.gate_passed is True
    assert promoted_res.promoted is True
    assert promoted_res.champion_version == 2
    assert promoted_res.previous_version == 1
    assert registry_rot.get_champion("credit_risk_batch").version == 2  # type: ignore[union-attr]
    assert registry_rot.get_previous("credit_risk_batch").version == 1  # type: ignore[union-attr]


def test_training_pipeline_cli_runs_against_feast_offline_store(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_dir = tmp_path / "feast_repo"
    data_dir = tmp_path / "feast_data"
    artifact_dir = tmp_path / "model_artifacts"

    feast_store = FeastFeatureStore(repo_path=repo_dir, data_dir=data_dir)
    sim = BankingEventSimulator(SimulatorConfig(num_customers=80, backfill_days=60, seed=42))
    batch = sim.generate_backfill(end_time=datetime(2026, 2, 1, tzinfo=UTC))
    FeaturePipeline(feast_store).run_batch(batch)

    exit_code = training_pipeline_mod.main(
        [
            "--repo-path",
            str(repo_dir),
            "--data-dir",
            str(data_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--num-boost-round",
            "50",
            "--seed",
            "42",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["model_name"] == "credit_risk_batch"
    assert payload["challenger_registered"] is True
    assert payload["convergence_passed"] is True
    assert payload["parity_pass"] is True
    assert payload["onnx_max_abs_diff"] < 1e-5
    assert payload["serving_smoke_passed"] is True
    assert (artifact_dir / "credit_risk_batch").exists()
