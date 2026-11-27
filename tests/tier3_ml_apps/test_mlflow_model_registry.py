"""Tier 3 offline SQLite round-trip tests for MlflowModelRegistry."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from banking_fti.adapters import MlflowModelRegistry
from banking_fti.adapters.in_memory import InMemoryFeatureStore
from banking_fti.core.contracts import (
    CARD_WINDOW_AGGS_CONTRACT,
    CREDIT_APPLICATIONS_CONTRACT,
    CUSTOMER_PROFILES_CONTRACT,
)
from banking_fti.ports.model_registry import ModelRegistry, RegisteredModel
from banking_fti.training.pipeline import TrainingPipeline
from banking_fti.training.preprocessor import TabularPreprocessor

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_artifacts(base_dir: Path, tag: str) -> tuple[Path, Path, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    native = base_dir / f"model_{tag}.txt"
    onnx = base_dir / f"model_{tag}.onnx"
    preproc = base_dir / f"preprocessor_{tag}.joblib"
    native.write_text(f"native-booster-{tag}", encoding="utf-8")
    onnx.write_bytes(f"onnx-bytes-{tag}".encode())
    preproc.write_bytes(f"joblib-bytes-{tag}".encode())
    return native, onnx, preproc


def _candidate(
    artifact_dir: Path,
    *,
    tag: str,
    run_id: str = "pipe-run-1",
    feature_view_version: str = "v1",
    roc_auc: float = 0.85,
) -> RegisteredModel:
    native, onnx, preproc = _make_artifacts(artifact_dir, tag)
    return RegisteredModel(
        model_name="credit_risk_batch",
        version=0,
        status="challenger",
        run_id=run_id,
        created_at="2026-01-15T12:00:00+00:00",
        feature_view_name="credit_applications",
        feature_view_version=feature_view_version,
        feature_schema_hash="schema-hash-abc123",
        dataset_sha256="dataset-sha-xyz789",
        native_model_path=str(native),
        onnx_model_path=str(onnx),
        preprocessor_path=str(preproc),
        git_sha="deadbeef1234",
        dvc_lock_hash="dvc-lock-hash-5678",
        pipeline_version="v0.1.0",
        compute_backend="cpu",
        executor="pytest_sqlite",
        timestamp_start="2026-01-15T11:59:00+00:00",
        timestamp_end="2026-01-15T12:00:00+00:00",
        metrics={
            "roc_auc": roc_auc,
            "default_recall": 0.78,
            "brier_score": 0.12,
        },
    )


def test_mlflow_registry_register_fetch_lineage_and_restore_artifacts_offline(
    tmp_path: Path,
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    registry = MlflowModelRegistry(
        tracking_uri=tracking_uri,
        experiment_name="banking_fti_test",
        environment="hom",
    )
    assert isinstance(registry, ModelRegistry)
    assert registry.get_challenger("credit_risk_batch") is None
    assert registry.get_champion("credit_risk_batch") is None
    assert registry.get_previous("credit_risk_batch") is None

    train_dir = tmp_path / "ephemeral_train"
    candidate = _candidate(train_dir, tag="v1")
    registered = registry.register_challenger(candidate)

    assert registered.version == 1
    assert registered.status == "challenger"
    assert registered.git_sha == "deadbeef1234"
    assert registered.dvc_lock_hash == "dvc-lock-hash-5678"
    assert registered.metrics["roc_auc"] == pytest.approx(0.85)

    # Delete original training directory and fetch from a brand-new registry instance
    shutil.rmtree(train_dir)
    fresh_registry = MlflowModelRegistry(
        tracking_uri=tracking_uri,
        experiment_name="banking_fti_test",
        environment="hom",
        artifact_cache_dir=tmp_path / "fresh_cache",
    )
    fetched = fresh_registry.get_challenger("credit_risk_batch")
    assert fetched is not None
    assert fetched.version == 1
    assert fetched.status == "challenger"
    assert fetched.feature_view_name == "credit_applications"
    assert fetched.feature_view_version == "v1"
    assert fetched.feature_schema_hash == "schema-hash-abc123"
    assert fetched.dataset_sha256 == "dataset-sha-xyz789"
    assert fetched.git_sha == "deadbeef1234"
    assert fetched.dvc_lock_hash == "dvc-lock-hash-5678"
    assert fetched.pipeline_version == "v0.1.0"
    assert fetched.compute_backend == "cpu"
    assert fetched.executor == "pytest_sqlite"
    assert fetched.timestamp_start == "2026-01-15T11:59:00+00:00"
    assert fetched.timestamp_end == "2026-01-15T12:00:00+00:00"
    assert fetched.metrics["default_recall"] == pytest.approx(0.78)
    assert Path(fetched.native_model_path).read_text(encoding="utf-8") == "native-booster-v1"
    assert Path(fetched.onnx_model_path).read_bytes() == b"onnx-bytes-v1"
    assert Path(fetched.preprocessor_path).read_bytes() == b"joblib-bytes-v1"

    downloaded = fresh_registry.download_artifacts(fetched, tmp_path / "explicit_dst")
    assert downloaded["native"].read_text(encoding="utf-8") == "native-booster-v1"
    assert downloaded["onnx"].read_bytes() == b"onnx-bytes-v1"
    assert downloaded["preprocessor"].read_bytes() == b"joblib-bytes-v1"

    # Verify MLflow run tags and experiment metadata directly
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(fetched.run_id)
    assert run.data.tags["start_time"] == "2026-01-15T11:59:00+00:00"
    assert run.data.tags["end_time"] == "2026-01-15T12:00:00+00:00"
    assert run.data.tags["environment"] == "hom"
    assert run.data.tags["git_sha"] == "deadbeef1234"
    assert not (_REPO_ROOT / "mlruns").exists()


def test_mlflow_registry_missing_artifact_raises_file_not_found(tmp_path: Path) -> None:
    registry = MlflowModelRegistry(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    missing = RegisteredModel(
        model_name="credit_risk_batch",
        version=0,
        status="challenger",
        run_id="r0",
        created_at="2026-01-01T00:00:00+00:00",
        feature_view_name="credit_applications",
        feature_view_version="v1",
        feature_schema_hash="h1",
        dataset_sha256="s1",
        native_model_path=str(tmp_path / "does_not_exist.txt"),
        onnx_model_path=str(tmp_path / "does_not_exist.onnx"),
        preprocessor_path=str(tmp_path / "does_not_exist.joblib"),
    )
    with pytest.raises(FileNotFoundError):
        registry.register_challenger(missing)


def test_mlflow_registry_promote_blocks_rotates_and_rolls_back_atomically(
    tmp_path: Path,
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    registry = MlflowModelRegistry(tracking_uri=tracking_uri)

    c1 = registry.register_challenger(_candidate(tmp_path / "art1", tag="v1", run_id="r1"))
    blocked = registry.promote_to_champion(
        "credit_risk_batch",
        challenger_version=c1.version,
        validation_passed=False,
        validation_summary="AUC too low",
    )
    assert blocked["ok"] is False
    assert blocked["error"] == "AUC too low"
    assert registry.get_champion("credit_risk_batch") is None
    assert registry.get_challenger("credit_risk_batch") is not None

    wrong_ver = registry.promote_to_champion(
        "credit_risk_batch",
        challenger_version=99,
        validation_passed=True,
    )
    assert wrong_ver["ok"] is False
    assert wrong_ver["error"] == "no challenger at version 99"

    # Promote v1 -> @champion
    p1 = registry.promote_to_champion(
        "credit_risk_batch",
        challenger_version=c1.version,
        validation_passed=True,
        validation_summary="v1 passed",
    )
    assert p1["ok"] is True
    assert p1["champion_version"] == 1
    assert p1["previous_version"] is None
    assert registry.get_challenger("credit_risk_batch") is None
    assert registry.get_champion("credit_risk_batch").version == 1  # type: ignore[union-attr]

    # Cannot rollback when @previous does not exist
    no_prev = registry.rollback_to_previous("credit_risk_batch")
    assert no_prev["ok"] is False
    assert no_prev["error"] == "no @previous to roll back to"

    # Register & promote v2 with feature_view_version="v2" -> rotates v1 to @previous
    c2 = registry.register_challenger(
        _candidate(
            tmp_path / "art2",
            tag="v2",
            run_id="r2",
            feature_view_version="v2",
            roc_auc=0.89,
        )
    )
    p2 = registry.promote_to_champion(
        "credit_risk_batch",
        challenger_version=c2.version,
        validation_passed=True,
    )
    assert p2["ok"] is True
    assert p2["champion_version"] == 2
    assert p2["previous_version"] == 1

    # Rollback without expected_feature_view_version must block when v1 != v2
    incompatible = registry.rollback_to_previous("credit_risk_batch")
    assert incompatible["ok"] is False
    assert "feature_view_version" in str(incompatible["error"])
    assert registry.get_champion("credit_risk_batch").version == 2  # type: ignore[union-attr]

    # Rollback with explicit expected_feature_view_version="v1" succeeds & restores v1 artifacts
    shutil.rmtree(tmp_path / "art1")
    shutil.rmtree(tmp_path / "art2")
    rb = registry.rollback_to_previous("credit_risk_batch", expected_feature_view_version="v1")
    assert rb["ok"] is True
    assert rb["champion_version"] == 1
    assert rb["previous_version"] == 2
    assert rb["feature_view_version"] == "v1"

    reloaded = MlflowModelRegistry(
        tracking_uri=tracking_uri,
        artifact_cache_dir=tmp_path / "rb_cache",
    )
    champ = reloaded.get_champion("credit_risk_batch")
    prev = reloaded.get_previous("credit_risk_batch")
    assert champ is not None and champ.version == 1
    assert prev is not None and prev.version == 2
    assert Path(champ.onnx_model_path).read_bytes() == b"onnx-bytes-v1"
    assert Path(prev.onnx_model_path).read_bytes() == b"onnx-bytes-v2"


def test_mlflow_registry_from_env_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.delenv("MLFLOW_ENVIRONMENT", raising=False)

    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        MlflowModelRegistry.from_env()

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'hom.db'}")
    monkeypatch.setenv("MLFLOW_ENVIRONMENT", "hom")
    hom_reg = MlflowModelRegistry.from_env()
    assert hom_reg.environment == "hom"
    assert hom_reg.experiment_name == "banking_fti_hom"

    monkeypatch.setenv("MLFLOW_ENVIRONMENT", "prd")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "custom_prd_exp")
    prd_reg = MlflowModelRegistry.from_env()
    assert prd_reg.environment == "prd"
    assert prd_reg.experiment_name == "custom_prd_exp"


def test_training_pipeline_end_to_end_with_mlflow_registry_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(17)
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    num_customers = 120
    num_apps = 320
    tiers = np.array([35_000.0, 75_000.0, 130_000.0])
    incomes = np.array([tiers[i % 3] + rng.uniform(-4000.0, 4000.0) for i in range(num_customers)])
    home_choices = np.array(["RENT", "MORTGAGE", "OWN", "OTHER"])
    homes = np.array([home_choices[i % 4] for i in range(num_customers)])
    cust_ids = [f"MLF-CUST-{i:04d}" for i in range(num_customers)]

    store = InMemoryFeatureStore()
    store.write_feature_group(
        "customer_profiles",
        pd.DataFrame(
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
        ),
        schema_hash=CUSTOMER_PROFILES_CONTRACT.schema_hash(),
    )
    store.write_feature_group(
        "card_window_aggs",
        pd.DataFrame(
            {
                "cc_num": [f"40001111{i:08d}" for i in range(num_customers)],
                "customer_id": cust_ids,
                "event_timestamp": [base_time] * num_customers,
                "tx_amount_mavg_7d": rng.uniform(40.0, 350.0, size=num_customers),
                "tx_amount_mstd_7d": rng.uniform(5.0, 90.0, size=num_customers),
                "tx_count_24h": rng.integers(0, 8, size=num_customers),
                "tx_count_7d": rng.integers(1, 35, size=num_customers),
                "loc_delta_mavg_7d": rng.uniform(0.1, 15.0, size=num_customers),
            }
        ),
        schema_hash=CARD_WINDOW_AGGS_CONTRACT.schema_hash(),
    )
    app_cust_idx = np.arange(num_apps) % num_customers
    dti = rng.uniform(0.05, 0.75, size=num_apps)
    util = rng.uniform(0.05, 0.90, size=num_apps)
    delinq = rng.integers(0, 4, size=num_apps)
    pub_rec = (rng.random(num_apps) < 0.25).astype(np.int64)
    risk = 2.6 * dti + 1.8 * util + 0.65 * delinq + 0.9 * pub_rec - 2.35
    labels = (risk > 0.0).astype(np.int64)
    flip = rng.random(num_apps) < 0.03
    labels = np.where(flip, 1 - labels, labels).astype(np.int64)
    apps = pd.DataFrame(
        {
            "application_id": [f"MLF-APP-{i:05d}" for i in range(num_apps)],
            "customer_id": [cust_ids[idx] for idx in app_cust_idx],
            "event_timestamp": [base_time + timedelta(hours=6 * (i + 1)) for i in range(num_apps)],
            "loan_amount": rng.uniform(5_000.0, 60_000.0, size=num_apps),
            "loan_term_months": rng.choice([12, 24, 36, 48, 60], size=num_apps),
            "dti_ratio": dti,
            "existing_delinquencies": delinq,
            "utilization_ratio": util,
            "pub_rec_flag": pub_rec,
            "default_label": labels,
        }
    )
    store.write_feature_group(
        "credit_applications",
        apps,
        schema_hash=CREDIT_APPLICATIONS_CONTRACT.schema_hash(),
    )

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    registry = MlflowModelRegistry(tracking_uri=tracking_uri, environment="local")
    raw_artifact_dir = tmp_path / "train_artifacts"
    result = TrainingPipeline(
        store,
        registry,
        artifact_dir=raw_artifact_dir,
        executor="pytest_mlflow",
    ).train(num_boost_round=60, random_state=42)

    assert result.gate_passed is True
    assert result.promoted is True
    assert result.champion_version == 1

    # Remove original artifact dir to verify MLflow artifact restoration from tracking store
    shutil.rmtree(raw_artifact_dir)
    fresh = MlflowModelRegistry(
        tracking_uri=tracking_uri,
        artifact_cache_dir=tmp_path / "restored_cache",
    )
    champion = fresh.get_champion("credit_risk_batch")
    assert champion is not None
    assert champion.version == 1
    assert champion.dataset_sha256 == result.dataset_sha256

    pp = TabularPreprocessor.load(champion.preprocessor_path)
    joined = store.get_historical_features(
        apps.loc[:4, ["application_id", "customer_id", "event_timestamp"]],
        ["customer_profiles", "card_window_aggs", "credit_applications"],
        exclude_labels=True,
    )
    X_sample = pp.transform_features(joined)
    sess = ort.InferenceSession(Path(champion.onnx_model_path).read_bytes())
    outputs = sess.run(None, {"input": X_sample})
    probs = np.asarray(outputs[1], dtype=np.float64)[:, 1]
    assert probs.shape == (5,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
