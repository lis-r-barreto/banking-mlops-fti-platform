"""Tier 3 offline round-trip tests for FeastFeatureStore (Offline Parquet) and Hopsworks seam."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from banking_fti.adapters.feast_store import FeastFeatureStore
from banking_fti.adapters.hopsworks_seam import HopsworksConfig, HopsworksFeatureStore
from banking_fti.adapters.in_memory import InMemoryFeatureStore
from banking_fti.core.contracts import (
    CARD_WINDOW_AGGS_CONTRACT,
    CREDIT_APPLICATIONS_CONTRACT,
    CUSTOMER_PROFILES_CONTRACT,
    FeatureGroupName,
)
from banking_fti.features.pipeline import FeaturePipeline
from banking_fti.features.simulator import BankingEventSimulator, SimulatorConfig
from banking_fti.ports.feature_store import FeatureStore, SchemaMismatchError


def _profiles_batch_1() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["CUST-1", "CUST-2"],
            "event_timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], utc=True
            ).astype("datetime64[ns, UTC]"),
            "age_years": np.array([32, 45], dtype=np.int64),
            "income_annual": np.array([70000.0, 110000.0], dtype=np.float64),
            "employment_years": np.array([5.0, 12.0], dtype=np.float64),
            "home_ownership": ["RENT", "MORTGAGE"],
            "tenure_months": np.array([24, 60], dtype=np.int64),
            "num_active_cards": np.array([1, 2], dtype=np.int64),
        }
    )


def _profiles_batch_2() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["CUST-1"],
            "event_timestamp": pd.to_datetime(["2026-01-10T00:00:00Z"], utc=True).astype(
                "datetime64[ns, UTC]"
            ),
            "age_years": np.array([33], dtype=np.int64),
            "income_annual": np.array([95000.0], dtype=np.float64),
            "employment_years": np.array([6.0], dtype=np.float64),
            "home_ownership": ["OWN"],
            "tenure_months": np.array([36], dtype=np.int64),
            "num_active_cards": np.array([2], dtype=np.int64),
        }
    )


def _profiles_with_future() -> pd.DataFrame:
    future = pd.DataFrame(
        {
            "customer_id": ["CUST-1"],
            "event_timestamp": pd.to_datetime(["2026-01-20T00:00:00Z"], utc=True).astype(
                "datetime64[ns, UTC]"
            ),
            "age_years": np.array([99], dtype=np.int64),
            "income_annual": np.array([999999.0], dtype=np.float64),
            "employment_years": np.array([40.0], dtype=np.float64),
            "home_ownership": ["OTHER"],
            "tenure_months": np.array([200], dtype=np.int64),
            "num_active_cards": np.array([9], dtype=np.int64),
        }
    )
    return pd.concat([_profiles_batch_1(), _profiles_batch_2(), future], ignore_index=True)


def _card_aggs_with_future() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cc_num": ["4000000000000001", "4000000000000001", "4000000000000001"],
            "customer_id": ["CUST-1", "CUST-1", "CUST-1"],
            "event_timestamp": pd.to_datetime(
                ["2026-01-02T00:00:00Z", "2026-01-12T00:00:00Z", "2026-01-20T00:00:00Z"],
                utc=True,
            ).astype("datetime64[ns, UTC]"),
            "tx_amount_mavg_7d": np.array([55.0, 400.0, 9999.0], dtype=np.float64),
            "tx_amount_mstd_7d": np.array([10.0, 50.0, 500.0], dtype=np.float64),
            "tx_count_24h": np.array([1, 4, 99], dtype=np.int64),
            "tx_count_7d": np.array([3, 9, 199], dtype=np.int64),
            "loc_delta_mavg_7d": np.array([1.2, 42.0, 900.0], dtype=np.float64),
        }
    )


def _credit_apps_with_future() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "application_id": ["APP-EARLY", "APP-LATE", "APP-FUTURE"],
            "customer_id": ["CUST-1", "CUST-1", "CUST-1"],
            "event_timestamp": pd.to_datetime(
                ["2026-01-05T00:00:00Z", "2026-01-15T00:00:00Z", "2026-01-25T00:00:00Z"],
                utc=True,
            ).astype("datetime64[ns, UTC]"),
            "loan_amount": np.array([10000.0, 20000.0, 50000.0], dtype=np.float64),
            "loan_term_months": np.array([24, 36, 60], dtype=np.int64),
            "dti_ratio": np.array([0.20, 0.35, 0.85], dtype=np.float64),
            "existing_delinquencies": np.array([0, 1, 5], dtype=np.int64),
            "utilization_ratio": np.array([0.25, 0.55, 0.99], dtype=np.float64),
            "pub_rec_flag": np.array([0, 0, 1], dtype=np.int64),
            "default_label": np.array([0, 1, 1], dtype=np.int64),
        }
    )


def test_feature_repo_and_feast_store_apply_stage1_entities_and_views(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    entities_mod = importlib.import_module("feature_repo.entities")
    views_mod = importlib.import_module("feature_repo.feature_views")

    assert entities_mod.customer.join_key == "customer_id"
    assert entities_mod.credit_card.join_key == "cc_num"
    assert entities_mod.loan_application.join_key == "application_id"

    assert views_mod.customer_profiles.online is False
    assert views_mod.card_window_aggs.online is False
    assert views_mod.card_window_aggs_by_customer.online is False
    assert views_mod.credit_applications.online is False

    store = FeastFeatureStore(tmp_path / "feast_repo")
    assert isinstance(store, FeatureStore)

    registered_entities = {entity.name for entity in store.list_entities()}
    assert {"customer", "credit_card", "loan_application"}.issubset(registered_entities)

    registered_views = {view.name for view in store.list_feature_views()}
    assert {
        "customer_profiles",
        "card_window_aggs",
        "card_window_aggs_by_customer",
        "credit_applications",
    }.issubset(registered_views)


def test_feast_store_appends_offline_parquet_with_deterministic_schema_and_reload(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "feast_repo"
    store = FeastFeatureStore(repo_dir)

    assert store.get_schema_hash("customer_profiles") is None
    empty = store.read_feature_group("customer_profiles")
    assert empty.empty
    assert tuple(empty.columns) == CUSTOMER_PROFILES_CONTRACT.all_columns()

    expected_hash = CUSTOMER_PROFILES_CONTRACT.schema_hash()
    first = store.write_feature_group(
        "customer_profiles",
        _profiles_batch_1(),
        schema_hash=expected_hash,
    )
    assert first.group_name == "customer_profiles"
    assert first.rows_written == 2
    assert first.total_rows == 2
    assert first.schema_hash == expected_hash

    parquet_path = store.data_dir / "customer_profiles.parquet"
    assert parquet_path.exists()
    schema_after_first = pq.read_schema(parquet_path)

    second = store.write_feature_group(
        "customer_profiles",
        _profiles_batch_2(),
        schema_hash=expected_hash,
    )
    assert second.rows_written == 1
    assert second.total_rows == 3

    schema_after_second = pq.read_schema(parquet_path)
    assert schema_after_second.equals(schema_after_first)
    assert schema_after_second.field("event_timestamp").type == pa.timestamp("ns", tz="UTC")
    assert schema_after_second.field("age_years").type == pa.int64()
    assert schema_after_second.field("income_annual").type == pa.float64()
    assert schema_after_second.field("home_ownership").type == pa.string()

    # Reload a fresh instance from disk and verify schema_hash persistence and reads
    reloaded = FeastFeatureStore(repo_dir)
    assert reloaded.get_schema_hash("customer_profiles") == expected_hash
    all_rows = reloaded.read_feature_group("customer_profiles")
    assert len(all_rows) == 3
    assert all_rows["home_ownership"].tolist() == ["RENT", "MORTGAGE", "OWN"]
    assert str(all_rows["event_timestamp"].dtype) == "datetime64[ns, UTC]"

    filtered = reloaded.read_feature_group(
        "customer_profiles",
        start_time=datetime(2026, 1, 5, 0, 0, tzinfo=UTC),
        end_time=datetime(2026, 1, 15, 0, 0, tzinfo=UTC),
    )
    assert len(filtered) == 1
    assert filtered["home_ownership"].tolist() == ["OWN"]

    with pytest.raises(ValueError, match="timezone-aware"):
        reloaded.read_feature_group(
            "customer_profiles",
            start_time=datetime(2026, 1, 5, 0, 0),
        )

    with pytest.raises(SchemaMismatchError):
        reloaded.write_feature_group(
            "customer_profiles",
            _profiles_batch_2(),
            schema_hash="0" * 64,
        )

    with pytest.raises(ValueError, match="columns"):
        reloaded.write_feature_group(
            "customer_profiles",
            _profiles_batch_2().drop(columns=["income_annual"]),
            schema_hash=expected_hash,
        )


def test_feast_store_point_in_time_historical_join_and_unlabelled_snapshots(
    tmp_path: Path,
) -> None:
    feast_store = FeastFeatureStore(tmp_path / "feast_repo")
    memory_store = InMemoryFeatureStore()

    datasets: dict[FeatureGroupName, tuple[pd.DataFrame, str]] = {
        "customer_profiles": (_profiles_with_future(), CUSTOMER_PROFILES_CONTRACT.schema_hash()),
        "card_window_aggs": (_card_aggs_with_future(), CARD_WINDOW_AGGS_CONTRACT.schema_hash()),
        "credit_applications": (
            _credit_apps_with_future(),
            CREDIT_APPLICATIONS_CONTRACT.schema_hash(),
        ),
    }
    for group_name, (frame, schema_hash) in datasets.items():
        feast_store.write_feature_group(group_name, frame, schema_hash=schema_hash)
        memory_store.write_feature_group(group_name, frame, schema_hash=schema_hash)

    entity_with_card = pd.DataFrame(
        {
            "application_id": ["APP-LATE", "APP-EARLY"],
            "customer_id": ["CUST-1", "CUST-1"],
            "cc_num": ["4000000000000001", "4000000000000001"],
            "event_timestamp": pd.to_datetime(
                ["2026-01-15T00:00:00Z", "2026-01-05T00:00:00Z"], utc=True
            ).astype("datetime64[ns, UTC]"),
        }
    )

    feast_joined = feast_store.get_historical_features(
        entity_with_card,
        ["customer_profiles", "card_window_aggs", "credit_applications"],
    )
    mem_joined = memory_store.get_historical_features(
        entity_with_card,
        ["customer_profiles", "card_window_aggs", "credit_applications"],
    )

    assert feast_joined["application_id"].tolist() == ["APP-LATE", "APP-EARLY"]
    assert feast_joined["home_ownership"].tolist() == ["OWN", "RENT"]
    assert feast_joined["tx_amount_mavg_7d"].tolist() == [400.0, 55.0]
    assert feast_joined["default_label"].tolist() == [1, 0]
    pd.testing.assert_frame_equal(
        feast_joined.loc[:, mem_joined.columns],
        mem_joined,
        check_dtype=False,
    )

    # Without cc_num in entity_df, Feast resolves card_window_aggs via customer_id
    entity_without_card = entity_with_card.drop(columns=["cc_num"])
    joined_by_customer = feast_store.get_historical_features(
        entity_without_card,
        ["customer_profiles", "card_window_aggs", "credit_applications"],
    )
    assert joined_by_customer["application_id"].tolist() == ["APP-LATE", "APP-EARLY"]
    assert joined_by_customer["tx_amount_mavg_7d"].tolist() == [400.0, 55.0]
    assert joined_by_customer["home_ownership"].tolist() == ["OWN", "RENT"]
    assert joined_by_customer["default_label"].tolist() == [1, 0]

    # Unlabelled batch snapshot via exclude_labels=True
    unlabelled_via_flag = feast_store.get_historical_features(
        entity_with_card,
        ["customer_profiles", "card_window_aggs", "credit_applications"],
        exclude_labels=True,
    )
    assert "default_label" not in unlabelled_via_flag.columns
    assert unlabelled_via_flag["loan_amount"].tolist() == [20000.0, 10000.0]
    assert unlabelled_via_flag["tx_amount_mavg_7d"].tolist() == [400.0, 55.0]

    # Unlabelled batch snapshot by passing unlabelled credit applications as entity_df
    unlabelled_apps = _credit_apps_with_future().iloc[[1, 0]].drop(columns=["default_label"])
    batch_snapshot = feast_store.get_historical_features(
        unlabelled_apps,
        ["customer_profiles", "card_window_aggs"],
    )
    assert "default_label" not in batch_snapshot.columns
    assert batch_snapshot["application_id"].tolist() == ["APP-LATE", "APP-EARLY"]
    assert batch_snapshot["home_ownership"].tolist() == ["OWN", "RENT"]
    assert batch_snapshot["tx_amount_mavg_7d"].tolist() == [400.0, 55.0]
    assert batch_snapshot["loan_amount"].tolist() == [20000.0, 10000.0]

    # Empty entity_df returns empty DataFrame with expected columns; missing timestamp fails
    empty_result = feast_store.get_historical_features(
        entity_with_card.iloc[0:0],
        ["customer_profiles", "card_window_aggs"],
    )
    assert empty_result.empty
    assert "home_ownership" in empty_result.columns
    assert "tx_amount_mavg_7d" in empty_result.columns

    with pytest.raises(ValueError, match="event_timestamp"):
        feast_store.get_historical_features(
            entity_with_card.drop(columns=["event_timestamp"]),
            ["customer_profiles"],
        )


def test_feast_store_historical_join_handles_unwritten_feature_group(tmp_path: Path) -> None:
    feast_store = FeastFeatureStore(tmp_path / "feast_repo")
    feast_store.write_feature_group(
        "customer_profiles",
        _profiles_batch_1(),
        schema_hash=CUSTOMER_PROFILES_CONTRACT.schema_hash(),
    )

    entity_df = pd.DataFrame(
        {
            "customer_id": ["CUST-1"],
            "cc_num": ["4000000000000001"],
            "event_timestamp": pd.to_datetime(["2026-01-05T00:00:00Z"], utc=True).astype(
                "datetime64[ns, UTC]"
            ),
        }
    )
    joined = feast_store.get_historical_features(
        entity_df,
        ["customer_profiles", "card_window_aggs"],
    )
    assert joined["home_ownership"].tolist() == ["RENT"]
    assert pd.isna(joined.loc[0, "tx_amount_mavg_7d"])


def test_feature_pipeline_end_to_end_round_trip_with_feast_offline_store(tmp_path: Path) -> None:
    repo_dir = tmp_path / "feast_repo"
    store = FeastFeatureStore(repo_dir)
    pipeline = FeaturePipeline(feature_store=store)

    sim = BankingEventSimulator(SimulatorConfig(num_customers=20, backfill_days=21, seed=42))
    t0 = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    backfill = sim.generate_backfill(end_time=t0)
    inc = sim.generate_incremental_batch(start_time=t0, window_hours=24, batch_seq=1)

    backfill_result = pipeline.run_batch(backfill)
    inc_result = pipeline.run_batch(
        inc,
        history_card_events=backfill.card_events,
        reference_stats=backfill_result.group_stats,
    )

    assert inc_result.receipts["customer_profiles"].total_rows == 40
    assert (store.data_dir / "customer_profiles.parquet").exists()
    assert (store.data_dir / "card_window_aggs.parquet").exists()
    assert (store.data_dir / "credit_applications.parquet").exists()

    reloaded = FeastFeatureStore(repo_dir)
    stored_apps = reloaded.read_feature_group("credit_applications")
    assert len(stored_apps) == len(backfill.credit_applications) + len(inc.credit_applications)

    # Labelled historical training join
    training_entity_df = stored_apps.loc[:, ["application_id", "customer_id", "event_timestamp"]]
    training_df = reloaded.get_historical_features(
        training_entity_df,
        ["customer_profiles", "card_window_aggs", "credit_applications"],
    )
    assert len(training_df) == len(stored_apps)
    assert training_df["application_id"].tolist() == stored_apps["application_id"].tolist()
    assert not training_df["home_ownership"].isna().any()
    assert "default_label" in training_df.columns

    # Unlabelled batch snapshot retrieval
    unlabelled_entity_df = (
        stored_apps.drop(columns=["default_label"]).tail(5).reset_index(drop=True)
    )
    batch_df = reloaded.get_historical_features(
        unlabelled_entity_df,
        ["customer_profiles", "card_window_aggs"],
    )
    assert len(batch_df) == 5
    assert "default_label" not in batch_df.columns
    assert batch_df["application_id"].tolist() == unlabelled_entity_df["application_id"].tolist()


def test_hopsworks_seam_conforms_to_feature_store_protocol_and_documents_swap_in() -> None:
    config = HopsworksConfig(
        project_name="banking_fti_platform",
        host="c.app.hopsworks.ai",
        feature_store_name="banking_fti_featurestore",
    )
    seam = HopsworksFeatureStore(config)
    assert isinstance(seam, FeatureStore)
    assert seam.config == config
    assert seam.get_schema_hash("customer_profiles") is None

    with pytest.raises(NotImplementedError, match="hsfs"):
        seam.write_feature_group(
            "customer_profiles",
            _profiles_batch_1(),
            schema_hash=CUSTOMER_PROFILES_CONTRACT.schema_hash(),
        )

    with pytest.raises(NotImplementedError, match="hsfs"):
        seam.read_feature_group("customer_profiles")

    with pytest.raises(NotImplementedError, match="hsfs"):
        seam.get_historical_features(
            _profiles_batch_1().loc[:, ["customer_id", "event_timestamp"]],
            ["customer_profiles"],
        )


def test_feature_repo_supports_native_feast_apply_in_temp_copy(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    copied_repo = tmp_path / "feature_repo"
    shutil.copytree(
        repo_root / "feature_repo",
        copied_repo,
        ignore=shutil.ignore_patterns("data", "__pycache__"),
    )
    (copied_repo / "data").mkdir(parents=True, exist_ok=True)

    feast_cli = str(Path(sys.executable).with_name("feast"))
    proc = subprocess.run(
        [feast_cli, "apply"],
        cwd=copied_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        "Created entity customer" in proc.stdout or (copied_repo / "data" / "registry.db").exists()
    )
    assert (copied_repo / "data" / "registry.db").exists()


def test_feast_store_supports_customer_keyed_credit_apps_long_ttl_and_custom_data_dir(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    custom_data_dir = tmp_path / "custom_data"
    store = FeastFeatureStore(repo_root / "feature_repo", data_dir=custom_data_dir)
    assert (custom_data_dir / "registry.db").exists()
    assert not (repo_root / "feature_repo" / "data").exists()

    old_profile = _profiles_batch_1().copy()
    old_profile["event_timestamp"] = pd.to_datetime(
        ["2020-01-01T00:00:00Z", "2020-01-01T00:00:00Z"], utc=True
    ).astype("datetime64[ns, UTC]")
    store.write_feature_group(
        "customer_profiles",
        old_profile,
        schema_hash=CUSTOMER_PROFILES_CONTRACT.schema_hash(),
    )
    store.write_feature_group(
        "credit_applications",
        _credit_apps_with_future(),
        schema_hash=CREDIT_APPLICATIONS_CONTRACT.schema_hash(),
    )

    customer_only_entity = pd.DataFrame(
        {
            "customer_id": ["CUST-1"],
            "event_timestamp": pd.to_datetime(["2026-01-16T00:00:00Z"], utc=True).astype(
                "datetime64[ns, UTC]"
            ),
        }
    )
    joined = store.get_historical_features(
        customer_only_entity,
        ["customer_profiles", "credit_applications"],
    )
    assert joined["home_ownership"].tolist() == ["RENT"]
    assert joined["loan_amount"].tolist() == [20000.0]
    assert joined["default_label"].tolist() == [1]
