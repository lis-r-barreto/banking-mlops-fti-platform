"""Tier 3 integration tests for the FeatureStore port and InMemoryFeatureStore adapter."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from banking_fti.adapters.in_memory import InMemoryFeatureStore
from banking_fti.core.contracts import (
    CARD_WINDOW_AGGS_CONTRACT,
    CREDIT_APPLICATIONS_CONTRACT,
    CUSTOMER_PROFILES_CONTRACT,
)
from banking_fti.ports.feature_store import FeatureStore, SchemaMismatchError


def _profiles_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["CUST-1", "CUST-1"],
            "event_timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-10T00:00:00Z"], utc=True
            ).astype("datetime64[ns, UTC]"),
            "age_years": np.array([32, 33], dtype=np.int64),
            "income_annual": np.array([70000.0, 95000.0], dtype=np.float64),
            "employment_years": np.array([5.0, 6.0], dtype=np.float64),
            "home_ownership": ["RENT", "OWN"],
            "tenure_months": np.array([24, 36], dtype=np.int64),
            "num_active_cards": np.array([1, 2], dtype=np.int64),
        }
    )


def _card_aggs_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cc_num": ["4000000000000001", "4000000000000001"],
            "customer_id": ["CUST-1", "CUST-1"],
            "event_timestamp": pd.to_datetime(
                ["2026-01-02T00:00:00Z", "2026-01-12T00:00:00Z"], utc=True
            ).astype("datetime64[ns, UTC]"),
            "tx_amount_mavg_7d": np.array([55.0, 400.0], dtype=np.float64),
            "tx_amount_mstd_7d": np.array([10.0, 50.0], dtype=np.float64),
            "tx_count_24h": np.array([1, 4], dtype=np.int64),
            "tx_count_7d": np.array([3, 9], dtype=np.int64),
            "loc_delta_mavg_7d": np.array([1.2, 42.0], dtype=np.float64),
        }
    )


def _credit_apps_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "application_id": ["APP-EARLY", "APP-LATE"],
            "customer_id": ["CUST-1", "CUST-1"],
            "event_timestamp": pd.to_datetime(
                ["2026-01-05T00:00:00Z", "2026-01-15T00:00:00Z"], utc=True
            ).astype("datetime64[ns, UTC]"),
            "loan_amount": np.array([10000.0, 20000.0], dtype=np.float64),
            "loan_term_months": np.array([24, 36], dtype=np.int64),
            "dti_ratio": np.array([0.20, 0.35], dtype=np.float64),
            "existing_delinquencies": np.array([0, 1], dtype=np.int64),
            "utilization_ratio": np.array([0.25, 0.55], dtype=np.float64),
            "pub_rec_flag": np.array([0, 0], dtype=np.int64),
            "default_label": np.array([0, 1], dtype=np.int64),
        }
    )


def test_in_memory_store_preserves_unencoded_features_schema_hash_and_defensive_copies() -> None:
    store = InMemoryFeatureStore()
    assert isinstance(store, FeatureStore)
    assert store.get_schema_hash("customer_profiles") is None

    profiles = _profiles_df()
    expected_hash = CUSTOMER_PROFILES_CONTRACT.schema_hash()
    receipt = store.write_feature_group("customer_profiles", profiles, schema_hash=expected_hash)
    assert receipt.group_name == "customer_profiles"
    assert receipt.rows_written == 2
    assert receipt.total_rows == 2
    assert receipt.schema_hash == expected_hash
    assert store.get_schema_hash("customer_profiles") == expected_hash

    # Mutate caller DataFrame and read result to confirm defensive copying
    profiles.loc[0, "home_ownership"] = "OTHER"
    stored = store.read_feature_group("customer_profiles")
    assert stored["home_ownership"].tolist() == ["RENT", "OWN"]
    stored.loc[0, "home_ownership"] = "OTHER"
    assert store.read_feature_group("customer_profiles")["home_ownership"].tolist() == [
        "RENT",
        "OWN",
    ]

    # Time-filtered read
    filtered = store.read_feature_group(
        "customer_profiles",
        start_time=datetime(2026, 1, 5, 0, 0, tzinfo=UTC),
        end_time=datetime(2026, 1, 15, 0, 0, tzinfo=UTC),
    )
    assert len(filtered) == 1
    assert filtered["home_ownership"].tolist() == ["OWN"]


def test_in_memory_store_rejects_schema_hash_mismatch_and_column_drift() -> None:
    store = InMemoryFeatureStore()
    profiles = _profiles_df()
    store.write_feature_group(
        "customer_profiles",
        profiles,
        schema_hash=CUSTOMER_PROFILES_CONTRACT.schema_hash(),
    )

    with pytest.raises(SchemaMismatchError):
        store.write_feature_group("customer_profiles", profiles, schema_hash="0" * 64)

    with pytest.raises(ValueError, match="columns"):
        store.write_feature_group(
            "customer_profiles",
            profiles.drop(columns=["income_annual"]),
            schema_hash=CUSTOMER_PROFILES_CONTRACT.schema_hash(),
        )


def test_get_historical_features_performs_leak_free_point_in_time_backward_join() -> None:
    store = InMemoryFeatureStore()
    store.write_feature_group(
        "customer_profiles",
        _profiles_df(),
        schema_hash=CUSTOMER_PROFILES_CONTRACT.schema_hash(),
    )
    store.write_feature_group(
        "card_window_aggs",
        _card_aggs_df(),
        schema_hash=CARD_WINDOW_AGGS_CONTRACT.schema_hash(),
    )
    store.write_feature_group(
        "credit_applications",
        _credit_apps_df(),
        schema_hash=CREDIT_APPLICATIONS_CONTRACT.schema_hash(),
    )

    entity_df = pd.DataFrame(
        {
            "application_id": ["APP-LATE", "APP-EARLY"],
            "customer_id": ["CUST-1", "CUST-1"],
            "cc_num": ["4000000000000001", "4000000000000001"],
            "event_timestamp": pd.to_datetime(
                ["2026-01-15T00:00:00Z", "2026-01-05T00:00:00Z"], utc=True
            ).astype("datetime64[ns, UTC]"),
        }
    )

    joined = store.get_historical_features(
        entity_df,
        ["customer_profiles", "card_window_aggs", "credit_applications"],
    )

    # Preserves entity_df row order (APP-LATE first, APP-EARLY second)
    assert joined["application_id"].tolist() == ["APP-LATE", "APP-EARLY"]
    # APP-LATE (Jan 15) gets Jan 10 profile (OWN) and Jan 12 card aggs (400.0)
    assert joined["home_ownership"].tolist() == ["OWN", "RENT"]
    assert joined["tx_amount_mavg_7d"].tolist() == [400.0, 55.0]
    assert joined["default_label"].tolist() == [1, 0]


def test_get_historical_features_supports_exclude_labels_and_unlabelled_batch_snapshots() -> None:
    store = InMemoryFeatureStore()
    store.write_feature_group(
        "customer_profiles",
        _profiles_df(),
        schema_hash=CUSTOMER_PROFILES_CONTRACT.schema_hash(),
    )
    store.write_feature_group(
        "card_window_aggs",
        _card_aggs_df(),
        schema_hash=CARD_WINDOW_AGGS_CONTRACT.schema_hash(),
    )
    store.write_feature_group(
        "credit_applications",
        _credit_apps_df(),
        schema_hash=CREDIT_APPLICATIONS_CONTRACT.schema_hash(),
    )

    entity_df = pd.DataFrame(
        {
            "application_id": ["APP-LATE", "APP-EARLY"],
            "customer_id": ["CUST-1", "CUST-1"],
            "cc_num": ["4000000000000001", "4000000000000001"],
            "event_timestamp": pd.to_datetime(
                ["2026-01-15T00:00:00Z", "2026-01-05T00:00:00Z"], utc=True
            ).astype("datetime64[ns, UTC]"),
        }
    )

    unlabelled_via_flag = store.get_historical_features(
        entity_df,
        ["customer_profiles", "card_window_aggs", "credit_applications"],
        exclude_labels=True,
    )
    assert "default_label" not in unlabelled_via_flag.columns
    assert unlabelled_via_flag["application_id"].tolist() == ["APP-LATE", "APP-EARLY"]
    assert unlabelled_via_flag["loan_amount"].tolist() == [20000.0, 10000.0]
    assert unlabelled_via_flag["home_ownership"].tolist() == ["OWN", "RENT"]
    assert unlabelled_via_flag["tx_amount_mavg_7d"].tolist() == [400.0, 55.0]

    unlabelled_apps = _credit_apps_df().drop(columns=["default_label"]).iloc[[1, 0]]
    batch_snapshot = store.get_historical_features(
        unlabelled_apps,
        ["customer_profiles", "card_window_aggs"],
    )
    assert "default_label" not in batch_snapshot.columns
    assert batch_snapshot["application_id"].tolist() == ["APP-LATE", "APP-EARLY"]
    assert batch_snapshot["home_ownership"].tolist() == ["OWN", "RENT"]
    assert batch_snapshot["tx_amount_mavg_7d"].tolist() == [400.0, 55.0]
    assert batch_snapshot["loan_amount"].tolist() == [20000.0, 10000.0]
