"""Tier 1 tests for canonical Feature Group contracts, schema_hash, and Pandera validation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from banking_fti.core.contracts import (
    CARD_WINDOW_AGGS_CONTRACT,
    CREDIT_APPLICATIONS_CONTRACT,
    CUSTOMER_PROFILES_CONTRACT,
    FEATURE_GROUP_CONTRACTS,
    HOME_OWNERSHIP_VALUES,
    ColumnSpec,
    RawBankingBatch,
    compute_combined_schema_hash,
    compute_schema_hash,
)
from banking_fti.core.validation import (
    SchemaValidationError,
    validate_card_events,
    validate_card_window_aggs,
    validate_credit_applications,
    validate_customer_profiles,
    validate_raw_batch,
)


def _valid_customer_profiles_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["CUST-00001"],
            "event_timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True).astype(
                "datetime64[ns, UTC]"
            ),
            "age_years": np.array([34], dtype=np.int64),
            "income_annual": np.array([85000.0], dtype=np.float64),
            "employment_years": np.array([6.5], dtype=np.float64),
            "home_ownership": ["MORTGAGE"],
            "tenure_months": np.array([48], dtype=np.int64),
            "num_active_cards": np.array([2], dtype=np.int64),
        }
    )


def _valid_card_events_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["EVT-00001"],
            "cc_num": ["4000000000000001"],
            "customer_id": ["CUST-00001"],
            "event_timestamp": pd.to_datetime(["2026-01-01T12:00:00Z"], utc=True).astype(
                "datetime64[ns, UTC]"
            ),
            "amount": np.array([125.50], dtype=np.float64),
            "latitude": np.array([40.7128], dtype=np.float64),
            "longitude": np.array([-74.0060], dtype=np.float64),
        }
    )


def _valid_credit_applications_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "application_id": ["APP-00001"],
            "customer_id": ["CUST-00001"],
            "event_timestamp": pd.to_datetime(["2026-01-02T09:00:00Z"], utc=True).astype(
                "datetime64[ns, UTC]"
            ),
            "loan_amount": np.array([15000.0], dtype=np.float64),
            "loan_term_months": np.array([36], dtype=np.int64),
            "dti_ratio": np.array([0.22], dtype=np.float64),
            "existing_delinquencies": np.array([0], dtype=np.int64),
            "utilization_ratio": np.array([0.31], dtype=np.float64),
            "pub_rec_flag": np.array([0], dtype=np.int64),
            "default_label": np.array([0], dtype=np.int64),
        }
    )


def _valid_card_window_aggs_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cc_num": ["4000000000000001"],
            "customer_id": ["CUST-00001"],
            "event_timestamp": pd.to_datetime(["2026-01-01T12:00:00Z"], utc=True).astype(
                "datetime64[ns, UTC]"
            ),
            "tx_amount_mavg_7d": np.array([98.25], dtype=np.float64),
            "tx_amount_mstd_7d": np.array([14.50], dtype=np.float64),
            "tx_count_24h": np.array([2], dtype=np.int64),
            "tx_count_7d": np.array([5], dtype=np.int64),
            "loc_delta_mavg_7d": np.array([3.4], dtype=np.float64),
        }
    )


def test_canonical_contracts_cover_poc1_groups_and_unencoded_columns() -> None:
    assert set(FEATURE_GROUP_CONTRACTS.keys()) == {
        "customer_profiles",
        "card_window_aggs",
        "credit_applications",
    }
    assert CUSTOMER_PROFILES_CONTRACT.entity_keys == ("customer_id",)
    assert CARD_WINDOW_AGGS_CONTRACT.entity_keys == ("cc_num", "customer_id")
    assert CREDIT_APPLICATIONS_CONTRACT.entity_keys == ("application_id", "customer_id")
    assert HOME_OWNERSHIP_VALUES == ("RENT", "MORTGAGE", "OWN", "OTHER")


def test_schema_hash_is_deterministic_and_sensitive_to_contract_mutations() -> None:
    baseline = compute_schema_hash(CUSTOMER_PROFILES_CONTRACT)
    assert len(baseline) == 64
    assert baseline == CUSTOMER_PROFILES_CONTRACT.schema_hash()

    dtype_changed = replace(
        CUSTOMER_PROFILES_CONTRACT,
        feature_columns=(
            ColumnSpec("age_years", "float64"),
            *CUSTOMER_PROFILES_CONTRACT.feature_columns[1:],
        ),
    )
    assert compute_schema_hash(dtype_changed) != baseline

    column_added = replace(
        CUSTOMER_PROFILES_CONTRACT,
        feature_columns=(
            *CUSTOMER_PROFILES_CONTRACT.feature_columns,
            ColumnSpec("extra_col", "int64"),
        ),
    )
    assert compute_schema_hash(column_added) != baseline

    key_changed = replace(
        CUSTOMER_PROFILES_CONTRACT,
        entity_keys=("customer_id", "cc_num"),
    )
    assert compute_schema_hash(key_changed) != baseline

    combined_a = compute_combined_schema_hash(
        [CUSTOMER_PROFILES_CONTRACT, CARD_WINDOW_AGGS_CONTRACT, CREDIT_APPLICATIONS_CONTRACT]
    )
    combined_b = compute_combined_schema_hash(
        [CREDIT_APPLICATIONS_CONTRACT, CUSTOMER_PROFILES_CONTRACT, CARD_WINDOW_AGGS_CONTRACT]
    )
    assert len(combined_a) == 64
    assert combined_a == combined_b


def test_pandera_validators_accept_valid_poc1_dataframes_and_batch() -> None:
    batch = RawBankingBatch(
        customer_profiles=_valid_customer_profiles_df(),
        card_events=_valid_card_events_df(),
        credit_applications=_valid_credit_applications_df(),
        batch_id="batch-001",
    )
    validated = validate_raw_batch(batch)
    assert validated.batch_id == "batch-001"
    assert validated.customer_profiles.loc[0, "home_ownership"] == "MORTGAGE"

    aggs = validate_card_window_aggs(_valid_card_window_aggs_df())
    assert aggs["tx_count_7d"].tolist() == [5]


def test_validate_customer_profiles_rejects_missing_extra_null_and_encoded_values() -> None:
    missing = _valid_customer_profiles_df().drop(columns=["income_annual"])
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_customer_profiles(missing)
    assert exc_info.value.dataset_name == "customer_profiles"

    extra = _valid_customer_profiles_df().assign(unexpected_col=1)
    with pytest.raises(SchemaValidationError):
        validate_customer_profiles(extra)

    null_row = _valid_customer_profiles_df()
    null_row.loc[0, "customer_id"] = None
    with pytest.raises(SchemaValidationError):
        validate_customer_profiles(null_row)

    encoded_home = _valid_customer_profiles_df()
    encoded_home.loc[0, "home_ownership"] = "0"
    with pytest.raises(SchemaValidationError):
        validate_customer_profiles(encoded_home)

    naive_ts = _valid_customer_profiles_df()
    naive_ts["event_timestamp"] = pd.to_datetime(["2026-01-01T00:00:00"])
    with pytest.raises(SchemaValidationError):
        validate_customer_profiles(naive_ts)


def test_validate_card_events_and_credit_applications_reject_non_finite_and_out_of_range() -> None:
    neg_amount = _valid_card_events_df()
    neg_amount.loc[0, "amount"] = -10.0
    with pytest.raises(SchemaValidationError):
        validate_card_events(neg_amount)

    inf_amount = _valid_card_events_df()
    inf_amount.loc[0, "amount"] = np.inf
    with pytest.raises(SchemaValidationError):
        validate_card_events(inf_amount)

    bad_lat = _valid_card_events_df()
    bad_lat.loc[0, "latitude"] = 95.0
    with pytest.raises(SchemaValidationError):
        validate_card_events(bad_lat)

    neg_dti = _valid_credit_applications_df()
    neg_dti.loc[0, "dti_ratio"] = -0.05
    with pytest.raises(SchemaValidationError):
        validate_credit_applications(neg_dti)

    bad_label = _valid_credit_applications_df()
    bad_label.loc[0, "default_label"] = 2
    with pytest.raises(SchemaValidationError):
        validate_credit_applications(bad_label)


def test_validate_card_window_aggs_rejects_24h_exceeding_7d_and_nan() -> None:
    bad_counts = _valid_card_window_aggs_df()
    bad_counts.loc[0, "tx_count_24h"] = 10
    bad_counts.loc[0, "tx_count_7d"] = 4
    with pytest.raises(SchemaValidationError):
        validate_card_window_aggs(bad_counts)

    nan_avg = _valid_card_window_aggs_df()
    nan_avg.loc[0, "tx_amount_mavg_7d"] = np.nan
    with pytest.raises(SchemaValidationError):
        validate_card_window_aggs(nan_avg)
