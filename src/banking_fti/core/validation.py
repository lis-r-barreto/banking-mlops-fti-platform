"""Fail-fast Pandera schema validation for raw banking batches and causal aggregations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

from banking_fti.core.contracts import HOME_OWNERSHIP_VALUES, RawBankingBatch


class SchemaValidationError(ValueError):
    """Raised when a banking DataFrame fails its Pandera schema contract."""

    def __init__(self, dataset_name: str, message: str) -> None:
        self.dataset_name = dataset_name
        super().__init__(f"[{dataset_name}] {message}")


def _is_non_empty_str(series: pd.Series) -> pd.Series:
    return series.astype(str).str.len() > 0


def _is_finite_float(series: pd.Series) -> pd.Series:
    values = series.to_numpy(dtype=np.float64, copy=False)
    return pd.Series(np.isfinite(values), index=series.index)


def _utc_timestamp_column() -> pa.Column:
    return pa.Column(
        pd.DatetimeTZDtype(tz="UTC"),
        nullable=False,
        coerce=False,
    )


def _non_empty_id_column() -> pa.Column:
    return pa.Column(
        str,
        nullable=False,
        coerce=False,
        checks=[pa.Check(_is_non_empty_str, error="identifier must be non-empty")],
    )


def _finite_float_column(*extra_checks: pa.Check) -> pa.Column:
    return pa.Column(
        np.float64,
        nullable=False,
        coerce=False,
        checks=[pa.Check(_is_finite_float, error="float values must be finite"), *extra_checks],
    )


RAW_CUSTOMER_PROFILES_SCHEMA = pa.DataFrameSchema(
    columns={
        "customer_id": _non_empty_id_column(),
        "event_timestamp": _utc_timestamp_column(),
        "age_years": pa.Column(np.int64, nullable=False, checks=pa.Check.in_range(18, 120)),
        "income_annual": _finite_float_column(pa.Check.gt(0.0)),
        "employment_years": _finite_float_column(pa.Check.in_range(0.0, 80.0)),
        "home_ownership": pa.Column(
            str,
            nullable=False,
            checks=pa.Check.isin(HOME_OWNERSHIP_VALUES),
        ),
        "tenure_months": pa.Column(np.int64, nullable=False, checks=pa.Check.ge(0)),
        "num_active_cards": pa.Column(np.int64, nullable=False, checks=pa.Check.ge(1)),
    },
    strict=True,
    coerce=False,
)

RAW_CARD_EVENTS_SCHEMA = pa.DataFrameSchema(
    columns={
        "event_id": _non_empty_id_column(),
        "cc_num": _non_empty_id_column(),
        "customer_id": _non_empty_id_column(),
        "event_timestamp": _utc_timestamp_column(),
        "amount": _finite_float_column(pa.Check.gt(0.0)),
        "latitude": _finite_float_column(pa.Check.in_range(-90.0, 90.0)),
        "longitude": _finite_float_column(pa.Check.in_range(-180.0, 180.0)),
    },
    strict=True,
    coerce=False,
)

RAW_CREDIT_APPLICATIONS_SCHEMA = pa.DataFrameSchema(
    columns={
        "application_id": _non_empty_id_column(),
        "customer_id": _non_empty_id_column(),
        "event_timestamp": _utc_timestamp_column(),
        "loan_amount": _finite_float_column(pa.Check.gt(0.0)),
        "loan_term_months": pa.Column(np.int64, nullable=False, checks=pa.Check.in_range(6, 360)),
        "dti_ratio": _finite_float_column(pa.Check.ge(0.0)),
        "existing_delinquencies": pa.Column(np.int64, nullable=False, checks=pa.Check.ge(0)),
        "utilization_ratio": _finite_float_column(pa.Check.ge(0.0)),
        "pub_rec_flag": pa.Column(np.int64, nullable=False, checks=pa.Check.isin((0, 1))),
        "default_label": pa.Column(np.int64, nullable=False, checks=pa.Check.isin((0, 1))),
    },
    strict=True,
    coerce=False,
)


def _check_24h_le_7d(df: pd.DataFrame) -> pd.Series:
    return df["tx_count_24h"] <= df["tx_count_7d"]


CARD_WINDOW_AGGS_SCHEMA = pa.DataFrameSchema(
    columns={
        "cc_num": _non_empty_id_column(),
        "customer_id": _non_empty_id_column(),
        "event_timestamp": _utc_timestamp_column(),
        "tx_amount_mavg_7d": _finite_float_column(pa.Check.ge(0.0)),
        "tx_amount_mstd_7d": _finite_float_column(pa.Check.ge(0.0)),
        "tx_count_24h": pa.Column(np.int64, nullable=False, checks=pa.Check.ge(0)),
        "tx_count_7d": pa.Column(np.int64, nullable=False, checks=pa.Check.ge(0)),
        "loc_delta_mavg_7d": _finite_float_column(pa.Check.ge(0.0)),
    },
    checks=[pa.Check(_check_24h_le_7d, error="tx_count_24h must not exceed tx_count_7d")],
    strict=True,
    coerce=False,
)


def _validate_with_schema(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    dataset_name: str,
) -> pd.DataFrame:
    try:
        return schema.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise SchemaValidationError(dataset_name, str(exc)) from exc


def validate_customer_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw customer_profiles DataFrame fail-fast against its Pandera schema."""
    return _validate_with_schema(df, RAW_CUSTOMER_PROFILES_SCHEMA, "customer_profiles")


def validate_card_events(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw card_events DataFrame fail-fast against its Pandera schema."""
    return _validate_with_schema(df, RAW_CARD_EVENTS_SCHEMA, "card_events")


def validate_credit_applications(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw credit_applications DataFrame fail-fast against its Pandera schema."""
    return _validate_with_schema(df, RAW_CREDIT_APPLICATIONS_SCHEMA, "credit_applications")


def validate_card_window_aggs(df: pd.DataFrame) -> pd.DataFrame:
    """Validate computed card_window_aggs DataFrame against its Pandera schema."""
    return _validate_with_schema(df, CARD_WINDOW_AGGS_SCHEMA, "card_window_aggs")


def validate_raw_batch(batch: RawBankingBatch) -> RawBankingBatch:
    """Validate all raw DataFrames in a RawBankingBatch before computation or persistence."""
    validated_profiles = validate_customer_profiles(batch.customer_profiles)
    validated_events = validate_card_events(batch.card_events)
    validated_apps = validate_credit_applications(batch.credit_applications)
    return RawBankingBatch(
        customer_profiles=validated_profiles,
        card_events=validated_events,
        credit_applications=validated_apps,
        batch_id=batch.batch_id,
        simulate_drift=batch.simulate_drift,
    )
