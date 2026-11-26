"""Shared validation, timestamp, and column-selection helpers for FeatureStore adapters."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from banking_fti.core.contracts import (
    EVENT_TIMESTAMP_COL,
    FEATURE_GROUP_CONTRACTS,
    FeatureGroupContract,
    FeatureGroupName,
)
from banking_fti.ports.feature_store import SchemaMismatchError


def to_utc_timestamp(dt: datetime) -> pd.Timestamp:
    """Convert a timezone-aware datetime to a UTC pandas Timestamp."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"Timestamp filter must be timezone-aware, got {dt!r}.")
    return pd.Timestamp(dt.astimezone(UTC))


def normalize_utc_ns(series: pd.Series) -> pd.Series:
    """Convert a timestamp Series to UTC `datetime64[ns, UTC]`."""
    return pd.to_datetime(series, utc=True).dt.as_unit("ns")


def verify_schema_hash(
    group_name: FeatureGroupName,
    existing_hash: str | None,
    schema_hash: str,
) -> None:
    """Raise `SchemaMismatchError` if an existing hash differs from `schema_hash`."""
    if existing_hash is not None and existing_hash != schema_hash:
        raise SchemaMismatchError(
            f"Schema hash mismatch for '{group_name}': "
            f"expected '{existing_hash}', got '{schema_hash}'."
        )


def prepare_write_frame(group_name: FeatureGroupName, df: pd.DataFrame) -> pd.DataFrame:
    """Validate exact columns against the contract and return a UTC-normalized defensive copy."""
    contract = FEATURE_GROUP_CONTRACTS[group_name]
    expected_cols = list(contract.all_columns())
    if set(df.columns) != set(expected_cols):
        missing = sorted(set(expected_cols) - set(df.columns))
        extra = sorted(set(df.columns) - set(expected_cols))
        raise ValueError(
            f"Feature Group '{group_name}' columns mismatch: missing={missing}, extra={extra}."
        )

    ordered = df.loc[:, expected_cols].copy(deep=True)
    ordered[contract.timestamp_col] = normalize_utc_ns(ordered[contract.timestamp_col])
    return ordered


def filter_by_time_bounds(
    df: pd.DataFrame,
    timestamp_col: str,
    *,
    start_time: datetime | None,
    end_time: datetime | None,
) -> pd.DataFrame:
    """Filter a Feature Group frame by optional inclusive UTC timestamp bounds."""
    mask = pd.Series(True, index=df.index)
    if start_time is not None:
        mask &= df[timestamp_col] >= to_utc_timestamp(start_time)
    if end_time is not None:
        mask &= df[timestamp_col] <= to_utc_timestamp(end_time)
    return df.loc[mask].copy(deep=True).reset_index(drop=True)


def normalize_entity_frame(entity_df: pd.DataFrame) -> pd.DataFrame:
    """Copy `entity_df` and ensure `event_timestamp` is present and normalized to UTC ns."""
    if EVENT_TIMESTAMP_COL not in entity_df.columns:
        raise ValueError(f"entity_df must contain '{EVENT_TIMESTAMP_COL}' column.")

    result = entity_df.copy(deep=True).reset_index(drop=True)
    result[EVENT_TIMESTAMP_COL] = normalize_utc_ns(result[EVENT_TIMESTAMP_COL])
    return result


def resolve_join_key(entity_columns: pd.Index, contract: FeatureGroupContract) -> str:
    """Select the primary join key present in `entity_columns` for `contract`."""
    for key in contract.entity_keys:
        if key in entity_columns:
            return key
    raise ValueError(
        f"entity_df is missing required join key from {contract.entity_keys} for '{contract.name}'."
    )


def select_feature_columns(
    contract: FeatureGroupContract,
    existing_columns: pd.Index,
    *,
    exclude_labels: bool,
) -> list[str]:
    """Return unjoined feature column names for `contract`, optionally excluding label columns."""
    excluded = set(contract.label_columns) if exclude_labels else set()
    return [
        col.name
        for col in contract.feature_columns
        if col.name not in existing_columns and col.name not in excluded
    ]
