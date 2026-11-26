"""In-memory FeatureStore adapter for offline Stage 1 tests and local runs."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from banking_fti.core.contracts import (
    EVENT_TIMESTAMP_COL,
    FEATURE_GROUP_CONTRACTS,
    FeatureGroupContract,
    FeatureGroupName,
)
from banking_fti.ports.feature_store import SchemaMismatchError, WriteReceipt


def _to_utc_timestamp(dt: datetime) -> pd.Timestamp:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"Timestamp filter must be timezone-aware, got {dt!r}.")
    return pd.Timestamp(dt.astimezone(UTC))


def _resolve_join_key(entity_columns: pd.Index, contract: FeatureGroupContract) -> str:
    for key in contract.entity_keys:
        if key in entity_columns:
            return key
    raise ValueError(
        f"entity_df is missing required join key from {contract.entity_keys} for '{contract.name}'."
    )


class InMemoryFeatureStore:
    """Thread-unsafe in-memory implementation of the FeatureStore Protocol."""

    def __init__(self) -> None:
        self._tables: dict[FeatureGroupName, pd.DataFrame] = {}
        self._schema_hashes: dict[FeatureGroupName, str] = {}

    def get_schema_hash(self, group_name: FeatureGroupName) -> str | None:
        """Return the registered schema_hash for `group_name`, or None if unwritten."""
        return self._schema_hashes.get(group_name)

    def write_feature_group(
        self,
        group_name: FeatureGroupName,
        df: pd.DataFrame,
        *,
        schema_hash: str,
    ) -> WriteReceipt:
        """Validate columns and schema_hash, then append defensive copies to memory."""
        contract = FEATURE_GROUP_CONTRACTS[group_name]
        expected_cols = list(contract.all_columns())
        if set(df.columns) != set(expected_cols):
            missing = sorted(set(expected_cols) - set(df.columns))
            extra = sorted(set(df.columns) - set(expected_cols))
            raise ValueError(
                f"Feature Group '{group_name}' columns mismatch: missing={missing}, extra={extra}."
            )

        existing_hash = self._schema_hashes.get(group_name)
        if existing_hash is not None and existing_hash != schema_hash:
            raise SchemaMismatchError(
                f"Schema hash mismatch for '{group_name}': "
                f"expected '{existing_hash}', got '{schema_hash}'."
            )

        ordered = df.loc[:, expected_cols].copy(deep=True)
        ordered[contract.timestamp_col] = pd.to_datetime(
            ordered[contract.timestamp_col], utc=True
        ).dt.as_unit("ns")

        if existing_hash is None:
            self._schema_hashes[group_name] = schema_hash
            self._tables[group_name] = ordered.reset_index(drop=True)
        else:
            self._tables[group_name] = pd.concat(
                [self._tables[group_name], ordered], ignore_index=True
            )

        return WriteReceipt(
            group_name=group_name,
            rows_written=len(ordered),
            total_rows=len(self._tables[group_name]),
            schema_hash=schema_hash,
        )

    def read_feature_group(
        self,
        group_name: FeatureGroupName,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Return a defensive copy of `group_name`, optionally filtered by UTC time bounds."""
        contract = FEATURE_GROUP_CONTRACTS[group_name]
        table = self._tables.get(group_name)
        if table is None:
            return contract.empty_frame()

        mask = pd.Series(True, index=table.index)
        if start_time is not None:
            mask &= table[contract.timestamp_col] >= _to_utc_timestamp(start_time)
        if end_time is not None:
            mask &= table[contract.timestamp_col] <= _to_utc_timestamp(end_time)

        return table.loc[mask].copy(deep=True).reset_index(drop=True)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_groups: Sequence[FeatureGroupName],
    ) -> pd.DataFrame:
        """Perform leak-free backward point-in-time joins preserving `entity_df` row order."""
        if EVENT_TIMESTAMP_COL not in entity_df.columns:
            raise ValueError(f"entity_df must contain '{EVENT_TIMESTAMP_COL}' column.")

        result = entity_df.copy(deep=True).reset_index(drop=True)
        result[EVENT_TIMESTAMP_COL] = pd.to_datetime(
            result[EVENT_TIMESTAMP_COL], utc=True
        ).dt.as_unit("ns")
        result["_entity_row_order"] = np.arange(len(result), dtype=np.int64)

        for group_name in feature_groups:
            contract = FEATURE_GROUP_CONTRACTS[group_name]
            join_key = _resolve_join_key(result.columns, contract)
            feature_table = self.read_feature_group(group_name)

            feature_cols = [
                col.name for col in contract.feature_columns if col.name not in result.columns
            ]
            if not feature_cols:
                continue

            right_cols = [join_key, EVENT_TIMESTAMP_COL, *feature_cols]
            right = feature_table.loc[:, right_cols].sort_values(
                [EVENT_TIMESTAMP_COL, join_key], kind="mergesort"
            )
            left = result.sort_values([EVENT_TIMESTAMP_COL, join_key], kind="mergesort")

            result = pd.merge_asof(
                left,
                right,
                on=EVENT_TIMESTAMP_COL,
                by=join_key,
                direction="backward",
                allow_exact_matches=True,
            )

        result = result.sort_values("_entity_row_order", kind="mergesort").drop(
            columns=["_entity_row_order"]
        )
        return result.reset_index(drop=True)
