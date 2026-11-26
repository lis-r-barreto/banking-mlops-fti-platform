"""In-memory FeatureStore adapter for offline Stage 1 tests and local runs."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd

from banking_fti.adapters._common import (
    filter_by_time_bounds,
    normalize_entity_frame,
    prepare_write_frame,
    resolve_join_key,
    select_feature_columns,
    verify_schema_hash,
)
from banking_fti.core.contracts import (
    EVENT_TIMESTAMP_COL,
    FEATURE_GROUP_CONTRACTS,
    FeatureGroupName,
)
from banking_fti.ports.feature_store import WriteReceipt


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
        ordered = prepare_write_frame(group_name, df)
        existing_hash = self._schema_hashes.get(group_name)
        verify_schema_hash(group_name, existing_hash, schema_hash)

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

        return filter_by_time_bounds(
            table,
            contract.timestamp_col,
            start_time=start_time,
            end_time=end_time,
        )

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_groups: Sequence[FeatureGroupName],
        *,
        exclude_labels: bool = False,
    ) -> pd.DataFrame:
        """Perform leak-free backward point-in-time joins preserving `entity_df` row order."""
        result = normalize_entity_frame(entity_df)
        result["_entity_row_order"] = np.arange(len(result), dtype=np.int64)

        for group_name in feature_groups:
            contract = FEATURE_GROUP_CONTRACTS[group_name]
            join_key = resolve_join_key(result.columns, contract)
            feature_table = self.read_feature_group(group_name)

            feature_cols = select_feature_columns(
                contract,
                result.columns,
                exclude_labels=exclude_labels,
            )
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
