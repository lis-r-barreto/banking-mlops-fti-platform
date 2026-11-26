"""FeatureStore port Protocol, receipts, and schema-compatibility errors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd

from banking_fti.core.contracts import FeatureGroupName


class SchemaMismatchError(ValueError):
    """Raised when a Feature Store write violates a registered Feature Group schema hash."""


@dataclass(frozen=True)
class WriteReceipt:
    """Confirmation metadata returned by FeatureStore.write_feature_group."""

    group_name: FeatureGroupName
    rows_written: int
    total_rows: int
    schema_hash: str


@runtime_checkable
class FeatureStore(Protocol):
    """Stateful Feature Store port decoupling Feature Pipeline from Training/Inference."""

    def write_feature_group(
        self,
        group_name: FeatureGroupName,
        df: pd.DataFrame,
        *,
        schema_hash: str,
    ) -> WriteReceipt:
        """Append unencoded feature rows for `group_name` and enforce `schema_hash`."""
        ...

    def read_feature_group(
        self,
        group_name: FeatureGroupName,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Read unencoded rows from `group_name`, optionally bounded by UTC timestamps."""
        ...

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_groups: Sequence[FeatureGroupName],
    ) -> pd.DataFrame:
        """Perform a point-in-time backward join across `feature_groups` for `entity_df`."""
        ...

    def get_schema_hash(self, group_name: FeatureGroupName) -> str | None:
        """Return the active schema_hash for `group_name`, or None if never written."""
        ...
