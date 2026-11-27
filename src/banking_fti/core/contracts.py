"""Canonical Feature Group contracts and deterministic schema hashing for Stage 1 PoC."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

FeatureGroupName = Literal["customer_profiles", "card_window_aggs", "credit_applications"]
ColumnDType = Literal["int64", "float64", "string"]

EVENT_TIMESTAMP_COL = "event_timestamp"
HOME_OWNERSHIP_VALUES: tuple[str, ...] = ("RENT", "MORTGAGE", "OWN", "OTHER")
RAW_CARD_EVENTS_COLUMNS: tuple[str, ...] = (
    "event_id",
    "cc_num",
    "customer_id",
    EVENT_TIMESTAMP_COL,
    "amount",
    "latitude",
    "longitude",
)


@dataclass(frozen=True)
class ColumnSpec:
    """Single feature column definition inside a canonical Feature Group contract."""

    name: str
    dtype: ColumnDType


_EMPTY_SERIES_BUILDERS: Mapping[ColumnDType, Callable[[], pd.Series]] = {
    "int64": lambda: pd.Series([], dtype=np.int64),
    "float64": lambda: pd.Series([], dtype=np.float64),
    "string": lambda: pd.Series([], dtype=str),
}


@dataclass(frozen=True)
class FeatureGroupContract:
    """Canonical schema contract for an unencoded Feature Group."""

    name: FeatureGroupName
    entity_keys: tuple[str, ...]
    timestamp_col: str
    feature_columns: tuple[ColumnSpec, ...]
    label_columns: tuple[str, ...] = ()

    def all_columns(self) -> tuple[str, ...]:
        """Return ordered entity key, timestamp, and feature column names."""
        return (
            *self.entity_keys,
            self.timestamp_col,
            *(col.name for col in self.feature_columns),
        )

    def schema_hash(self) -> str:
        """Compute deterministic SHA-256 hash of this contract's structural definition."""
        return compute_schema_hash(self)

    def empty_frame(self) -> pd.DataFrame:
        """Return an empty DataFrame matching all columns and canonical dtypes."""
        data: dict[str, pd.Series] = {key: pd.Series([], dtype=str) for key in self.entity_keys}
        data[self.timestamp_col] = pd.Series(pd.DatetimeIndex([], tz="UTC").as_unit("ns"))
        for col in self.feature_columns:
            data[col.name] = _EMPTY_SERIES_BUILDERS[col.dtype]()
        return pd.DataFrame(data).loc[:, list(self.all_columns())]


def compute_schema_hash(contract: FeatureGroupContract) -> str:
    """Compute a deterministic SHA-256 hex digest for a FeatureGroupContract."""
    payload = {
        "name": contract.name,
        "entity_keys": list(contract.entity_keys),
        "timestamp_col": contract.timestamp_col,
        "feature_columns": [[col.name, col.dtype] for col in contract.feature_columns],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def compute_combined_schema_hash(contracts: Sequence[FeatureGroupContract]) -> str:
    """Compute a deterministic SHA-256 hex digest across multiple contracts sorted by name."""
    ordered = sorted(contracts, key=lambda contract: contract.name)
    payload = [
        {"name": contract.name, "schema_hash": compute_schema_hash(contract)}
        for contract in ordered
    ]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


CUSTOMER_PROFILES_CONTRACT = FeatureGroupContract(
    name="customer_profiles",
    entity_keys=("customer_id",),
    timestamp_col=EVENT_TIMESTAMP_COL,
    feature_columns=(
        ColumnSpec("age_years", "int64"),
        ColumnSpec("income_annual", "float64"),
        ColumnSpec("employment_years", "float64"),
        ColumnSpec("home_ownership", "string"),
        ColumnSpec("tenure_months", "int64"),
        ColumnSpec("num_active_cards", "int64"),
    ),
)

CARD_WINDOW_AGGS_CONTRACT = FeatureGroupContract(
    name="card_window_aggs",
    entity_keys=("cc_num", "customer_id"),
    timestamp_col=EVENT_TIMESTAMP_COL,
    feature_columns=(
        ColumnSpec("tx_amount_mavg_7d", "float64"),
        ColumnSpec("tx_amount_mstd_7d", "float64"),
        ColumnSpec("tx_count_24h", "int64"),
        ColumnSpec("tx_count_7d", "int64"),
        ColumnSpec("loc_delta_mavg_7d", "float64"),
    ),
)

CREDIT_APPLICATIONS_CONTRACT = FeatureGroupContract(
    name="credit_applications",
    entity_keys=("application_id", "customer_id"),
    timestamp_col=EVENT_TIMESTAMP_COL,
    feature_columns=(
        ColumnSpec("loan_amount", "float64"),
        ColumnSpec("loan_term_months", "int64"),
        ColumnSpec("dti_ratio", "float64"),
        ColumnSpec("existing_delinquencies", "int64"),
        ColumnSpec("utilization_ratio", "float64"),
        ColumnSpec("pub_rec_flag", "int64"),
        ColumnSpec("default_label", "int64"),
    ),
    label_columns=("default_label",),
)

FEATURE_GROUP_CONTRACTS: Mapping[FeatureGroupName, FeatureGroupContract] = {
    "customer_profiles": CUSTOMER_PROFILES_CONTRACT,
    "card_window_aggs": CARD_WINDOW_AGGS_CONTRACT,
    "credit_applications": CREDIT_APPLICATIONS_CONTRACT,
}


@dataclass(frozen=True)
class RawBankingBatch:
    """Raw incoming banking batch before causal aggregation and Feature Store persistence."""

    customer_profiles: pd.DataFrame
    card_events: pd.DataFrame
    credit_applications: pd.DataFrame
    batch_id: str
    simulate_drift: bool = False
