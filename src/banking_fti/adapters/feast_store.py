"""Feast Offline Feature Store (Parquet) adapter for Stage 1 PoC."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast import FeatureStore as _FeastClient
from feast.repo_config import load_repo_config
from feast.types import Float64, Int64, String

from banking_fti.adapters._common import (
    filter_by_time_bounds,
    normalize_entity_frame,
    normalize_utc_ns,
    prepare_write_frame,
    resolve_join_key,
    select_feature_columns,
    verify_schema_hash,
)
from banking_fti.core.contracts import (
    CARD_WINDOW_AGGS_CONTRACT,
    CREDIT_APPLICATIONS_CONTRACT,
    CUSTOMER_PROFILES_CONTRACT,
    EVENT_TIMESTAMP_COL,
    FEATURE_GROUP_CONTRACTS,
    ColumnDType,
    FeatureGroupContract,
    FeatureGroupName,
)
from banking_fti.ports.feature_store import SchemaMismatchError, WriteReceipt

DEFAULT_FEAST_PROJECT = "banking_feature_store"
_SCHEMA_HASHES_FILENAME = "_schema_hashes.json"

_FEAST_DTYPES: Mapping[ColumnDType, Any] = {
    "int64": Int64,
    "float64": Float64,
    "string": String,
}

_ARROW_DTYPES: Mapping[ColumnDType, pa.DataType] = {
    "int64": pa.int64(),
    "float64": pa.float64(),
    "string": pa.string(),
}


@dataclass(frozen=True)
class FeastEntities:
    """Canonical Stage 1 PoC Feast entities."""

    customer: Entity
    credit_card: Entity
    loan_application: Entity

    def all(self) -> tuple[Entity, ...]:
        """Return all Stage 1 entities as a tuple."""
        return (self.customer, self.credit_card, self.loan_application)


@dataclass(frozen=True)
class FeastRepoObjects:
    """Entities, Parquet FileSources, FeatureViews, and join-key routing for Stage 1 PoC."""

    entities: FeastEntities
    sources: Mapping[FeatureGroupName, FileSource]
    feature_views: Mapping[str, FeatureView]
    view_lookup: Mapping[tuple[FeatureGroupName, str], str]

    def apply_objects(self) -> list[Any]:
        """Return all Feast objects to register via `FeatureStore.apply`."""
        return [*self.entities.all(), *self.feature_views.values()]


def build_feast_entities() -> FeastEntities:
    """Build Stage 1 PoC banking entities (`customer`, `credit_card`, `loan_application`)."""
    return FeastEntities(
        customer=Entity(
            name="customer",
            join_keys=["customer_id"],
            value_type=ValueType.STRING,
            description="Primary banking customer identifier.",
        ),
        credit_card=Entity(
            name="credit_card",
            join_keys=["cc_num"],
            value_type=ValueType.STRING,
            description="Credit card account identifier linked to a customer.",
        ),
        loan_application=Entity(
            name="loan_application",
            join_keys=["application_id"],
            value_type=ValueType.STRING,
            description="Credit risk loan application identifier.",
        ),
    )


def _contract_arrow_schema(contract: FeatureGroupContract) -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field(key, pa.string(), nullable=False) for key in contract.entity_keys
    ]
    fields.append(pa.field(contract.timestamp_col, pa.timestamp("ns", tz="UTC"), nullable=False))
    for col in contract.feature_columns:
        fields.append(pa.field(col.name, _ARROW_DTYPES[col.dtype], nullable=False))
    return pa.schema(fields)


def _atomic_write_bytes(target_path: Path, writer: Any) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target_path.parent, prefix=f".{target_path.name}.")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        os.replace(tmp_path, target_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _contract_fields(contract: FeatureGroupContract) -> list[Field]:
    return [
        Field(name=col.name, dtype=_FEAST_DTYPES[col.dtype]) for col in contract.feature_columns
    ]


def _make_feature_view(
    *,
    name: str,
    entities: list[Entity],
    contract: FeatureGroupContract,
    source: FileSource,
    extra_tags: Mapping[str, str] | None = None,
) -> FeatureView:
    tags = {
        "schema_hash": contract.schema_hash(),
        "unencoded": "true",
        "stage": "poc1",
    }
    if extra_tags is not None:
        tags.update(extra_tags)
    return FeatureView(
        name=name,
        entities=entities,
        ttl=None,
        schema=_contract_fields(contract),
        online=False,
        offline=True,
        source=source,
        tags=tags,
    )


def build_feast_objects(data_dir: Path) -> FeastRepoObjects:
    """Build canonical Stage 1 Feast FileSources and unencoded FeatureViews for `data_dir`."""
    resolved_dir = data_dir.resolve()
    entities = build_feast_entities()

    sources: dict[FeatureGroupName, FileSource] = {
        "customer_profiles": FileSource(
            name="customer_profiles_source",
            path=str(resolved_dir / "customer_profiles.parquet"),
            timestamp_field=EVENT_TIMESTAMP_COL,
            description="Unencoded customer demographic and tenure snapshots.",
        ),
        "card_window_aggs": FileSource(
            name="card_window_aggs_source",
            path=str(resolved_dir / "card_window_aggs.parquet"),
            timestamp_field=EVENT_TIMESTAMP_COL,
            description="Strictly causal t - 1 rolling card-window aggregations.",
        ),
        "credit_applications": FileSource(
            name="credit_applications_source",
            path=str(resolved_dir / "credit_applications.parquet"),
            timestamp_field=EVENT_TIMESTAMP_COL,
            description="Unencoded loan application features and default labels.",
        ),
    }

    feature_views: dict[str, FeatureView] = {
        "customer_profiles": _make_feature_view(
            name="customer_profiles",
            entities=[entities.customer],
            contract=CUSTOMER_PROFILES_CONTRACT,
            source=sources["customer_profiles"],
        ),
        "card_window_aggs": _make_feature_view(
            name="card_window_aggs",
            entities=[entities.credit_card],
            contract=CARD_WINDOW_AGGS_CONTRACT,
            source=sources["card_window_aggs"],
        ),
        "card_window_aggs_by_customer": _make_feature_view(
            name="card_window_aggs_by_customer",
            entities=[entities.customer],
            contract=CARD_WINDOW_AGGS_CONTRACT,
            source=sources["card_window_aggs"],
            extra_tags={"source_group": "card_window_aggs"},
        ),
        "credit_applications": _make_feature_view(
            name="credit_applications",
            entities=[entities.loan_application],
            contract=CREDIT_APPLICATIONS_CONTRACT,
            source=sources["credit_applications"],
        ),
        "credit_applications_by_customer": _make_feature_view(
            name="credit_applications_by_customer",
            entities=[entities.customer],
            contract=CREDIT_APPLICATIONS_CONTRACT,
            source=sources["credit_applications"],
            extra_tags={"source_group": "credit_applications"},
        ),
    }
    view_lookup: dict[tuple[FeatureGroupName, str], str] = {
        ("customer_profiles", "customer_id"): "customer_profiles",
        ("card_window_aggs", "cc_num"): "card_window_aggs",
        ("card_window_aggs", "customer_id"): "card_window_aggs_by_customer",
        ("credit_applications", "application_id"): "credit_applications",
        ("credit_applications", "customer_id"): "credit_applications_by_customer",
    }
    return FeastRepoObjects(
        entities=entities,
        sources=sources,
        feature_views=feature_views,
        view_lookup=view_lookup,
    )


class FeastFeatureStore:
    """Concrete Feast Offline Parquet adapter implementing the FeatureStore Protocol."""

    def __init__(
        self,
        repo_path: str | Path,
        *,
        data_dir: str | Path | None = None,
        project: str = DEFAULT_FEAST_PROJECT,
        apply_on_init: bool = True,
    ) -> None:
        self._repo_path = Path(repo_path).resolve()
        self._data_dir = (
            Path(data_dir).resolve() if data_dir is not None else self._repo_path / "data"
        )
        self._repo_path.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        yaml_path = self._repo_path / "feature_store.yaml"
        if not yaml_path.exists():
            registry_path = (self._data_dir / "registry.db").resolve()
            yaml_path.write_text(
                "\n".join(
                    [
                        f"project: {project}",
                        f"registry: {registry_path}",
                        "provider: local",
                        "online_store: null",
                        "offline_store:",
                        "  type: file",
                        "entity_key_serialization_version: 3",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

        self._objects = build_feast_objects(self._data_dir)
        repo_config = load_repo_config(self._repo_path, yaml_path)
        if data_dir is not None:
            repo_config.registry_config = str((self._data_dir / "registry.db").resolve())
        self._feast_store = _FeastClient(config=repo_config)
        if apply_on_init:
            self.apply()

    @property
    def repo_path(self) -> Path:
        """Return the Feast repository directory path."""
        return self._repo_path

    @property
    def data_dir(self) -> Path:
        """Return the directory storing Offline Parquet files and schema metadata."""
        return self._data_dir

    def apply(self) -> None:
        """Apply Stage 1 PoC Feast entities and FeatureViews to the local registry."""
        self._feast_store.apply(self._objects.apply_objects())

    def list_entities(self) -> list[Entity]:
        """Return entities currently registered in the Feast registry."""
        return list(self._feast_store.list_entities())

    def list_feature_views(self) -> list[FeatureView]:
        """Return FeatureViews currently registered in the Feast registry."""
        return list(self._feast_store.list_feature_views())

    def _schema_hashes_path(self) -> Path:
        return self._data_dir / _SCHEMA_HASHES_FILENAME

    def _load_schema_hashes(self) -> dict[str, str]:
        path = self._schema_hashes_path()
        if not path.exists():
            return {}
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Corrupt schema hash metadata in {path}.")
        return {str(key): str(val) for key, val in loaded.items()}

    def _save_schema_hashes(self, hashes: Mapping[str, str]) -> None:
        payload = json.dumps(dict(sorted(hashes.items())), indent=2, sort_keys=True) + "\n"
        _atomic_write_bytes(
            self._schema_hashes_path(),
            lambda tmp_path: tmp_path.write_text(payload, encoding="utf-8"),
        )

    def _parquet_path(self, group_name: FeatureGroupName) -> Path:
        return self._data_dir / f"{group_name}.parquet"

    def get_schema_hash(self, group_name: FeatureGroupName) -> str | None:
        """Return the persisted schema_hash for `group_name`, or None if unwritten."""
        return self._load_schema_hashes().get(group_name)

    def write_feature_group(
        self,
        group_name: FeatureGroupName,
        df: pd.DataFrame,
        *,
        schema_hash: str,
    ) -> WriteReceipt:
        """Append unencoded rows to Offline Parquet and enforce `schema_hash`."""
        contract = FEATURE_GROUP_CONTRACTS[group_name]
        ordered = prepare_write_frame(group_name, df)

        hashes = self._load_schema_hashes()
        existing_hash = hashes.get(group_name)
        verify_schema_hash(group_name, existing_hash, schema_hash)

        arrow_schema = _contract_arrow_schema(contract)
        new_table = pa.Table.from_pandas(ordered, schema=arrow_schema, preserve_index=False)
        parquet_path = self._parquet_path(group_name)

        if parquet_path.exists():
            existing_table = pq.read_table(parquet_path)
            if not existing_table.schema.equals(arrow_schema, check_metadata=False):
                raise SchemaMismatchError(
                    f"Existing Parquet schema for '{group_name}' at {parquet_path} "
                    "does not match canonical FeatureGroupContract."
                )
            combined_table = pa.concat_tables([existing_table, new_table])
        else:
            combined_table = new_table

        _atomic_write_bytes(
            parquet_path,
            lambda tmp_path: pq.write_table(combined_table, tmp_path),
        )
        if existing_hash is None:
            hashes[group_name] = schema_hash
            self._save_schema_hashes(hashes)

        return WriteReceipt(
            group_name=group_name,
            rows_written=len(ordered),
            total_rows=int(combined_table.num_rows),
            schema_hash=schema_hash,
        )

    def read_feature_group(
        self,
        group_name: FeatureGroupName,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Read unencoded rows from Offline Parquet, optionally bounded by UTC timestamps."""
        contract = FEATURE_GROUP_CONTRACTS[group_name]
        parquet_path = self._parquet_path(group_name)
        if not parquet_path.exists():
            return contract.empty_frame()

        frame = cast(pd.DataFrame, pq.read_table(parquet_path).to_pandas())
        frame = frame.loc[:, list(contract.all_columns())]
        frame[contract.timestamp_col] = normalize_utc_ns(frame[contract.timestamp_col])
        return filter_by_time_bounds(
            frame,
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
        """Perform leak-free Feast point-in-time joins across `feature_groups`."""
        base_df = normalize_entity_frame(entity_df)
        seen_cols = pd.Index(list(base_df.columns))
        feature_refs: list[str] = []
        added_cols: list[str] = []
        unwritten_cols: list[str] = []

        for group_name in feature_groups:
            contract = FEATURE_GROUP_CONTRACTS[group_name]
            join_key = resolve_join_key(seen_cols, contract)
            feature_cols = select_feature_columns(
                contract,
                seen_cols,
                exclude_labels=exclude_labels,
            )
            if not feature_cols:
                continue

            seen_cols = seen_cols.append(pd.Index(feature_cols))
            added_cols.extend(feature_cols)

            if not self._parquet_path(group_name).exists():
                unwritten_cols.extend(feature_cols)
                continue

            view_name = self._objects.view_lookup[(group_name, join_key)]
            feature_refs.extend(f"{view_name}:{col}" for col in feature_cols)

        if base_df.empty:
            empty_result = base_df.copy()
            for col in added_cols:
                empty_result[col] = pd.Series(dtype=object)
            return empty_result.loc[:, [*base_df.columns, *added_cols]]

        if feature_refs:
            row_order_col = "_banking_fti_row_order"
            feast_entity_df = base_df.copy()
            feast_entity_df[row_order_col] = np.arange(len(feast_entity_df), dtype=np.int64)
            joined = self._feast_store.get_historical_features(
                entity_df=feast_entity_df,
                features=feature_refs,
            ).to_df()
            joined = (
                joined.sort_values(row_order_col, kind="mergesort")
                .drop(columns=[row_order_col])
                .reset_index(drop=True)
            )
            joined[EVENT_TIMESTAMP_COL] = normalize_utc_ns(joined[EVENT_TIMESTAMP_COL])
        else:
            joined = base_df.copy()

        for col in unwritten_cols:
            if col not in joined.columns:
                joined[col] = np.nan

        ordered_cols = [*base_df.columns, *added_cols]
        return joined.loc[:, ordered_cols]
