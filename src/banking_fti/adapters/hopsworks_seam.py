"""Documented Hopsworks Feature Store (`hsfs`) swap-in seam for the FeatureStore port.

This module intentionally requires zero Hopsworks SDK dependency or cloud credentials at
Stage 1. It defines the exact configuration surface and `FeatureStore` Protocol mapping so an
enterprise deployment can replace `FeastFeatureStore` with Hopsworks without modifying the
Feature, Training, or Inference pipelines:

Protocol-to-Hopsworks (`hsfs`) Mapping
--------------------------------------
- ``write_feature_group(group_name, df, *, schema_hash)``:
  1. Look up ``contract = FEATURE_GROUP_CONTRACTS[group_name]`` and validate columns via
     ``prepare_write_frame(group_name, df)``.
  2. Obtain ``fg = fs.get_or_create_feature_group(name=group_name, version=1,
     primary_key=list(contract.entity_keys), event_time=contract.timestamp_col,
     online_enabled=False)``.
  3. Enforce schema compatibility via feature-group tag ``fg.get_tag("schema_hash")``; raise
     ``SchemaMismatchError`` on mismatch, or set ``fg.set_tag("schema_hash", schema_hash)``
     on first write.
  4. Persist unencoded rows via ``fg.insert(ordered_df, write_options={"wait_for_job": True})``.
- ``read_feature_group(group_name, *, start_time=None, end_time=None)``:
  1. Build a query ``query = fs.get_feature_group(group_name, version=1).select_all()``.
  2. Apply optional UTC bounds via ``query.filter(...)`` or read and apply
     ``filter_by_time_bounds(...)``.
  3. Execute ``query.read()`` returning an unencoded DataFrame.
- ``get_historical_features(entity_df, feature_groups, *, exclude_labels=False)``:
  1. Normalize ``entity_df`` UTC timestamps via ``normalize_entity_frame(entity_df)``.
  2. Build or fetch a Hopsworks ``FeatureView`` joining the requested ``feature_groups`` on
     each contract's resolved join key and ``event_timestamp`` (omitting
     ``contract.label_columns`` when ``exclude_labels=True``).
  3. Retrieve point-in-time historical snapshots via ``fv.get_batch_data(...)`` or
     ``fv.create_training_data(...)`` while preserving ``entity_df`` row order.
- ``get_schema_hash(group_name)``:
  Read the persisted ``schema_hash`` tag from ``fs.get_feature_group(group_name, version=1)``
  or return ``None`` if the Feature Group does not yet exist.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from banking_fti.core.contracts import FeatureGroupName
from banking_fti.ports.feature_store import WriteReceipt


@dataclass(frozen=True)
class HopsworksConfig:
    """Connection configuration for swapping in a Hopsworks Feature Store via `hsfs`."""

    project_name: str
    host: str = "c.app.hopsworks.ai"
    feature_store_name: str | None = None
    feature_group_version: int = 1
    api_key_env_var: str = "HOPSWORKS_API_KEY"


class HopsworksFeatureStore:
    """Documented enterprise Hopsworks swap-in adapter conforming to `FeatureStore`."""

    def __init__(self, config: HopsworksConfig) -> None:
        self._config = config

    @property
    def config(self) -> HopsworksConfig:
        """Return Hopsworks connection configuration."""
        return self._config

    def _not_configured_error(self, operation: str) -> NotImplementedError:
        return NotImplementedError(
            f"HopsworksFeatureStore.{operation} is a documented swap-in seam for `hsfs`. "
            f"Install `hopsworks`/`hsfs`, provide `{self._config.api_key_env_var}`, and bind "
            f"project `{self._config.project_name}` at `{self._config.host}`."
        )

    def get_schema_hash(self, group_name: FeatureGroupName) -> str | None:
        """Return `None` until bound to a live Hopsworks Feature Group tag."""
        _ = group_name
        return None

    def write_feature_group(
        self,
        group_name: FeatureGroupName,
        df: pd.DataFrame,
        *,
        schema_hash: str,
    ) -> WriteReceipt:
        """Swap-in seam for `hsfs` `FeatureGroup.insert` with `schema_hash` tag enforcement."""
        _ = (group_name, df, schema_hash)
        raise self._not_configured_error("write_feature_group")

    def read_feature_group(
        self,
        group_name: FeatureGroupName,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Swap-in seam for `hsfs` `FeatureGroup.select_all().read()`."""
        _ = (group_name, start_time, end_time)
        raise self._not_configured_error("read_feature_group")

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_groups: Sequence[FeatureGroupName],
        *,
        exclude_labels: bool = False,
    ) -> pd.DataFrame:
        """Swap-in seam for `hsfs` `FeatureView.get_batch_data` / `create_training_data`."""
        _ = (entity_df, feature_groups, exclude_labels)
        raise self._not_configured_error("get_historical_features")
