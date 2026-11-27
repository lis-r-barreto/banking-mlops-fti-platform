"""Feature Pipeline (F): Pandera validation, causal aggregations, stats, and drift events."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd

from banking_fti.core.contracts import (
    FEATURE_GROUP_CONTRACTS,
    FeatureGroupName,
    RawBankingBatch,
)
from banking_fti.core.validation import (
    validate_card_events,
    validate_card_window_aggs,
    validate_raw_batch,
)
from banking_fti.features.causal_aggs import compute_card_window_aggs
from banking_fti.ports.feature_store import FeatureStore, WriteReceipt

_LOGGER = logging.getLogger(__name__)
_STD_FLOOR = 1e-6

DriftMetric = Literal["standardized_mean_shift", "total_variation_distance"]
PipelineEventType = Literal["new_features_ready", "drift_detected"]


@dataclass(frozen=True)
class NumericFeatureStats:
    """Summary distribution statistics for a numeric feature column."""

    count: int
    mean: float
    std: float
    min: float
    p50: float
    p95: float
    max: float


@dataclass(frozen=True)
class CategoricalFeatureStats:
    """Normalized category distribution for a string feature column."""

    count: int
    proportions: Mapping[str, float]


@dataclass(frozen=True)
class FeatureGroupStats:
    """Upstream feature distribution statistics emitted per Feature Group."""

    group_name: FeatureGroupName
    schema_hash: str
    row_count: int
    numeric: Mapping[str, NumericFeatureStats]
    categorical: Mapping[str, CategoricalFeatureStats]


@dataclass(frozen=True)
class FeatureDriftFinding:
    """Single upstream feature drift finding exceeding configured threshold."""

    group_name: FeatureGroupName
    feature_name: str
    metric: DriftMetric
    score: float
    threshold: float


@dataclass(frozen=True)
class FeaturePipelineEvent:
    """Structured event emitted by FeaturePipeline after batch ingestion."""

    event_type: PipelineEventType
    batch_id: str
    emitted_at: datetime
    schema_hashes: Mapping[FeatureGroupName, str]
    rows_written: Mapping[FeatureGroupName, int]
    drifted_features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize event into a JSON-compatible dictionary."""
        return {
            "event_type": self.event_type,
            "batch_id": self.batch_id,
            "emitted_at": self.emitted_at.astimezone(UTC).isoformat(),
            "schema_hashes": dict(self.schema_hashes),
            "rows_written": dict(self.rows_written),
            "drifted_features": list(self.drifted_features),
        }


@dataclass(frozen=True)
class FeaturePipelineResult:
    """End-to-end execution result of a FeaturePipeline.run_batch call."""

    batch_id: str
    receipts: Mapping[FeatureGroupName, WriteReceipt]
    group_stats: Mapping[FeatureGroupName, FeatureGroupStats]
    drift_findings: tuple[FeatureDriftFinding, ...]
    drift_detected: bool
    events: tuple[FeaturePipelineEvent, ...]


def compute_group_stats(
    group_name: FeatureGroupName,
    df: pd.DataFrame,
    *,
    schema_hash: str,
) -> FeatureGroupStats:
    """Compute upstream numeric and categorical distribution statistics for a Feature Group."""
    contract = FEATURE_GROUP_CONTRACTS[group_name]
    numeric_stats: dict[str, NumericFeatureStats] = {}
    categorical_stats: dict[str, CategoricalFeatureStats] = {}

    for col_spec in contract.feature_columns:
        if col_spec.dtype in {"int64", "float64"}:
            values = df[col_spec.name].to_numpy(dtype=np.float64, copy=False)
            if len(values) == 0:
                numeric_stats[col_spec.name] = NumericFeatureStats(
                    count=0,
                    mean=0.0,
                    std=0.0,
                    min=0.0,
                    p50=0.0,
                    p95=0.0,
                    max=0.0,
                )
            else:
                numeric_stats[col_spec.name] = NumericFeatureStats(
                    count=len(values),
                    mean=float(np.mean(values)),
                    std=float(np.std(values, ddof=0)),
                    min=float(np.min(values)),
                    p50=float(np.percentile(values, 50.0)),
                    p95=float(np.percentile(values, 95.0)),
                    max=float(np.max(values)),
                )
        else:
            counts = df[col_spec.name].astype(str).value_counts(normalize=True)
            proportions = {str(key): float(val) for key, val in counts.items()}
            categorical_stats[col_spec.name] = CategoricalFeatureStats(
                count=len(df),
                proportions=MappingProxyType(proportions),
            )

    return FeatureGroupStats(
        group_name=group_name,
        schema_hash=schema_hash,
        row_count=len(df),
        numeric=MappingProxyType(numeric_stats),
        categorical=MappingProxyType(categorical_stats),
    )


def detect_upstream_drift(
    current_stats: Mapping[FeatureGroupName, FeatureGroupStats],
    reference_stats: Mapping[FeatureGroupName, FeatureGroupStats],
    *,
    numeric_z_threshold: float = 0.75,
    categorical_tvd_threshold: float = 0.15,
) -> tuple[FeatureDriftFinding, ...]:
    """Compare current batch stats against reference stats and return drift findings."""
    findings: list[FeatureDriftFinding] = []

    for group_name, cur_group in current_stats.items():
        ref_group = reference_stats.get(group_name)
        if ref_group is None:
            continue

        for feat_name, cur_num in cur_group.numeric.items():
            ref_num = ref_group.numeric.get(feat_name)
            if ref_num is None or cur_num.count == 0 or ref_num.count == 0:
                continue
            score = abs(cur_num.mean - ref_num.mean) / max(ref_num.std, _STD_FLOOR)
            if score >= numeric_z_threshold:
                findings.append(
                    FeatureDriftFinding(
                        group_name=group_name,
                        feature_name=feat_name,
                        metric="standardized_mean_shift",
                        score=float(score),
                        threshold=numeric_z_threshold,
                    )
                )

        for feat_name, cur_cat in cur_group.categorical.items():
            ref_cat = ref_group.categorical.get(feat_name)
            if ref_cat is None or cur_cat.count == 0 or ref_cat.count == 0:
                continue
            all_keys = set(cur_cat.proportions.keys()) | set(ref_cat.proportions.keys())
            tvd = 0.5 * sum(
                abs(cur_cat.proportions.get(key, 0.0) - ref_cat.proportions.get(key, 0.0))
                for key in all_keys
            )
            if tvd >= categorical_tvd_threshold:
                findings.append(
                    FeatureDriftFinding(
                        group_name=group_name,
                        feature_name=feat_name,
                        metric="total_variation_distance",
                        score=float(tvd),
                        threshold=categorical_tvd_threshold,
                    )
                )

    return tuple(findings)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class FeaturePipeline:
    """Feature Pipeline (F) orchestrating validation, causal aggregation, persistence, and drift."""

    def __init__(
        self,
        feature_store: FeatureStore,
        *,
        numeric_z_threshold: float = 0.75,
        categorical_tvd_threshold: float = 0.15,
        event_sink: Callable[[FeaturePipelineEvent], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._feature_store = feature_store
        self._numeric_z_threshold = numeric_z_threshold
        self._categorical_tvd_threshold = categorical_tvd_threshold
        self._event_sink = event_sink
        self._clock = clock or _utc_now

    def _emit_event(self, event: FeaturePipelineEvent) -> None:
        _LOGGER.info("%s", json.dumps(event.to_dict(), sort_keys=True))
        if self._event_sink is not None:
            self._event_sink(event)

    def run_batch(
        self,
        raw_batch: RawBankingBatch,
        *,
        history_card_events: pd.DataFrame | None = None,
        reference_stats: Mapping[FeatureGroupName, FeatureGroupStats] | None = None,
    ) -> FeaturePipelineResult:
        """Validate batch fail-fast, compute causal aggs, persist groups, and emit events."""
        validated = validate_raw_batch(raw_batch)
        validated_history = (
            validate_card_events(history_card_events)
            if history_card_events is not None and not history_card_events.empty
            else None
        )

        raw_aggs = compute_card_window_aggs(
            validated.card_events,
            history_events=validated_history,
        )
        card_window_aggs = validate_card_window_aggs(raw_aggs)

        group_frames: dict[FeatureGroupName, pd.DataFrame] = {
            "customer_profiles": validated.customer_profiles,
            "card_window_aggs": card_window_aggs,
            "credit_applications": validated.credit_applications,
        }

        receipts: dict[FeatureGroupName, WriteReceipt] = {}
        stats: dict[FeatureGroupName, FeatureGroupStats] = {}
        schema_hashes: dict[FeatureGroupName, str] = {}
        rows_written: dict[FeatureGroupName, int] = {}

        for group_name, frame in group_frames.items():
            schema_hash = FEATURE_GROUP_CONTRACTS[group_name].schema_hash()
            receipt = self._feature_store.write_feature_group(
                group_name,
                frame,
                schema_hash=schema_hash,
            )
            receipts[group_name] = receipt
            schema_hashes[group_name] = schema_hash
            rows_written[group_name] = receipt.rows_written
            stats[group_name] = compute_group_stats(
                group_name,
                frame,
                schema_hash=schema_hash,
            )

        findings: tuple[FeatureDriftFinding, ...] = ()
        if reference_stats is not None:
            findings = detect_upstream_drift(
                stats,
                reference_stats,
                numeric_z_threshold=self._numeric_z_threshold,
                categorical_tvd_threshold=self._categorical_tvd_threshold,
            )

        emitted_at = self._clock()
        frozen_hashes = MappingProxyType(schema_hashes)
        frozen_rows = MappingProxyType(rows_written)

        events: list[FeaturePipelineEvent] = [
            FeaturePipelineEvent(
                event_type="new_features_ready",
                batch_id=validated.batch_id,
                emitted_at=emitted_at,
                schema_hashes=frozen_hashes,
                rows_written=frozen_rows,
                drifted_features=(),
            )
        ]
        if findings:
            events.append(
                FeaturePipelineEvent(
                    event_type="drift_detected",
                    batch_id=validated.batch_id,
                    emitted_at=emitted_at,
                    schema_hashes=frozen_hashes,
                    rows_written=frozen_rows,
                    drifted_features=tuple(
                        f"{finding.group_name}.{finding.feature_name}" for finding in findings
                    ),
                )
            )

        for event in events:
            self._emit_event(event)

        return FeaturePipelineResult(
            batch_id=validated.batch_id,
            receipts=MappingProxyType(receipts),
            group_stats=MappingProxyType(stats),
            drift_findings=findings,
            drift_detected=bool(findings),
            events=tuple(events),
        )


@dataclass(frozen=True)
class Stage1IngestionSummary:
    """Typed execution summary emitted by `run_stage1_ingestion` and the CLI."""

    repo_path: str
    data_dir: str
    data_source: str
    batches_run: int
    drift_detected: bool
    rows_written: Mapping[FeatureGroupName, int]
    total_rows: Mapping[FeatureGroupName, int]
    schema_hashes: Mapping[FeatureGroupName, str]

    def to_dict(self) -> dict[str, object]:
        """Serialize ingestion summary to a JSON-compatible dictionary."""
        return {
            "repo_path": self.repo_path,
            "data_dir": self.data_dir,
            "data_source": self.data_source,
            "batches_run": self.batches_run,
            "drift_detected": self.drift_detected,
            "rows_written": dict(self.rows_written),
            "total_rows": dict(self.total_rows),
            "schema_hashes": dict(self.schema_hashes),
        }


def run_stage1_ingestion(
    *,
    repo_path: str | Path,
    data_dir: str | Path | None = None,
    data_source: Literal["auto", "csv", "simulator"] = "auto",
    raw_dir: str | Path | None = None,
    num_customers: int = 120,
    backfill_days: int = 90,
    seed: int = 42,
    incremental_batches: int = 1,
    simulate_drift: bool = False,
) -> Stage1IngestionSummary:
    """Run deterministic Stage 1 backfill and incremental batches into FeastFeatureStore."""
    if incremental_batches < 0:
        raise ValueError("incremental_batches must be >= 0.")

    from banking_fti.adapters.feast_store import FeastFeatureStore
    from banking_fti.features.csv_loader import (
        generate_incremental_from_backfill,
        has_tutorial_csvs,
        load_tutorial_csv_backfill,
    )
    from banking_fti.features.simulator import BankingEventSimulator, SimulatorConfig

    resolved_raw_dir = Path(raw_dir) if raw_dir is not None else Path("data/raw")
    resolved_source: Literal["csv", "simulator"]
    if data_source == "auto":
        resolved_source = "csv" if has_tutorial_csvs(resolved_raw_dir) else "simulator"
    else:
        resolved_source = data_source

    store = FeastFeatureStore(repo_path=repo_path, data_dir=data_dir)
    pipeline = FeaturePipeline(feature_store=store)

    if resolved_source == "csv":
        backfill = load_tutorial_csv_backfill(resolved_raw_dir)
        simulator: BankingEventSimulator | None = None
    else:
        simulator = BankingEventSimulator(
            SimulatorConfig(
                num_customers=num_customers,
                backfill_days=backfill_days,
                seed=seed,
            )
        )
        end_time = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
        backfill = simulator.generate_backfill(end_time=end_time)

    backfill_result = pipeline.run_batch(backfill)

    rows_written: dict[FeatureGroupName, int] = {
        group_name: receipt.rows_written for group_name, receipt in backfill_result.receipts.items()
    }
    latest_result = backfill_result
    history_events = backfill.card_events
    latest_batch = backfill
    any_drift = backfill_result.drift_detected

    for seq in range(1, incremental_batches + 1):
        apply_drift = simulate_drift and seq == incremental_batches
        if simulator is None:
            inc_batch = generate_incremental_from_backfill(
                latest_batch,
                window_hours=24,
                simulate_drift=apply_drift,
                batch_seq=seq,
                seed=seed,
            )
        else:
            inc_start = datetime(2026, 2, 1, 0, 0, tzinfo=UTC) + timedelta(hours=24 * (seq - 1))
            inc_batch = simulator.generate_incremental_batch(
                start_time=inc_start,
                window_hours=24,
                simulate_drift=apply_drift,
                batch_seq=seq,
            )
        latest_result = pipeline.run_batch(
            inc_batch,
            history_card_events=history_events,
            reference_stats=backfill_result.group_stats,
        )
        history_events = pd.concat([history_events, inc_batch.card_events], ignore_index=True)
        latest_batch = inc_batch
        any_drift = any_drift or latest_result.drift_detected
        for group_name, receipt in latest_result.receipts.items():
            rows_written[group_name] += receipt.rows_written

    total_rows = {
        group_name: receipt.total_rows for group_name, receipt in latest_result.receipts.items()
    }
    schema_hashes = {
        group_name: receipt.schema_hash for group_name, receipt in latest_result.receipts.items()
    }
    return Stage1IngestionSummary(
        repo_path=str(store.repo_path),
        data_dir=str(store.data_dir),
        data_source=resolved_source,
        batches_run=1 + incremental_batches,
        drift_detected=any_drift,
        rows_written=MappingProxyType(rows_written),
        total_rows=MappingProxyType(total_rows),
        schema_hashes=MappingProxyType(schema_hashes),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 1 Feature Pipeline ingestion into Feast Offline Parquet."
    )
    parser.add_argument(
        "--repo-path",
        default=os.environ.get("FEAST_REPO_PATH", "feature_repo"),
        help="Path to the Feast repository directory.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("FEAST_DATA_DIR"),
        help="Optional directory for Offline Parquet files and Feast registry.",
    )
    parser.add_argument(
        "--data-source",
        choices=("auto", "csv", "simulator"),
        default="auto",
        help="Backfill source: tutorial CSVs, synthetic simulator, or auto-detect.",
    )
    parser.add_argument(
        "--raw-dir",
        default=os.environ.get("RAW_DATA_DIR", "data/raw"),
        help="Directory containing downloaded Hopsworks Tutorial CSV files.",
    )
    parser.add_argument("--num-customers", type=int, default=120)
    parser.add_argument("--backfill-days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--incremental-batches", type=int, default=1)
    parser.add_argument("--simulate-drift", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for containerized and local Stage 1 Feature Pipeline ingestion."""
    args = _build_arg_parser().parse_args(argv)
    summary = run_stage1_ingestion(
        repo_path=args.repo_path,
        data_dir=args.data_dir,
        data_source=args.data_source,
        raw_dir=args.raw_dir,
        num_customers=args.num_customers,
        backfill_days=args.backfill_days,
        seed=args.seed,
        incremental_batches=args.incremental_batches,
        simulate_drift=args.simulate_drift,
    )
    print(json.dumps(summary.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
