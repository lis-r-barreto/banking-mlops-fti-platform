"""Tier 1 tests for Feature Pipeline (F), fail-fast gating, stats, and drift events."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from banking_fti.adapters.in_memory import InMemoryFeatureStore
from banking_fti.core.contracts import FEATURE_GROUP_CONTRACTS, RawBankingBatch
from banking_fti.core.validation import SchemaValidationError
from banking_fti.features.pipeline import FeaturePipeline, FeaturePipelineEvent
from banking_fti.features.simulator import BankingEventSimulator, SimulatorConfig


def test_feature_pipeline_backfill_persists_unencoded_features_stats_and_event() -> None:
    store = InMemoryFeatureStore()
    emitted: list[FeaturePipelineEvent] = []
    fixed_clock = datetime(2026, 1, 1, 0, 5, tzinfo=UTC)
    pipeline = FeaturePipeline(
        feature_store=store,
        event_sink=emitted.append,
        clock=lambda: fixed_clock,
    )

    sim = BankingEventSimulator(SimulatorConfig(num_customers=30, backfill_days=90, seed=42))
    backfill = sim.generate_backfill(end_time=datetime(2026, 1, 1, 0, 0, tzinfo=UTC))

    result = pipeline.run_batch(backfill)

    assert result.batch_id == backfill.batch_id
    assert result.drift_detected is False
    assert len(result.events) == 1
    assert result.events[0].event_type == "new_features_ready"
    assert result.events[0].emitted_at == fixed_clock
    assert len(emitted) == 1
    # Ensure structured event serializes cleanly to JSON
    serialized = json.dumps(emitted[0].to_dict())
    assert "new_features_ready" in serialized

    for group_name, contract in FEATURE_GROUP_CONTRACTS.items():
        expected_hash = contract.schema_hash()
        assert store.get_schema_hash(group_name) == expected_hash
        assert result.receipts[group_name].schema_hash == expected_hash
        assert result.group_stats[group_name].schema_hash == expected_hash
        stored = store.read_feature_group(group_name)
        assert not stored.empty
        assert tuple(stored.columns) == contract.all_columns()

    stored_profiles = store.read_feature_group("customer_profiles")
    assert set(stored_profiles["home_ownership"]).issubset({"RENT", "MORTGAGE", "OWN", "OTHER"})


def test_feature_pipeline_fails_fast_without_writing_any_rows_on_invalid_batch() -> None:
    store = InMemoryFeatureStore()
    emitted: list[FeaturePipelineEvent] = []
    pipeline = FeaturePipeline(feature_store=store, event_sink=emitted.append)

    sim = BankingEventSimulator(SimulatorConfig(num_customers=15, backfill_days=10, seed=7))
    batch = sim.generate_backfill(end_time=datetime(2026, 1, 1, 0, 0, tzinfo=UTC))

    corrupt_apps = batch.credit_applications.copy()
    corrupt_apps.loc[0, "dti_ratio"] = -0.50
    corrupt_batch = RawBankingBatch(
        customer_profiles=batch.customer_profiles,
        card_events=batch.card_events,
        credit_applications=corrupt_apps,
        batch_id="bad-batch",
    )

    with pytest.raises(SchemaValidationError) as exc_info:
        pipeline.run_batch(corrupt_batch)

    assert exc_info.value.dataset_name == "credit_applications"
    assert len(emitted) == 0
    for group_name in FEATURE_GROUP_CONTRACTS:
        assert store.get_schema_hash(group_name) is None
        assert store.read_feature_group(group_name).empty


def test_feature_pipeline_emits_drift_event_on_simulate_drift_and_stays_quiet_on_nominal() -> None:
    store = InMemoryFeatureStore()
    emitted: list[FeaturePipelineEvent] = []
    pipeline = FeaturePipeline(feature_store=store, event_sink=emitted.append)

    sim = BankingEventSimulator(
        SimulatorConfig(
            num_customers=60,
            backfill_days=30,
            avg_applications_per_day=30.0,
            seed=42,
        )
    )
    t0 = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    backfill = sim.generate_backfill(end_time=t0)
    ref_result = pipeline.run_batch(backfill)

    nominal_inc = sim.generate_incremental_batch(
        start_time=t0, window_hours=24, simulate_drift=False, batch_seq=1
    )
    nominal_result = pipeline.run_batch(
        nominal_inc,
        history_card_events=backfill.card_events,
        reference_stats=ref_result.group_stats,
    )
    assert nominal_result.drift_detected is False
    assert [e.event_type for e in nominal_result.events] == ["new_features_ready"]

    drifted_inc = sim.generate_incremental_batch(
        start_time=t0, window_hours=24, simulate_drift=True, batch_seq=2
    )
    drifted_result = pipeline.run_batch(
        drifted_inc,
        history_card_events=backfill.card_events,
        reference_stats=ref_result.group_stats,
    )
    assert drifted_result.drift_detected is True
    assert [e.event_type for e in drifted_result.events] == [
        "new_features_ready",
        "drift_detected",
    ]
    drift_event = drifted_result.events[1]
    assert "credit_applications.dti_ratio" in drift_event.drifted_features
    assert len(emitted) == 4
