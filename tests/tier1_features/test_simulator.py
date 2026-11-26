"""Tier 1 tests for the deterministic Banking Event Simulator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from banking_fti.core.contracts import HOME_OWNERSHIP_VALUES
from banking_fti.core.validation import validate_raw_batch
from banking_fti.features.simulator import BankingEventSimulator, SimulatorConfig


def test_simulator_is_deterministic_across_instances_for_same_seed() -> None:
    config = SimulatorConfig(num_customers=30, backfill_days=90, seed=1234)
    end_time = datetime(2026, 1, 15, 0, 0, tzinfo=UTC)

    sim_a = BankingEventSimulator(config)
    sim_b = BankingEventSimulator(config)

    backfill_a = sim_a.generate_backfill(end_time=end_time)
    backfill_b = sim_b.generate_backfill(end_time=end_time)

    assert backfill_a.batch_id == backfill_b.batch_id
    pd.testing.assert_frame_equal(
        backfill_a.customer_profiles, backfill_b.customer_profiles, check_exact=True
    )
    pd.testing.assert_frame_equal(backfill_a.card_events, backfill_b.card_events, check_exact=True)
    pd.testing.assert_frame_equal(
        backfill_a.credit_applications, backfill_b.credit_applications, check_exact=True
    )

    inc_a = sim_a.generate_incremental_batch(start_time=end_time, window_hours=24, batch_seq=1)
    inc_b = sim_b.generate_incremental_batch(start_time=end_time, window_hours=24, batch_seq=1)
    pd.testing.assert_frame_equal(inc_a.card_events, inc_b.card_events, check_exact=True)
    pd.testing.assert_frame_equal(
        inc_a.credit_applications, inc_b.credit_applications, check_exact=True
    )


def test_simulator_maintains_consistent_customer_to_card_identity_graph() -> None:
    sim = BankingEventSimulator(SimulatorConfig(num_customers=40, backfill_days=90, seed=7))
    end_time = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)

    backfill = sim.generate_backfill(end_time=end_time)
    incremental = sim.generate_incremental_batch(start_time=end_time, window_hours=48, batch_seq=2)

    known_customers = set(sim.customer_card_map.keys())
    known_cards = set(sim.card_to_customer.keys())
    assert len(known_customers) == 40
    assert len(known_cards) >= 40

    for batch in (backfill, incremental):
        validate_raw_batch(batch)
        assert set(batch.customer_profiles["customer_id"]).issubset(known_customers)
        assert set(batch.credit_applications["customer_id"]).issubset(known_customers)
        assert set(batch.card_events["cc_num"]).issubset(known_cards)

        for cc_num, customer_id in zip(
            batch.card_events["cc_num"].tolist(),
            batch.card_events["customer_id"].tolist(),
            strict=True,
        ):
            assert sim.card_to_customer[cc_num] == customer_id
            assert cc_num in sim.customer_card_map[customer_id]

        assert set(batch.customer_profiles["home_ownership"]).issubset(HOME_OWNERSHIP_VALUES)


def test_simulator_backfill_spans_90_days_and_incremental_respects_window() -> None:
    sim = BankingEventSimulator(SimulatorConfig(num_customers=35, backfill_days=90, seed=99))
    end_time = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)

    backfill = sim.generate_backfill(end_time=end_time)
    start_bound = pd.Timestamp(end_time - timedelta(days=90))
    end_bound = pd.Timestamp(end_time)

    assert backfill.card_events["event_timestamp"].min() >= start_bound
    assert backfill.card_events["event_timestamp"].max() <= end_bound
    assert backfill.credit_applications["event_timestamp"].min() >= start_bound
    assert backfill.credit_applications["event_timestamp"].max() <= end_bound
    span_days = (
        backfill.card_events["event_timestamp"].max()
        - backfill.card_events["event_timestamp"].min()
    ).days
    assert span_days >= 80

    inc = sim.generate_incremental_batch(start_time=end_time, window_hours=24, batch_seq=3)
    inc_end = pd.Timestamp(end_time + timedelta(hours=24))
    assert inc.card_events["event_timestamp"].min() > end_bound
    assert inc.card_events["event_timestamp"].max() <= inc_end
    assert inc.credit_applications["event_timestamp"].min() > end_bound
    assert inc.credit_applications["event_timestamp"].max() <= inc_end


def test_simulator_drift_mode_shifts_macro_dti_and_utilization_upward() -> None:
    sim = BankingEventSimulator(
        SimulatorConfig(
            num_customers=80,
            avg_applications_per_day=40.0,
            seed=42,
            drift_dti_shift=0.30,
            drift_utilization_shift=0.22,
        )
    )
    start_time = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)

    nominal = sim.generate_incremental_batch(
        start_time=start_time, window_hours=24, simulate_drift=False, batch_seq=1
    )
    drifted = sim.generate_incremental_batch(
        start_time=start_time, window_hours=24, simulate_drift=True, batch_seq=1
    )

    assert nominal.simulate_drift is False
    assert drifted.simulate_drift is True
    validate_raw_batch(nominal)
    validate_raw_batch(drifted)

    dti_delta = float(
        drifted.credit_applications["dti_ratio"].mean()
        - nominal.credit_applications["dti_ratio"].mean()
    )
    util_delta = float(
        drifted.credit_applications["utilization_ratio"].mean()
        - nominal.credit_applications["utilization_ratio"].mean()
    )
    assert dti_delta >= 0.20
    assert util_delta >= 0.15


def test_simulator_rejects_naive_timestamps() -> None:
    sim = BankingEventSimulator(SimulatorConfig(num_customers=10, seed=1))
    naive = datetime(2026, 1, 1, 0, 0)

    with pytest.raises(ValueError, match="timezone-aware"):
        sim.generate_backfill(end_time=naive)

    with pytest.raises(ValueError, match="timezone-aware"):
        sim.generate_incremental_batch(start_time=naive)
