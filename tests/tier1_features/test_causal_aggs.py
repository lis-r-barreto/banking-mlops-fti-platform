"""Tier 1 tests for strictly causal t - 1 rolling card-window aggregations."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from banking_fti.core.contracts import CARD_WINDOW_AGGS_CONTRACT
from banking_fti.core.validation import validate_card_window_aggs
from banking_fti.features.causal_aggs import compute_card_window_aggs
from banking_fti.features.simulator import BankingEventSimulator, SimulatorConfig


def _make_events_df(rows: list[tuple[str, str, str, str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [r[0] for r in rows],
            "cc_num": [r[1] for r in rows],
            "customer_id": [r[2] for r in rows],
            "event_timestamp": pd.to_datetime([r[3] for r in rows], utc=True).astype(
                "datetime64[ns, UTC]"
            ),
            "amount": np.array([r[4] for r in rows], dtype=np.float64),
            "latitude": np.array([r[5] for r in rows], dtype=np.float64),
            "longitude": np.array([r[6] for r in rows], dtype=np.float64),
        }
    )


def test_causal_aggs_match_hand_worked_24h_and_7d_windows_and_simultaneous_events() -> None:
    events = _make_events_df(
        [
            ("E0", "CARD-1", "CUST-1", "2026-01-01T00:00:00Z", 100.0, 0.0, 0.0),
            ("E1", "CARD-1", "CUST-1", "2026-01-01T12:00:00Z", 200.0, 0.0, 1.0),
            ("E2", "CARD-1", "CUST-1", "2026-01-03T00:00:00Z", 300.0, 0.0, 1.0),
            ("E2B", "CARD-1", "CUST-1", "2026-01-03T00:00:00Z", 999.0, 10.0, 10.0),
            ("E3", "CARD-1", "CUST-1", "2026-01-09T00:00:00Z", 400.0, 0.0, 1.0),
        ]
    )

    aggs = compute_card_window_aggs(events)
    validate_card_window_aggs(aggs)
    assert tuple(aggs.columns) == CARD_WINDOW_AGGS_CONTRACT.all_columns()

    # Row 0 (cold start): no prior events
    assert aggs["tx_count_24h"].tolist()[0] == 0
    assert aggs["tx_count_7d"].tolist()[0] == 0
    assert aggs["tx_amount_mavg_7d"].tolist()[0] == 0.0
    assert aggs["tx_amount_mstd_7d"].tolist()[0] == 0.0
    assert aggs["loc_delta_mavg_7d"].tolist()[0] == 0.0

    # Row 1 (+12h): sees only E0
    assert aggs["tx_count_24h"].tolist()[1] == 1
    assert aggs["tx_count_7d"].tolist()[1] == 1
    assert aggs["tx_amount_mavg_7d"].tolist()[1] == pytest.approx(100.0)
    assert aggs["tx_amount_mstd_7d"].tolist()[1] == pytest.approx(0.0)
    assert aggs["loc_delta_mavg_7d"].tolist()[1] == pytest.approx(0.0)

    # Rows 2 and 3 (simultaneous at +48h): neither sees the other; both see E0 + E1
    for row_idx in (2, 3):
        assert aggs["tx_count_24h"].tolist()[row_idx] == 0
        assert aggs["tx_count_7d"].tolist()[row_idx] == 2
        assert aggs["tx_amount_mavg_7d"].tolist()[row_idx] == pytest.approx(150.0)
        assert aggs["tx_amount_mstd_7d"].tolist()[row_idx] == pytest.approx(50.0)
        # E0 loc_delta = 0.0, E1 loc_delta = 1 deg lon at equator (~111.195 km) -> mean ~55.5975 km
        assert aggs["loc_delta_mavg_7d"].tolist()[row_idx] == pytest.approx(55.5975, abs=0.05)

    # Row 4 (+8d): E0 & E1 expired beyond 7d; only E2 & E2B remain in 7d window
    assert aggs["tx_count_24h"].tolist()[4] == 0
    assert aggs["tx_count_7d"].tolist()[4] == 2
    assert aggs["tx_amount_mavg_7d"].tolist()[4] == pytest.approx(649.5)


def test_mutating_future_t_plus_1_rows_leaves_t_features_byte_for_byte_identical() -> None:
    sim = BankingEventSimulator(SimulatorConfig(num_customers=25, backfill_days=30, seed=11))
    batch = sim.generate_backfill(end_time=datetime(2026, 2, 1, 0, 0, tzinfo=UTC))
    events = batch.card_events.sort_values(["event_timestamp", "event_id"]).reset_index(drop=True)

    cutoff_idx = len(events) // 2
    cutoff_ts = events["event_timestamp"].iloc[cutoff_idx]

    baseline_aggs = compute_card_window_aggs(events)

    mutated_events = events.copy()
    future_mask = mutated_events["event_timestamp"] > cutoff_ts
    assert bool(future_mask.any())
    mutated_events.loc[future_mask, "amount"] = 99999.99
    mutated_events.loc[future_mask, "latitude"] = -45.0
    mutated_events.loc[future_mask, "longitude"] = 120.0

    extra_future = _make_events_df(
        [
            (
                "EXTRA-FUTURE-1",
                str(events["cc_num"].iloc[0]),
                str(events["customer_id"].iloc[0]),
                "2026-02-02T00:00:00Z",
                50000.0,
                -30.0,
                150.0,
            )
        ]
    )
    extended_mutated = pd.concat([mutated_events, extra_future], ignore_index=True)
    mutated_aggs = compute_card_window_aggs(extended_mutated)

    past_mask_base = events["event_timestamp"] <= cutoff_ts
    past_mask_mut = extended_mutated["event_timestamp"] <= cutoff_ts

    pd.testing.assert_frame_equal(
        baseline_aggs.loc[past_mask_base].reset_index(drop=True),
        mutated_aggs.loc[past_mask_mut].reset_index(drop=True),
        check_exact=True,
    )


def test_incremental_history_events_matches_full_sequence_computation() -> None:
    events = _make_events_df(
        [
            ("E0", "CARD-A", "CUST-A", "2026-01-01T00:00:00Z", 50.0, 35.0, -80.0),
            ("E1", "CARD-A", "CUST-A", "2026-01-02T06:00:00Z", 75.0, 35.5, -80.2),
            ("E2", "CARD-A", "CUST-A", "2026-01-02T18:00:00Z", 125.0, 36.0, -79.8),
            ("E3", "CARD-A", "CUST-A", "2026-01-03T10:00:00Z", 250.0, 36.1, -79.9),
        ]
    )
    full_aggs = compute_card_window_aggs(events)

    history = events.iloc[:2].copy()
    new_batch = events.iloc[2:].copy()
    inc_aggs = compute_card_window_aggs(new_batch, history_events=history)

    pd.testing.assert_frame_equal(
        full_aggs.iloc[2:].reset_index(drop=True),
        inc_aggs.reset_index(drop=True),
        check_exact=True,
    )


def test_causal_aggs_handles_empty_input_and_emits_no_nan_or_inf() -> None:
    empty = _make_events_df([])
    empty_aggs = compute_card_window_aggs(empty)
    assert empty_aggs.empty
    assert tuple(empty_aggs.columns) == CARD_WINDOW_AGGS_CONTRACT.all_columns()

    shuffled = _make_events_df(
        [
            ("E2", "CARD-Z", "CUST-Z", "2026-01-05T10:00:00Z", 80.0, 10.0, 10.0),
            ("E1", "CARD-Z", "CUST-Z", "2026-01-04T10:00:00Z", 40.0, 10.1, 10.1),
        ]
    )
    aggs = compute_card_window_aggs(shuffled)
    validate_card_window_aggs(aggs)
    # Row 0 (E2) has prior E1; Row 1 (E1) is cold start -> preserves input row alignment
    assert aggs["tx_count_7d"].tolist() == [1, 0]
    assert np.isfinite(aggs["tx_amount_mavg_7d"].to_numpy()).all()
    assert np.isfinite(aggs["tx_amount_mstd_7d"].to_numpy()).all()
    assert np.isfinite(aggs["loc_delta_mavg_7d"].to_numpy()).all()
