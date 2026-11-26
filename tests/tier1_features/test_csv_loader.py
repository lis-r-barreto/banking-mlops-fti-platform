"""Tier 1 offline tests for Hopsworks Tutorial CSV backfill loader and incremental generator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from banking_fti.adapters.in_memory import InMemoryFeatureStore
from banking_fti.core.contracts import (
    CUSTOMER_PROFILES_CONTRACT,
    HOME_OWNERSHIP_VALUES,
    RAW_CARD_EVENTS_COLUMNS,
)
from banking_fti.core.validation import validate_raw_batch
from banking_fti.features.csv_loader import (
    generate_incremental_from_backfill,
    has_tutorial_csvs,
    load_tutorial_csv_backfill,
)
from banking_fti.features.pipeline import FeaturePipeline


def _write_mini_tutorial_csvs(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    profiles = pd.DataFrame(
        {
            "name": ["Alice Smith", "Bob Jones", "Carol Lee", "Dan Cruz", "Eve Park", "Fay Wu"],
            "sex": ["F", "M", "F", "M", "F", "F"],
            "mail": [f"user{i}@example.com" for i in range(6)],
            "birthdate": [
                "1988-05-10",
                "1975-02-20",
                "1992-11-01",
                "1960-07-15",
                "1983-09-30",
                "1990-01-12",
            ],
            "City": ["Austin", "Boston", "Chicago", "Denver", "Seattle", "Miami"],
            "Country": ["US"] * 6,
            "cc_num": [
                4796807885357879,
                4473593503484549,
                4000000000000003,
                4000000000000004,
                4000000000000005,
                4000000000000006,
            ],
        }
    )
    profiles.to_csv(raw_dir / "profiles.csv", index=False)

    cards = pd.DataFrame(
        {
            "cc_num": profiles["cc_num"],
            "provider": ["visa", "mastercard", "visa", "amex", "visa", "discover"],
            "expires": ["05/25", "09/26", "11/24", "01/27", "03/28", "08/25"],
        }
    )
    cards.to_csv(raw_dir / "credit_cards.csv", index=False)

    applications = pd.DataFrame(
        {
            "sk_id_curr": [100001, 100002, 100003, 100004, 100005, 100006],
            "target": [0, 1, 0, 1, 0, 0],
            "amt_income_total": [180000.0, 65000.0, 120000.0, 50000.0, 95000.0, 140000.0],
            "amt_credit": [450000.0, 320000.0, 240000.0, 180000.0, 300000.0, 500000.0],
            "amt_annuity": [22500.0, 16000.0, np.nan, 9000.0, 15000.0, 25000.0],
            "amt_goods_price": [400000.0, 300000.0, 220000.0, 160000.0, 280000.0, 450000.0],
            "name_housing_type": [
                "House / apartment",
                "Rented apartment",
                "With parents",
                "Municipal apartment",
                "Co-op apartment",
                "Office apartment",
            ],
            "flag_own_realty": ["Y", "N", "N", "N", "Y", "N"],
            "days_birth": [-12009, -15406, -11000, -22000, -14000, -13000],
            "days_employed": [-2844, -892, -1549, 365243, -3650, -730],
            "days_registration": [-4000, -3000, -2000, -5000, -1000, -2500],
            "def_30_cnt_social_circle": [0.0, 2.0, np.nan, 1.0, 0.0, 0.0],
            "amt_req_credit_bureau_year": [1.0, 4.0, 0.0, np.nan, 2.0, 1.0],
            "ext_source_2": [0.7, 0.2, 0.6, 0.15, 0.55, 0.8],
            "ext_source_3": [0.65, 0.25, np.nan, 0.1, 0.5, 0.75],
            "datetime": [
                "2020-01-01 00:00:00",
                "2020-06-01 00:00:00",
                "2020-12-01 00:00:00",
                "2021-04-01 00:00:00",
                "2021-09-01 00:00:00",
                "2022-01-01 00:00:00",
            ],
        }
    )
    applications.to_csv(raw_dir / "applications.csv", index=False)

    tx_rows = []
    base_ts = datetime(2022, 1, 1, 0, 0, tzinfo=UTC)
    for day in range(30):
        for idx, cc in enumerate(profiles["cc_num"]):
            n_day_tx = (day + idx) % 3
            for sub in range(n_day_tx):
                tx_rows.append(
                    {
                        "tid": f"tx-{day}-{idx}-{sub}",
                        "datetime": (
                            base_ts + timedelta(days=day, hours=(idx * 3 + sub * 7 + day) % 24)
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "cc_num": cc,
                        "category": "Shopping",
                        "amount": 50.0 + ((day + sub) % 5) * 4.0 + idx * 5.0,
                        "latitude": 40.7 + idx * 0.1,
                        "longitude": -74.0 - idx * 0.1,
                        "city": "New York",
                        "country": "US",
                        "fraud_label": 0,
                    }
                )
    pd.DataFrame(tx_rows).to_csv(raw_dir / "transactions.csv", index=False)


def test_load_tutorial_csv_backfill_maps_contracts_and_aligns_timeline(tmp_path: Path) -> None:
    assert has_tutorial_csvs(tmp_path) is False
    _write_mini_tutorial_csvs(tmp_path)
    assert has_tutorial_csvs(tmp_path) is True

    batch = load_tutorial_csv_backfill(tmp_path)
    validated = validate_raw_batch(batch)

    assert validated.batch_id == "tutorial-csv-backfill"
    assert len(validated.customer_profiles) == 6
    assert len(validated.credit_applications) == 6
    assert tuple(validated.customer_profiles.columns) == CUSTOMER_PROFILES_CONTRACT.all_columns()
    assert tuple(validated.card_events.columns) == RAW_CARD_EVENTS_COLUMNS
    assert set(validated.customer_profiles["home_ownership"]).issubset(HOME_OWNERSHIP_VALUES)
    # Pensioner sentinel 365243 must sanitize to 0.0 employment_years, not ~1000 years
    assert validated.customer_profiles["employment_years"].max() <= 80.0
    assert (validated.customer_profiles["employment_years"] >= 0.0).all()
    # Real default labels preserved in chronological order
    assert validated.credit_applications["default_label"].tolist() == [0, 1, 0, 1, 0, 0]
    # Every application happens after profile snapshot and at least 7 days after first tx
    tx_min = validated.card_events["event_timestamp"].min()
    tx_max = validated.card_events["event_timestamp"].max()
    assert (validated.customer_profiles["event_timestamp"] == tx_min).all()
    min_application_ts = tx_min.to_pydatetime() + timedelta(days=7)
    assert (validated.credit_applications["event_timestamp"] >= min_application_ts).all()
    assert (validated.credit_applications["event_timestamp"] <= tx_max).all()


def test_hybrid_incremental_batch_bootstraps_from_csv_and_triggers_drift(tmp_path: Path) -> None:
    _write_mini_tutorial_csvs(tmp_path)
    backfill = load_tutorial_csv_backfill(tmp_path)

    pipeline = FeaturePipeline(InMemoryFeatureStore())
    ref_result = pipeline.run_batch(backfill)

    nominal = generate_incremental_from_backfill(
        backfill, window_hours=24, simulate_drift=False, batch_seq=1, seed=42
    )
    validate_raw_batch(nominal)
    assert (
        nominal.card_events["event_timestamp"].min() > backfill.card_events["event_timestamp"].max()
    )
    assert (
        nominal.credit_applications["event_timestamp"].min()
        > backfill.credit_applications["event_timestamp"].max()
    )
    nominal_res = pipeline.run_batch(
        nominal,
        history_card_events=backfill.card_events,
        reference_stats=ref_result.group_stats,
    )
    assert nominal_res.drift_detected is False

    drifted = generate_incremental_from_backfill(
        backfill, window_hours=24, simulate_drift=True, batch_seq=2, seed=42
    )
    validate_raw_batch(drifted)
    drifted_res = pipeline.run_batch(
        drifted,
        history_card_events=backfill.card_events,
        reference_stats=ref_result.group_stats,
    )
    assert drifted_res.drift_detected is True
    assert "credit_applications.dti_ratio" in drifted_res.events[1].drifted_features
