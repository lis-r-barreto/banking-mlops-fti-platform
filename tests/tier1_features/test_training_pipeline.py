"""Integration tests for TrainingPipeline via InMemoryFeatureStore."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from banking_fti.adapters.in_memory import InMemoryFeatureStore
from banking_fti.adapters.in_memory_model_registry import InMemoryModelRegistry
from banking_fti.core.contracts import (
    CARD_WINDOW_AGGS_CONTRACT,
    CREDIT_APPLICATIONS_CONTRACT,
    CUSTOMER_PROFILES_CONTRACT,
)
from banking_fti.training.pipeline import TrainingPipeline

pytest.importorskip("lightgbm")


def _store_with_synthetic_credit(num_customers: int = 200) -> InMemoryFeatureStore:
    store = InMemoryFeatureStore()
    rng = np.random.default_rng(7)
    base_time = datetime(2026, 1, 1, tzinfo=UTC)

    home = rng.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], size=num_customers)
    income = rng.normal(80000, 20000, size=num_customers).clip(25000, 160000)
    profiles = pd.DataFrame(
        {
            "customer_id": [f"cust-{i}" for i in range(num_customers)],
            "event_timestamp": base_time,
            "age_years": rng.integers(22, 60, size=num_customers),
            "income_annual": income,
            "employment_years": rng.integers(0, 30, size=num_customers),
            "home_ownership": home,
            "tenure_months": rng.integers(0, 120, size=num_customers),
            "num_active_cards": rng.integers(1, 5, size=num_customers),
        }
    )

    app_count = num_customers * 2
    cust_ids = [f"cust-{i}" for i in range(num_customers)]
    incomes_for_apps = rng.choice(income, size=app_count)
    loan_amounts = rng.normal(300000, 150000, size=app_count).clip(50000, 800000)
    default_prob = np.clip(
        0.05 + (50000 - incomes_for_apps) / 120000 + (loan_amounts - 300000) / 1500000, 0.01, 0.9
    )
    default_labels = (rng.random(app_count) < default_prob).astype(int)

    apps = pd.DataFrame(
        {
            "application_id": [f"app-{i}" for i in range(app_count)],
            "customer_id": rng.choice(cust_ids, size=app_count),
            "event_timestamp": [
                base_time + timedelta(days=int(d)) for d in rng.integers(0, 90, size=app_count)
            ],
            "loan_amount": loan_amounts,
            "loan_term_months": rng.integers(12, 60, size=app_count),
            "dti_ratio": rng.uniform(0.1, 0.6, size=app_count),
            "existing_delinquencies": rng.integers(0, 5, size=app_count),
            "utilization_ratio": rng.uniform(0.0, 0.8, size=app_count),
            "pub_rec_flag": rng.integers(0, 2, size=app_count),
            "default_label": default_labels,
        }
    )

    aggs = pd.DataFrame(
        {
            "cc_num": [f"400000000000000{i}" for i in range(1, app_count + 1)],
            "customer_id": rng.choice(cust_ids, size=app_count),
            "event_timestamp": base_time,
            "tx_amount_mavg_7d": rng.exponential(100, size=app_count),
            "tx_amount_mstd_7d": rng.exponential(50, size=app_count),
            "tx_count_24h": rng.integers(0, 10, size=app_count),
            "tx_count_7d": rng.integers(0, 20, size=app_count),
            "loc_delta_mavg_7d": rng.exponential(5, size=app_count),
        }
    )

    store.write_feature_group(
        "customer_profiles", profiles, schema_hash=CUSTOMER_PROFILES_CONTRACT.schema_hash()
    )
    store.write_feature_group(
        "credit_applications", apps, schema_hash=CREDIT_APPLICATIONS_CONTRACT.schema_hash()
    )
    store.write_feature_group(
        "card_window_aggs", aggs, schema_hash=CARD_WINDOW_AGGS_CONTRACT.schema_hash()
    )
    return store


def test_training_pipeline_runs_end_to_end_and_registers_challenger() -> None:
    registry = InMemoryModelRegistry()
    store = _store_with_synthetic_credit()
    pipeline = TrainingPipeline(store, registry, model_name="credit_risk_batch")

    result = pipeline.train(test_size=0.25, random_state=42)

    assert result.model_name == "credit_risk_batch"
    assert result.challenger_registered is True
    assert isinstance(result.gate_passed, bool)
    assert isinstance(result.promoted, bool)
    assert isinstance(result.holdout_auc, float)
    assert isinstance(result.dataset_sha256, str)
    assert result.dataset_sha256 != ""
