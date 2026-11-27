"""Deterministic Banking Event Simulator for Stage 1 PoC living banking data."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import numpy as np
import pandas as pd

from banking_fti.core.contracts import (
    CREDIT_APPLICATIONS_CONTRACT,
    CUSTOMER_PROFILES_CONTRACT,
    HOME_OWNERSHIP_VALUES,
    RAW_CARD_EVENTS_COLUMNS,
    RawBankingBatch,
)

_HOME_OWNERSHIP_PROBS = (0.38, 0.42, 0.16, 0.04)
_LOAN_TERMS_MONTHS = (12, 24, 36, 48, 60)
_DEFAULT_END_TIME = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)


@dataclass(frozen=True)
class CustomerIdentity:
    """Immutable synthetic customer node inside the banking customer <-> card graph."""

    customer_id: str
    cc_nums: tuple[str, ...]
    age_years: int
    income_annual: float
    employment_years: float
    home_ownership: str
    tenure_months: int
    home_lat: float
    home_lon: float
    base_spend: float


@dataclass(frozen=True)
class SimulatorConfig:
    """Configuration for deterministic banking event generation and drift injection."""

    num_customers: int = 120
    backfill_days: int = 90
    avg_card_events_per_day: float = 0.6
    avg_applications_per_day: float = 4.0
    seed: int = 42
    drift_dti_shift: float = 0.28
    drift_utilization_shift: float = 0.20


def _ensure_utc(dt: datetime, *, param_name: str) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{param_name} must be timezone-aware UTC datetime, got naive {dt!r}.")
    return dt.astimezone(UTC)


def _derive_seed(base_seed: int, *labels: object) -> int:
    digest = hashlib.sha256(
        "|".join([str(base_seed), *(str(label) for label in labels)]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _to_utc_ns_series(timestamps: list[datetime]) -> pd.Series:
    if not timestamps:
        return pd.Series(pd.DatetimeIndex([], tz="UTC").as_unit("ns"))
    return pd.Series(pd.to_datetime(timestamps, utc=True).as_unit("ns"))


class BankingEventSimulator:
    """Deterministic simulator generating linked customer profiles, card events, and credit apps."""

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self._config = config or SimulatorConfig()
        if self._config.num_customers < 1:
            raise ValueError("num_customers must be >= 1.")
        if self._config.backfill_days < 1:
            raise ValueError("backfill_days must be >= 1.")

        self._identities = self._build_identity_graph()
        customer_map: dict[str, tuple[str, ...]] = {}
        card_map: dict[str, str] = {}
        for identity in self._identities:
            customer_map[identity.customer_id] = identity.cc_nums
            for cc_num in identity.cc_nums:
                card_map[cc_num] = identity.customer_id
        self._customer_card_map: Mapping[str, tuple[str, ...]] = MappingProxyType(customer_map)
        self._card_to_customer: Mapping[str, str] = MappingProxyType(card_map)

    @property
    def config(self) -> SimulatorConfig:
        """Return simulator configuration."""
        return self._config

    @property
    def customer_card_map(self) -> Mapping[str, tuple[str, ...]]:
        """Read-only mapping from customer_id to owned cc_num tuple."""
        return self._customer_card_map

    @property
    def card_to_customer(self) -> Mapping[str, str]:
        """Read-only mapping from cc_num to owning customer_id."""
        return self._card_to_customer

    def _build_identity_graph(self) -> tuple[CustomerIdentity, ...]:
        rng = np.random.default_rng(_derive_seed(self._config.seed, "identity_graph"))
        identities: list[CustomerIdentity] = []
        card_counter = 0

        for idx in range(self._config.num_customers):
            customer_id = f"CUST-{idx:05d}"
            num_cards = int(rng.choice((1, 2, 3), p=(0.45, 0.40, 0.15)))
            cc_nums = tuple(f"40000000{card_counter + offset:08d}" for offset in range(num_cards))
            card_counter += num_cards

            age_years = int(rng.integers(21, 72))
            max_emp = max(1.0, float(age_years - 18))
            employment_years = float(
                np.round(np.clip(rng.gamma(shape=2.2, scale=3.2), 0.0, min(45.0, max_emp)), 2)
            )
            income_annual = float(np.round(np.exp(rng.normal(11.15, 0.48)), 2))
            home_ownership = str(rng.choice(HOME_OWNERSHIP_VALUES, p=_HOME_OWNERSHIP_PROBS))
            tenure_months = int(rng.integers(3, 180))
            home_lat = float(np.round(rng.uniform(25.0, 48.0), 5))
            home_lon = float(np.round(rng.uniform(-123.0, -71.0), 5))
            base_spend = float(np.round(np.clip(income_annual / 1200.0, 25.0, 450.0), 2))

            identities.append(
                CustomerIdentity(
                    customer_id=customer_id,
                    cc_nums=cc_nums,
                    age_years=age_years,
                    income_annual=income_annual,
                    employment_years=employment_years,
                    home_ownership=home_ownership,
                    tenure_months=tenure_months,
                    home_lat=home_lat,
                    home_lon=home_lon,
                    base_spend=base_spend,
                )
            )

        return tuple(identities)

    def _build_customer_profiles_df(self, snapshot_time: datetime) -> pd.DataFrame:
        ts_list = [snapshot_time] * len(self._identities)
        df = pd.DataFrame(
            {
                "customer_id": [c.customer_id for c in self._identities],
                "event_timestamp": _to_utc_ns_series(ts_list),
                "age_years": np.array([c.age_years for c in self._identities], dtype=np.int64),
                "income_annual": np.array(
                    [c.income_annual for c in self._identities], dtype=np.float64
                ),
                "employment_years": np.array(
                    [c.employment_years for c in self._identities], dtype=np.float64
                ),
                "home_ownership": [c.home_ownership for c in self._identities],
                "tenure_months": np.array(
                    [c.tenure_months for c in self._identities], dtype=np.int64
                ),
                "num_active_cards": np.array(
                    [len(c.cc_nums) for c in self._identities], dtype=np.int64
                ),
            }
        )
        return df.loc[:, list(CUSTOMER_PROFILES_CONTRACT.all_columns())]

    def _generate_window(
        self,
        *,
        window_start: datetime,
        window_end: datetime,
        inclusive_start: bool,
        simulate_drift: bool,
        batch_id: str,
        seed_tag: str,
    ) -> RawBankingBatch:
        rng = np.random.default_rng(
            _derive_seed(self._config.seed, seed_tag, window_start.isoformat())
        )
        total_seconds = int((window_end - window_start).total_seconds())
        if total_seconds < 1:
            raise ValueError("Simulation window duration must be at least 1 second.")
        min_offset = 0 if inclusive_start else 1

        days = total_seconds / 86400.0
        n_card_events = max(
            1,
            int(
                rng.poisson(
                    self._config.num_customers * self._config.avg_card_events_per_day * days
                )
            ),
        )
        n_apps = max(1, int(rng.poisson(self._config.avg_applications_per_day * days)))

        card_rows: list[dict[str, object]] = []
        event_offsets = np.sort(rng.integers(min_offset, total_seconds + 1, size=n_card_events))
        for seq, offset_sec in enumerate(event_offsets):
            identity = self._identities[int(rng.integers(0, len(self._identities)))]
            cc_num = identity.cc_nums[int(rng.integers(0, len(identity.cc_nums)))]
            event_ts = window_start + timedelta(seconds=int(offset_sec))
            amount = float(
                np.round(np.clip(rng.lognormal(np.log(identity.base_spend), 0.65), 1.5, 9500.0), 2)
            )
            is_travel = bool(rng.random() < 0.08)
            lat_jitter = float(rng.normal(0.0, 2.5 if is_travel else 0.05))
            lon_jitter = float(rng.normal(0.0, 3.0 if is_travel else 0.05))
            latitude = float(np.round(np.clip(identity.home_lat + lat_jitter, -89.9, 89.9), 5))
            longitude = float(np.round(np.clip(identity.home_lon + lon_jitter, -179.9, 179.9), 5))

            card_rows.append(
                {
                    "event_id": f"{batch_id}-EVT-{seq:06d}",
                    "cc_num": cc_num,
                    "customer_id": identity.customer_id,
                    "event_timestamp": event_ts,
                    "amount": amount,
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )

        card_events_df = pd.DataFrame(card_rows)
        card_events_df["event_timestamp"] = _to_utc_ns_series(
            [row["event_timestamp"] for row in card_rows]  # type: ignore[misc]
        )
        card_events_df["amount"] = card_events_df["amount"].astype(np.float64)
        card_events_df["latitude"] = card_events_df["latitude"].astype(np.float64)
        card_events_df["longitude"] = card_events_df["longitude"].astype(np.float64)
        card_events_df = card_events_df.loc[:, list(RAW_CARD_EVENTS_COLUMNS)]

        app_rows: list[dict[str, object]] = []
        app_offsets = np.sort(rng.integers(min_offset, total_seconds + 1, size=n_apps))
        dti_shift = self._config.drift_dti_shift if simulate_drift else 0.0
        util_shift = self._config.drift_utilization_shift if simulate_drift else 0.0
        loan_scale = 1.20 if simulate_drift else 1.0

        for seq, offset_sec in enumerate(app_offsets):
            identity = self._identities[int(rng.integers(0, len(self._identities)))]
            event_ts = window_start + timedelta(seconds=int(offset_sec))

            base_dti = float(np.clip(rng.beta(2.2, 5.2) * 0.65, 0.02, 0.95))
            dti_ratio = float(np.round(np.clip(base_dti + dti_shift, 0.02, 1.50), 4))
            base_util = float(np.clip(rng.beta(2.0, 3.8), 0.01, 1.0))
            utilization_ratio = float(np.round(np.clip(base_util + util_shift, 0.01, 1.50), 4))
            existing_delinquencies = int(rng.poisson(0.35 + (0.25 if simulate_drift else 0.0)))
            pub_rec_flag = int(rng.random() < 0.08)
            loan_term_months = int(rng.choice(_LOAN_TERMS_MONTHS))
            loan_amount = float(
                np.round(
                    np.clip(
                        identity.income_annual * rng.uniform(0.10, 0.45) * loan_scale,
                        1000.0,
                        120000.0,
                    ),
                    2,
                )
            )

            income_norm = (identity.income_annual - 65000.0) / 35000.0
            logit = (
                -2.3
                + 3.1 * dti_ratio
                + 1.6 * utilization_ratio
                + 0.75 * existing_delinquencies
                + 0.90 * pub_rec_flag
                - 0.45 * income_norm
            )
            default_prob = 1.0 / (1.0 + float(np.exp(-logit)))
            default_label = int(rng.random() < default_prob)

            app_rows.append(
                {
                    "application_id": f"{batch_id}-APP-{seq:06d}",
                    "customer_id": identity.customer_id,
                    "event_timestamp": event_ts,
                    "loan_amount": loan_amount,
                    "loan_term_months": loan_term_months,
                    "dti_ratio": dti_ratio,
                    "existing_delinquencies": existing_delinquencies,
                    "utilization_ratio": utilization_ratio,
                    "pub_rec_flag": pub_rec_flag,
                    "default_label": default_label,
                }
            )

        apps_df = pd.DataFrame(app_rows)
        apps_df["event_timestamp"] = _to_utc_ns_series(
            [row["event_timestamp"] for row in app_rows]  # type: ignore[misc]
        )
        apps_df["loan_amount"] = apps_df["loan_amount"].astype(np.float64)
        apps_df["loan_term_months"] = apps_df["loan_term_months"].astype(np.int64)
        apps_df["dti_ratio"] = apps_df["dti_ratio"].astype(np.float64)
        apps_df["existing_delinquencies"] = apps_df["existing_delinquencies"].astype(np.int64)
        apps_df["utilization_ratio"] = apps_df["utilization_ratio"].astype(np.float64)
        apps_df["pub_rec_flag"] = apps_df["pub_rec_flag"].astype(np.int64)
        apps_df["default_label"] = apps_df["default_label"].astype(np.int64)
        apps_df = apps_df.loc[:, list(CREDIT_APPLICATIONS_CONTRACT.all_columns())]

        profiles_ts = window_start if inclusive_start else window_end
        profiles_df = self._build_customer_profiles_df(profiles_ts)

        return RawBankingBatch(
            customer_profiles=profiles_df,
            card_events=card_events_df,
            credit_applications=apps_df,
            batch_id=batch_id,
            simulate_drift=simulate_drift,
        )

    def generate_backfill(self, *, end_time: datetime | None = None) -> RawBankingBatch:
        """Generate a deterministic 90-day (configurable) backfill batch ending at `end_time`."""
        resolved_end = (
            _DEFAULT_END_TIME if end_time is None else _ensure_utc(end_time, param_name="end_time")
        )
        window_start = resolved_end - timedelta(days=self._config.backfill_days)
        return self._generate_window(
            window_start=window_start,
            window_end=resolved_end,
            inclusive_start=True,
            simulate_drift=False,
            batch_id=f"backfill-{self._config.backfill_days}d",
            seed_tag="backfill",
        )

    def generate_incremental_batch(
        self,
        *,
        start_time: datetime,
        window_hours: int = 24,
        simulate_drift: bool = False,
        batch_seq: int = 1,
    ) -> RawBankingBatch:
        """Generate an incremental batch in (start_time, start_time + window_hours]."""
        if window_hours < 1:
            raise ValueError("window_hours must be >= 1.")
        if batch_seq < 1:
            raise ValueError("batch_seq must be >= 1.")

        resolved_start = _ensure_utc(start_time, param_name="start_time")
        window_end = resolved_start + timedelta(hours=window_hours)
        drift_tag = "drift" if simulate_drift else "nominal"
        return self._generate_window(
            window_start=resolved_start,
            window_end=window_end,
            inclusive_start=False,
            simulate_drift=simulate_drift,
            batch_id=f"inc-{batch_seq:04d}-{drift_tag}",
            seed_tag=f"inc-{batch_seq}",
        )
