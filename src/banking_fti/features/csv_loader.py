"""Hybrid Hopsworks Tutorial CSV loader and incremental/drift batch generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from banking_fti.core.contracts import (
    CREDIT_APPLICATIONS_CONTRACT,
    CUSTOMER_PROFILES_CONTRACT,
    RAW_CARD_EVENTS_COLUMNS,
    RawBankingBatch,
)

DEFAULT_TUTORIAL_CSV_URLS: dict[str, str] = {
    "applications.csv": "https://repo.hops.works/dev/davit/credit_scores/applications.csv",
    "profiles.csv": (
        "https://repo.hops.works/master/hopsworks-tutorials/data/card_fraud_data/profiles.csv"
    ),
    "credit_cards.csv": (
        "https://repo.hops.works/master/hopsworks-tutorials/data/card_fraud_data/credit_cards.csv"
    ),
    "transactions.csv": (
        "https://repo.hops.works/master/hopsworks-tutorials/data/card_fraud_data/transactions.csv"
    ),
}

_HOUSING_MAP: dict[tuple[str, str], str] = {
    ("Rented apartment", "N"): "RENT",
    ("Rented apartment", "Y"): "RENT",
    ("Municipal apartment", "N"): "RENT",
    ("Municipal apartment", "Y"): "RENT",
    ("House / apartment", "Y"): "OWN",
    ("House / apartment", "N"): "MORTGAGE",
    ("Co-op apartment", "Y"): "OWN",
    ("Co-op apartment", "N"): "MORTGAGE",
}


@dataclass(frozen=True)
class TutorialCsvSources:
    """Filenames and download URLs for Stage 1 Hopsworks Tutorial CSV datasets."""

    applications_url: str = DEFAULT_TUTORIAL_CSV_URLS["applications.csv"]
    profiles_url: str = DEFAULT_TUTORIAL_CSV_URLS["profiles.csv"]
    credit_cards_url: str = DEFAULT_TUTORIAL_CSV_URLS["credit_cards.csv"]
    transactions_url: str = DEFAULT_TUTORIAL_CSV_URLS["transactions.csv"]

    def as_mapping(self) -> dict[str, str]:
        """Return ordered mapping from local CSV filename to source URL."""
        return {
            "applications.csv": self.applications_url,
            "profiles.csv": self.profiles_url,
            "credit_cards.csv": self.credit_cards_url,
            "transactions.csv": self.transactions_url,
        }


def has_tutorial_csvs(raw_dir: str | Path) -> bool:
    """Return True if all required Stage 1 tutorial CSV files exist in `raw_dir`."""
    base = Path(raw_dir)
    return all((base / filename).is_file() for filename in DEFAULT_TUTORIAL_CSV_URLS)


def download_tutorial_csvs(
    raw_dir: str | Path,
    *,
    sources: TutorialCsvSources | None = None,
    force: bool = False,
    timeout_seconds: float = 60.0,
) -> dict[str, Path]:
    """Download Stage 1 Hopsworks Tutorial CSV files atomically into `raw_dir`."""
    target_dir = Path(raw_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_sources = sources or TutorialCsvSources()

    downloaded: dict[str, Path] = {}
    for filename, url in resolved_sources.as_mapping().items():
        dest = target_dir / filename
        if dest.is_file() and not force:
            downloaded[filename] = dest
            continue

        fd, tmp_name = tempfile.mkstemp(dir=target_dir, prefix=f".{filename}.")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response, tmp_path.open(
                "wb"
            ) as handle:
                while chunk := response.read(65536):
                    handle.write(chunk)
            os.replace(tmp_path, dest)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        downloaded[filename] = dest

    return downloaded


def _map_home_ownership(housing_type: object, own_realty: object) -> str:
    return _HOUSING_MAP.get((str(housing_type), str(own_realty)), "OTHER")


def _derive_seed(seed: int, *labels: object) -> int:
    digest = hashlib.sha256(
        "|".join([str(seed), *(str(label) for label in labels)]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _align_application_timestamps(
    raw_datetimes: pd.Series,
    *,
    tx_min: pd.Timestamp,
    tx_max: pd.Timestamp,
) -> pd.Series:
    n_rows = len(raw_datetimes)
    if n_rows == 0:
        return pd.Series(pd.DatetimeIndex([], tz="UTC").as_unit("ns"))

    span = tx_max - tx_min
    align_start = tx_min + timedelta(days=7) if span >= timedelta(days=7) else tx_min
    if n_rows == 1 or align_start >= tx_max:
        return pd.Series([tx_max] * n_rows, dtype="datetime64[ns, UTC]")

    parsed = pd.to_datetime(raw_datetimes, utc=True).dt.as_unit("ns").astype(np.int64)
    min_raw = int(parsed.min())
    max_raw = int(parsed.max())
    start_ns = int(align_start.as_unit("ns").value)
    end_ns = int(tx_max.as_unit("ns").value)

    if max_raw <= min_raw:
        fractions = np.linspace(0.0, 1.0, n_rows, dtype=np.float64)
    else:
        fractions = np.asarray(
            (parsed.to_numpy(dtype=np.float64) - float(min_raw)) / float(max_raw - min_raw),
            dtype=np.float64,
        )

    aligned_ns = np.round(start_ns + fractions * float(end_ns - start_ns)).astype(np.int64)
    return pd.Series(pd.to_datetime(aligned_ns, utc=True, unit="ns").as_unit("ns"))


def load_tutorial_csv_backfill(
    raw_dir: str | Path,
    *,
    max_transactions: int | None = None,
) -> RawBankingBatch:
    """Load real Hopsworks Tutorial CSVs from `raw_dir` into a canonical `RawBankingBatch`."""
    base = Path(raw_dir)
    if not has_tutorial_csvs(base):
        missing = [
            name for name in DEFAULT_TUTORIAL_CSV_URLS if not (base / name).is_file()
        ]
        raise FileNotFoundError(
            f"Missing tutorial CSVs in {base}: {missing}. Run `make fetch-data` first."
        )

    apps_raw = pd.read_csv(base / "applications.csv")
    profiles_raw = pd.read_csv(base / "profiles.csv", dtype={"cc_num": str})
    cards_raw = pd.read_csv(base / "credit_cards.csv", dtype={"cc_num": str})
    tx_raw = pd.read_csv(base / "transactions.csv", dtype={"tid": str, "cc_num": str})

    if max_transactions is not None and max_transactions > 0 and len(tx_raw) > max_transactions:
        tx_raw = tx_raw.head(max_transactions).copy()

    apps_sorted = apps_raw.sort_values(["datetime", "sk_id_curr"], kind="mergesort").reset_index(
        drop=True
    )

    known_cards = set(cards_raw["cc_num"].dropna().astype(str)) & set(
        profiles_raw["cc_num"].dropna().astype(str)
    )
    tx_cards_ordered = [
        cc for cc in tx_raw["cc_num"].dropna().astype(str).drop_duplicates() if cc in known_cards
    ]
    tx_card_set = set(tx_cards_ordered)
    remaining_cards = sorted(known_cards - tx_card_set)
    available_cards = [*tx_cards_ordered, *remaining_cards]
    if len(available_cards) < len(apps_sorted):
        raise ValueError(
            f"Not enough linked credit cards ({len(available_cards)}) for "
            f"applications ({len(apps_sorted)})."
        )

    customer_ids = [f"CUST-{int(sk_id)}" for sk_id in apps_sorted["sk_id_curr"]]
    assigned_cards = available_cards[: len(customer_ids)]
    card_to_customer = dict(zip(assigned_cards, customer_ids, strict=True))

    tx_filtered = tx_raw.loc[tx_raw["cc_num"].astype(str).isin(card_to_customer)].copy()
    tx_filtered["amount"] = pd.to_numeric(tx_filtered["amount"], errors="coerce").fillna(0.0)
    tx_filtered = tx_filtered.loc[np.isfinite(tx_filtered["amount"]) & (tx_filtered["amount"] > 0)]
    tx_filtered = tx_filtered.sort_values(["datetime", "tid"], kind="mergesort").reset_index(
        drop=True
    )
    if tx_filtered.empty:
        raise ValueError("No valid card transactions remained after linking to tutorial cards.")

    tx_timestamps = pd.to_datetime(tx_filtered["datetime"], utc=True).dt.as_unit("ns")
    tx_min = pd.Timestamp(tx_timestamps.min()).tz_convert("UTC")
    tx_max = pd.Timestamp(tx_timestamps.max()).tz_convert("UTC")

    card_events_df = pd.DataFrame(
        {
            "event_id": tx_filtered["tid"].astype(str),
            "cc_num": tx_filtered["cc_num"].astype(str),
            "customer_id": tx_filtered["cc_num"].astype(str).map(card_to_customer),
            "event_timestamp": tx_timestamps,
            "amount": tx_filtered["amount"].to_numpy(dtype=np.float64),
            "latitude": np.clip(
                pd.to_numeric(tx_filtered["latitude"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float64),
                -90.0,
                90.0,
            ),
            "longitude": np.clip(
                pd.to_numeric(tx_filtered["longitude"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float64),
                -180.0,
                180.0,
            ),
        }
    ).loc[:, list(RAW_CARD_EVENTS_COLUMNS)]

    days_birth = pd.to_numeric(apps_sorted["days_birth"], errors="coerce").fillna(-12000)
    age_years = np.clip(np.round(-days_birth.to_numpy(dtype=np.float64) / 365.25), 18, 120).astype(
        np.int64
    )

    days_emp = pd.to_numeric(apps_sorted["days_employed"], errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float64
    )
    valid_emp = (days_emp < 0.0) & (days_emp > -365.25 * 80.0)
    employment_years = np.where(
        valid_emp,
        np.clip(np.round(-days_emp / 365.25, 2), 0.0, 80.0),
        0.0,
    ).astype(np.float64)

    income_annual = np.clip(
        pd.to_numeric(apps_sorted["amt_income_total"], errors="coerce")
        .fillna(50000.0)
        .to_numpy(dtype=np.float64),
        1.0,
        None,
    )
    home_ownership = [
        _map_home_ownership(housing, realty)
        for housing, realty in zip(
            apps_sorted["name_housing_type"],
            apps_sorted["flag_own_realty"],
            strict=True,
        )
    ]
    days_reg = pd.to_numeric(apps_sorted["days_registration"], errors="coerce").fillna(-365.0)
    tenure_months = np.clip(
        np.round(-days_reg.to_numpy(dtype=np.float64) / 30.4375),
        0,
        600,
    ).astype(np.int64)

    profile_ts = pd.Series([tx_min] * len(customer_ids), dtype="datetime64[ns, UTC]").dt.as_unit(
        "ns"
    )
    customer_profiles_df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "event_timestamp": profile_ts,
            "age_years": age_years,
            "income_annual": income_annual,
            "employment_years": employment_years,
            "home_ownership": home_ownership,
            "tenure_months": tenure_months,
            "num_active_cards": np.ones(len(customer_ids), dtype=np.int64),
        }
    ).loc[:, list(CUSTOMER_PROFILES_CONTRACT.all_columns())]

    loan_amount = np.clip(
        pd.to_numeric(apps_sorted["amt_credit"], errors="coerce")
        .fillna(10000.0)
        .to_numpy(dtype=np.float64),
        1.0,
        None,
    )
    annuity = pd.to_numeric(apps_sorted["amt_annuity"], errors="coerce").to_numpy(dtype=np.float64)
    valid_annuity = np.isfinite(annuity) & (annuity > 0.0)
    raw_terms = np.where(valid_annuity, np.round((loan_amount / annuity) * 12.0), 36.0)
    loan_term_months = np.clip(raw_terms, 6, 360).astype(np.int64)

    dti_ratio = np.round(
        np.where(valid_annuity, np.clip(annuity / income_annual, 0.0, 2.0), 0.0),
        4,
    ).astype(np.float64)

    delinq_col = (
        apps_sorted["def_30_cnt_social_circle"]
        if "def_30_cnt_social_circle" in apps_sorted.columns
        else pd.Series(0.0, index=apps_sorted.index)
    )
    existing_delinquencies = (
        pd.to_numeric(delinq_col, errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.int64)
    )

    goods_price = pd.to_numeric(apps_sorted["amt_goods_price"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    ext2 = (
        pd.to_numeric(apps_sorted["ext_source_2"], errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype=np.float64)
    )
    valid_goods = np.isfinite(goods_price) & (goods_price > 0.0)
    utilization_ratio = np.round(
        np.clip(
            np.where(valid_goods, (loan_amount / goods_price) * 0.45, 1.0 - ext2),
            0.0,
            2.0,
        ),
        4,
    ).astype(np.float64)

    bureau_col = (
        apps_sorted["amt_req_credit_bureau_year"]
        if "amt_req_credit_bureau_year" in apps_sorted.columns
        else pd.Series(0.0, index=apps_sorted.index)
    )
    pub_rec_flag = (
        pd.to_numeric(bureau_col, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) >= 3.0
    ).astype(np.int64)
    default_label = (
        pd.to_numeric(apps_sorted["target"], errors="coerce")
        .fillna(0)
        .clip(0, 1)
        .to_numpy(dtype=np.int64)
    )

    app_timestamps = _align_application_timestamps(
        apps_sorted["datetime"],
        tx_min=tx_min,
        tx_max=tx_max,
    )
    credit_applications_df = pd.DataFrame(
        {
            "application_id": [f"APP-{int(sk_id)}" for sk_id in apps_sorted["sk_id_curr"]],
            "customer_id": customer_ids,
            "event_timestamp": app_timestamps,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "dti_ratio": dti_ratio,
            "existing_delinquencies": existing_delinquencies,
            "utilization_ratio": utilization_ratio,
            "pub_rec_flag": pub_rec_flag,
            "default_label": default_label,
        }
    ).loc[:, list(CREDIT_APPLICATIONS_CONTRACT.all_columns())]

    return RawBankingBatch(
        customer_profiles=customer_profiles_df,
        card_events=card_events_df,
        credit_applications=credit_applications_df,
        batch_id="tutorial-csv-backfill",
        simulate_drift=False,
    )


def generate_incremental_from_backfill(
    backfill: RawBankingBatch,
    *,
    window_hours: int = 24,
    simulate_drift: bool = False,
    batch_seq: int = 1,
    seed: int = 42,
    drift_dti_shift: float = 0.28,
    drift_utilization_shift: float = 0.20,
) -> RawBankingBatch:
    """Bootstrap a hybrid incremental batch from a real CSV backfill with optional drift."""
    if window_hours < 1:
        raise ValueError("window_hours must be >= 1.")
    if batch_seq < 1:
        raise ValueError("batch_seq must be >= 1.")

    rng = np.random.default_rng(_derive_seed(seed, "hybrid_inc", batch_seq))
    tx_min = pd.Timestamp(backfill.card_events["event_timestamp"].min()).tz_convert("UTC")
    tx_max = pd.Timestamp(backfill.card_events["event_timestamp"].max()).tz_convert("UTC")
    app_max = pd.Timestamp(backfill.credit_applications["event_timestamp"].max()).tz_convert("UTC")
    window_start = max(tx_max, app_max).to_pydatetime()
    window_end = window_start + timedelta(hours=window_hours)
    total_seconds = int((window_end - window_start).total_seconds())

    profiles_df = backfill.customer_profiles.copy()
    profiles_df["event_timestamp"] = pd.Series(
        [pd.Timestamp(window_end)] * len(profiles_df),
        dtype="datetime64[ns, UTC]",
    ).dt.as_unit("ns")

    backfill_hours = max(float(window_hours), (tx_max - tx_min).total_seconds() / 3600.0)
    events_per_hour = len(backfill.card_events) / backfill_hours
    n_events = max(1, min(len(backfill.card_events), round(events_per_hour * window_hours)))
    ev_indices = rng.integers(0, len(backfill.card_events), size=n_events)
    ev_offsets = np.sort(rng.integers(1, total_seconds + 1, size=n_events))
    sampled_events = backfill.card_events.iloc[ev_indices].reset_index(drop=True).copy()
    drift_tag = "drift" if simulate_drift else "nominal"
    batch_id = f"hybrid-inc-{batch_seq:04d}-{drift_tag}"
    sampled_events["event_id"] = [f"{batch_id}-EVT-{idx:06d}" for idx in range(n_events)]
    sampled_events["event_timestamp"] = pd.Series(
        [window_start + timedelta(seconds=int(off)) for off in ev_offsets],
        dtype="datetime64[ns, UTC]",
    ).dt.as_unit("ns")
    sampled_events["amount"] = np.round(
        np.clip(
            sampled_events["amount"].to_numpy(dtype=np.float64) * rng.uniform(0.9, 1.1, n_events),
            1.0,
            None,
        ),
        2,
    )

    n_apps = max(24, min(len(backfill.credit_applications), 120))
    app_indices = rng.integers(0, len(backfill.credit_applications), size=n_apps)
    app_offsets = np.sort(rng.integers(1, total_seconds + 1, size=n_apps))
    sampled_apps = backfill.credit_applications.iloc[app_indices].reset_index(drop=True).copy()
    sampled_apps["application_id"] = [f"{batch_id}-APP-{idx:06d}" for idx in range(n_apps)]
    sampled_apps["event_timestamp"] = pd.Series(
        [window_start + timedelta(seconds=int(off)) for off in app_offsets],
        dtype="datetime64[ns, UTC]",
    ).dt.as_unit("ns")

    dti_shift = drift_dti_shift if simulate_drift else 0.0
    util_shift = drift_utilization_shift if simulate_drift else 0.0
    sampled_apps["dti_ratio"] = np.round(
        np.clip(
            sampled_apps["dti_ratio"].to_numpy(dtype=np.float64)
            + rng.normal(0.0, 0.01, n_apps)
            + dti_shift,
            0.0,
            2.5,
        ),
        4,
    )
    sampled_apps["utilization_ratio"] = np.round(
        np.clip(
            sampled_apps["utilization_ratio"].to_numpy(dtype=np.float64)
            + rng.normal(0.0, 0.01, n_apps)
            + util_shift,
            0.0,
            2.5,
        ),
        4,
    )
    if simulate_drift:
        sampled_apps["loan_amount"] = np.round(
            sampled_apps["loan_amount"].to_numpy(dtype=np.float64) * 1.15, 2
        )

    return RawBankingBatch(
        customer_profiles=profiles_df.loc[:, list(CUSTOMER_PROFILES_CONTRACT.all_columns())],
        card_events=sampled_events.loc[:, list(RAW_CARD_EVENTS_COLUMNS)],
        credit_applications=sampled_apps.loc[:, list(CREDIT_APPLICATIONS_CONTRACT.all_columns())],
        batch_id=batch_id,
        simulate_drift=simulate_drift,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Download Stage 1 Hopsworks Tutorial CSVs into `--raw-dir` and print JSON manifest."""
    parser = argparse.ArgumentParser(
        description="Fetch Hopsworks Tutorial CSV datasets for Stage 1 hybrid backfill."
    )
    parser.add_argument(
        "--raw-dir",
        default=os.environ.get("RAW_DATA_DIR", "data/raw"),
        help="Destination directory for downloaded tutorial CSV files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    args = parser.parse_args(argv)
    paths = download_tutorial_csvs(args.raw_dir, force=args.force)
    manifest = {name: str(path.resolve()) for name, path in paths.items()}
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
