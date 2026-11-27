"""Strictly causal t - 1 rolling card-window aggregations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from banking_fti.core.contracts import CARD_WINDOW_AGGS_CONTRACT

_EARTH_RADIUS_KM = 6371.0088
_ONE_DAY_NS = 24 * 3600 * 1_000_000_000
_SEVEN_DAYS_NS = 7 * _ONE_DAY_NS


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers between two WGS84 coordinates."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    sin_half_phi = math.sin(d_phi / 2.0)
    sin_half_lambda = math.sin(d_lambda / 2.0)
    a = sin_half_phi * sin_half_phi + math.cos(phi1) * math.cos(phi2) * (
        sin_half_lambda * sin_half_lambda
    )
    return 2.0 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def compute_card_window_aggs(
    card_events: pd.DataFrame,
    *,
    history_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute strictly causal (t - 1) rolling 24h and 7d card-window aggregations.

    For every target event at timestamp `t`, only events with strictly earlier timestamps
    (`< t`) contribute to counts, amount averages/standard deviations, and location deltas.
    Input row order of `card_events` is preserved in the returned DataFrame.
    """
    n_target = len(card_events)
    if n_target == 0:
        return CARD_WINDOW_AGGS_CONTRACT.empty_frame()

    target_frame = card_events.reset_index(drop=True).copy()
    target_frame["_target_pos"] = np.arange(n_target, dtype=np.int64)

    if history_events is not None and not history_events.empty:
        hist_frame = history_events.reset_index(drop=True).copy()
        hist_frame["_target_pos"] = np.full(len(hist_frame), -1, dtype=np.int64)
        combined = pd.concat([hist_frame, target_frame], ignore_index=True)
    else:
        combined = target_frame

    ts_ns_series = pd.to_datetime(combined["event_timestamp"], utc=True).dt.as_unit("ns")
    combined["_ts_ns"] = ts_ns_series.astype(np.int64)

    mavg_7d = np.zeros(n_target, dtype=np.float64)
    mstd_7d = np.zeros(n_target, dtype=np.float64)
    count_24h = np.zeros(n_target, dtype=np.int64)
    count_7d = np.zeros(n_target, dtype=np.int64)
    loc_delta_mavg_7d = np.zeros(n_target, dtype=np.float64)

    for _, group in combined.groupby("cc_num", sort=False):
        ordered = group.sort_values(["_ts_ns", "_target_pos"], kind="mergesort")
        ts = ordered["_ts_ns"].to_numpy(dtype=np.int64)
        amounts = ordered["amount"].to_numpy(dtype=np.float64)
        lats = ordered["latitude"].to_numpy(dtype=np.float64)
        lons = ordered["longitude"].to_numpy(dtype=np.float64)
        positions = ordered["_target_pos"].to_numpy(dtype=np.int64)

        n_group = len(ts)
        loc_deltas = np.zeros(n_group, dtype=np.float64)
        for idx in range(n_group):
            hi = int(np.searchsorted(ts, ts[idx], side="left"))
            if hi > 0:
                prev_idx = hi - 1
                loc_deltas[idx] = haversine_km(
                    float(lats[prev_idx]),
                    float(lons[prev_idx]),
                    float(lats[idx]),
                    float(lons[idx]),
                )

        for idx in range(n_group):
            pos = int(positions[idx])
            if pos < 0:
                continue

            t_val = int(ts[idx])
            hi = int(np.searchsorted(ts, t_val, side="left"))
            lo_7d = int(np.searchsorted(ts, t_val - _SEVEN_DAYS_NS, side="left"))
            lo_24h = int(np.searchsorted(ts, t_val - _ONE_DAY_NS, side="left"))

            n_7d = hi - lo_7d
            n_24h = hi - lo_24h
            count_7d[pos] = n_7d
            count_24h[pos] = n_24h

            if n_7d > 0:
                window_amounts = amounts[lo_7d:hi]
                window_loc_deltas = loc_deltas[lo_7d:hi]
                mavg_7d[pos] = float(np.mean(window_amounts))
                mstd_7d[pos] = float(np.std(window_amounts, ddof=0)) if n_7d >= 2 else 0.0
                loc_delta_mavg_7d[pos] = float(np.mean(window_loc_deltas))

    result = pd.DataFrame(
        {
            "cc_num": target_frame["cc_num"].astype(str),
            "customer_id": target_frame["customer_id"].astype(str),
            "event_timestamp": pd.to_datetime(target_frame["event_timestamp"], utc=True).dt.as_unit(
                "ns"
            ),
            "tx_amount_mavg_7d": mavg_7d,
            "tx_amount_mstd_7d": mstd_7d,
            "tx_count_24h": count_24h,
            "tx_count_7d": count_7d,
            "loc_delta_mavg_7d": loc_delta_mavg_7d,
        }
    )
    return result.loc[:, list(CARD_WINDOW_AGGS_CONTRACT.all_columns())]
