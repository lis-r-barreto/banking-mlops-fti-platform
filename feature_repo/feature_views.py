"""Feast Offline Parquet FeatureView definitions for Stage 1 PoC."""

from __future__ import annotations

from pathlib import Path

from banking_fti.adapters.feast_store import build_feast_objects

_DATA_DIR = Path(__file__).resolve().parent / "data"
_objects = build_feast_objects(_DATA_DIR)

customer_profiles_source = _objects.sources["customer_profiles"]
card_window_aggs_source = _objects.sources["card_window_aggs"]
credit_applications_source = _objects.sources["credit_applications"]

customer_profiles = _objects.feature_views["customer_profiles"]
card_window_aggs = _objects.feature_views["card_window_aggs"]
card_window_aggs_by_customer = _objects.feature_views["card_window_aggs_by_customer"]
credit_applications = _objects.feature_views["credit_applications"]
credit_applications_by_customer = _objects.feature_views["credit_applications_by_customer"]
