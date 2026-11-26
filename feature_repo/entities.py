"""Feast entity definitions for Stage 1 PoC banking Feature Groups."""

from __future__ import annotations

from banking_fti.adapters.feast_store import build_feast_entities

_entities = build_feast_entities()

customer = _entities.customer
credit_card = _entities.credit_card
loan_application = _entities.loan_application
