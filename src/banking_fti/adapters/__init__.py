"""Concrete adapters implementing stateful ports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from banking_fti.adapters.feast_store import FeastFeatureStore
from banking_fti.adapters.in_memory import InMemoryFeatureStore
from banking_fti.adapters.in_memory_model_registry import InMemoryModelRegistry

if TYPE_CHECKING:
    from banking_fti.adapters.mlflow_registry import MlflowModelRegistry

__all__ = [
    "FeastFeatureStore",
    "InMemoryFeatureStore",
    "InMemoryModelRegistry",
    "MlflowModelRegistry",
]


def __getattr__(name: str) -> object:
    # Lazy-load MLflow adapter so Feature-only containers (--extra feature) stay isolated.
    if name == "MlflowModelRegistry":
        from banking_fti.adapters.mlflow_registry import MlflowModelRegistry

        return MlflowModelRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
