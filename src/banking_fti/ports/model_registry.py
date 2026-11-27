"""ModelRegistry port for multi-model champion/challenger promotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RegisteredModel:
    """A registered model version carrying full lineage metadata for alias lifecycle."""

    model_name: str
    version: int
    status: str  # "challenger" | "champion" | "previous"
    run_id: str
    created_at: str

    feature_view_name: str
    feature_view_version: str
    feature_schema_hash: str
    dataset_sha256: str
    native_model_path: str
    onnx_model_path: str
    preprocessor_path: str

    git_sha: str | None = None
    pipeline_version: str | None = None
    compute_backend: str | None = None
    executor: str | None = None
    timestamp_start: str | None = None
    timestamp_end: str | None = None


@runtime_checkable
class ModelRegistry(Protocol):
    """Multi-model alias registry (challenger/champion/previous) for FTI models."""

    def register_challenger(self, model: RegisteredModel) -> RegisteredModel:
        """Register a validated candidate as the active `@challenger` for its model."""
        ...

    def get_champion(self, model_name: str) -> RegisteredModel | None:
        """Return the currently promoted `@champion` for `model_name`, if any."""
        ...

    def get_challenger(self, model_name: str) -> RegisteredModel | None:
        """Return the registered `@challenger` for `model_name`, if any."""
        ...

    def get_previous(self, model_name: str) -> RegisteredModel | None:
        """Return the last demoted `@previous` for `model_name`, if any."""
        ...

    def promote_to_champion(
        self,
        model_name: str,
        challenger_version: int,
        *,
        validation_passed: bool,
        validation_summary: str | None = None,
    ) -> dict[str, object]:
        """Atomically promote a validated challenger.

        Returns a dict with keys ``ok``, ``champion_version``, ``previous_version``,
        and optionally ``error`` when blocked.
        """
        ...

    def rollback_to_previous(self, model_name: str) -> dict[str, object]:
        """Atomically demote the champion and promote the `@previous` alias."""
        ...
