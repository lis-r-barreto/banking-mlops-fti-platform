"""ModelRegistry port for multi-model champion/challenger promotion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
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
    dvc_lock_hash: str | None = None
    pipeline_version: str | None = None
    compute_backend: str | None = None
    executor: str | None = None
    timestamp_start: str | None = None
    timestamp_end: str | None = None
    metrics: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))


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

    def rollback_to_previous(
        self,
        model_name: str,
        *,
        expected_feature_view_version: str | None = None,
    ) -> dict[str, object]:
        """Atomically demote the champion and promote the `@previous` alias."""
        ...


def validate_rollback_compatibility(
    *,
    model_name: str,
    current_champion: RegisteredModel | None,
    previous: RegisteredModel | None,
    expected_feature_view_version: str | None,
) -> dict[str, object] | None:
    """Return a structured failure payload if rollback is invalid, else ``None``."""
    champion_version = current_champion.version if current_champion is not None else None
    if previous is None:
        return {
            "ok": False,
            "model_name": model_name,
            "champion_version": champion_version,
            "previous_version": None,
            "error": "no @previous to roll back to",
        }
    target_fv = (
        expected_feature_view_version
        if expected_feature_view_version is not None
        else (current_champion.feature_view_version if current_champion is not None else None)
    )
    if target_fv is not None and previous.feature_view_version != target_fv:
        return {
            "ok": False,
            "model_name": model_name,
            "champion_version": champion_version,
            "previous_version": previous.version,
            "feature_view_version": previous.feature_view_version,
            "error": (
                "incompatible feature_view_version: "
                f"previous={previous.feature_view_version!r}, expected={target_fv!r}"
            ),
        }
    return None
