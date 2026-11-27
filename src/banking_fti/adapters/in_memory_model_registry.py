"""In-memory ModelRegistry adapter for Stage 1 offline tests and local runs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType

from banking_fti.ports.model_registry import RegisteredModel, validate_rollback_compatibility


@dataclass
class _RegistryEntry:
    model_name: str
    challenger: RegisteredModel | None = None
    champion: RegisteredModel | None = None
    previous: RegisteredModel | None = None
    next_version: int = 1
    history: list[RegisteredModel] = field(default_factory=list)


class InMemoryModelRegistry:
    """Thread-unsafe in-memory implementation of the ModelRegistry protocol.

    Alias lifecycle:
    - ``@champion`` is the currently active model version.
    - ``@challenger`` is a validated candidate awaiting promotion.
    - ``@previous`` is the last demoted version.

    Atomic promotion rotates ``champion -> previous`` and ``challenger -> champion``.
    Atomic rollback demotes ``champion -> previous`` and promotes ``previous -> champion``.
    """

    backend_name: str = "in_memory"

    def __init__(self) -> None:
        self._entries: dict[str, _RegistryEntry] = {}

    def _entry(self, model_name: str) -> _RegistryEntry:
        entry = self._entries.get(model_name)
        if entry is None:
            entry = _RegistryEntry(model_name=model_name)
            self._entries[model_name] = entry
        return entry

    def register_challenger(self, model: RegisteredModel) -> RegisteredModel:
        entry = self._entry(model.model_name)
        if model.version <= 0:
            assigned_version = entry.next_version
            entry.next_version += 1
        else:
            assigned_version = model.version
            entry.next_version = max(entry.next_version, model.version + 1)
        challenger = replace(
            model,
            version=assigned_version,
            status="challenger",
            metrics=MappingProxyType(dict(model.metrics)),
        )
        entry.challenger = challenger
        entry.history.append(challenger)
        return challenger

    def get_champion(self, model_name: str) -> RegisteredModel | None:
        return self._entry(model_name).champion

    def get_challenger(self, model_name: str) -> RegisteredModel | None:
        return self._entry(model_name).challenger

    def get_previous(self, model_name: str) -> RegisteredModel | None:
        return self._entry(model_name).previous

    def promote_to_champion(
        self,
        model_name: str,
        challenger_version: int,
        *,
        validation_passed: bool,
        validation_summary: str | None = None,
    ) -> dict[str, object]:
        entry = self._entry(model_name)
        if not validation_passed:
            return {
                "ok": False,
                "model_name": model_name,
                "champion_version": self._version_of(entry.champion),
                "previous_version": self._version_of(entry.previous),
                "error": validation_summary or "validation gate not met",
            }
        if entry.challenger is None or entry.challenger.version != challenger_version:
            return {
                "ok": False,
                "model_name": model_name,
                "champion_version": self._version_of(entry.champion),
                "previous_version": self._version_of(entry.previous),
                "error": f"no challenger at version {challenger_version}",
            }

        old_champion = entry.champion
        new_champion = self._with_status(entry.challenger, "champion")
        entry.champion = new_champion
        entry.previous = (
            old_champion if old_champion is None else replace(old_champion, status="previous")
        )
        entry.challenger = None
        return {
            "ok": True,
            "model_name": model_name,
            "champion_version": new_champion.version,
            "previous_version": self._version_of(entry.previous),
            "error": None,
        }

    def rollback_to_previous(
        self,
        model_name: str,
        *,
        expected_feature_view_version: str | None = None,
    ) -> dict[str, object]:
        entry = self._entry(model_name)
        failure = validate_rollback_compatibility(
            model_name=model_name,
            current_champion=entry.champion,
            previous=entry.previous,
            expected_feature_view_version=expected_feature_view_version,
        )
        if failure is not None:
            return failure
        assert entry.previous is not None
        demoted_champion = entry.champion
        new_champion = self._with_status(entry.previous, "champion")
        entry.champion = new_champion
        entry.previous = (
            demoted_champion
            if demoted_champion is None
            else replace(demoted_champion, status="previous")
        )
        entry.challenger = None
        return {
            "ok": True,
            "model_name": model_name,
            "champion_version": new_champion.version,
            "previous_version": self._version_of(entry.previous),
            "feature_view_version": new_champion.feature_view_version,
            "error": None,
        }

    @staticmethod
    def _with_status(model: RegisteredModel, status: str) -> RegisteredModel:
        return model if model.status == status else replace(model, status=status)

    @staticmethod
    def _version_of(model: RegisteredModel | None) -> int | None:
        return model.version if model is not None else None
