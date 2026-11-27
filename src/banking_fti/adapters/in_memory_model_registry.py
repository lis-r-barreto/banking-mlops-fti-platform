"""In-memory ModelRegistry adapter for Stage 1 offline tests and local runs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from banking_fti.ports.model_registry import RegisteredModel


@dataclass
class _RegistryEntry:
    model_name: str
    challenger: RegisteredModel | None = None
    champion: RegisteredModel | None = None
    previous: RegisteredModel | None = None
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
        challenger = self._with_status(model, "challenger")
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

    def rollback_to_previous(self, model_name: str) -> dict[str, object]:
        entry = self._entry(model_name)
        if entry.previous is None:
            return {
                "ok": False,
                "model_name": model_name,
                "champion_version": self._version_of(entry.champion),
                "previous_version": None,
                "error": "no @previous to roll back to",
            }
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
            "error": None,
        }

    @staticmethod
    def _with_status(model: RegisteredModel, status: str) -> RegisteredModel:
        return model if model.status == status else replace(model, status=status)

    @staticmethod
    def _version_of(model: RegisteredModel | None) -> int | None:
        return model.version if model is not None else None
