"""Tests for ModelRegistry: alias lifecycle (challenger/champion/previous)."""

from __future__ import annotations

from banking_fti.adapters.in_memory_model_registry import InMemoryModelRegistry
from banking_fti.ports.model_registry import RegisteredModel


def _challenge(model_name: str, version: int) -> RegisteredModel:
    return RegisteredModel(
        model_name=model_name,
        version=version,
        status="challenger",
        run_id="run-0",
        created_at="2026-01-01T00:00:00+00:00",
        feature_view_name="credit_applications",
        feature_view_version="v1",
        feature_schema_hash="hash-1",
        dataset_sha256="sha-1",
        native_model_path="native.txt",
        onnx_model_path="onnx.onnx",
        preprocessor_path="preprocessor.joblib",
    )


def _champion(model_name: str, version: int) -> RegisteredModel:
    return RegisteredModel(
        model_name=model_name,
        version=version,
        status="champion",
        run_id="run-0",
        created_at="2026-01-01T00:00:00+00:00",
        feature_view_name="credit_applications",
        feature_view_version="v1",
        feature_schema_hash="hash-1",
        dataset_sha256="sha-1",
        native_model_path="native.txt",
        onnx_model_path="onnx.onnx",
        preprocessor_path="preprocessor.joblib",
    )


def _previous(model_name: str, version: int) -> RegisteredModel:
    return RegisteredModel(
        model_name=model_name,
        version=version,
        status="previous",
        run_id="run-0",
        created_at="2026-01-01T00:00:00+00:00",
        feature_view_name="credit_applications",
        feature_view_version="v1",
        feature_schema_hash="hash-1",
        dataset_sha256="sha-1",
        native_model_path="native.txt",
        onnx_model_path="onnx.onnx",
        preprocessor_path="preprocessor.joblib",
    )


def test_register_challenger_records_challenger_only() -> None:
    registry = InMemoryModelRegistry()
    model = _challenge("credit_risk_batch", 1)
    registered = registry.register_challenger(model)

    assert registered.status == "challenger"
    assert registered.version == 1
    challenger = registry.get_challenger("credit_risk_batch")
    assert challenger is not None
    assert challenger.version == 1
    assert registry.get_champion("credit_risk_batch") is None
    assert registry.get_previous("credit_risk_batch") is None


def test_promote_validated_challenger_becomes_champion_previous_demoted() -> None:
    registry = InMemoryModelRegistry()
    existing = _champion("credit_risk_batch", 0)
    registry._entry("credit_risk_batch").champion = existing
    challenger = _challenge("credit_risk_batch", 1)
    registry.register_challenger(challenger)

    result = registry.promote_to_champion(
        "credit_risk_batch", challenger_version=1, validation_passed=True
    )

    assert result["ok"] is True
    assert result["champion_version"] == 1
    assert result["previous_version"] == 0
    champion = registry.get_champion("credit_risk_batch")
    assert champion is not None
    assert champion.version == 1
    assert champion.status == "champion"
    previous = registry.get_previous("credit_risk_batch")
    assert previous is not None
    assert previous.version == 0
    assert registry.get_challenger("credit_risk_batch") is None


def test_promote_blocked_when_validation_fails_champion_unchanged() -> None:
    registry = InMemoryModelRegistry()
    champion = _champion("credit_risk_batch", 0)
    registry._entry("credit_risk_batch").champion = champion
    challenger = _challenge("credit_risk_batch", 1)
    registry.register_challenger(challenger)

    result = registry.promote_to_champion(
        "credit_risk_batch", challenger_version=1, validation_passed=False
    )

    assert result["ok"] is False
    assert result["error"] == "validation gate not met"
    champion = registry.get_champion("credit_risk_batch")  # type: ignore[assignment]
    assert champion is not None
    assert champion.version == 0
    challenger = registry.get_challenger("credit_risk_batch")  # type: ignore[assignment]
    assert challenger is not None
    assert challenger.version == 1


def test_rollback_promotes_previous_to_champion() -> None:
    registry = InMemoryModelRegistry()
    previous = _previous("credit_risk_batch", 0)
    champion = _champion("credit_risk_batch", 1)
    registry._entry("credit_risk_batch").champion = champion
    registry._entry("credit_risk_batch").previous = previous

    result = registry.rollback_to_previous("credit_risk_batch")

    assert result["ok"] is True
    assert result["champion_version"] == 0
    assert result["previous_version"] == 1
    champion = registry.get_champion("credit_risk_batch")  # type: ignore[assignment]
    assert champion is not None
    assert champion.version == 0
    assert champion.status == "champion"


def test_rollback_without_previous_fails() -> None:
    registry = InMemoryModelRegistry()
    champion = _champion("credit_risk_batch", 1)
    registry._entry("credit_risk_batch").champion = champion

    result = registry.rollback_to_previous("credit_risk_batch")

    assert result["ok"] is False
    assert result["error"] == "no @previous to roll back to"
    champion = registry.get_champion("credit_risk_batch")  # type: ignore[assignment]
    assert champion is not None
    assert champion.version == 1


def test_register_challenger_assigns_monotonic_versions_and_preserves_lineage() -> None:
    registry = InMemoryModelRegistry()

    first = registry.register_challenger(
        RegisteredModel(
            model_name="credit_risk_batch",
            version=0,
            status="challenger",
            run_id="run-1",
            created_at="2026-01-01T00:00:00+00:00",
            feature_view_name="credit_applications",
            feature_view_version="v1",
            feature_schema_hash="hash-1",
            dataset_sha256="sha-1",
            native_model_path="model.txt",
            onnx_model_path="model.onnx",
            preprocessor_path="preprocessor.joblib",
            dvc_lock_hash="dvc-123",
            metrics={"roc_auc": 0.84, "default_recall": 0.72, "min_cohort_recall": 0.65},
        )
    )
    second = registry.register_challenger(
        RegisteredModel(
            model_name="credit_risk_batch",
            version=0,
            status="challenger",
            run_id="run-2",
            created_at="2026-01-02T00:00:00+00:00",
            feature_view_name="credit_applications",
            feature_view_version="v1",
            feature_schema_hash="hash-1",
            dataset_sha256="sha-2",
            native_model_path="model2.txt",
            onnx_model_path="model2.onnx",
            preprocessor_path="preprocessor2.joblib",
            dvc_lock_hash="dvc-456",
            metrics={"roc_auc": 0.86},
        )
    )

    assert first.version == 1
    assert first.dvc_lock_hash == "dvc-123"
    assert first.metrics["roc_auc"] == 0.84
    assert second.version == 2
    assert second.dvc_lock_hash == "dvc-456"
    assert registry.get_challenger("credit_risk_batch") == second
