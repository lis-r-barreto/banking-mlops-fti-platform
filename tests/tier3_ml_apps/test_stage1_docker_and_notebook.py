"""Tier 3 tests for Stage 1 containerized Feast, Feature Pipeline CLI, and Notebook."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from banking_fti.features import pipeline as feature_pipeline_mod

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_CONTAINER_TOKENS = (
    "--extra train",
    "--extra serve",
    "xgboost",
    "lightgbm",
    "onnxmltools",
    "onnxruntime",
    "tensorflow",
    "torch",
)
_CLOUD_SECRET_PREFIXES = ("AWS_", "DAGSHUB_", "COLAB_", "RENDER_", "NEON_")


def test_feature_pipeline_cli_runs_backfill_and_incremental_drift_into_feast(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_dir = tmp_path / "feast_repo"
    data_dir = tmp_path / "feast_data"

    exit_code = feature_pipeline_mod.main(
        [
            "--repo-path",
            str(repo_dir),
            "--data-dir",
            str(data_dir),
            "--num-customers",
            "15",
            "--backfill-days",
            "14",
            "--incremental-batches",
            "1",
            "--simulate-drift",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["batches_run"] == 2
    assert payload["drift_detected"] is True
    assert payload["rows_written"]["customer_profiles"] == 30
    assert (data_dir / "customer_profiles.parquet").exists()
    assert (data_dir / "card_window_aggs.parquet").exists()
    assert (data_dir / "credit_applications.parquet").exists()
    assert (data_dir / "registry.db").exists()


def test_stage1_dockerfiles_and_compose_wire_feast_feature_and_notebook() -> None:
    required_files = [
        _REPO_ROOT / ".dockerignore",
        _REPO_ROOT / "docker" / "Dockerfile.feast",
        _REPO_ROOT / "docker" / "Dockerfile.feature",
        _REPO_ROOT / "docker" / "Dockerfile.notebook",
        _REPO_ROOT / "docker-compose.yml",
    ]
    for path in required_files:
        assert path.exists(), f"Missing required container file: {path}"

    for dockerfile_name in ("Dockerfile.feast", "Dockerfile.feature"):
        content = (_REPO_ROOT / "docker" / dockerfile_name).read_text(encoding="utf-8")
        assert "--extra feature" in content
        for forbidden in _FORBIDDEN_CONTAINER_TOKENS:
            assert forbidden not in content

    compose = yaml.safe_load((_REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert "feast_data" in compose["volumes"]

    services = compose["services"]
    assert set(services.keys()) == {"feast", "feature", "notebook"}

    for svc_name in ("feast", "feature", "notebook"):
        assert "feast_data:/app/feature_repo/data" in services[svc_name]["volumes"]
        env_entries = services[svc_name].get("environment", {})
        env_keys = list(env_entries.keys()) if isinstance(env_entries, dict) else list(env_entries)
        for key in env_keys:
            assert not any(str(key).startswith(prefix) for prefix in _CLOUD_SECRET_PREFIXES)

    assert (
        services["feature"]["depends_on"]["feast"]["condition"] == "service_completed_successfully"
    )
    assert (
        services["notebook"]["depends_on"]["feature"]["condition"]
        == "service_completed_successfully"
    )
    assert "8888:8888" in services["notebook"]["ports"]
    assert "./notebooks:/app/notebooks" in services["notebook"]["volumes"]

    makefile_text = (_REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("up:", "down:", "notebook:", "smoke-stage1:"):
        assert target in makefile_text


def test_stage1_validation_notebook_executes_end_to_end_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook_path = _REPO_ROOT / "notebooks" / "01_stage1_features_and_feast.ipynb"
    assert notebook_path.exists()

    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 4

    data_dir = tmp_path / "nb_feast_data"
    monkeypatch.setenv("FEAST_REPO_PATH", str(_REPO_ROOT / "feature_repo"))
    monkeypatch.setenv("FEAST_DATA_DIR", str(data_dir))

    namespace: dict[str, object] = {}
    for cell in code_cells:
        source = "".join(cell["source"])
        exec(compile(source, str(notebook_path), "exec"), namespace)

    entities = namespace["registered_entities"]
    assert isinstance(entities, list)
    assert {"customer", "credit_card", "loan_application"}.issubset(set(entities))
    assert namespace["causal_exact_match"] is True
    assert namespace["pandera_rejected_invalid"] is True
    assert namespace["drift_detected_on_inc"] is True
    assert namespace["train_has_label"] is True
    assert namespace["batch_omits_label"] is True
    assert not (_REPO_ROOT / "feature_repo" / "data").exists()
