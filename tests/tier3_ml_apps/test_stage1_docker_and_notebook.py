"""Tier 3 tests for Stage 1 containerized Feast, Feature Pipeline CLI, and Notebook."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from banking_fti.adapters import FeastFeatureStore
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
            "--data-source",
            "simulator",
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
        _REPO_ROOT / "docker" / "Dockerfile.mlflow",
        _REPO_ROOT / "docker" / "Dockerfile.notebook",
        _REPO_ROOT / "docker" / "Dockerfile.train",
        _REPO_ROOT / "docker-compose.yml",
    ]
    for path in required_files:
        assert path.exists(), f"Missing required container file: {path}"

    for dockerfile_name in ("Dockerfile.feast", "Dockerfile.feature"):
        content = (_REPO_ROOT / "docker" / dockerfile_name).read_text(encoding="utf-8")
        assert "--extra feature" in content
        for forbidden in _FORBIDDEN_CONTAINER_TOKENS:
            assert forbidden not in content

    mlflow_dockerfile = (_REPO_ROOT / "docker" / "Dockerfile.mlflow").read_text(encoding="utf-8")
    assert "mlflow" in mlflow_dockerfile
    assert "--backend-store-uri" in mlflow_dockerfile
    assert "--serve-artifacts" in mlflow_dockerfile
    assert "--allowed-hosts" in mlflow_dockerfile
    assert "EXPOSE 5000" in mlflow_dockerfile
    assert "HEALTHCHECK" in mlflow_dockerfile
    for forbidden in _FORBIDDEN_CONTAINER_TOKENS:
        assert forbidden not in mlflow_dockerfile

    train_dockerfile = (_REPO_ROOT / "docker" / "Dockerfile.train").read_text(encoding="utf-8")
    assert "libgomp1" in train_dockerfile
    assert "--extra feature --extra train" in train_dockerfile
    assert "banking_fti.training.pipeline" in train_dockerfile

    notebook_dockerfile = (_REPO_ROOT / "docker" / "Dockerfile.notebook").read_text(
        encoding="utf-8"
    )
    assert "libgomp1" in notebook_dockerfile
    assert "--extra feature --extra train" in notebook_dockerfile

    compose = yaml.safe_load((_REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert "feast_data" in compose["volumes"]
    assert "model_artifacts" in compose["volumes"]
    assert "mlflow_data" in compose["volumes"]

    services = compose["services"]
    assert set(services.keys()) == {"feast", "feature", "mlflow", "notebook", "feast-ui", "train"}

    for svc_name in ("feast", "feature", "notebook", "train"):
        assert "feast_data:/app/feature_repo/data" in services[svc_name]["volumes"]
        env_entries = services[svc_name].get("environment", {})
        env_keys = list(env_entries.keys()) if isinstance(env_entries, dict) else list(env_entries)
        for key in env_keys:
            assert not any(str(key).startswith(prefix) for prefix in _CLOUD_SECRET_PREFIXES)

    mlflow_svc = services["mlflow"]
    assert "5000:5000" in mlflow_svc["ports"]
    assert "mlflow_data:/mlflow" in mlflow_svc["volumes"]
    assert "healthcheck" in mlflow_svc
    mlflow_env = mlflow_svc.get("environment", {})
    mlflow_env_keys = list(mlflow_env.keys()) if isinstance(mlflow_env, dict) else list(mlflow_env)
    for key in mlflow_env_keys:
        assert not any(str(key).startswith(prefix) for prefix in _CLOUD_SECRET_PREFIXES)

    assert (
        services["feature"]["depends_on"]["feast"]["condition"] == "service_completed_successfully"
    )
    assert (
        services["train"]["depends_on"]["feature"]["condition"] == "service_completed_successfully"
    )
    assert services["train"]["depends_on"]["mlflow"]["condition"] == "service_healthy"
    assert "model_artifacts:/app/artifacts/models" in services["train"]["volumes"]
    assert services["train"]["environment"]["MODEL_ARTIFACT_DIR"] == "/app/artifacts/models"
    assert services["train"]["environment"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert (
        services["notebook"]["depends_on"]["feature"]["condition"]
        == "service_completed_successfully"
    )
    assert (
        services["notebook"]["depends_on"]["train"]["condition"] == "service_completed_successfully"
    )
    assert services["notebook"]["depends_on"]["mlflow"]["condition"] == "service_healthy"
    assert services["notebook"]["environment"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert "8888:8888" in services["notebook"]["ports"]
    notebook_volumes = services["notebook"]["volumes"]
    assert "./notebooks:/app/notebooks" in notebook_volumes
    assert "./feature_repo:/app/feature_repo" in notebook_volumes
    assert "./src:/app/src" in notebook_volumes
    assert "model_artifacts:/app/artifacts/models" in notebook_volumes
    assert notebook_volumes.index("./feature_repo:/app/feature_repo") < notebook_volumes.index(
        "feast_data:/app/feature_repo/data"
    )

    makefile_text = (_REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("fetch-data:", "up:", "down:", "notebook:", "train:", "mlflow:"):
        assert target in makefile_text

    assert compose["services"]["feature"]["environment"]["RAW_DATA_DIR"] == "/app/data/raw"
    assert "./data/raw:/app/data/raw:ro" in compose["services"]["feature"]["volumes"]
    assert "./data/raw:/app/data/raw:ro" in notebook_volumes


def _write_sample_tutorial_csvs(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cc_nums = [f"400000000000000{i}" for i in range(1, 7)]
    pd.DataFrame(
        {
            "name": [f"Customer {i}" for i in range(1, 7)],
            "sex": ["F", "M", "F", "M", "F", "F"],
            "mail": [f"c{i}@example.com" for i in range(1, 7)],
            "birthdate": ["1988-01-01"] * 6,
            "City": ["Austin"] * 6,
            "Country": ["US"] * 6,
            "cc_num": cc_nums,
        }
    ).to_csv(raw_dir / "profiles.csv", index=False)
    pd.DataFrame({"cc_num": cc_nums, "provider": ["visa"] * 6, "expires": ["05/26"] * 6}).to_csv(
        raw_dir / "credit_cards.csv", index=False
    )
    pd.DataFrame(
        {
            "sk_id_curr": [1001, 1002, 1003, 1004, 1005, 1006],
            "target": [0, 1, 0, 1, 0, 0],
            "amt_income_total": [180000.0, 65000.0, 120000.0, 50000.0, 95000.0, 140000.0],
            "amt_credit": [450000.0, 320000.0, 240000.0, 180000.0, 300000.0, 500000.0],
            "amt_annuity": [22500.0, 16000.0, np.nan, 9000.0, 15000.0, 25000.0],
            "amt_goods_price": [400000.0, 300000.0, 220000.0, 160000.0, 280000.0, 450000.0],
            "name_housing_type": ["House / apartment", "Rented apartment"] * 3,
            "flag_own_realty": ["Y", "N"] * 3,
            "days_birth": [-12000] * 6,
            "days_employed": [-2500, -900, -1500, 365243, -3600, -800],
            "days_registration": [-3000] * 6,
            "def_30_cnt_social_circle": [0.0, 1.0, 0.0, 2.0, 0.0, 0.0],
            "amt_req_credit_bureau_year": [1.0, 4.0, 0.0, 1.0, 2.0, 0.0],
            "ext_source_2": [0.7, 0.2, 0.6, 0.2, 0.5, 0.8],
            "datetime": [f"2021-0{i}-01 00:00:00" for i in range(1, 7)],
        }
    ).to_csv(raw_dir / "applications.csv", index=False)

    tx_rows = []
    base_ts = datetime(2022, 1, 1, 0, 0, tzinfo=UTC)
    for day in range(25):
        for idx, cc in enumerate(cc_nums):
            tx_rows.append(
                {
                    "tid": f"tx-{day}-{idx}",
                    "datetime": (base_ts + timedelta(days=day, hours=idx)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "cc_num": cc,
                    "category": "Shopping",
                    "amount": 50.0 + (day % 5) * 3.0 + idx * 4.0,
                    "latitude": 40.7 + idx * 0.1,
                    "longitude": -74.0 - idx * 0.1,
                    "city": "New York",
                    "country": "US",
                    "fraud_label": 0,
                }
            )
    pd.DataFrame(tx_rows).to_csv(raw_dir / "transactions.csv", index=False)


def test_feature_pipeline_cli_supports_csv_data_source_and_feast_join(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_dir = tmp_path / "raw_csvs"
    repo_dir = tmp_path / "feast_repo"
    data_dir = tmp_path / "feast_data"
    _write_sample_tutorial_csvs(raw_dir)

    exit_code = feature_pipeline_mod.main(
        [
            "--data-source",
            "csv",
            "--raw-dir",
            str(raw_dir),
            "--repo-path",
            str(repo_dir),
            "--data-dir",
            str(data_dir),
            "--incremental-batches",
            "1",
            "--simulate-drift",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["data_source"] == "csv"
    assert payload["batches_run"] == 2
    assert payload["drift_detected"] is True

    store = FeastFeatureStore(repo_dir, data_dir=data_dir)
    apps = store.read_feature_group("credit_applications")
    joined = store.get_historical_features(
        apps.loc[:, ["application_id", "customer_id", "event_timestamp"]],
        ["customer_profiles", "card_window_aggs", "credit_applications"],
    )
    assert not joined["home_ownership"].isna().any()
    assert not joined["tx_amount_mavg_7d"].isna().any()
    assert "default_label" in joined.columns


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

    shared_dir = tmp_path / "shared_feast_data"
    sandbox_dir = tmp_path / "nb_sandbox_data"
    monkeypatch.setenv("FEAST_REPO_PATH", str(_REPO_ROOT / "feature_repo"))
    monkeypatch.setenv("FEAST_DATA_DIR", str(shared_dir))
    monkeypatch.setenv("FEAST_NOTEBOOK_SANDBOX_DIR", str(sandbox_dir))

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
    assert not (shared_dir / "credit_applications.parquet").exists()
    repo_data_dir = _REPO_ROOT / "feature_repo" / "data"
    assert not repo_data_dir.exists() or list(repo_data_dir.iterdir()) == []


def test_stage1_training_notebook_executes_end_to_end_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("lightgbm")
    old_notebook_path = _REPO_ROOT / "notebooks" / "02_stage1_credit_risk_training.ipynb"
    assert not old_notebook_path.exists(), "Replaced training notebook must not remain as an orphan"

    notebook_path = _REPO_ROOT / "notebooks" / "02_stage1_credit_risk_training_and_mlflow.ipynb"
    assert notebook_path.exists()

    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 4

    shared_dir = tmp_path / "shared_feast_data"
    shared_art_dir = tmp_path / "shared_model_artifacts"
    sandbox_feast_dir = tmp_path / "nb_sandbox_feast"
    sandbox_art_dir = tmp_path / "nb_sandbox_artifacts"
    sandbox_mlflow_dir = tmp_path / "nb_sandbox_mlflow"
    monkeypatch.setenv("FEAST_REPO_PATH", str(_REPO_ROOT / "feature_repo"))
    monkeypatch.setenv("FEAST_DATA_DIR", str(shared_dir))
    monkeypatch.setenv("MODEL_ARTIFACT_DIR", str(shared_art_dir))
    monkeypatch.setenv("TRAINING_NOTEBOOK_SANDBOX_FEAST_DIR", str(sandbox_feast_dir))
    monkeypatch.setenv("TRAINING_NOTEBOOK_SANDBOX_ARTIFACT_DIR", str(sandbox_art_dir))
    monkeypatch.setenv("TRAINING_NOTEBOOK_SANDBOX_MLFLOW_DIR", str(sandbox_mlflow_dir))

    namespace: dict[str, object] = {}
    for cell in code_cells:
        source = "".join(cell["source"])
        exec(compile(source, str(notebook_path), "exec"), namespace)

    sandbox_result = namespace["sandbox_result"]
    assert sandbox_result.gate_passed is True  # type: ignore[attr-defined]
    assert sandbox_result.parity_pass is True  # type: ignore[attr-defined]
    assert namespace["onnx_scores_valid"] is True
    assert namespace["rotation_verified"] is True
    assert namespace["incompatible_rollback_blocked"] is True
    assert namespace["rollback_verified"] is True
    assert not (shared_dir / "credit_applications.parquet").exists()
    assert not (_REPO_ROOT / "mlruns").exists()
    repo_data_dir = _REPO_ROOT / "feature_repo" / "data"
    assert not repo_data_dir.exists() or list(repo_data_dir.iterdir()) == []
