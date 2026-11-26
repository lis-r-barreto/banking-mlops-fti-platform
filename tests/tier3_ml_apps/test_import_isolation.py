"""Tier 3 import-isolation tests verifying zero ML training/serving libs in Feature runtime."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

FORBIDDEN_LIBRARIES = frozenset(
    {
        "xgboost",
        "lightgbm",
        "onnxmltools",
        "onnxruntime",
        "tensorflow",
        "torch",
    }
)


def _top_level_module(name: str) -> str:
    return name.split(".", maxsplit=1)[0]


def test_feature_pipeline_runtime_never_imports_ml_training_or_serving_libs(
    tmp_path: Path,
) -> None:
    feast_repo = tmp_path / "feast_repo"
    snippet = (
        "import sys\n"
        "from datetime import datetime, timezone\n"
        "from banking_fti.adapters.feast_store import FeastFeatureStore\n"
        "from banking_fti.adapters.hopsworks_seam import HopsworksConfig, HopsworksFeatureStore\n"
        "from banking_fti.adapters.in_memory import InMemoryFeatureStore\n"
        "from banking_fti.core import contracts, validation\n"
        "from banking_fti.features import causal_aggs, pipeline, simulator\n"
        "from banking_fti.ports import feature_store\n"
        "cfg = simulator.SimulatorConfig(num_customers=10, backfill_days=7)\n"
        "sim = simulator.BankingEventSimulator(cfg)\n"
        "batch = sim.generate_backfill(end_time=datetime(2026, 1, 1, tzinfo=timezone.utc))\n"
        "fp = pipeline.FeaturePipeline(InMemoryFeatureStore())\n"
        "fp.run_batch(batch)\n"
        f"feast_store = FeastFeatureStore({str(feast_repo)!r})\n"
        "pipeline.FeaturePipeline(feast_store).run_batch(batch)\n"
        "HopsworksFeatureStore(HopsworksConfig(project_name='banking_fti'))\n"
        "loaded = sorted(name for name in sys.modules)\n"
        "print('\\n'.join(loaded))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded_roots = {_top_level_module(mod) for mod in proc.stdout.splitlines() if mod}
    violations = sorted(loaded_roots & FORBIDDEN_LIBRARIES)
    assert violations == []


def test_feature_runtime_source_files_contain_no_forbidden_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "src" / "banking_fti"
    runtime_dirs = ("core", "ports", "adapters", "features")

    violations: list[str] = []
    for subdir in runtime_dirs:
        for py_file in sorted((package_root / subdir).rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if _top_level_module(alias.name) in FORBIDDEN_LIBRARIES:
                            violations.append(f"{py_file}: import {alias.name}")
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module is not None
                    and _top_level_module(node.module) in FORBIDDEN_LIBRARIES
                ):
                    violations.append(f"{py_file}: from {node.module} import ...")

    assert violations == []
