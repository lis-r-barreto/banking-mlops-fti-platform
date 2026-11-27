"""MLflow 3 ModelRegistry adapter supporting local SQLite, MLflow server, and DagsHub."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from banking_fti.ports.model_registry import RegisteredModel, validate_rollback_compatibility

_ARTIFACT_BUNDLE_DIR = "bundle"
_MISSING_ALIAS_ERRORS = frozenset({"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"})
_ARTIFACT_SPECS = (
    ("native", "native_artifact_relpath", "model.txt"),
    ("onnx", "onnx_artifact_relpath", "model.onnx"),
    ("preprocessor", "preprocessor_artifact_relpath", "preprocessor.joblib"),
)


class MlflowModelRegistry:
    """MLflow 3 implementation of the ``ModelRegistry`` protocol.

    Supports three deployment modes through ``tracking_uri``:
    - Offline SQLite (`sqlite:///...`) for fast hermetic round-trip tests
    - Containerized MLflow server (`http://mlflow:5000`) for local Docker validation
    - DagsHub MLflow (`https://dagshub.com/...mlflow`) for `hom` / `prd` cloud registries
    """

    backend_name: str = "mlflow"

    def __init__(
        self,
        tracking_uri: str,
        *,
        experiment_name: str = "banking_fti",
        environment: str = "local",
        registry_uri: str | None = None,
        artifact_location: str | None = None,
        artifact_cache_dir: str | Path | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.environment = environment
        self.registry_uri = registry_uri

        resolved_artifact_location: str | None
        if tracking_uri.startswith("sqlite:///"):
            db_path = Path(tracking_uri.removeprefix("sqlite:///")).resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_artifact_location = (
                artifact_location
                if artifact_location is not None
                else (db_path.parent / "mlartifacts").resolve().as_uri()
            )
            default_cache_dir = db_path.parent / ".mlflow_artifact_cache"
        else:
            resolved_artifact_location = artifact_location
            default_cache_dir = Path(tempfile.mkdtemp(prefix="banking_fti_mlflow_cache_"))

        self._artifact_location = resolved_artifact_location
        self._cache_dir = (
            Path(artifact_cache_dir).resolve()
            if artifact_cache_dir is not None
            else default_cache_dir.resolve()
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)

    @classmethod
    def from_env(
        cls,
        *,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        environment: str | None = None,
    ) -> MlflowModelRegistry:
        """Construct an ``MlflowModelRegistry`` from environment variables."""
        resolved_uri = (
            tracking_uri if tracking_uri is not None else os.environ.get("MLFLOW_TRACKING_URI", "")
        ).strip()
        if not resolved_uri:
            raise ValueError("MLFLOW_TRACKING_URI must be set to configure MlflowModelRegistry")

        resolved_env = (
            environment
            if environment is not None
            else os.environ.get("MLFLOW_ENVIRONMENT", "local")
        ).strip() or "local"
        default_exp = (
            f"banking_fti_{resolved_env}" if resolved_env in {"hom", "prd"} else "banking_fti"
        )
        resolved_exp = (
            experiment_name
            if experiment_name is not None
            else os.environ.get("MLFLOW_EXPERIMENT_NAME", "")
        ).strip() or default_exp
        registry_uri = os.environ.get("MLFLOW_REGISTRY_URI") or None
        cache_dir = os.environ.get("MLFLOW_ARTIFACT_CACHE_DIR") or None
        return cls(
            tracking_uri=resolved_uri,
            experiment_name=resolved_exp,
            environment=resolved_env,
            registry_uri=registry_uri,
            artifact_cache_dir=cache_dir,
        )

    def register_challenger(self, model: RegisteredModel) -> RegisteredModel:
        """Log run, upload ONNX/native/preprocessor bundle, and assign `@challenger`."""
        artifact_paths = {
            "native": Path(model.native_model_path),
            "onnx": Path(model.onnx_model_path),
            "preprocessor": Path(model.preprocessor_path),
        }
        for kind, path in artifact_paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {kind} artifact for registration: {path}")

        tags = self._build_lineage_tags(model, artifact_paths)
        experiment_id = self._ensure_experiment()
        run = self._client.create_run(experiment_id=experiment_id, tags=tags)
        mlflow_run_id = run.info.run_id
        try:
            for metric_name, metric_value in model.metrics.items():
                self._client.log_metric(mlflow_run_id, metric_name, float(metric_value))
            for path in artifact_paths.values():
                self._client.log_artifact(
                    mlflow_run_id,
                    str(path),
                    artifact_path=_ARTIFACT_BUNDLE_DIR,
                )
        except Exception:
            self._client.set_terminated(mlflow_run_id, status="FAILED")
            raise
        self._client.set_terminated(mlflow_run_id, status="FINISHED")

        self._ensure_registered_model(model.model_name)
        mv = self._client.create_model_version(
            name=model.model_name,
            source=f"runs:/{mlflow_run_id}/{_ARTIFACT_BUNDLE_DIR}",
            run_id=mlflow_run_id,
            tags=tags,
        )
        version_int = int(mv.version)
        self._client.set_registered_model_alias(model.model_name, "challenger", str(version_int))
        return replace(
            model,
            version=version_int,
            status="challenger",
            run_id=mlflow_run_id,
            metrics=MappingProxyType(dict(model.metrics)),
        )

    def get_by_alias(self, model_name: str, alias: str) -> RegisteredModel | None:
        """Fetch a registered model version by alias and restore its artifact bundle."""
        try:
            mv = self._client.get_model_version_by_alias(model_name, alias)
        except MlflowException as exc:
            if exc.error_code in _MISSING_ALIAS_ERRORS:
                return None
            raise
        return self._hydrate_model_version(mv, status=alias)

    def get_champion(self, model_name: str) -> RegisteredModel | None:
        """Return the currently promoted `@champion` for ``model_name``, if any."""
        return self.get_by_alias(model_name, "champion")

    def get_challenger(self, model_name: str) -> RegisteredModel | None:
        """Return the registered `@challenger` for ``model_name``, if any."""
        return self.get_by_alias(model_name, "challenger")

    def get_previous(self, model_name: str) -> RegisteredModel | None:
        """Return the last demoted `@previous` for ``model_name``, if any."""
        return self.get_by_alias(model_name, "previous")

    def promote_to_champion(
        self,
        model_name: str,
        challenger_version: int,
        *,
        validation_passed: bool,
        validation_summary: str | None = None,
    ) -> dict[str, object]:
        """Atomically rotate `@champion -> @previous` and `@challenger -> @champion`."""
        current_champion = self.get_champion(model_name)
        current_previous = self.get_previous(model_name)

        if not validation_passed:
            return {
                "ok": False,
                "model_name": model_name,
                "champion_version": self._version_of(current_champion),
                "previous_version": self._version_of(current_previous),
                "error": validation_summary or "validation gate not met",
            }

        challenger = self.get_challenger(model_name)
        if challenger is None or challenger.version != challenger_version:
            return {
                "ok": False,
                "model_name": model_name,
                "champion_version": self._version_of(current_champion),
                "previous_version": self._version_of(current_previous),
                "error": f"no challenger at version {challenger_version}",
            }

        if validation_summary:
            self._client.set_model_version_tag(
                model_name, str(challenger_version), "validation_summary", validation_summary
            )

        previous_version: int | None = None
        try:
            if current_champion is not None:
                previous_version = current_champion.version
                self._client.set_registered_model_alias(
                    model_name, "previous", str(previous_version)
                )
            self._client.set_registered_model_alias(model_name, "champion", str(challenger_version))
            self._delete_alias_if_present(model_name, "challenger")
        except Exception:
            self._restore_aliases(
                model_name,
                champion_version=self._version_of(current_champion),
                previous_version=self._version_of(current_previous),
                challenger_version=challenger.version,
            )
            raise
        return {
            "ok": True,
            "model_name": model_name,
            "champion_version": challenger_version,
            "previous_version": previous_version,
            "error": None,
        }

    def rollback_to_previous(
        self,
        model_name: str,
        *,
        expected_feature_view_version: str | None = None,
    ) -> dict[str, object]:
        """Atomically restore `@champion <- @previous` after checking `feature_view_version`."""
        current_champion = self.get_champion(model_name)
        previous = self.get_previous(model_name)
        failure = validate_rollback_compatibility(
            model_name=model_name,
            current_champion=current_champion,
            previous=previous,
            expected_feature_view_version=expected_feature_view_version,
        )
        if failure is not None:
            return failure
        assert previous is not None

        new_previous_version: int | None = None
        try:
            if current_champion is not None:
                new_previous_version = current_champion.version
                self._client.set_registered_model_alias(
                    model_name, "previous", str(new_previous_version)
                )
            else:
                self._delete_alias_if_present(model_name, "previous")

            self._client.set_registered_model_alias(model_name, "champion", str(previous.version))
        except Exception:
            self._restore_aliases(
                model_name,
                champion_version=self._version_of(current_champion),
                previous_version=previous.version,
            )
            raise
        return {
            "ok": True,
            "model_name": model_name,
            "champion_version": previous.version,
            "previous_version": new_previous_version,
            "feature_view_version": previous.feature_view_version,
            "error": None,
        }

    def download_artifacts(self, model: RegisteredModel, dst_dir: str | Path) -> dict[str, Path]:
        """Download ONNX, native, and preprocessor artifacts for ``model`` into ``dst_dir``."""
        target = Path(dst_dir)
        target.mkdir(parents=True, exist_ok=True)
        mv = self._client.get_model_version(model.model_name, str(model.version))
        tags = mv.tags or {}
        run_id = mv.run_id or model.run_id
        fallbacks = {
            "native": Path(model.native_model_path).name,
            "onnx": Path(model.onnx_model_path).name,
            "preprocessor": Path(model.preprocessor_path).name,
        }
        return self._download_bundle(run_id, tags, target, fallback_names=fallbacks)

    def _ensure_experiment(self) -> str:
        exp = self._client.get_experiment_by_name(self.experiment_name)
        if exp is not None:
            return str(exp.experiment_id)
        return str(
            self._client.create_experiment(
                self.experiment_name,
                artifact_location=self._artifact_location,
            )
        )

    def _ensure_registered_model(self, model_name: str) -> None:
        try:
            self._client.create_registered_model(model_name)
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise

    def _delete_alias_if_present(self, model_name: str, alias: str) -> None:
        try:
            self._client.delete_registered_model_alias(model_name, alias)
        except MlflowException as exc:
            if exc.error_code not in _MISSING_ALIAS_ERRORS:
                raise

    def _build_lineage_tags(
        self,
        model: RegisteredModel,
        artifact_paths: dict[str, Path],
    ) -> dict[str, str]:
        tags = {
            "feature_view_name": model.feature_view_name,
            "feature_view_version": model.feature_view_version,
            "feature_schema_hash": model.feature_schema_hash,
            "dataset_sha256": model.dataset_sha256,
            "created_at": model.created_at,
            "environment": self.environment,
            "pipeline_run_id": model.run_id,
            "native_artifact_relpath": f"{_ARTIFACT_BUNDLE_DIR}/{artifact_paths['native'].name}",
            "onnx_artifact_relpath": f"{_ARTIFACT_BUNDLE_DIR}/{artifact_paths['onnx'].name}",
            "preprocessor_artifact_relpath": (
                f"{_ARTIFACT_BUNDLE_DIR}/{artifact_paths['preprocessor'].name}"
            ),
        }
        optional_tags = {
            "git_sha": model.git_sha,
            "dvc_lock_hash": model.dvc_lock_hash,
            "pipeline_version": model.pipeline_version,
            "compute_backend": model.compute_backend,
            "executor": model.executor,
            "start_time": model.timestamp_start,
            "end_time": model.timestamp_end,
        }
        for key, value in optional_tags.items():
            if value is not None:
                tags[key] = value
        return tags

    def _hydrate_model_version(self, mv: ModelVersion, *, status: str) -> RegisteredModel:
        tags = mv.tags or {}
        run_id = mv.run_id or ""
        metrics: dict[str, float] = {}
        if run_id:
            try:
                run = self._client.get_run(run_id)
                metrics = {k: float(v) for k, v in run.data.metrics.items()}
                for k, v in run.data.tags.items():
                    tags.setdefault(k, v)
            except MlflowException:
                pass

        version_cache = self._cache_dir / mv.name / f"v{mv.version}"
        version_cache.mkdir(parents=True, exist_ok=True)
        bundle = self._download_bundle(run_id, tags, version_cache)
        return RegisteredModel(
            model_name=mv.name,
            version=int(mv.version),
            status=status,
            run_id=run_id,
            created_at=tags.get("created_at", ""),
            feature_view_name=tags.get("feature_view_name", ""),
            feature_view_version=tags.get("feature_view_version", ""),
            feature_schema_hash=tags.get("feature_schema_hash", ""),
            dataset_sha256=tags.get("dataset_sha256", ""),
            native_model_path=str(bundle["native"]),
            onnx_model_path=str(bundle["onnx"]),
            preprocessor_path=str(bundle["preprocessor"]),
            git_sha=tags.get("git_sha"),
            dvc_lock_hash=tags.get("dvc_lock_hash"),
            pipeline_version=tags.get("pipeline_version"),
            compute_backend=tags.get("compute_backend"),
            executor=tags.get("executor"),
            timestamp_start=tags.get("start_time"),
            timestamp_end=tags.get("end_time"),
            metrics=MappingProxyType(metrics),
        )

    def _download_bundle(
        self,
        run_id: str,
        tags: Mapping[str, str],
        dst_dir: Path,
        *,
        fallback_names: Mapping[str, str] | None = None,
    ) -> dict[str, Path]:
        resolved: dict[str, Path] = {}
        for kind, tag_key, default_filename in _ARTIFACT_SPECS:
            filename = (
                fallback_names.get(kind, default_filename)
                if fallback_names is not None
                else default_filename
            )
            relpath = tags.get(tag_key, f"{_ARTIFACT_BUNDLE_DIR}/{filename}")
            local = self._client.download_artifacts(run_id, relpath, dst_path=str(dst_dir))
            resolved[kind] = Path(local)
        return resolved

    def _restore_aliases(
        self,
        model_name: str,
        *,
        champion_version: int | None,
        previous_version: int | None,
        challenger_version: int | None = None,
    ) -> None:
        alias_targets = (
            ("champion", champion_version),
            ("previous", previous_version),
            ("challenger", challenger_version),
        )
        for alias, version in alias_targets:
            try:
                if version is not None:
                    self._client.set_registered_model_alias(model_name, alias, str(version))
                elif alias != "challenger":
                    self._delete_alias_if_present(model_name, alias)
            except MlflowException:
                pass

    @staticmethod
    def _version_of(model: RegisteredModel | None) -> int | None:
        return model.version if model is not None else None


__all__: list[str] = ["MlflowModelRegistry"]
