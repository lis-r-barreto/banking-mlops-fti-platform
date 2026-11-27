"""Training Pipeline (T) for PoC Model 1 (credit_risk_batch): LightGBM -> ONNX.

Runs point-in-time Feast joins, value-skew checks, cohort fairness gate, and artifact persistence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import onnxmltools
import onnxruntime as ort
import pandas as pd
from onnxmltools.convert.common.data_types import FloatTensorType

from banking_fti.core.contracts import (
    CARD_WINDOW_AGGS_CONTRACT,
    CREDIT_APPLICATIONS_CONTRACT,
    CUSTOMER_PROFILES_CONTRACT,
    EVENT_TIMESTAMP_COL,
    compute_combined_schema_hash,
)
from banking_fti.ports.feature_store import FeatureStore
from banking_fti.ports.model_registry import ModelRegistry, RegisteredModel
from banking_fti.training.gates import (
    CreditRiskGateConfig,
    CreditRiskGateResult,
    ValueSkewConfig,
    check_value_skew,
    evaluate_credit_risk_gate,
)
from banking_fti.training.preprocessor import TabularPreprocessor


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of a TrainingPipeline run."""

    model_name: str
    gate_passed: bool
    holdout_auc: float
    holdout_recall: float
    brier: float
    onnx_max_abs_diff: float | None
    parity_pass: bool | None
    champion_version: int | None
    previous_version: int | None
    challenger_registered: bool
    dataset_sha256: str
    promoted: bool
    challenger_version: int | None = None
    rows_used: int = 0
    convergence_passed: bool = False
    nan_inf_passed: bool = False
    value_skew_passed: bool = False
    fairness_passed: bool = False
    serving_smoke_passed: bool = False
    champion_comparison_passed: bool = False
    gate_reasons: tuple[str, ...] = ()
    artifact_dir: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize training result into a JSON-compatible dictionary."""
        return {
            "model_name": self.model_name,
            "gate_passed": self.gate_passed,
            "holdout_auc": self.holdout_auc,
            "holdout_recall": self.holdout_recall,
            "brier": self.brier,
            "onnx_max_abs_diff": self.onnx_max_abs_diff,
            "parity_pass": self.parity_pass,
            "champion_version": self.champion_version,
            "previous_version": self.previous_version,
            "challenger_registered": self.challenger_registered,
            "challenger_version": self.challenger_version,
            "dataset_sha256": self.dataset_sha256,
            "promoted": self.promoted,
            "rows_used": self.rows_used,
            "convergence_passed": self.convergence_passed,
            "nan_inf_passed": self.nan_inf_passed,
            "value_skew_passed": self.value_skew_passed,
            "fairness_passed": self.fairness_passed,
            "serving_smoke_passed": self.serving_smoke_passed,
            "champion_comparison_passed": self.champion_comparison_passed,
            "gate_reasons": list(self.gate_reasons),
            "artifact_dir": self.artifact_dir,
        }


class TrainingPipeline:
    """PoC Model 1 (credit_risk_batch) Training Pipeline (T)."""

    def __init__(
        self,
        feature_store: FeatureStore,
        model_registry: ModelRegistry,
        model_name: str = "credit_risk_batch",
        *,
        artifact_dir: str | Path | None = None,
        gate_config: CreditRiskGateConfig | None = None,
        value_skew_config: ValueSkewConfig | None = None,
        compute_backend: str = "cpu",
        executor: str = "local_python",
        pipeline_version: str = "v0.1.0",
        dvc_lock_path: str | Path | None = None,
    ) -> None:
        self.feature_store = feature_store
        self.model_registry = model_registry
        self.model_name = model_name
        self._artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self._gate_config = gate_config or CreditRiskGateConfig()
        self._value_skew_config = value_skew_config or ValueSkewConfig()
        self._compute_backend = compute_backend
        self._executor = executor
        self._pipeline_version = pipeline_version
        self._dvc_lock_path = Path(dvc_lock_path) if dvc_lock_path is not None else Path("dvc.lock")
        self.preprocessor = TabularPreprocessor()
        self._native_model: lgb.LGBMClassifier | None = None
        self._onnx_session: ort.InferenceSession | None = None

    def train(
        self,
        *,
        test_size: float = 0.25,
        random_state: int = 42,
        num_boost_round: int = 100,
    ) -> TrainingResult:
        """Execute point-in-time join, preprocessing, LightGBM training, ONNX export, and gating."""
        started_at = datetime.now(UTC).isoformat()

        # 1. Read entity spine and fetch leak-free point-in-time joined snapshot from FeatureStore
        credit_apps = self.feature_store.read_feature_group("credit_applications")
        if credit_apps.empty or "default_label" not in credit_apps.columns:
            return self._empty_result()

        entity_df = (
            credit_apps.loc[:, ["application_id", "customer_id", EVENT_TIMESTAMP_COL]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        df = self.feature_store.get_historical_features(
            entity_df,
            ["customer_profiles", "card_window_aggs", "credit_applications"],
            exclude_labels=False,
        )
        if df.empty or "default_label" not in df.columns:
            return self._empty_result()

        # 2. Temporal split: train on earlier rows, holdout on latest rows
        df = df.sort_values([EVENT_TIMESTAMP_COL, "application_id"], kind="mergesort").reset_index(
            drop=True
        )
        cutoff = max(1, min(len(df) - 1, int(len(df) * (1.0 - test_size))))
        train_df = df.iloc[:cutoff].reset_index(drop=True)
        holdout_df = df.iloc[cutoff:].reset_index(drop=True)

        # 3. Pre-training statistical value-skew & finiteness check
        skew_report = check_value_skew(
            train_df,
            holdout_df,
            target_column="default_label",
            config=self._value_skew_config,
        )
        dataset_sha = self._compute_dataset_sha256(df)
        run_id = self._build_run_id(dataset_sha)
        run_dir = self._resolve_run_dir(run_id)

        numeric_finite = not any(finding.metric == "inf_count" for finding in skew_report.findings)

        # 4. Fit TabularPreprocessor solely on training split (guard non-finite inputs)
        can_fit = False
        if not numeric_finite:
            nan_inf_passed = False
            convergence_passed = False
            max_abs_diff = None
            serving_smoke_passed = False
            onnx_bytes = b""
            y_holdout = holdout_df["default_label"].to_numpy(dtype=np.int64)
            holdout_probs = np.zeros(len(holdout_df), dtype=np.float64)
        else:
            self.preprocessor.fit(train_df, target_column="default_label")
            X_train, y_train = self.preprocessor.transform(train_df)
            X_holdout, y_holdout = self.preprocessor.transform(holdout_df)
            nan_inf_passed = bool(np.isfinite(X_train).all() and np.isfinite(X_holdout).all())
            can_fit = (
                nan_inf_passed
                and train_df["default_label"].nunique() >= 2
                and holdout_df["default_label"].nunique() >= 2
            )

            if not can_fit:
                convergence_passed = False
                max_abs_diff = None
                serving_smoke_passed = False
                onnx_bytes = b""
                holdout_probs = np.zeros(len(holdout_df), dtype=np.float64)
            else:
                convergence_passed, onnx_bytes, max_abs_diff, serving_smoke_passed = (
                    self._fit_and_export(
                        X_train=X_train,
                        y_train=y_train,
                        X_holdout=X_holdout,
                        y_holdout=y_holdout,
                        num_boost_round=num_boost_round,
                        random_state=random_state,
                    )
                )
                holdout_probs = self._predict_native(X_holdout)
                nan_inf_passed = nan_inf_passed and bool(np.isfinite(holdout_probs).all())

        # 5. Evaluate Credit Risk & Cohort Fairness Validation Gate
        current_champion = self.model_registry.get_champion(self.model_name)
        gate_result = evaluate_credit_risk_gate(
            holdout_df=holdout_df,
            y_true=y_holdout,
            y_prob=holdout_probs,
            onnx_max_abs_diff=max_abs_diff,
            convergence_passed=convergence_passed,
            nan_inf_passed=nan_inf_passed,
            value_skew_passed=skew_report.passed,
            serving_smoke_passed=serving_smoke_passed,
            current_champion=current_champion,
            config=self._gate_config,
        )

        # 6. Persist artifacts and register @challenger only when training built a bundle
        if not can_fit or not onnx_bytes or not self.preprocessor.is_fitted:
            return self._build_training_result(
                gate_result=gate_result,
                dataset_sha=dataset_sha,
                rows_used=len(df),
                challenger_registered=False,
                challenger_version=None,
                champion_version=self._version_of(
                    self.model_registry.get_champion(self.model_name)
                ),
                previous_version=self._version_of(
                    self.model_registry.get_previous(self.model_name)
                ),
                promoted=False,
                artifact_dir=None,
            )

        native_path, onnx_path, preproc_path = self._persist_artifacts(run_dir, onnx_bytes)

        # 7. Register challenger with full lineage metadata and promote only if gate passes
        ended_at = datetime.now(UTC).isoformat()
        candidate = self._build_registered_model(
            run_id=run_id,
            dataset_sha=dataset_sha,
            df=df,
            native_path=native_path,
            onnx_path=onnx_path,
            preproc_path=preproc_path,
            started_at=started_at,
            ended_at=ended_at,
            gate_result=gate_result,
        )
        registered_challenger = self.model_registry.register_challenger(candidate)

        promoted = False
        champion_version = self._version_of(self.model_registry.get_champion(self.model_name))
        previous_version = self._version_of(self.model_registry.get_previous(self.model_name))
        if gate_result.passed:
            promote_result = self.model_registry.promote_to_champion(
                model_name=self.model_name,
                challenger_version=registered_challenger.version,
                validation_passed=True,
                validation_summary=(
                    f"AUC={gate_result.roc_auc:.3f}, "
                    f"Recall={gate_result.default_recall:.3f}, "
                    f"Brier={gate_result.brier_score:.3f}, "
                    f"min_cohort_recall={gate_result.min_cohort_recall:.3f}, "
                    f"parity={gate_result.parity_passed}"
                ),
            )
            promoted = bool(promote_result.get("ok"))
            champion_version = self._as_int(promote_result.get("champion_version"))
            previous_version = self._as_int(promote_result.get("previous_version"))

        return self._build_training_result(
            gate_result=gate_result,
            dataset_sha=dataset_sha,
            rows_used=len(df),
            challenger_registered=True,
            challenger_version=registered_challenger.version,
            champion_version=champion_version,
            previous_version=previous_version,
            promoted=promoted,
            artifact_dir=str(run_dir),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_and_export(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_holdout: np.ndarray,
        y_holdout: np.ndarray,
        num_boost_round: int,
        random_state: int,
    ) -> tuple[bool, bytes, float | None, bool]:
        self._native_model = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
            n_estimators=num_boost_round,
            class_weight="balanced",
            verbosity=-1,
            random_state=random_state,
        )
        evals_result: dict[str, dict[str, list[float]]] = {}
        self._native_model.fit(
            X_train,
            y_train,
            eval_X=cast(Any, (X_train, X_holdout)),
            eval_y=cast(Any, (y_train, y_holdout)),
            eval_names=["train", "holdout"],
            eval_metric="binary_logloss",
            callbacks=[
                lgb.record_evaluation(evals_result),
                lgb.early_stopping(stopping_rounds=20, verbose=False),
            ],
        )

        train_losses = evals_result.get("train", {}).get("binary_logloss", [])
        convergence_passed = bool(
            len(train_losses) >= 2
            and all(np.isfinite(loss) for loss in train_losses)
            and float(train_losses[-1]) < float(train_losses[0])
        )

        onnx_bytes, max_abs_diff = self._export_and_verify_onnx(X_holdout)
        serving_smoke_passed = self._run_serving_smoke_check(onnx_bytes, X_holdout)
        return convergence_passed, onnx_bytes, max_abs_diff, serving_smoke_passed

    def _export_and_verify_onnx(self, X_holdout: np.ndarray) -> tuple[bytes, float | None]:
        if self._native_model is None:
            return b"", None
        try:
            onnx_model = onnxmltools.convert_lightgbm(
                self._native_model,
                initial_types=[("input", FloatTensorType([None, X_holdout.shape[1]]))],
                zipmap=False,
            )
            # Set dynamic batch dim on graph outputs to avoid VerifyOutputSizes warnings
            for output in onnx_model.graph.output:
                dims = output.type.tensor_type.shape.dim
                if dims:
                    dims[0].ClearField("dim_value")
                    dims[0].dim_param = "N"
            onnx_bytes = bytes(onnx_model.SerializeToString())
            sess = ort.InferenceSession(onnx_bytes)
            self._onnx_session = sess
        except Exception:
            return b"", None

        native = self._predict_native(X_holdout)
        onnx = self._predict_onnx(X_holdout)
        max_abs_diff = float(np.max(np.abs(native - onnx)))
        return onnx_bytes, max_abs_diff

    def _run_serving_smoke_check(self, onnx_bytes: bytes, X_holdout: np.ndarray) -> bool:
        if not onnx_bytes or len(X_holdout) == 0:
            return False
        try:
            sample = X_holdout[: min(8, len(X_holdout))].astype(np.float32)
            sess = ort.InferenceSession(onnx_bytes)
            outputs = sess.run(None, {"input": sample})
            probs = self._extract_positive_proba(outputs, len(sample))
            return bool(
                probs.shape == (len(sample),)
                and np.isfinite(probs).all()
                and np.all((probs >= 0.0) & (probs <= 1.0))
            )
        except Exception:
            return False

    def _predict_native(self, X: np.ndarray) -> np.ndarray:
        if self._native_model is None:
            return np.zeros(len(X), dtype=np.float64)
        return np.asarray(self._native_model.predict_proba(X), dtype=np.float64)[:, 1]

    def _predict_onnx(self, X: np.ndarray) -> np.ndarray:
        if self._onnx_session is None:
            return np.zeros(len(X), dtype=np.float64)
        outputs = self._onnx_session.run(None, {"input": X.astype(np.float32)})
        return self._extract_positive_proba(outputs, len(X))

    @staticmethod
    def _extract_positive_proba(outputs: Sequence[Any], expected_rows: int) -> np.ndarray:
        if len(outputs) >= 2:
            second = outputs[1]
            if isinstance(second, list):
                return np.asarray([float(row.get(1, 0.0)) for row in second], dtype=np.float64)
            arr = np.asarray(second, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, 1]
        for out in outputs:
            arr = np.asarray(out, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, 1]
        return np.zeros(expected_rows, dtype=np.float64)

    def _resolve_run_dir(self, run_id: str) -> Path:
        base_dir = (
            self._artifact_dir
            if self._artifact_dir is not None
            else Path(tempfile.mkdtemp(prefix="banking_fti_models_"))
        )
        run_dir = base_dir / self.model_name / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _persist_artifacts(self, run_dir: Path, onnx_bytes: bytes) -> tuple[str, str, str]:
        native_path = run_dir / "model.txt"
        onnx_path = run_dir / "model.onnx"
        preproc_path = run_dir / "preprocessor.joblib"

        if self._native_model is not None:
            self._native_model.booster_.save_model(str(native_path))
        if onnx_bytes:
            onnx_path.write_bytes(onnx_bytes)
        if self.preprocessor.is_fitted:
            self.preprocessor.save(preproc_path)

        return str(native_path), str(onnx_path), str(preproc_path)

    def _build_registered_model(
        self,
        *,
        run_id: str,
        dataset_sha: str,
        df: pd.DataFrame,
        native_path: str,
        onnx_path: str,
        preproc_path: str,
        started_at: str,
        ended_at: str,
        gate_result: CreditRiskGateResult,
    ) -> RegisteredModel:
        git_sha = self._resolve_git_sha()
        dvc_lock_hash = self._resolve_dvc_lock_hash(self._dvc_lock_path)
        schema_hash = compute_combined_schema_hash(
            [
                CUSTOMER_PROFILES_CONTRACT,
                CARD_WINDOW_AGGS_CONTRACT,
                CREDIT_APPLICATIONS_CONTRACT,
            ]
        )
        return RegisteredModel(
            model_name=self.model_name,
            version=0,
            status="challenger",
            run_id=run_id,
            created_at=ended_at,
            feature_view_name="credit_applications",
            feature_view_version="v1",
            feature_schema_hash=schema_hash,
            dataset_sha256=dataset_sha,
            native_model_path=native_path,
            onnx_model_path=onnx_path,
            preprocessor_path=preproc_path,
            git_sha=git_sha,
            dvc_lock_hash=dvc_lock_hash,
            pipeline_version=self._pipeline_version,
            compute_backend=self._compute_backend,
            executor=self._executor,
            timestamp_start=started_at,
            timestamp_end=ended_at,
            metrics={
                "roc_auc": gate_result.roc_auc,
                "default_recall": gate_result.default_recall,
                "brier_score": gate_result.brier_score,
                "min_cohort_recall": gate_result.min_cohort_recall,
                "max_cohort_recall_gap": gate_result.max_cohort_recall_gap,
            },
        )

    @staticmethod
    def _compute_dataset_sha256(df: pd.DataFrame) -> str:
        df_copy = df.copy()
        if EVENT_TIMESTAMP_COL in df_copy.columns:
            df_copy[EVENT_TIMESTAMP_COL] = pd.to_datetime(
                df_copy[EVENT_TIMESTAMP_COL], utc=True
            ).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        payload = json.dumps(
            df_copy.to_dict(orient="records"),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _build_training_result(
        self,
        *,
        gate_result: CreditRiskGateResult,
        dataset_sha: str,
        rows_used: int,
        challenger_registered: bool,
        challenger_version: int | None,
        champion_version: int | None,
        previous_version: int | None,
        promoted: bool,
        artifact_dir: str | None,
    ) -> TrainingResult:
        return TrainingResult(
            model_name=self.model_name,
            gate_passed=gate_result.passed,
            holdout_auc=gate_result.roc_auc,
            holdout_recall=gate_result.default_recall,
            brier=gate_result.brier_score,
            onnx_max_abs_diff=gate_result.onnx_max_abs_diff,
            parity_pass=gate_result.parity_passed,
            champion_version=champion_version,
            previous_version=previous_version,
            challenger_registered=challenger_registered,
            challenger_version=challenger_version,
            dataset_sha256=dataset_sha,
            promoted=promoted,
            rows_used=rows_used,
            convergence_passed=gate_result.convergence_passed,
            nan_inf_passed=gate_result.nan_inf_passed,
            value_skew_passed=gate_result.value_skew_passed,
            fairness_passed=gate_result.fairness_passed,
            serving_smoke_passed=gate_result.serving_smoke_passed,
            champion_comparison_passed=gate_result.champion_comparison_passed,
            gate_reasons=gate_result.reasons,
            artifact_dir=artifact_dir,
        )

    @staticmethod
    def _build_run_id(dataset_sha: str) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}-{dataset_sha[:8]}"

    @staticmethod
    def _resolve_git_sha() -> str:
        env_sha = os.environ.get("GIT_SHA")
        if env_sha:
            return env_sha.strip()
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    @staticmethod
    def _resolve_dvc_lock_hash(dvc_lock_path: Path) -> str:
        if dvc_lock_path.exists():
            return hashlib.sha256(dvc_lock_path.read_bytes()).hexdigest()
        return os.environ.get("DVC_LOCK_HASH", "untracked")

    def _empty_result(self) -> TrainingResult:
        return TrainingResult(
            model_name=self.model_name,
            gate_passed=False,
            holdout_auc=0.0,
            holdout_recall=0.0,
            brier=1.0,
            onnx_max_abs_diff=None,
            parity_pass=None,
            champion_version=None,
            previous_version=None,
            challenger_registered=False,
            dataset_sha256="",
            promoted=False,
        )

    @staticmethod
    def _as_int(value: Any) -> int | None:
        return int(value) if isinstance(value, int) else None

    @staticmethod
    def _version_of(model: RegisteredModel | None) -> int | None:
        return model.version if model is not None else None


def run_stage1_training(
    *,
    repo_path: str | Path,
    data_dir: str | Path | None = None,
    artifact_dir: str | Path = "artifacts/models",
    model_name: str = "credit_risk_batch",
    test_size: float = 0.25,
    num_boost_round: int = 100,
    seed: int = 42,
    compute_backend: str = "cpu",
    executor: str = "local_python",
) -> TrainingResult:
    """Run Stage 1 Training Pipeline against FeastFeatureStore and InMemoryModelRegistry."""
    from banking_fti.adapters.feast_store import FeastFeatureStore
    from banking_fti.adapters.in_memory_model_registry import InMemoryModelRegistry

    store = FeastFeatureStore(repo_path=repo_path, data_dir=data_dir)
    registry = InMemoryModelRegistry()
    pipeline = TrainingPipeline(
        feature_store=store,
        model_registry=registry,
        model_name=model_name,
        artifact_dir=artifact_dir,
        compute_backend=compute_backend,
        executor=executor,
    )
    return pipeline.train(
        test_size=test_size,
        random_state=seed,
        num_boost_round=num_boost_round,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 1 Credit Risk Training Pipeline (LightGBM -> ONNX)."
    )
    parser.add_argument(
        "--repo-path",
        default=os.environ.get("FEAST_REPO_PATH", "feature_repo"),
        help="Path to the Feast repository directory.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("FEAST_DATA_DIR"),
        help="Optional directory for Offline Parquet files and Feast registry.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=os.environ.get("MODEL_ARTIFACT_DIR", "artifacts/models"),
        help="Directory for persisted native, ONNX, and preprocessor artifacts.",
    )
    parser.add_argument("--model", default="credit_risk_batch")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--num-boost-round", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compute-backend", default="cpu")
    parser.add_argument(
        "--executor",
        default=os.environ.get("TRAINING_EXECUTOR", "local_python"),
    )
    parser.add_argument(
        "--require-promotion",
        action="store_true",
        help="Exit with code 1 if the model validation gate blocks promotion.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for containerized and local Stage 1 Training Pipeline runs."""
    args = _build_arg_parser().parse_args(argv)
    result = run_stage1_training(
        repo_path=args.repo_path,
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        model_name=args.model,
        test_size=args.test_size,
        num_boost_round=args.num_boost_round,
        seed=args.seed,
        compute_backend=args.compute_backend,
        executor=args.executor,
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    if args.require_promotion and not result.promoted:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
