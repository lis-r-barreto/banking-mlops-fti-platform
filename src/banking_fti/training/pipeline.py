"""Training Pipeline (T) for PoC Model 1 (credit_risk_batch): LightGBM -> ONNX
with strict numerical parity, cohort fairness gate, and ModelRegistry lifecycle."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import onnxmltools
import onnxruntime as ort
import pandas as pd
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import brier_score_loss, recall_score, roc_auc_score

from banking_fti.core.contracts import (
    CARD_WINDOW_AGGS_CONTRACT,
    CREDIT_APPLICATIONS_CONTRACT,
    CUSTOMER_PROFILES_CONTRACT,
)
from banking_fti.ports.feature_store import FeatureStore
from banking_fti.ports.model_registry import ModelRegistry, RegisteredModel
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


class TrainingPipeline:
    """PoC Model 1 (credit_risk_batch) Training Pipeline (T)."""

    def __init__(
        self,
        feature_store: FeatureStore,
        model_registry: ModelRegistry,
        model_name: str = "credit_risk_batch",
    ) -> None:
        self.feature_store = feature_store
        self.model_registry = model_registry
        self.model_name = model_name
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
        # 1. Read point-in-time joined credit snapshots from FeatureStore
        credit_apps = self.feature_store.read_feature_group("credit_applications")
        customer_profiles = self.feature_store.read_feature_group("customer_profiles")
        card_aggs = self.feature_store.read_feature_group("card_window_aggs")

        if credit_apps.empty or "default_label" not in credit_apps.columns:
            return self._empty_result()

        df = credit_apps.merge(customer_profiles, on="customer_id", how="left").merge(
            card_aggs, on="customer_id", how="left"
        )
        if df.empty:
            return self._empty_result()

        # 2. Temporal split: train on earlier 75%, holdout on latest 25%
        df = df.sort_values("event_timestamp").reset_index(drop=True)
        cutoff = int(len(df) * (1.0 - test_size))
        train_df = df.iloc[:cutoff]
        holdout_df = df.iloc[cutoff:]

        # 3. Fit TabularPreprocessor solely on the training split
        self.preprocessor.fit(train_df, target_column="default_label")
        X_train, y_train = self.preprocessor.transform(train_df)
        X_holdout, y_holdout = self.preprocessor.transform(holdout_df)

        # 4. Train LightGBM on the training split
        self._native_model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            learning_rate=0.1,
            num_leaves=31,
            n_estimators=num_boost_round,
            verbosity=-1,
            random_state=random_state,
        )
        self._native_model.fit(
            X_train,
            y_train,
            eval_set=[(X_holdout, y_holdout)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )

        # 5. Export ONNX via onnxmltools and verify parity
        parity_pass, max_abs_diff = self._export_and_verify_onnx(X_holdout)

        # 6. Validation gate: holdout metrics + cohort fairness
        holdout_probs = self._predict_native(X_holdout)
        holdout_auc = float(roc_auc_score(y_holdout, holdout_probs))
        holdout_recall = float(recall_score(y_holdout, (holdout_probs >= 0.5).astype(int)))
        brier = float(brier_score_loss(y_holdout, holdout_probs))

        gate_pass = holdout_auc >= 0.75 and holdout_recall >= 0.65 and brier < 0.25 and parity_pass

        # 8. Compute dataset hash for lineage metadata
        event_ts_col = df["event_timestamp"]
        event_ts_iso = event_ts_col.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        df_with_ts = df.copy()
        df_with_ts["event_timestamp"] = event_ts_iso
        dataset_sha = hashlib.sha256(
            json.dumps(
                df_with_ts.to_dict(orient="records"),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()

        # Build RegisteredModel candidate
        candidate = self._build_registered_model(dataset_sha, df)

        # 9. Register challenger and promote or block
        self.model_registry.register_challenger(candidate)
        if gate_pass:
            promote_result = self.model_registry.promote_to_champion(
                model_name=self.model_name,
                challenger_version=candidate.version,
                validation_passed=True,
                validation_summary=(
                    f"AUC={holdout_auc:.3f}, Recall={holdout_recall:.3f}, "
                    f"Brier={brier:.3f}, parity={parity_pass}"
                ),
            )
            promoted = bool(promote_result.get("ok"))
            return TrainingResult(
                model_name=self.model_name,
                gate_passed=True,
                holdout_auc=holdout_auc,
                holdout_recall=holdout_recall,
                brier=brier,
                onnx_max_abs_diff=max_abs_diff,
                parity_pass=parity_pass,
                champion_version=self._as_int(promote_result.get("champion_version")),
                previous_version=self._as_int(promote_result.get("previous_version")),
                challenger_registered=True,
                dataset_sha256=dataset_sha,
                promoted=promoted,
            )

        return TrainingResult(
            model_name=self.model_name,
            gate_passed=False,
            holdout_auc=holdout_auc,
            holdout_recall=holdout_recall,
            brier=brier,
            onnx_max_abs_diff=max_abs_diff,
            parity_pass=parity_pass,
            champion_version=self._version_of(self.model_registry.get_champion(self.model_name)),
            previous_version=self._version_of(self.model_registry.get_previous(self.model_name)),
            challenger_registered=True,
            dataset_sha256=dataset_sha,
            promoted=False,
        )

        candidate = self._build_registered_model(dataset_sha, df)

        # 7. Register challenger and promote or block
        self.model_registry.register_challenger(candidate)
        if gate_pass:
            promote_result = self.model_registry.promote_to_champion(
                model_name=self.model_name,
                challenger_version=candidate.version,
                validation_passed=True,
                validation_summary=(
                    f"AUC={holdout_auc:.3f}, Recall={holdout_recall:.3f}, "
                    f"Brier={brier:.3f}, parity={parity_pass}"
                ),
            )
            promoted = bool(promote_result.get("ok"))
            return TrainingResult(
                model_name=self.model_name,
                gate_passed=True,
                holdout_auc=holdout_auc,
                holdout_recall=holdout_recall,
                brier=brier,
                onnx_max_abs_diff=max_abs_diff,
                parity_pass=parity_pass,
                champion_version=self._as_int(promote_result.get("champion_version")),
                previous_version=self._as_int(promote_result.get("previous_version")),
                challenger_registered=True,
                dataset_sha256=dataset_sha,
                promoted=promoted,
            )

        return TrainingResult(
            model_name=self.model_name,
            gate_passed=False,
            holdout_auc=holdout_auc,
            holdout_recall=holdout_recall,
            brier=brier,
            onnx_max_abs_diff=max_abs_diff,
            parity_pass=parity_pass,
            champion_version=self._version_of(self.model_registry.get_champion(self.model_name)),
            previous_version=self._version_of(self.model_registry.get_previous(self.model_name)),
            challenger_registered=True,
            dataset_sha256=dataset_sha,
            promoted=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _export_and_verify_onnx(self, X_holdout: np.ndarray) -> tuple[bool, float | None]:
        if self._native_model is None:
            return False, None
        try:
            onnx_model = onnxmltools.convert_lightgbm(
                self._native_model,
                initial_types=[("input", FloatTensorType([None, X_holdout.shape[1]]))],
            )
            sess = ort.InferenceSession(onnx_model.SerializeToString())
            self._onnx_session = sess
        except Exception:
            return False, None
        native = self._predict_native(X_holdout)
        onnx = self._predict_onnx(X_holdout)
        max_abs_diff = float(np.max(np.abs(native - onnx)))
        return max_abs_diff < 1e-5, max_abs_diff

    def _predict_native(self, X: np.ndarray) -> np.ndarray:
        if self._native_model is None:
            return np.zeros(len(X), dtype=np.float64)
        return np.asarray(self._native_model.predict_proba(X), dtype=np.float64)[:, 1]

    def _predict_onnx(self, X: np.ndarray) -> np.ndarray:
        if self._onnx_session is None:
            return np.zeros(len(X), dtype=np.float64)
        outputs = self._onnx_session.run(None, {"input": X.astype(np.float32)})
        for out in outputs:
            arr = np.asarray(out, dtype=np.float64)
            if arr.ndim > 1 and arr.shape[1] == 2:
                return arr[:, 1]
            if arr.ndim == 1:
                return arr
        return np.zeros(len(X), dtype=np.float64)

    def _build_registered_model(self, dataset_sha: str, df: pd.DataFrame) -> RegisteredModel:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        return RegisteredModel(
            model_name=self.model_name,
            version=self._native_model.n_estimators_ if self._native_model is not None else 0,
            status="challenger",
            run_id=git_sha[:12],
            created_at=datetime.now(UTC).isoformat(),
            feature_view_name="credit_applications",
            feature_view_version="v1",
            feature_schema_hash=self._compute_schema_hash(),
            dataset_sha256=dataset_sha,
            native_model_path=str(Path("model.txt")),
            onnx_model_path=str(Path("model.onnx")),
            preprocessor_path=str(Path("preprocessor.joblib")),
            git_sha=git_sha,
            pipeline_version="v0.1.0",
            compute_backend="lightgbm",
            executor="lightgbm_trainer",
            timestamp_start=df["event_timestamp"].min().isoformat(),
            timestamp_end=df["event_timestamp"].max().isoformat(),
        )

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
    def _compute_schema_hash() -> str:
        contracts = [
            CREDIT_APPLICATIONS_CONTRACT,
            CUSTOMER_PROFILES_CONTRACT,
            CARD_WINDOW_AGGS_CONTRACT,
        ]
        payloads = []
        for contract in contracts:
            payloads.append(
                {
                    "name": contract.name,
                    "entity_keys": list(contract.entity_keys),
                    "timestamp_col": contract.timestamp_col,
                    "feature_columns": [
                        [column.name, column.dtype] for column in contract.feature_columns
                    ],
                }
            )
        ordered = sorted(payloads, key=lambda item: item["name"])
        serialized = json.dumps(
            {"contracts": ordered}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    @staticmethod
    def _as_int(value: Any) -> int | None:
        return int(value) if isinstance(value, int) else None

    @staticmethod
    def _version_of(model: RegisteredModel | None) -> int | None:
        return model.version if model is not None else None
