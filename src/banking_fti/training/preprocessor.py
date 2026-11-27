"""Train-split-only TabularPreprocessor for credit risk features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_EXCLUDED_COLUMNS: tuple[str, ...] = (
    "application_id",
    "customer_id",
    "cc_num",
    "tx_id",
    "event_timestamp",
)


@dataclass(frozen=True)
class PreprocessorBundle:
    """Fitted transformers and feature metadata for deterministic serving."""

    transformer: ColumnTransformer
    input_columns: tuple[str, ...]
    feature_names: tuple[str, ...]
    target_column: str
    fitted_at: str


class TabularPreprocessor:
    """Column-aware preprocessor that fits ONLY on the training split."""

    def __init__(self) -> None:
        self._fitted: PreprocessorBundle | None = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted is not None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._fitted.feature_names if self._fitted is not None else ()

    @property
    def input_columns(self) -> tuple[str, ...]:
        return self._fitted.input_columns if self._fitted is not None else ()

    def fit(
        self,
        df: pd.DataFrame,
        target_column: str = "default_label",
        exclude_columns: Sequence[str] = DEFAULT_EXCLUDED_COLUMNS,
    ) -> PreprocessorBundle:
        """Fit the preprocessor on a training DataFrame (unencoded features)."""
        if target_column not in df.columns:
            raise ValueError(f"target_column '{target_column}' not in DataFrame")

        excluded = set(exclude_columns) | {target_column}
        feature_cols = [
            column
            for column in df.columns
            if column not in excluded and not pd.api.types.is_datetime64_any_dtype(df[column])
        ]
        if not feature_cols:
            raise ValueError("no feature columns after excluding target and metadata columns")

        numeric_cols = [
            column for column in feature_cols if pd.api.types.is_numeric_dtype(df[column])
        ]
        categorical_cols = [
            column for column in feature_cols if not pd.api.types.is_numeric_dtype(df[column])
        ]

        transformers: list[tuple[str, object, list[str]]] = []
        if numeric_cols:
            numeric_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("num", numeric_pipe, numeric_cols))
        if categorical_cols:
            categorical_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("cat", categorical_pipe, categorical_cols))

        ct = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        prepared = self._prepare_inputs(df, feature_cols, categorical_cols)
        ct.fit(prepared)

        bundle = PreprocessorBundle(
            transformer=ct,
            input_columns=tuple(feature_cols),
            feature_names=tuple(ct.get_feature_names_out()),
            target_column=target_column,
            fitted_at=pd.Timestamp.now(tz="UTC").isoformat(),
        )
        self._fitted = bundle
        return bundle

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform feature columns into a float32 numpy matrix without requiring target."""
        bundle = self._fitted
        if bundle is None:
            raise RuntimeError("preprocessor not fitted; call fit() first")

        missing = [col for col in bundle.input_columns if col not in df.columns]
        if missing:
            raise ValueError(f"missing required feature columns: {missing}")

        categorical_cols = [
            col for col in bundle.input_columns if not pd.api.types.is_numeric_dtype(df[col])
        ]
        prepared = self._prepare_inputs(df, list(bundle.input_columns), categorical_cols)
        X = bundle.transformer.transform(prepared)
        return np.asarray(X, dtype=np.float32)

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Transform features and extract the target array."""
        bundle = self._fitted
        if bundle is None:
            raise RuntimeError("preprocessor not fitted; call fit() first")
        if bundle.target_column not in df.columns:
            raise ValueError(f"target_column '{bundle.target_column}' not in DataFrame")

        target = df[bundle.target_column].to_numpy(dtype=np.int64)
        X = self.transform_features(df)
        return X, target

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: str = "default_label",
        exclude_columns: Sequence[str] = DEFAULT_EXCLUDED_COLUMNS,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.fit(df, target_column, exclude_columns=exclude_columns)
        return self.transform(df)

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("nothing to save: preprocessor not fitted")
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._fitted, target_path)

    @classmethod
    def load(cls, path: str | Path) -> TabularPreprocessor:
        bundle = joblib.load(Path(path))
        instance = cls()
        instance._fitted = bundle
        return instance

    @staticmethod
    def _prepare_inputs(
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        categorical_cols: Sequence[str],
    ) -> pd.DataFrame:
        prepared = df.loc[:, list(feature_cols)].copy()
        for col in categorical_cols:
            prepared[col] = prepared[col].astype(object).where(prepared[col].notna(), np.nan)
        return prepared
