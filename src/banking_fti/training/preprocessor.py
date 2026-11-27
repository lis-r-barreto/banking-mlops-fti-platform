"""Train-split-only TabularPreprocessor for credit risk features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class PreprocessorBundle:
    """Fitted transformers and feature metadata for deterministic serving."""

    transformer: ColumnTransformer
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

    def fit(self, df: pd.DataFrame, target_column: str = "default_label") -> PreprocessorBundle:
        """Fit the preprocessor on a training DataFrame (unencoded features)."""
        if target_column not in df.columns:
            raise ValueError(f"target_column '{target_column}' not in DataFrame")

        feature_cols = [column for column in df.columns if column != target_column]
        if not feature_cols:
            raise ValueError("no feature columns after excluding target")

        numeric_cols = [
            column for column in feature_cols if pd.api.types.is_numeric_dtype(df[column])
        ]
        categorical_cols = [
            column for column in feature_cols if not pd.api.types.is_numeric_dtype(df[column])
        ]

        transformers: list[tuple[str, object, list[str]]] = []
        if numeric_cols:
            transformers.append(("num", StandardScaler(), numeric_cols))
        if categorical_cols:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_cols,
                )
            )

        ct = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        ct.fit(df[feature_cols])

        bundle = PreprocessorBundle(
            transformer=ct,
            feature_names=tuple(ct.get_feature_names_out()),
            target_column=target_column,
            fitted_at=pd.Timestamp.now(tz="UTC").isoformat(),
        )
        self._fitted = bundle
        return bundle

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Transform features and extract the target array."""
        bundle = self._fitted
        if bundle is None:
            raise RuntimeError("preprocessor not fitted; call fit() first")

        target = df[bundle.target_column].to_numpy(dtype=np.int64)
        feature_cols = [column for column in df.columns if column != bundle.target_column]
        X = bundle.transformer.transform(df[feature_cols])
        return X, target

    def fit_transform(
        self, df: pd.DataFrame, target_column: str = "default_label"
    ) -> tuple[np.ndarray, np.ndarray]:
        self.fit(df, target_column)
        return self.transform(df)

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("nothing to save: preprocessor not fitted")
        joblib.dump(self._fitted, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> TabularPreprocessor:
        bundle = joblib.load(Path(path))
        instance = cls()
        instance._fitted = bundle
        return instance
