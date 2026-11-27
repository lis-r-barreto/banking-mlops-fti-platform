"""Tests for TabularPreprocessor: fit on training split only."""

from __future__ import annotations

import pandas as pd

from banking_fti.training.preprocessor import TabularPreprocessor


def test_fit_and_transform_roundtrip() -> None:
    df = pd.DataFrame(
        {
            "loan_amount": [450000.0, 320000.0, 240000.0, 180000.0, 300000.0],
            "loan_term_months": [36, 24, 36, 36, 60],
            "dti_ratio": [0.2, 0.5, 0.3, 0.4, 0.6],
            "home_ownership": ["RENT", "MORTGAGE", "RENT", "OWN", "RENT"],
            "default_label": [0, 1, 0, 1, 0],
        }
    )
    preprocessor = TabularPreprocessor()
    assert preprocessor.is_fitted is False

    X_train, y_train = preprocessor.fit_transform(df)
    assert preprocessor.is_fitted is True
    assert X_train.shape[0] == len(df)
    assert y_train.tolist() == [0, 1, 0, 1, 0]
    assert len(preprocessor.feature_names) == X_train.shape[1]


def test_transform_does_not_refit_and_matches_shape() -> None:
    train_df = pd.DataFrame(
        {
            "loan_amount": [450000.0, 320000.0, 240000.0],
            "loan_term_months": [36, 24, 36],
            "dti_ratio": [0.2, 0.5, 0.3],
            "home_ownership": ["RENT", "MORTGAGE", "RENT"],
            "default_label": [0, 1, 0],
        }
    )
    preprocessor = TabularPreprocessor()
    preprocessor.fit(train_df)

    new_df = pd.DataFrame(
        {
            "loan_amount": [500000.0],
            "loan_term_months": [60],
            "dti_ratio": [0.7],
            "home_ownership": ["OWN"],
            "default_label": [1],
        }
    )
    X_new, y_new = preprocessor.transform(new_df)
    assert y_new.tolist() == [1]
    assert X_new.shape[1] == preprocessor.feature_names.__len__()


def test_fit_uses_training_split_only() -> None:
    train_df = pd.DataFrame(
        {
            "loan_amount": [100000.0, 200000.0, 300000.0],
            "loan_term_months": [36, 36, 36],
            "dti_ratio": [0.2, 0.3, 0.4],
            "home_ownership": ["RENT", "RENT", "RENT"],
            "default_label": [0, 0, 0],
        }
    )
    holdout_df = pd.DataFrame(
        {
            "loan_amount": [1000000.0],
            "loan_term_months": [360],
            "dti_ratio": [0.9],
            "home_ownership": ["OWN"],
            "default_label": [1],
        }
    )
    preprocessor = TabularPreprocessor()
    preprocessor.fit(train_df)

    X_train, _ = preprocessor.transform(train_df)
    X_holdout, _ = preprocessor.transform(holdout_df)
    assert X_train.shape[1] == X_holdout.shape[1]
    assert X_train.shape[0] == len(train_df)
    assert X_holdout.shape[0] == len(holdout_df)
