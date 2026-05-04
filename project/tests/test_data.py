"""Sanity checks for data loading and preprocessing."""
from pathlib import Path

import pandas as pd

from src.data.load import load_clean
from src.data.preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_preprocessor,
    train_test_split_stratified,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"


def test_data_file_exists():
    assert DATA_PATH.exists(), f"Dataset missing at {DATA_PATH}"


def test_load_clean_no_nans():
    df = load_clean(DATA_PATH)
    assert df.isna().sum().sum() == 0
    assert TARGET in df.columns
    assert df[TARGET].isin([0, 1]).all()


def test_load_clean_expected_columns():
    df = load_clean(DATA_PATH)
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        assert col in df.columns, f"Missing column: {col}"
    assert "customerID" not in df.columns


def test_train_test_split_stratified_shapes():
    df = load_clean(DATA_PATH)
    x_tr, x_te, y_tr, y_te = train_test_split_stratified(df, test_size=0.2)
    assert len(x_tr) + len(x_te) == len(df)
    assert abs(y_tr.mean() - y_te.mean()) < 0.01


def test_preprocessor_fit_transform():
    df = load_clean(DATA_PATH)
    x_tr, _, _, _ = train_test_split_stratified(df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(x_tr)
    assert transformed.shape[0] == len(x_tr)
    assert transformed.shape[1] > len(NUMERIC_FEATURES)
