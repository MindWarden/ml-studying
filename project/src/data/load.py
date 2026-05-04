"""Loading and basic cleaning of the Telco Customer Churn dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path("data/raw/telco_churn.csv")


def load_raw(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges приходит строкой и для tenure=0 содержит пробелы
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Целевая переменная Yes/No -> 1/0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def load_clean(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))
