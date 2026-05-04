"""Train multiple models, compare metrics, save the best one."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data.load import load_clean
from src.data.preprocess import build_preprocessor, train_test_split_stratified

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

RANDOM_STATE = 42


def get_models() -> dict[str, object]:
    return {
        "logreg": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def train_and_compare() -> dict:
    log.info("Loading data from %s", DATA_PATH)
    df = load_clean(DATA_PATH)
    log.info("Data shape: %s, churn rate: %.3f", df.shape, df["Churn"].mean())

    x_train, x_test, y_train, y_test = train_test_split_stratified(df)
    log.info("Train: %d, Test: %d", len(x_train), len(x_test))

    results: dict[str, dict] = {}
    pipelines: dict[str, Pipeline] = {}

    for name, model in get_models().items():
        log.info("Training %s ...", name)
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor()), ("model", model)]
        )
        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_test)
        y_proba = pipeline.predict_proba(x_test)[:, 1]
        metrics = evaluate(y_test, y_pred, y_proba)

        log.info(
            "%s: ROC-AUC=%.4f F1=%.4f P=%.4f R=%.4f Acc=%.4f",
            name,
            metrics["roc_auc"],
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            metrics["accuracy"],
        )
        results[name] = metrics
        pipelines[name] = pipeline

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    log.info("Best model by ROC-AUC: %s", best_name)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipelines[best_name], MODEL_PATH)
    log.info("Saved best pipeline to %s", MODEL_PATH)

    summary = {
        "best_model": best_name,
        "metrics": results,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "churn_rate_train": float(np.mean(y_train)),
        "churn_rate_test": float(np.mean(y_test)),
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("Saved metrics to %s", METRICS_PATH)

    return summary


if __name__ == "__main__":
    train_and_compare()
