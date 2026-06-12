"""Train baseline and tuned models, pick the best one, tune the decision threshold.

Pipeline:
1. Baseline models (params from configs/config.yaml, section `models`).
2. Tuned models via GridSearchCV (grids from section `tuning`), incl. MLP as a neural baseline.
3. The best model is selected by cross-validated ROC-AUC on train.
4. The decision threshold is tuned on out-of-fold train predictions (F-beta, beta=2)
   so the test set stays untouched until the final evaluation.
5. Artifacts: model.pkl (full Pipeline), metrics.json, threshold.json.
6. Every run is optionally tracked in MLflow (local ./mlruns, disable with DISABLE_MLFLOW=1).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.data.load import load_clean
from src.data.preprocess import build_preprocessor, train_test_split_stratified

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"

ESTIMATORS = {
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "mlp": MLPClassifier,
}


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pipeline(name: str, params: dict) -> Pipeline:
    if "hidden_layer_sizes" in params and isinstance(params["hidden_layer_sizes"], list):
        params = {**params, "hidden_layer_sizes": tuple(params["hidden_layer_sizes"])}
    model = ESTIMATORS[name](**params)
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])


def evaluate(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _mlflow():
    if os.getenv("DISABLE_MLFLOW", "").lower() in ("1", "true", "yes"):
        return None
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow is not installed, experiment tracking disabled")
        return None
    # mlflow-skinny поддерживает только файловый бэкенд; для локального
    # учебного трекинга он достаточен
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file:{(PROJECT_ROOT / 'mlruns').as_posix()}")
    mlflow.set_experiment("telco-churn")
    return mlflow


def _log_run(mlflow, run_name: str, params: dict, metrics: dict[str, float]) -> None:
    if mlflow is None:
        return
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics(metrics)


def tune_threshold(pipeline: Pipeline, x_train, y_train, cv, beta: float) -> dict:
    """Pick the threshold maximizing F-beta on out-of-fold train predictions."""
    oof_proba = cross_val_predict(
        clone(pipeline), x_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    grid = np.arange(0.05, 0.951, 0.01)
    scores = [fbeta_score(y_train, oof_proba >= t, beta=beta) for t in grid]
    best_idx = int(np.argmax(scores))
    return {
        "threshold": float(round(grid[best_idx], 2)),
        "beta": beta,
        "oof_fbeta": float(scores[best_idx]),
        "oof_fbeta_at_05": float(fbeta_score(y_train, oof_proba >= 0.5, beta=beta)),
    }


def train_and_compare() -> dict:
    cfg = load_config()
    mlflow = _mlflow()

    log.info("Loading data from %s", DATA_PATH)
    df = load_clean(DATA_PATH)
    log.info("Data shape: %s, churn rate: %.3f", df.shape, df["Churn"].mean())

    x_train, x_test, y_train, y_test = train_test_split_stratified(
        df,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )
    log.info("Train: %d, Test: %d", len(x_train), len(x_test))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["data"]["random_state"])

    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    cv_auc: dict[str, float] = {}

    # --- 1. Baselines -------------------------------------------------------
    for name, spec in cfg["models"].items():
        if not spec.get("enabled", True):
            continue
        run_name = f"baseline_{name}"
        log.info("Training %s ...", run_name)
        pipeline = build_pipeline(name, spec.get("params", {}))
        pipeline.fit(x_train, y_train)

        y_proba = pipeline.predict_proba(x_test)[:, 1]
        metrics = evaluate(y_test, pipeline.predict(x_test), y_proba)
        log.info("%s: test ROC-AUC=%.4f recall=%.4f", run_name, metrics["roc_auc"], metrics["recall"])

        results[run_name] = {"params": spec.get("params", {}), "test": metrics}
        _log_run(mlflow, run_name, spec.get("params", {}), metrics)

    # --- 2. Tuned models (GridSearchCV) + MLP neural baseline ---------------
    for name, spec in cfg["tuning"].items():
        if not spec.get("enabled", True):
            continue
        run_name = f"tuned_{name}"
        log.info("GridSearchCV for %s ...", run_name)
        pipeline = build_pipeline(name, spec.get("base_params", {}))
        grid = {f"model__{k}": v for k, v in spec["param_grid"].items()}
        if "model__hidden_layer_sizes" in grid:
            grid["model__hidden_layer_sizes"] = [
                tuple(v) for v in grid["model__hidden_layer_sizes"]
            ]
        search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        search.fit(x_train, y_train)

        best_params = {
            k.removeprefix("model__"): v for k, v in search.best_params_.items()
        }
        y_proba = search.predict_proba(x_test)[:, 1]
        metrics = evaluate(y_test, search.predict(x_test), y_proba)
        log.info(
            "%s: CV ROC-AUC=%.4f, test ROC-AUC=%.4f, best params: %s",
            run_name, search.best_score_, metrics["roc_auc"], best_params,
        )

        cv_auc[run_name] = float(search.best_score_)
        results[run_name] = {
            "params": {**spec.get("base_params", {}), **best_params},
            "cv_roc_auc": float(search.best_score_),
            "test": metrics,
        }
        fitted[run_name] = search.best_estimator_
        _log_run(
            mlflow, run_name,
            results[run_name]["params"],
            {**metrics, "cv_roc_auc": float(search.best_score_)},
        )

    # --- 3. Best model by CV ROC-AUC ----------------------------------------
    best_name = max(cv_auc, key=cv_auc.get)
    best_pipeline = fitted[best_name]
    log.info("Best model by CV ROC-AUC: %s (%.4f)", best_name, cv_auc[best_name])

    # --- 4. Threshold tuning on out-of-fold train predictions ----------------
    beta = float(cfg["threshold"]["beta"])
    thr_info = tune_threshold(best_pipeline, x_train, y_train, cv, beta)
    log.info(
        "Tuned threshold: %.2f (OOF F%.0f %.4f vs %.4f at 0.5)",
        thr_info["threshold"], beta, thr_info["oof_fbeta"], thr_info["oof_fbeta_at_05"],
    )

    y_proba_test = best_pipeline.predict_proba(x_test)[:, 1]
    test_at_tuned = evaluate(y_test, y_proba_test >= thr_info["threshold"], y_proba_test)
    log.info(
        "Final test metrics at threshold %.2f: recall=%.4f precision=%.4f f2=%.4f",
        thr_info["threshold"], test_at_tuned["recall"],
        test_at_tuned["precision"], test_at_tuned["f2"],
    )
    _log_run(
        mlflow, f"final_{best_name}_thr{thr_info['threshold']}",
        {**results[best_name]["params"], "threshold": thr_info["threshold"]},
        test_at_tuned,
    )

    # --- 5. Save artifacts ----------------------------------------------------
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    log.info("Saved best pipeline to %s", MODEL_PATH)

    THRESHOLD_PATH.write_text(json.dumps(thr_info, indent=2))

    summary = {
        "best_model": best_name,
        "threshold": thr_info,
        "test_at_tuned_threshold": test_at_tuned,
        "results": results,
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
