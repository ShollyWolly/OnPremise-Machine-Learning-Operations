"""
Production Training Script - Credit Risk Classifier
====================================================

Loads ALL available cleaned data from dwh_clean.cleaned_features,
trains a model, evaluates it, logs everything to MLflow, and optionally
promotes to Production if --promote flag is set.

This script is the canonical training logic used by:
  - The retraining DAG (dag_05_retraining.py)
  - Manual runs from CLI

The experiment notebook (notebooks/experiment.ipynb) is for exploration only.
Production training logic lives HERE, not in the notebook.

Usage:
    python train.py                          # trains, logs to MLflow (registered, no alias)
    python train.py --promote                # trains + promotes to Production alias if gates pass
    python train.py --model xgboost          # model type: logistic | rf | xgboost

MLflow Artifacts logged:
    - model/               (sklearn pyfunc flavor)
    - reference_data.parquet  (training distribution, used by drift detector)
    - feature_importance.json
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import compute_metrics, passes_promotion_gate

_SERVICES_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "services")
if _SERVICES_ROOT not in sys.path:
    sys.path.insert(0, _SERVICES_ROOT)

from platform_config import (  # noqa: E402
    FEATURE_COLUMNS,
    MODEL_REGISTRY_NAME_DEFAULT,
    TARGET_COLUMN,
    TRAINING_EXPERIMENT,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_REGISTRY_NAME = os.getenv("MODEL_REGISTRY_NAME", MODEL_REGISTRY_NAME_DEFAULT)
EXPERIMENT_NAME = TRAINING_EXPERIMENT

PARAM_DISTRIBUTIONS = {
    "xgboost": {
        "n_estimators": [100, 150, 200, 250, 300],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.03, 0.05, 0.07, 0.1, 0.15],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    },
    "rf": {
        "n_estimators": [100, 150, 200, 300],
        "max_depth": [5, 6, 8, 10, 12, None],
        "min_samples_leaf": [3, 5, 8, 10, 15],
        "max_features": ["sqrt", "log2", 0.5],
    },
    "logistic": {
        "clf__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
    },
}


def get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        return int(raw_value)
    except ValueError:
        log.warning("Invalid integer for %s=%r; using %s", name, raw_value, default)
        return default


def get_optional_int_env(name: str) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return None

    try:
        return int(raw_value)
    except ValueError:
        log.warning("Invalid integer for %s=%r; ignoring it", name, raw_value)
        return None


def load_training_data() -> pd.DataFrame:
    """Load all available cleaned features from PostgreSQL."""
    query = """
        SELECT {cols}, {target}
        FROM dwh_clean.cleaned_features
        WHERE {target} IS NOT NULL
        ORDER BY created_at
    """.format(
        cols=", ".join(FEATURE_COLUMNS),
        target=TARGET_COLUMN,
    )

    conn_params = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "dbname": os.getenv("POSTGRES_DB", "mlops"),
        "user": os.getenv("POSTGRES_USER", "mlops_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "changeme_secure_password"),
    }

    with psycopg2.connect(**conn_params) as conn:
        df = pd.read_sql(query, conn)

    log.info("Loaded %d training records (all available)", len(df))
    if len(df) < 100:
        raise ValueError(f"Insufficient training data: {len(df)} records (need >= 100)")
    return df


def build_model(model_type: str) -> object:
    """Return an untrained sklearn estimator or pipeline."""
    if model_type == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])
    elif model_type == "rf":
        rf_n_jobs = get_optional_int_env("RF_N_JOBS")
        if rf_n_jobs is None:
            rf_n_jobs = get_int_env("TRAINING_MODEL_N_JOBS", -1)

        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=rf_n_jobs,
        )
    elif model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        xgboost_n_jobs = get_optional_int_env("XGBOOST_N_JOBS")
        if xgboost_n_jobs is None:
            xgboost_n_jobs = get_optional_int_env("TRAINING_MODEL_N_JOBS")

        model_kwargs = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 3,  # handles class imbalance
            "random_state": 42,
            "eval_metric": "auc",
            "verbosity": 0,
        }
        if xgboost_n_jobs is not None:
            model_kwargs["n_jobs"] = xgboost_n_jobs

        return XGBClassifier(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose: logistic | rf | xgboost")


def get_production_baseline() -> dict | None:
    """Return metrics of the current Production model from MLflow registry."""
    try:
        client = mlflow.MlflowClient()
        version = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, "production")
        run = client.get_run(version.run_id)
        return {
            "roc_auc": float(run.data.metrics.get("test_roc_auc", 0.0)),
            "f1_score": float(run.data.metrics.get("test_f1_score", 0.0)),
        }
    except Exception as exc:
        log.warning("Could not fetch production baseline: %s", exc)
        return None


def train(
    model_type: str = "xgboost",
    promote: bool = False,
    test_size: float = 0.2,
) -> dict:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_training_data()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    search_n_jobs = get_int_env("TRAINING_SEARCH_N_JOBS", -1)
    model = build_model(model_type)
    baseline = get_production_baseline()

    run_name = f"{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "model_type": model_type,
            "train_size": len(X_train),
            "test_size_rows": len(X_test),
            "default_rate_train": float(y_train.mean()),
            "search_n_jobs": search_n_jobs,
        })

        # HP search with 5-fold CV on training set (test set held out completely)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            model,
            PARAM_DISTRIBUTIONS[model_type],
            n_iter=12,
            cv=cv,
            scoring="roc_auc",
            random_state=42,
            n_jobs=search_n_jobs,
            refit=True,
            verbose=0,
        )
        log.info(
            "Starting HP search (n_iter=12, cv=5, n_jobs=%s) on %d training samples…",
            search_n_jobs,
            len(X_train),
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_

        best_params = {f"hp_{k}": v for k, v in search.best_params_.items()}
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_roc_auc_mean", float(search.best_score_))
        log.info("Best CV ROC-AUC: %.4f | Best params: %s", search.best_score_, search.best_params_)

        # Test set evaluation
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        test_metrics = compute_metrics(y_test.values, y_pred, y_prob)

        for name, val in test_metrics.items():
            mlflow.log_metric(f"test_{name}", val)

        log.info("Test metrics: %s", test_metrics)

        # Log feature importance (RF / XGBoost)
        feature_importance = {}
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
        elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf", None), "coef_"):
            coefs = model.named_steps["clf"].coef_[0].tolist()
            feature_importance = dict(zip(FEATURE_COLUMNS, coefs))

        if feature_importance:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(feature_importance, f, indent=2)
                mlflow.log_artifact(f.name, artifact_path="feature_importance")

        # Log reference dataset (used by drift detector)
        # Write to an explicitly-named file so MLflow stores it as reference_data.parquet
        # (not inside a subdirectory of that name, which happens with artifact_path=)
        ref_df = X_train.copy()
        _ref_tmp = os.path.join(tempfile.gettempdir(), "reference_data.parquet")
        ref_df.to_parquet(_ref_tmp, index=False, engine="pyarrow")
        mlflow.log_artifact(_ref_tmp)  # logged at root as reference_data.parquet

        # Log model to registry
        if model_type == "xgboost":
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name=MODEL_REGISTRY_NAME,
            )
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=MODEL_REGISTRY_NAME,
            )

        run_id = run.info.run_id

    # Get the registered version (just created by log_model)
    client = mlflow.MlflowClient()
    # Give registry a moment to register
    import time; time.sleep(2)
    all_versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
    new_version = max(all_versions, key=lambda v: int(v.version)) if all_versions else None

    promotion_passed = False
    promotion_reason = "Not attempted"

    if promote and new_version:
        promotion_passed, promotion_reason = passes_promotion_gate(
            metrics=test_metrics,
            baseline_metrics=baseline,
        )
        if promotion_passed:
            try:
                prev = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, "production")
                client.set_model_version_tag(MODEL_REGISTRY_NAME, prev.version, "stage", "archived")
            except Exception:
                pass
            client.set_registered_model_alias(MODEL_REGISTRY_NAME, "production", new_version.version)
            client.set_model_version_tag(MODEL_REGISTRY_NAME, new_version.version, "stage", "production")
            log.info("Model version %s promoted to Production (alias: production)", new_version.version)
        else:
            log.warning("Promotion FAILED: %s", promotion_reason)
            client.set_model_version_tag(MODEL_REGISTRY_NAME, new_version.version, "stage", "rejected")

    result = {
        "run_id": run_id,
        "model_type": model_type,
        "test_metrics": test_metrics,
        "promoted": promotion_passed,
        "promotion_reason": promotion_reason,
        "model_version": new_version.version if new_version else None,
    }

    log.info("Training complete: %s", json.dumps(result, indent=2))
    print(json.dumps(result))  # captured as Airflow XCom
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production training script")
    parser.add_argument("--model", default="xgboost",
                        choices=["logistic", "rf", "xgboost"])
    parser.add_argument("--promote", action="store_true",
                        help="Promote to Production if evaluation passes")
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_type=args.model,
        promote=args.promote,
        test_size=args.test_size,
    )
