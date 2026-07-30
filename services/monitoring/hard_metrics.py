"""
Monitoring Module 1: Hard Metrics
===================================
Reads predictions + ground truth from dwh_history.prediction_ground_truth.
Computes all metrics defined in services/metrics.py.
Primary metric is read from the MLflow model-version tag 'primary_metric'
(set at training / challenger promotion time), defaulting to roc_auc.
Writes results to parquet and dwh_monitoring_hard.results.
Logs to MLflow experiment 'monitoring_hard'.
Outputs JSON for Airflow XCom capture.

Usage:
    python hard_metrics.py --run-index 42
    python hard_metrics.py --batch-run-id <uuid>
"""

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv
from evidently import BinaryClassification, Dataset, DataDefinition, Report
from evidently.core.report import Snapshot
from evidently.metrics import Accuracy, F1Score, Precision, Recall, RocAuc
from evidently.tests import gte
from evidently.sdk import panels
from evidently.sdk.models import DashboardModel, DashboardTabModel, PanelMetric, ProjectModel
from evidently.ui.workspace import Workspace

_METRIC_TYPE_PREFIX = "evidently:metric_v2:"

# Locate services/ root so we can import the shared metrics module
_SERVICES_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _SERVICES_ROOT)
from metrics import (  # noqa: E402
    DEFAULT_PRIMARY,
    REGISTRY,
    compute_all_metrics,
    display_name,
)
from platform_config import (  # noqa: E402
    EVIDENTLY_HARD_PROJECT_ID,
    EVIDENTLY_HARD_PROJECT_NAME,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME           = os.getenv("MODEL_REGISTRY_NAME", "credit-risk-classifier")
MONITORING_EXPERIMENT = "monitoring_hard"


def _db_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "mlops"),
        user=os.getenv("POSTGRES_USER", "mlops_user"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )


def load_eval_data(run_index: int | None, batch_run_id: str | None) -> pd.DataFrame:
    """Load predictions + ground truth from history table."""
    if run_index is not None:
        where = "WHERE run_index = %(run_index)s"
        params = {"run_index": run_index}
    elif batch_run_id:
        where = "WHERE batch_run_id = %(batch_run_id)s"
        params = {"batch_run_id": batch_run_id}
    else:
        where = """WHERE run_index = (
            SELECT MAX(run_index) FROM dwh_history.prediction_ground_truth
        )"""
        params = {}

    query = f"""
        SELECT run_index, batch_run_id, default_probability, default_flag_predicted,
               actual_default_flag, model_version, decision_threshold
        FROM dwh_history.prediction_ground_truth
        {where}
        AND actual_default_flag IS NOT NULL
    """
    with _db_conn() as conn:
        df = pd.read_sql(query, conn, params=params)

    log.info("Loaded %d records for evaluation (run_index=%s)", len(df), run_index)
    return df


def get_primary_metric_from_registry(model_version: str) -> str:
    """Read primary_metric tag from MLflow registry for the given model version."""
    try:
        client = mlflow.MlflowClient()
        mv = client.get_model_version(MODEL_NAME, str(model_version))
        tag = mv.tags.get("primary_metric", DEFAULT_PRIMARY)
        if tag not in REGISTRY:
            log.warning("Unknown primary_metric tag '%s' - falling back to %s", tag, DEFAULT_PRIMARY)
            return DEFAULT_PRIMARY
        return tag
    except Exception as exc:
        log.warning("Could not read primary_metric from registry: %s - using %s", exc, DEFAULT_PRIMARY)
        return DEFAULT_PRIMARY


def build_classification_report(df: pd.DataFrame) -> tuple[Snapshot, dict]:
    """
    Run Accuracy/Precision/Recall/F1Score/RocAuc (with a test bound to RocAuc against
    MIN_ROC_AUC) against predictions + ground truth.
    Returns (run, quality) where quality maps metric class name -> scalar value
    (Accuracy/Precision/Recall/F1Score/RocAuc).
    """
    definition = DataDefinition(classification=[BinaryClassification(
        target="actual_default_flag",
        prediction_probas="default_probability",
        pos_label=1,
    )])
    dataset = Dataset.from_pandas(df, data_definition=definition)

    # Match the actual decision threshold used at inference time (falls back to 0.5)
    # so accuracy/precision/recall/f1 reflect the real production decision, not
    # Evidently's own default cutoff.
    probas_threshold = float(df["decision_threshold"].iloc[0]) if "decision_threshold" in df.columns else 0.5
    # RocAuc's own test is graded against the same MIN_ROC_AUC threshold as the retrain gate;
    # the rest are descriptive (no fixed business threshold defined elsewhere for them).
    min_roc_auc = float(os.getenv("MIN_ROC_AUC", 0.70))

    report = Report([
        RocAuc(probas_threshold=probas_threshold, tests=[gte(min_roc_auc)]),
        Accuracy(probas_threshold=probas_threshold),
        Precision(probas_threshold=probas_threshold),
        Recall(probas_threshold=probas_threshold),
        F1Score(probas_threshold=probas_threshold),
    ])
    run = report.run(dataset, None)
    run_dict = run.dict()

    quality = {
        m["config"]["type"][len(_METRIC_TYPE_PREFIX):]: m["value"]
        for m in run_dict["metrics"]
    }
    return run, quality


def compute_metrics(df: pd.DataFrame) -> tuple[dict, str, dict, Snapshot]:
    """
    Compute all metrics. Returns (metrics_dict, primary_metric_key, raw_metrics, run).
    primary_metric is resolved from the MLflow model-version tag.

    accuracy/precision/recall/f1/roc_auc come from Evidently's classification metrics;
    pr_auc is kept from services/metrics.py (Evidently 0.7.x doesn't expose a scalar PR-AUC).
    """
    y_true  = df["actual_default_flag"].values.astype(int)
    y_score = df["default_probability"].values.astype(float)
    y_pred  = df["default_flag_predicted"].values.astype(int)

    all_m = compute_all_metrics(y_true, y_score, y_pred=y_pred)
    run, quality = build_classification_report(df)

    # Map to DB column naming convention (existing columns use _score suffix)
    db_metrics = {
        "roc_auc":          quality["RocAuc"],
        "pr_auc":           all_m["pr_auc"],
        "f1_score":         quality["F1Score"],
        "precision_score":  quality["Precision"],
        "recall_score":     quality["Recall"],
        "accuracy":         quality["Accuracy"],
        "n_records":        int(len(df)),
    }

    # Determine primary metric from model registry
    model_version = str(df["model_version"].iloc[0]) if "model_version" in df.columns else None
    primary_metric = get_primary_metric_from_registry(model_version) if model_version else DEFAULT_PRIMARY
    log.info("Primary metric: %s (%s)", primary_metric, display_name(primary_metric))

    return db_metrics, primary_metric, all_m, run


def should_retrain(metrics: dict, primary_metric: str = DEFAULT_PRIMARY) -> tuple[bool, str]:
    """Trigger retraining when the primary metric falls below MIN_ROC_AUC threshold."""
    threshold = float(os.getenv("MIN_ROC_AUC", 0.70))

    # Map primary_metric key to db_metrics key
    _key_map = {
        "roc_auc":   "roc_auc",
        "pr_auc":    "pr_auc",
        "f1":        "f1_score",
        "precision": "precision_score",
        "recall":    "recall_score",
        "accuracy":  "accuracy",
    }
    db_key = _key_map.get(primary_metric, "roc_auc")
    value = metrics.get(db_key, 0.0)

    if value < threshold:
        reason = f"{display_name(primary_metric)}={value:.4f} < threshold={threshold}"
        return True, reason
    return False, "All metrics within thresholds"


def _configure_hard_metrics_dashboard(ws: Workspace, project) -> None:
    """One-time dashboard setup: an Overview tab (5-counter grid, one per metric) and a
    Trends tab (5 separate line plots, one per metric, instead of one crowded combined chart).

    Built as a plain DashboardModel and pushed via ws.save_dashboard() rather than repeated
    project.dashboard.add_panel() calls - add_panel duplicates a tab's panel-id list on every
    call after the tab already exists (confirmed against evidently 0.7.21), so this is the
    only way to end up with clean, non-duplicated tabs."""
    overview_panels = []
    trend_panels = []
    for metric_name, legend in [
        ("RocAuc", "ROC-AUC"), ("Accuracy", "Accuracy"), ("Precision", "Precision"),
        ("Recall", "Recall"), ("F1Score", "F1"),
    ]:
        overview_panels.append(panels.counter_panel(
            title=f"Latest {legend}",
            values=[PanelMetric(metric=metric_name, legend=legend)],
            aggregation="last", size="half",
        ))
        trend_panels.append(panels.line_plot_panel(
            title=f"{legend} Trend",
            values=[PanelMetric(metric=metric_name, legend=legend)],
            size="half",
        ))

    dashboard = DashboardModel(
        tabs=[
            DashboardTabModel(title="Overview", panels=[p.id for p in overview_panels]),
            DashboardTabModel(title="Trends", panels=[p.id for p in trend_panels]),
        ],
        panels=[*overview_panels, *trend_panels],
    )
    ws.save_dashboard(project.id, dashboard)


def _reload_evidently_project(project_id) -> None:
    """Best-effort nudge so the running evidently-ui server's file-watcher-backed cache picks
    up the snapshot we just wrote immediately, instead of only on its next filesystem event
    (observed to be unreliable for test suite files specifically on evidently 0.4.30)."""
    base_url = os.getenv("EVIDENTLY_INTERNAL_URL", "http://evidently-ui:8000")
    try:
        requests.get(f"{base_url}/api/projects/{project_id}/reload", timeout=5)
    except requests.RequestException as exc:
        log.warning("Could not reload evidently-ui project %s: %s", project_id, exc)


def push_to_workspace(run: Snapshot) -> str:
    """Push the run (metrics + bound RocAuc test) as a snapshot to the Evidently self-hosted
    UI's Workspace. Returns the snapshot id for building a deep link to this exact run."""
    workspace_path = os.getenv("EVIDENTLY_WORKSPACE_PATH", "/data/monitoring/evidently_workspace")
    ws = Workspace.create(workspace_path)
    project = ws.get_project(EVIDENTLY_HARD_PROJECT_ID)
    if project is None:
        project = ws.add_project(ProjectModel(id=EVIDENTLY_HARD_PROJECT_ID, name=EVIDENTLY_HARD_PROJECT_NAME))
    if not ws.get_dashboard(project.id).panels:
        _configure_hard_metrics_dashboard(ws, project)
    snapshot_ref = ws.add_run(EVIDENTLY_HARD_PROJECT_ID, run)
    _reload_evidently_project(EVIDENTLY_HARD_PROJECT_ID)
    log.info("Hard metrics run pushed to Evidently workspace (project=%s, snapshot=%s)",
             EVIDENTLY_HARD_PROJECT_ID, snapshot_ref.id)
    return str(snapshot_ref.id)


def write_parquet(metrics: dict, run_index: int, batch_run_id: str,
                  primary_metric: str, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"run_{run_index:05d}.parquet"
    row = {
        **metrics,
        "run_index":     run_index,
        "batch_run_id":  batch_run_id,
        "primary_metric": primary_metric,
        "evaluated_at":  datetime.utcnow().isoformat(),
    }
    pd.DataFrame([row]).to_parquet(path, index=False, engine="pyarrow")
    log.info("Hard metrics parquet → %s", path)
    return str(path)


def write_db(run_index: int, batch_run_id: str, metrics: dict,
             primary_metric: str, retraining_triggered: bool,
             retraining_reason: str, mlflow_run_id: str, parquet_path: str,
             evidently_snapshot_id: str | None) -> None:
    record = (
        str(uuid.uuid4()), run_index, batch_run_id,
        metrics["n_records"],
        metrics["accuracy"], metrics["f1_score"],
        metrics["precision_score"], metrics["recall_score"],
        metrics["roc_auc"], metrics.get("pr_auc"),
        primary_metric,
        retraining_triggered, retraining_reason,
        mlflow_run_id, parquet_path, evidently_snapshot_id,
        datetime.utcnow(),
    )
    insert_sql = """
        INSERT INTO dwh_monitoring_hard.results
        (id, run_index, batch_run_id, n_records, accuracy, f1_score,
         precision_score, recall_score, roc_auc, pr_auc, primary_metric,
         retraining_triggered, retraining_reason, mlflow_run_id, parquet_path,
         evidently_snapshot_id, evaluated_at)
        VALUES %s
    """
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM dwh_monitoring_hard.results WHERE run_index = %s",
                (run_index,),
            )
            psycopg2.extras.execute_values(cur, insert_sql, [record])
        conn.commit()
    log.info("Hard metrics upserted to dwh_monitoring_hard.results (run_index=%s)", run_index)


def run_hard_metrics(run_index: int | None = None, batch_run_id: str | None = None) -> dict:
    log.info("=== Hard Metrics START (run_index=%s) ===", run_index)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MONITORING_EXPERIMENT)

    df = load_eval_data(run_index, batch_run_id)
    if df.empty:
        log.warning("No evaluation data found.")
        result = {"run_index": run_index, "roc_auc": 0.0, "primary_metric": DEFAULT_PRIMARY,
                  "retraining_triggered": False, "reason": "no_data", "n_records": 0}
        print(json.dumps(result))
        return result

    actual_run_index   = int(df["run_index"].iloc[0])
    actual_batch_run_id = df["batch_run_id"].iloc[0]

    metrics, primary_metric, raw_metrics, evidently_run = compute_metrics(df)
    log.info("Metrics: %s", {k: v for k, v in metrics.items() if k != "n_records"})

    retraining_triggered, reason = should_retrain(metrics, primary_metric)

    output_dir = os.getenv("DATA_MONITORING_HARD_PATH", "/data/monitoring/hard")
    parquet_path = write_parquet(metrics, actual_run_index, actual_batch_run_id,
                                 primary_metric, output_dir)
    evidently_snapshot_id = push_to_workspace(evidently_run)

    with mlflow.start_run(run_name=f"hard_metrics_run_{actual_run_index:05d}") as run:
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "n_records"})
        mlflow.log_metrics({
            "pr_auc":    raw_metrics["pr_auc"],
            "f1":        raw_metrics["f1"],
            "precision": raw_metrics["precision"],
            "recall":    raw_metrics["recall"],
        })
        mlflow.log_param("run_index",           actual_run_index)
        mlflow.log_param("batch_run_id",        actual_batch_run_id)
        mlflow.log_param("primary_metric",      primary_metric)
        mlflow.log_param("retraining_triggered", retraining_triggered)
        mlflow.log_param("evidently_snapshot_id", evidently_snapshot_id)
        mlflow.set_tag("primary_metric",        primary_metric)
        mlflow.log_artifact(parquet_path, artifact_path="hard_metrics")
        mlflow_run_id = run.info.run_id

    write_db(actual_run_index, actual_batch_run_id, metrics,
             primary_metric, retraining_triggered, reason,
             mlflow_run_id, parquet_path, evidently_snapshot_id)

    result = {
        "run_index":           actual_run_index,
        "batch_run_id":        actual_batch_run_id,
        "primary_metric":      primary_metric,
        "primary_value":       metrics.get({"roc_auc":"roc_auc","pr_auc":"pr_auc","f1":"f1_score",
                                            "precision":"precision_score","recall":"recall_score",
                                            "accuracy":"accuracy"}.get(primary_metric, "roc_auc"), 0.0),
        "roc_auc":             metrics["roc_auc"],
        "pr_auc":              metrics.get("pr_auc"),
        "f1_score":            metrics["f1_score"],
        "accuracy":            metrics["accuracy"],
        "n_records":           metrics["n_records"],
        "retraining_triggered": retraining_triggered,
        "reason":              reason,
        "evidently_snapshot_id": evidently_snapshot_id,
    }
    log.info("=== Hard Metrics COMPLETE (primary=%s) ===", primary_metric)
    print(json.dumps(result))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard metrics monitoring module")
    parser.add_argument("--run-index",    type=int, default=None)
    parser.add_argument("--batch-run-id", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_hard_metrics(run_index=args.run_index, batch_run_id=args.batch_run_id)
