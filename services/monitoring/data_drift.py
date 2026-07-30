"""
Monitoring Module 2: Data Drift
=================================
Compares current batch feature distributions against the reference (training) dataset.
Reference dataset is loaded from the MLflow artifact store of the current Production model.

Writes results to parquet and dwh_monitoring_drift.results.

Usage:
    python data_drift.py --run-index 42
    python data_drift.py --batch-run-id <uuid>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv
from evidently import Dataset, DataDefinition, Report
from evidently.core.report import Snapshot
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently.tests import lte
from evidently.sdk import panels
from evidently.sdk.models import DashboardModel, DashboardTabModel, PanelMetric, ProjectModel
from evidently.ui.workspace import Workspace

_DRIFTED_COLUMNS_COUNT_TYPE = "evidently:metric_v2:DriftedColumnsCount"
_VALUE_DRIFT_TYPE = "evidently:metric_v2:ValueDrift"

_SERVICES_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _SERVICES_ROOT not in sys.path:
    sys.path.insert(0, _SERVICES_ROOT)

from platform_config import (  # noqa: E402
    EVIDENTLY_DRIFT_PROJECT_ID,
    EVIDENTLY_DRIFT_PROJECT_NAME,
    FEATURE_COLUMNS,
    MONITORING_EXPERIMENT_DRIFT,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DRIFT_MONITORING_EXPERIMENT = MONITORING_EXPERIMENT_DRIFT


def _db_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "mlops"),
        user=os.getenv("POSTGRES_USER", "mlops_user"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )


def load_current_features(run_index: int | None, batch_run_id: str | None) -> pd.DataFrame:
    if run_index is not None:
        where = "WHERE run_index = %(run_index)s"
        params = {"run_index": run_index}
    elif batch_run_id:
        where = "WHERE batch_run_id = %(batch_run_id)s"
        params = {"batch_run_id": batch_run_id}
    else:
        where = "WHERE run_index = (SELECT MAX(run_index) FROM dwh_history.prediction_ground_truth)"
        params = {}

    query = f"""
        SELECT run_index, batch_run_id, {", ".join(FEATURE_COLUMNS)}
        FROM dwh_history.prediction_ground_truth
        {where}
    """
    with _db_conn() as conn:
        df = pd.read_sql(query, conn, params=params)
    log.info("Loaded %d current feature rows for drift check", len(df))
    return df


def load_reference_dataset(model_name: str, stage: str = "Production") -> tuple[pd.DataFrame, str]:
    """Load reference dataset from MLflow artifact of current Production model."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    try:
        version_info = client.get_model_version_by_alias(model_name, stage.lower())
    except Exception:
        raise RuntimeError(f"No '{stage.lower()}' alias found for '{model_name}'")

    run_id = version_info.run_id
    with tempfile.TemporaryDirectory() as tmp:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="reference_data.parquet", dst_path=tmp
        )
        ref_df = pd.read_parquet(local_path, engine="pyarrow")

    log.info("Reference dataset: %d rows from run %s", len(ref_df), run_id)
    return ref_df[FEATURE_COLUMNS], run_id


def compute_drift(current_df: pd.DataFrame, reference_df: pd.DataFrame) -> dict:
    current = current_df[FEATURE_COLUMNS].copy()
    reference = reference_df[FEATURE_COLUMNS].copy()

    threshold = float(os.getenv("MAX_DRIFT_FEATURE_FRACTION", 0.30))
    definition = DataDefinition(numerical_columns=FEATURE_COLUMNS)
    current_ds = Dataset.from_pandas(current, data_definition=definition)
    reference_ds = Dataset.from_pandas(reference, data_definition=definition)

    # lte(threshold): test PASSES while the drifted-column share stays within budget,
    # FAILS once it crosses the same threshold that drives our own retrain gate.
    report = Report([
        DriftedColumnsCount(columns=FEATURE_COLUMNS, drift_share=threshold, tests=[lte(threshold)]),
        *[ValueDrift(column=col) for col in FEATURE_COLUMNS],
    ])
    run = report.run(current_ds, reference_ds)
    run_dict = run.dict()

    count_metric = next(
        m for m in run_dict["metrics"] if m["config"]["type"] == _DRIFTED_COLUMNS_COUNT_TYPE
    )
    share = float(count_metric["value"]["share"])

    # Derive drifted_feature_names from the per-column ValueDrift metrics (same underlying
    # stattest/threshold DriftedColumnsCount uses internally) rather than trusting its own
    # "count" field blindly - keeps num_drifted_features and the name list guaranteed consistent.
    drifted_names = [
        m["config"]["column"]
        for m in run_dict["metrics"]
        if m["config"]["type"] == _VALUE_DRIFT_TYPE and float(m["value"]) < float(m["config"]["threshold"])
    ]

    result = {
        "drift_detected": share >= threshold,
        "drift_score": round(share, 4),
        "num_drifted_features": len(drifted_names),
        "drifted_feature_names": drifted_names,
        "run": run,
    }
    log.info("Drift: detected=%s score=%.4f (share of drifted columns, drift_share=%.2f) features=%s",
             result["drift_detected"], result["drift_score"], threshold, drifted_names)
    return result


def _configure_drift_dashboard(ws: Workspace, project) -> None:
    """One-time dashboard setup: an Overview tab (2x2 grid of counters + trends) and a
    Per-Feature Drift tab (one combined trend across all 10 features' p-values) so a single
    run's page shows the same per-feature detail the old custom Streamlit charts used to.

    Built as a plain DashboardModel and pushed via ws.save_dashboard() rather than repeated
    project.dashboard.add_panel() calls - add_panel duplicates a tab's panel-id list on every
    call after the tab already exists (confirmed against evidently 0.7.21), so this is the
    only way to end up with clean, non-duplicated tabs."""
    # metric_labels={"value_type": "share"|"count"} is the real convention for selecting a
    # sub-field of a CountMetric-derived metric's {"count":.., "share":..} value - confirmed
    # against Evidently's own bundled demo project (ui/service/demo_projects/bikes.py), not
    # guessed. An earlier {"share": "share"}/{"count": "count"} attempt used the wrong key
    # entirely and crashed the dashboard frontend ("t.sources[d.index] is undefined").
    drift_score_counter = panels.counter_panel(
        title="Latest Drift Score",
        values=[PanelMetric(metric="DriftedColumnsCount", metric_labels={"value_type": "share"}, legend="share")],
        aggregation="last", size="half",
    )
    drifted_count_counter = panels.counter_panel(
        title="Latest Drifted Feature Count",
        values=[PanelMetric(metric="DriftedColumnsCount", metric_labels={"value_type": "count"}, legend="count")],
        aggregation="last", size="half",
    )
    drift_score_trend = panels.line_plot_panel(
        title="Drift Score Trend",
        values=[PanelMetric(metric="DriftedColumnsCount", metric_labels={"value_type": "share"}, legend="drift score")],
        size="half",
    )
    drifted_count_trend = panels.line_plot_panel(
        title="Drifted Feature Count Trend",
        values=[PanelMetric(metric="DriftedColumnsCount", metric_labels={"value_type": "count"}, legend="drifted features")],
        size="half",
    )
    per_feature_drift = panels.line_plot_panel(
        title="Per-Feature Drift (statistical test value per column, lower = more drift)",
        values=[
            PanelMetric(metric="ValueDrift", metric_labels={"column": col}, legend=col)
            for col in FEATURE_COLUMNS
        ],
        size="full",
    )

    overview_panels = [drift_score_counter, drifted_count_counter, drift_score_trend, drifted_count_trend]
    dashboard = DashboardModel(
        tabs=[
            DashboardTabModel(title="Overview", panels=[p.id for p in overview_panels]),
            DashboardTabModel(title="Per-Feature Drift", panels=[per_feature_drift.id]),
        ],
        panels=[*overview_panels, per_feature_drift],
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
    """Push the run (Report + its test-bound metric results) as a snapshot to the Evidently
    self-hosted UI's Workspace. Returns the snapshot id for building a deep link to this run."""
    workspace_path = os.getenv("EVIDENTLY_WORKSPACE_PATH", "/data/monitoring/evidently_workspace")
    ws = Workspace.create(workspace_path)
    project = ws.get_project(EVIDENTLY_DRIFT_PROJECT_ID)
    if project is None:
        project = ws.add_project(ProjectModel(id=EVIDENTLY_DRIFT_PROJECT_ID, name=EVIDENTLY_DRIFT_PROJECT_NAME))
    if not ws.get_dashboard(project.id).panels:
        _configure_drift_dashboard(ws, project)
    snapshot_ref = ws.add_run(EVIDENTLY_DRIFT_PROJECT_ID, run)
    _reload_evidently_project(EVIDENTLY_DRIFT_PROJECT_ID)
    log.info("Drift run pushed to Evidently workspace (project=%s, snapshot=%s)",
             EVIDENTLY_DRIFT_PROJECT_ID, snapshot_ref.id)
    return str(snapshot_ref.id)


def write_parquet(drift_result: dict, current_df: pd.DataFrame,
                  run_index: int, batch_run_id: str, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"run_{run_index:05d}.parquet"
    row = {
        "run_index": run_index,
        "batch_run_id": batch_run_id,
        "drift_score": drift_result["drift_score"],
        "drift_detected": drift_result["drift_detected"],
        "num_drifted_features": drift_result["num_drifted_features"],
        "drifted_feature_names": ",".join(drift_result["drifted_feature_names"]),
        "n_records": len(current_df),
        "evaluated_at": datetime.utcnow().isoformat(),
    }
    pd.DataFrame([row]).to_parquet(path, index=False, engine="pyarrow")
    log.info("Drift parquet → %s", path)
    return str(path)


def write_db(run_index: int, batch_run_id: str, drift_result: dict,
             reference_run_id: str, n_records: int, parquet_path: str,
             evidently_snapshot_id: str | None) -> None:
    record = (
        str(uuid.uuid4()), run_index, batch_run_id,
        drift_result["drift_detected"], drift_result["drift_score"],
        drift_result["num_drifted_features"],
        ",".join(drift_result["drifted_feature_names"]),
        reference_run_id, n_records, parquet_path, evidently_snapshot_id, datetime.utcnow(),
    )
    insert_sql = """
        INSERT INTO dwh_monitoring_drift.results
        (id, run_index, batch_run_id, drift_detected, drift_score,
         num_drifted_features, drifted_feature_names, reference_run_id,
         n_records, parquet_path, evidently_snapshot_id, evaluated_at)
        VALUES %s
    """
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM dwh_monitoring_drift.results WHERE run_index = %s",
                (run_index,),
            )
            psycopg2.extras.execute_values(cur, insert_sql, [record])
        conn.commit()
    log.info("Drift results upserted to dwh_monitoring_drift.results")


def run_data_drift(run_index: int | None = None, batch_run_id: str | None = None) -> dict:
    log.info("=== Data Drift START (run_index=%s) ===", run_index)

    model_name = os.getenv("MODEL_REGISTRY_NAME", "credit-risk-classifier")
    current_df = load_current_features(run_index, batch_run_id)

    if current_df.empty:
        log.warning("No current data found.")
        result = {"run_index": run_index, "drift_detected": False, "drift_score": 0.0,
                  "drifted_features": [], "reason": "no_data"}
        print(json.dumps(result))
        return result

    actual_run_index = int(current_df["run_index"].iloc[0])
    actual_batch_run_id = current_df["batch_run_id"].iloc[0]

    ref_df = None
    try:
        ref_df, reference_run_id = load_reference_dataset(model_name)
        drift_result = compute_drift(current_df, ref_df)
    except Exception as exc:
        log.warning("Drift detection failed: %s", exc)
        drift_result = {"drift_detected": False, "drift_score": 0.0,
                        "num_drifted_features": 0, "drifted_feature_names": [], "run": None}
        reference_run_id = "unknown"

    run = drift_result.pop("run", None)

    output_dir = os.getenv("DATA_MONITORING_DRIFT_PATH", "/data/monitoring/drift")
    parquet_path = write_parquet(drift_result, current_df, actual_run_index,
                                  actual_batch_run_id, output_dir)
    evidently_snapshot_id = push_to_workspace(run) if run is not None else None
    write_db(actual_run_index, actual_batch_run_id, drift_result,
             reference_run_id, len(current_df), parquet_path, evidently_snapshot_id)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(DRIFT_MONITORING_EXPERIMENT)
    with mlflow.start_run(run_name=f"drift_run_{actual_run_index:05d}"):
        mlflow.log_metrics({
            "drift_score": drift_result["drift_score"],
            "num_drifted_features": float(drift_result["num_drifted_features"]),
            "drift_detected": float(drift_result["drift_detected"]),
        })
        mlflow.log_param("run_index", actual_run_index)
        mlflow.log_param("n_records", len(current_df))
        mlflow.log_param("drifted_features",
                         ",".join(drift_result["drifted_feature_names"]) or "none")
        for feat in FEATURE_COLUMNS:
            if feat in current_df.columns and pd.api.types.is_numeric_dtype(current_df[feat]):
                mlflow.log_metric(f"curr_{feat}_mean", float(current_df[feat].mean()))
                mlflow.log_metric(f"curr_{feat}_std", float(current_df[feat].std()))
        if ref_df is not None:
            for feat in FEATURE_COLUMNS:
                if feat in ref_df.columns and pd.api.types.is_numeric_dtype(ref_df[feat]):
                    mlflow.log_metric(f"ref_{feat}_mean", float(ref_df[feat].mean()))
                    mlflow.log_metric(f"ref_{feat}_std", float(ref_df[feat].std()))
        mlflow.log_artifact(parquet_path, artifact_path="drift_metrics")
        if evidently_snapshot_id:
            mlflow.log_param("evidently_snapshot_id", evidently_snapshot_id)
    log.info("Drift results logged to MLflow experiment '%s'", DRIFT_MONITORING_EXPERIMENT)

    result = {
        "run_index": actual_run_index,
        "batch_run_id": actual_batch_run_id,
        "drift_detected": drift_result["drift_detected"],
        "drift_score": drift_result["drift_score"],
        "num_drifted_features": drift_result["num_drifted_features"],
        "drifted_feature_names": drift_result["drifted_feature_names"],
        "n_records": len(current_df),
        "evidently_snapshot_id": evidently_snapshot_id,
    }
    log.info("=== Data Drift COMPLETE ===")
    print(json.dumps(result))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data drift monitoring module")
    parser.add_argument("--run-index", type=int, default=None)
    parser.add_argument("--batch-run-id", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_data_drift(run_index=args.run_index, batch_run_id=args.batch_run_id)
