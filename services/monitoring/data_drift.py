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

import argparse
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from scipy.stats import ks_2samp

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DRIFT_MONITORING_EXPERIMENT = "monitoring_drift"

FEATURE_COLUMNS = [
    "age", "annual_income", "credit_score", "loan_amount",
    "loan_term_months", "employment_length_years", "home_ownership_encoded",
    "debt_to_income_ratio", "num_credit_lines", "payment_history_score",
]


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

    threshold = float(os.getenv("MAX_DRIFT_FEATURE_FRACTION", 0.50))

    ks_per_feature = {}
    for feat in FEATURE_COLUMNS:
        if feat in current.columns and feat in reference.columns:
            ks_stat, _ = ks_2samp(reference[feat].dropna(), current[feat].dropna())
            ks_per_feature[feat] = round(float(ks_stat), 4)

    ks_scores = list(ks_per_feature.values())
    mean_ks = round(float(sum(ks_scores) / len(ks_scores)), 4) if ks_scores else 0.0

    drifted_names = [feat for feat, ks in ks_per_feature.items() if ks >= threshold]
    num_drifted = len(drifted_names)

    result = {
        "drift_detected": mean_ks >= threshold,
        "drift_score": mean_ks,
        "num_drifted_features": num_drifted,
        "drifted_feature_names": drifted_names,
    }
    log.info("Drift: detected=%s score=%.4f (mean KS, threshold=%.2f) features=%s",
             result["drift_detected"], result["drift_score"], threshold, drifted_names)
    return result


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
             reference_run_id: str, n_records: int, parquet_path: str) -> None:
    record = (
        str(uuid.uuid4()), run_index, batch_run_id,
        drift_result["drift_detected"], drift_result["drift_score"],
        drift_result["num_drifted_features"],
        ",".join(drift_result["drifted_feature_names"]),
        reference_run_id, n_records, parquet_path, datetime.utcnow(),
    )
    insert_sql = """
        INSERT INTO dwh_monitoring_drift.results
        (id, run_index, batch_run_id, drift_detected, drift_score,
         num_drifted_features, drifted_feature_names, reference_run_id,
         n_records, parquet_path, evaluated_at)
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
                        "num_drifted_features": 0, "drifted_feature_names": []}
        reference_run_id = "unknown"

    output_dir = os.getenv("DATA_MONITORING_DRIFT_PATH", "/data/monitoring/drift")
    parquet_path = write_parquet(drift_result, current_df, actual_run_index,
                                  actual_batch_run_id, output_dir)
    write_db(actual_run_index, actual_batch_run_id, drift_result,
             reference_run_id, len(current_df), parquet_path)

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
    log.info("Drift results logged to MLflow experiment '%s'", DRIFT_MONITORING_EXPERIMENT)

    result = {
        "run_index": actual_run_index,
        "batch_run_id": actual_batch_run_id,
        "drift_detected": drift_result["drift_detected"],
        "drift_score": drift_result["drift_score"],
        "num_drifted_features": drift_result["num_drifted_features"],
        "drifted_feature_names": drift_result["drifted_feature_names"],
        "n_records": len(current_df),
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
