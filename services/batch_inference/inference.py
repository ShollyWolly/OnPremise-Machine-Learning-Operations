"""
Batch Inference Pipeline
=========================
1. Reads latest cleaned data from data/processed/ (most recent parquet)
2. Sends records to Flask /predict endpoint in batches
3. Writes predictions to data/predictions/inference_run_NNNNN.parquet
4. Writes predictions to dwh_predictions.batch_predictions
5. Joins with ground truth and writes to dwh_history.prediction_ground_truth

Usage:
    python inference.py --run-index 42
    python inference.py --run-index 42 --batch-size 100
"""

import argparse
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

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


def load_clean_data(processed_dir: str) -> pd.DataFrame:
    """Load most recent processed parquet file."""
    path = Path(processed_dir)
    files = sorted(path.glob("*.parquet"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No processed parquet files found in {processed_dir}")
    target_file = files[0]
    df = pd.read_parquet(target_file, engine="pyarrow")
    log.info("Loaded %d records from %s", len(df), target_file.name)
    return df


def call_predict_endpoint(records: list[dict], endpoint: str, threshold: float) -> tuple:
    payload = {"records": records, "threshold": threshold}
    response = requests.post(f"{endpoint}/predict", json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["predictions"], data.get("model_name", ""), data.get("model_version", "")


def write_predictions_parquet(predictions_df: pd.DataFrame, output_dir: str, run_index: int) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"inference_run_{run_index:05d}.parquet"
    predictions_df.to_parquet(path, index=False, engine="pyarrow")
    log.info("Predictions parquet → %s (%d rows)", path, len(predictions_df))
    return str(path)


def write_predictions_postgres(predictions_df: pd.DataFrame) -> None:
    columns = [
        "prediction_id", "record_id", "customer_id",
        "default_probability", "default_flag_predicted", "decision_threshold",
        "model_name", "model_version", "batch_run_id", "predicted_at",
    ]
    available = [c for c in columns if c in predictions_df.columns]
    records = [tuple(row) for row in predictions_df[available].itertuples(index=False, name=None)]

    sql = f"""
        INSERT INTO dwh_predictions.batch_predictions ({", ".join(available)})
        VALUES %s
        ON CONFLICT (prediction_id) DO NOTHING
    """
    with _db_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, records, page_size=500)
        conn.commit()
    log.info("Inserted %d rows into dwh_predictions.batch_predictions", len(predictions_df))


def write_history(predictions_df: pd.DataFrame, clean_df: pd.DataFrame, run_index: int) -> None:
    """Join predictions with cleaned features + ground truth → dwh_history.prediction_ground_truth."""
    merged = predictions_df.merge(
        clean_df[["record_id"] + FEATURE_COLUMNS + ["default_flag"]],
        on="record_id", how="left",
    )

    columns = [
        "id", "run_index", "batch_run_id", "record_id", "customer_id",
        "age", "annual_income", "credit_score", "loan_amount", "loan_term_months",
        "employment_length_years", "home_ownership_encoded", "debt_to_income_ratio",
        "num_credit_lines", "payment_history_score",
        "default_probability", "default_flag_predicted", "decision_threshold",
        "model_name", "model_version", "actual_default_flag", "predicted_at",
    ]

    records = []
    for _, row in merged.iterrows():
        records.append((
            str(uuid.uuid4()), run_index, row.get("batch_run_id"),
            row.get("record_id"), row.get("customer_id"),
            row.get("age"), row.get("annual_income"), row.get("credit_score"),
            row.get("loan_amount"), row.get("loan_term_months"),
            row.get("employment_length_years"), row.get("home_ownership_encoded"),
            row.get("debt_to_income_ratio"), row.get("num_credit_lines"),
            row.get("payment_history_score"),
            row.get("default_probability"), row.get("default_flag_predicted"),
            row.get("decision_threshold"), row.get("model_name"), row.get("model_version"),
            row.get("default_flag"),  # actual_default_flag
            row.get("predicted_at"),
        ))

    sql = f"""
        INSERT INTO dwh_history.prediction_ground_truth ({", ".join(columns)})
        VALUES %s
        ON CONFLICT (id) DO NOTHING
    """
    with _db_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, records, page_size=500)
        conn.commit()
    log.info("Inserted %d rows into dwh_history.prediction_ground_truth", len(records))


def update_run_registry(run_index: int, batch_run_id: str, model_name: str,
                        model_version: str, n_records: int) -> None:
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE dwh_history.run_registry
                SET batch_run_id = %s, model_name = %s, model_version = %s,
                    n_records = %s, completed_at = NOW(), status = 'success'
                WHERE run_index = %s
            """, (batch_run_id, model_name, model_version, n_records, run_index))
        conn.commit()


def run_inference(
    processed_dir: str,
    predictions_dir: str,
    endpoint: str,
    batch_size: int,
    threshold: float,
    run_index: int,
) -> dict:
    log.info("=== Batch Inference START (run_index=%s) ===", run_index)
    batch_run_id = str(uuid.uuid4())

    clean_df = load_clean_data(processed_dir)
    clean_df["record_id"] = clean_df["record_id"].astype(str)

    all_predictions = []
    model_name, model_version = "", ""

    for start in range(0, len(clean_df), batch_size):
        chunk = clean_df.iloc[start:start + batch_size]
        records = chunk[["customer_id", "record_id"] + FEATURE_COLUMNS].to_dict(orient="records")
        preds, model_name, model_version = call_predict_endpoint(records, endpoint, threshold)

        for pred, (_, row) in zip(preds, chunk.iterrows()):
            all_predictions.append({
                "prediction_id": str(uuid.uuid4()),
                "record_id": str(row.get("record_id")),
                "customer_id": pred["customer_id"],
                "default_probability": pred["default_probability"],
                "default_flag_predicted": pred["default_flag_predicted"],
                "decision_threshold": pred["threshold_used"],
                "model_name": model_name,
                "model_version": str(model_version),
                "batch_run_id": batch_run_id,
                "predicted_at": datetime.utcnow().isoformat(),
            })
        log.info("Processed chunk %d–%d", start, min(start + batch_size, len(clean_df)))

    if not all_predictions:
        log.warning("No predictions generated.")
        return {"batch_run_id": batch_run_id, "n_predictions": 0, "run_index": run_index}

    pred_df = pd.DataFrame(all_predictions)
    write_predictions_parquet(pred_df, predictions_dir, run_index)
    write_predictions_postgres(pred_df)
    write_history(pred_df, clean_df, run_index)
    update_run_registry(run_index, batch_run_id, model_name, model_version, len(pred_df))

    result = {
        "batch_run_id": batch_run_id,
        "n_predictions": len(pred_df),
        "run_index": run_index,
        "model_name": model_name,
        "model_version": model_version,
    }
    log.info("=== Batch Inference COMPLETE: %d predictions, run_index=%s ===", len(pred_df), run_index)
    print(json.dumps(result))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference pipeline")
    parser.add_argument("--run-index", type=int, required=True, help="Run index from dwh_history.run_registry")
    parser.add_argument("--processed-dir", default=os.getenv("DATA_PROCESSED_PATH", "/data/processed"))
    parser.add_argument("--predictions-dir", default=os.getenv("DATA_PREDICTIONS_PATH", "/data/predictions"))
    parser.add_argument("--endpoint", default=os.getenv("FLASK_ENDPOINT", "http://flask-api:5001"))
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=float(os.getenv("DECISION_THRESHOLD", 0.5)))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        processed_dir=args.processed_dir,
        predictions_dir=args.predictions_dir,
        endpoint=args.endpoint,
        batch_size=args.batch_size,
        threshold=args.threshold,
        run_index=args.run_index,
    )
