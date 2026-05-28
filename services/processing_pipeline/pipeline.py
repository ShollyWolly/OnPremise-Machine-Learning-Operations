"""
Processing Pipeline Entrypoint
================================

Runs Stage 1 (ingest) → Stage 2 (clean) → writes outputs.

Outputs:
  - data/processed/YYYY-MM-DD_HHMMSS.parquet
  - dwh_clean.cleaned_features (PostgreSQL)

Usage:
    python pipeline.py
    python pipeline.py --raw-dir /data/raw --processed-dir /data/processed
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from clean import clean
from ingest import ingest

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def write_clean_parquet(df: pd.DataFrame, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    path = Path(output_dir) / f"{timestamp}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    log.info("Processed parquet → %s (%d rows)", path, len(df))
    return str(path)


def write_clean_postgres(df: pd.DataFrame) -> None:
    conn_params = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "dbname": os.getenv("POSTGRES_DB", "mlops"),
        "user": os.getenv("POSTGRES_USER", "mlops_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "changeme_secure_password"),
    }

    columns = [
        "record_id", "customer_id", "age", "annual_income", "credit_score",
        "loan_amount", "loan_term_months", "employment_length_years",
        "home_ownership_encoded", "debt_to_income_ratio", "num_credit_lines",
        "payment_history_score", "default_flag",
        "created_at", "ground_truth_available_at",
    ]
    if "source_parquet_file" in df.columns:
        columns.append("source_parquet_file")

    existing_cols = [c for c in columns if c in df.columns]
    records = [tuple(row) for row in df[existing_cols].itertuples(index=False, name=None)]

    insert_sql = f"""
        INSERT INTO dwh_clean.cleaned_features
        ({', '.join(existing_cols)})
        VALUES %s
        ON CONFLICT (record_id) DO NOTHING
    """

    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, insert_sql, records, page_size=500)
        conn.commit()

    log.info("PostgreSQL: inserted %d rows into dwh_clean.cleaned_features", len(df))


def run_pipeline(raw_dir: str, processed_dir: str) -> None:
    log.info("=== Processing Pipeline START ===")

    raw_df = ingest(raw_dir=raw_dir)
    if raw_df.empty:
        log.info("Nothing to process. Exiting.")
        return

    clean_df = clean(raw_df)
    if clean_df.empty:
        log.warning("All records dropped during cleaning. Exiting.")
        return

    write_clean_parquet(clean_df, processed_dir)
    write_clean_postgres(clean_df)

    log.info("=== Processing Pipeline COMPLETE: %d records processed ===", len(clean_df))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps processing pipeline")
    parser.add_argument("--raw-dir", default=os.getenv("DATA_RAW_PATH", "/data/raw"))
    parser.add_argument("--processed-dir", default=os.getenv("DATA_PROCESSED_PATH", "/data/processed"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(raw_dir=args.raw_dir, processed_dir=args.processed_dir)
