"""
Stage 1: Raw Ingestion
======================

Reads new (unprocessed) records from:
  - Parquet files in DATA_RAW_PATH that haven't been merged yet
  - PostgreSQL dwh_raw.raw_loan_applications

Merges on record_id, deduplicates, and returns a unified DataFrame.
Records already present in dwh_clean.cleaned_features are skipped.
"""

import logging
import os
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)


def _db_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "mlops"),
        user=os.getenv("POSTGRES_USER", "mlops_user"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme_secure_password"),
    )


def load_already_processed_ids() -> set:
    """Return record_ids already present in dwh_clean.cleaned_features."""
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT record_id FROM dwh_clean.cleaned_features")
            return {str(row[0]) for row in cur.fetchall()}


def load_raw_from_db(exclude_ids: set) -> pd.DataFrame:
    """Load all raw records from dwh_raw not yet cleaned."""
    query = """
        SELECT
            record_id, customer_id, age, annual_income, credit_score,
            loan_amount, loan_term_months, employment_length_years,
            home_ownership, debt_to_income_ratio, num_credit_lines,
            payment_history_score, default_flag, drift_applied, drift_factor,
            created_at, ground_truth_available_at
        FROM dwh_raw.raw_loan_applications
    """
    with _db_conn() as conn:
        df = pd.read_sql(query, conn)

    df["record_id"] = df["record_id"].astype(str)

    if exclude_ids:
        before = len(df)
        df = df[~df["record_id"].isin(exclude_ids)]
        log.info("DB: %d raw records, %d already processed, %d new",
                 before, before - len(df), len(df))
    return df


def load_raw_from_parquet(raw_dir: str, exclude_ids: set) -> pd.DataFrame:
    """Load all parquet files from raw_dir, excluding already-processed ids."""
    raw_path = Path(raw_dir)
    files = sorted(raw_path.glob("*.parquet"))

    if not files:
        log.info("No parquet files found in %s", raw_dir)
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_parquet(f, engine="pyarrow")
        df["source_parquet_file"] = f.name
        df["record_id"] = df["record_id"].astype(str)
        frames.append(df)
        log.debug("Loaded %s (%d rows)", f.name, len(df))

    combined = pd.concat(frames, ignore_index=True)

    if exclude_ids:
        combined = combined[~combined["record_id"].isin(exclude_ids)]

    log.info("Parquet: %d new records from %d file(s)", len(combined), len(files))
    return combined


def ingest(raw_dir: str | None = None) -> pd.DataFrame:
    """
    Main ingestion entry point.
    Returns a deduplicated DataFrame of unprocessed raw records.
    """
    raw_dir = raw_dir or os.getenv("DATA_RAW_PATH", "/data/raw")

    already_processed = load_already_processed_ids()
    log.info("%d records already cleaned; will skip them", len(already_processed))

    db_df = load_raw_from_db(already_processed)
    parquet_df = load_raw_from_parquet(raw_dir, already_processed)

    if db_df.empty and parquet_df.empty:
        log.info("No new records to process.")
        return pd.DataFrame()

    # Merge both sources; DB is authoritative — parquet fills in parquet filename
    if "source_parquet_file" not in db_df.columns:
        db_df["source_parquet_file"] = None

    merged = pd.concat([db_df, parquet_df], ignore_index=True)
    merged = merged.drop_duplicates(subset="record_id", keep="first")

    log.info("Ingestion: %d unique new records ready for cleaning", len(merged))
    return merged
