"""
Synthetic Loan Application Data Generator
==========================================

Generates realistic credit risk data and writes it to:
  1. Parquet files at DATA_RAW_PATH/YYYY-MM-DD_HH.parquet
  2. PostgreSQL table dwh_raw.raw_loan_applications

Usage:
    python generator.py --mode stable --n-records 1000
    python generator.py --mode drift --drift-factor 0.5 --n-records 1000

Arguments:
    --mode          stable | drift
    --drift-factor  float in [0.0, 1.0] (only used in drift mode, default 0.5)
    --n-records     number of rows to generate (default 1000)
    --no-db         skip writing to PostgreSQL (parquet only)
    --no-parquet    skip writing parquet (DB only)
"""

from __future__ import annotations

import argparse
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from config import DEFAULT_COEFFICIENTS, DRIFT_DELTAS, STABLE_PARAMS

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distribution samplers
# ---------------------------------------------------------------------------

def _sample_feature(rng: np.random.Generator, params: dict, n: int, drift_factor: float = 0.0) -> np.ndarray:
    """Sample n values from distribution described by params, with optional drift."""
    p = dict(params)  # shallow copy

    if p["dist"] == "normal":
        delta = DRIFT_DELTAS.get(None, {})
        mean = p["mean"]
        std = p["std"]
        values = rng.normal(mean, std, n)
        values = np.clip(values, p["clip_min"], p["clip_max"])

    elif p["dist"] == "lognormal":
        values = rng.lognormal(p["mean"], p["std"], n)
        values = np.clip(values, p["clip_min"], p["clip_max"])

    elif p["dist"] == "gamma":
        values = rng.gamma(p["shape"], p["scale"], n)
        values = np.clip(values, p["clip_min"], p["clip_max"])

    elif p["dist"] == "beta":
        values = rng.beta(p["a"], p["b"], n) * p["scale"]
        values = np.clip(values, p["clip_min"], p["clip_max"])

    elif p["dist"] == "poisson":
        values = rng.poisson(p["lam"], n).astype(float)
        values = np.clip(values, p["clip_min"], p["clip_max"])

    elif p["dist"] == "choice":
        weights = np.array(p["weights"]) / sum(p["weights"])
        values = rng.choice(p["choices"], n, p=weights)
        return values  # no clip for categoricals

    else:
        raise ValueError(f"Unknown distribution: {p['dist']}")

    if p.get("dtype") == "int":
        values = values.astype(int)
    return values


def _apply_drift(params: dict, feature_name: str, drift_factor: float) -> dict:
    """Linearly interpolate distribution params toward drift target based on drift_factor."""
    p = dict(params)
    deltas = DRIFT_DELTAS.get(feature_name, {})
    if not deltas or drift_factor == 0.0:
        return p

    if "mean_delta" in deltas:
        p["mean"] = p["mean"] + deltas["mean_delta"] * drift_factor
    if "std_delta" in deltas:
        p["std"] = max(p["std"] + deltas["std_delta"] * drift_factor, 1.0)
    if "shape_delta" in deltas:
        p["shape"] = max(p.get("shape", 1.0) + deltas["shape_delta"] * drift_factor, 0.5)
    if "scale_delta" in deltas:
        p["scale"] = max(p.get("scale", 1.0) + deltas["scale_delta"] * drift_factor, 0.1)
    if "a_delta" in deltas:
        p["a"] = max(p.get("a", 1.0) + deltas["a_delta"] * drift_factor, 0.5)
    if "weights_drift" in deltas:
        stable_w = np.array(p["weights"])
        drift_w = np.array(deltas["weights_drift"])
        p["weights"] = list(stable_w + (drift_w - stable_w) * drift_factor)
    return p


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_records(n: int, drift_factor: float = 0.0, seed: int | None = None) -> pd.DataFrame:
    """
    Generate n synthetic loan applications.

    drift_factor=0.0 → stable distribution
    drift_factor=1.0 → maximum covariate shift (target relationship unchanged)

    When seed is None and drift_factor == 0, a date-based seed is used so that
    stable-mode batches generated on the same day share the same base distribution,
    keeping monitoring metrics stable and avoiding false-positive drift alerts.
    Drift mode always draws fresh randomness so covariate shift is visible.
    """
    if seed is None and drift_factor == 0.0:
        seed = int(datetime.utcnow().strftime("%Y%m%d"))
    rng = np.random.default_rng(seed)

    rows: dict[str, np.ndarray] = {}

    for feature, params in STABLE_PARAMS.items():
        effective_params = _apply_drift(params, feature, drift_factor)
        rows[feature] = _sample_feature(rng, effective_params, n, drift_factor)

    df = pd.DataFrame(rows)

    # Derived feature: debt-to-income ratio
    df["debt_to_income_ratio"] = (
        df["loan_amount"] / df["annual_income"]
    ).round(4).clip(0.0, 5.0)

    # Deterministic target from logistic model + noise
    log_odds = DEFAULT_COEFFICIENTS["intercept"]
    for feat, coef in DEFAULT_COEFFICIENTS.items():
        if feat == "intercept":
            continue
        if feat in df.columns:
            log_odds = log_odds + coef * df[feat].astype(float)

    prob = 1.0 / (1.0 + np.exp(-log_odds))
    # Tiny perturbation - keeps labels stochastic without degrading signal
    noise = rng.normal(0, 0.002, n)
    prob_noisy = np.clip(prob + noise, 0.0, 1.0)
    df["default_flag"] = (rng.random(n) < prob_noisy).astype(int)

    # IDs and timestamps - ground truth immediately available for infinite demo
    now = datetime.utcnow()

    _CUST_NS = uuid.UUID("12345678-1234-5678-1234-567812345678")
    customer_pool_size = 10_000
    cust_indices = rng.integers(0, customer_pool_size, n)

    df.insert(0, "record_id", [str(uuid.uuid4()) for _ in range(n)])
    df.insert(1, "customer_id", [str(uuid.uuid5(_CUST_NS, f"cust_{i}")) for i in cust_indices])
    df["drift_applied"] = drift_factor > 0.0
    df["drift_factor"] = drift_factor
    df["created_at"] = now
    df["ground_truth_available_at"] = now
    df["ingested_at"] = now

    log.info(
        "Generated %d records | drift_factor=%.2f | default_rate=%.3f",
        n, drift_factor, df["default_flag"].mean(),
    )
    return df


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_parquet(df: pd.DataFrame, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    path = Path(output_dir) / f"{timestamp}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    log.info("Parquet written → %s (%d rows)", path, len(df))
    return str(path)


def write_postgres(df: pd.DataFrame) -> None:
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
        "home_ownership", "debt_to_income_ratio", "num_credit_lines",
        "payment_history_score", "default_flag", "drift_applied",
        "drift_factor", "created_at", "ground_truth_available_at", "ingested_at",
    ]

    records = [tuple(row) for row in df[columns].itertuples(index=False, name=None)]

    insert_sql = f"""
        INSERT INTO dwh_raw.raw_loan_applications
        ({', '.join(columns)})
        VALUES %s
        ON CONFLICT (record_id) DO NOTHING
    """

    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, insert_sql, records, page_size=500)
        conn.commit()

    log.info("PostgreSQL: inserted %d rows into dwh_raw.raw_loan_applications", len(df))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps synthetic data generator")
    parser.add_argument("--mode", choices=["stable", "drift"], default="stable",
                        help="stable = fixed distribution; drift = shifted distribution")
    parser.add_argument("--drift-factor", type=float, default=0.5,
                        help="Shift magnitude [0.0–1.0]. Only used in drift mode.")
    parser.add_argument("--n-records", type=int, default=1000,
                        help="Number of records to generate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip PostgreSQL write")
    parser.add_argument("--no-parquet", action="store_true",
                        help="Skip Parquet write")
    parser.add_argument("--output-dir",
                        default=os.getenv("DATA_RAW_PATH", "/data/raw"),
                        help="Directory for Parquet output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drift_factor = args.drift_factor if args.mode == "drift" else 0.0

    df = generate_records(
        n=args.n_records,
        drift_factor=drift_factor,
        seed=args.seed,
    )

    if not args.no_parquet:
        write_parquet(df, args.output_dir)

    if not args.no_db:
        write_postgres(df)

    log.info("Generation complete.")


if __name__ == "__main__":
    main()
