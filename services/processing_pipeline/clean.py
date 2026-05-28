"""
Stage 2: Data Cleaning
======================

Applies to the raw ingested DataFrame:
  1. Imputation  - median for numeric, mode for categorical
  2. Encoding    - OrdinalEncoder for home_ownership
  3. Outlier removal - IQR-based on loan_amount and annual_income
  4. Validation  - assert no nulls, check value ranges

Returns a cleaned DataFrame compatible with dwh_clean.cleaned_features schema.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HOME_OWNERSHIP_MAP = {"RENT": 0, "MORTGAGE": 1, "OWN": 2, "OTHER": 3}

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "credit_score",
    "loan_amount",
    "loan_term_months",
    "employment_length_years",
    "debt_to_income_ratio",
    "num_credit_lines",
    "payment_history_score",
]

EXPECTED_RANGES = {
    "age": (18, 75),
    "credit_score": (300, 850),
    "debt_to_income_ratio": (0.0, 5.0),
    "payment_history_score": (0.0, 100.0),
    "default_flag": (0, 1),
}


def _impute_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        missing = df[col].isna().sum()
        if missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log.info("Imputed %d nulls in '%s' with median=%.4f", missing, col, median_val)
    return df


def _impute_categorical(df: pd.DataFrame) -> pd.DataFrame:
    if "home_ownership" in df.columns:
        missing = df["home_ownership"].isna().sum()
        if missing > 0:
            mode_val = df["home_ownership"].mode()[0]
            df["home_ownership"] = df["home_ownership"].fillna(mode_val)
            log.info("Imputed %d nulls in 'home_ownership' with mode='%s'", missing, mode_val)
    return df


def _remove_outliers_iqr(df: pd.DataFrame, columns: list[str], k: float = 3.0) -> pd.DataFrame:
    """Remove rows where any of columns is more than k*IQR from Q1/Q3."""
    mask = pd.Series(True, index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        col_mask = df[col].between(lower, upper)
        removed = (~col_mask).sum()
        if removed > 0:
            log.info("Outlier removal: %d rows dropped for '%s' outside [%.2f, %.2f]",
                     removed, col, lower, upper)
        mask &= col_mask
    return df[mask].copy()


def _encode_home_ownership(df: pd.DataFrame) -> pd.DataFrame:
    df["home_ownership_encoded"] = (
        df["home_ownership"]
        .str.upper()
        .str.strip()
        .map(HOME_OWNERSHIP_MAP)
        .fillna(HOME_OWNERSHIP_MAP["OTHER"])
        .astype(int)
    )
    return df


def _validate(df: pd.DataFrame) -> None:
    null_counts = df[NUMERIC_FEATURES + ["home_ownership_encoded"]].isnull().sum()
    nulls_found = null_counts[null_counts > 0]
    if not nulls_found.empty:
        raise ValueError(f"Null values remain after cleaning: {nulls_found.to_dict()}")

    for col, (lo, hi) in EXPECTED_RANGES.items():
        if col not in df.columns:
            continue
        violations = (~df[col].between(lo, hi)).sum()
        if violations > 0:
            log.warning("Range violation: %d rows in '%s' outside [%s, %s]",
                        violations, col, lo, hi)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline.

    Input:  raw ingested DataFrame (from ingest.py)
    Output: cleaned DataFrame ready for dwh_clean.cleaned_features
    """
    if df.empty:
        return df

    log.info("Cleaning %d records", len(df))
    before = len(df)

    df = _impute_numeric(df)
    df = _impute_categorical(df)
    df = _remove_outliers_iqr(df, ["loan_amount", "annual_income"])
    df = _encode_home_ownership(df)

    _validate(df)

    log.info("Cleaning complete: %d → %d records (%.1f%% retained)",
             before, len(df), 100 * len(df) / before)

    # Select and order output columns
    output_cols = [
        "record_id", "customer_id", "age", "annual_income", "credit_score",
        "loan_amount", "loan_term_months", "employment_length_years",
        "home_ownership_encoded", "debt_to_income_ratio", "num_credit_lines",
        "payment_history_score", "default_flag",
        "created_at", "ground_truth_available_at",
    ]
    if "source_parquet_file" in df.columns:
        output_cols.append("source_parquet_file")

    return df[output_cols].copy()
