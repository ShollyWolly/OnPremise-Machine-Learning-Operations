"""Shared fixtures for the MLOps test suite."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).parent.parent
# Make service modules importable without installing them
for _p in [
    _ROOT / "services",
    _ROOT / "services" / "data_generator",
    _ROOT / "services" / "processing_pipeline",
    _ROOT / "services" / "monitoring",
    _ROOT / "services" / "batch_inference",
    _ROOT / "services" / "model_serving",
    _ROOT / "training",
]:
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)


@pytest.fixture
def perfect_binary_labels():
    """y_true and y_score where the model is perfect."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    return y_true, y_score


@pytest.fixture
def balanced_labels():
    """50/50 split, moderate model performance."""
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 50 + [1] * 50)
    y_score = np.where(y_true == 1, rng.uniform(0.4, 0.9, 100), rng.uniform(0.1, 0.6, 100))
    return y_true, y_score


@pytest.fixture
def raw_loan_df():
    """Minimal raw loan DataFrame matching the schema expected by clean.py."""
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame({
        "record_id": [f"r{i}" for i in range(n)],
        "customer_id": [f"c{i}" for i in range(n)],
        "age": rng.integers(20, 65, n),
        "annual_income": rng.uniform(30_000, 200_000, n),
        "credit_score": rng.integers(400, 820, n),
        "loan_amount": rng.uniform(5_000, 80_000, n),
        "loan_term_months": rng.choice([24, 36, 48, 60], n),
        "employment_length_years": rng.uniform(0.5, 20.0, n),
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], n),
        "debt_to_income_ratio": rng.uniform(0.05, 2.5, n),
        "num_credit_lines": rng.integers(2, 20, n),
        "payment_history_score": rng.uniform(20.0, 98.0, n),
        "default_flag": rng.integers(0, 2, n),
        "created_at": pd.Timestamp("2024-01-01"),
        "ground_truth_available_at": pd.Timestamp("2024-01-01"),
    })
