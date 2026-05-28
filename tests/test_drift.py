"""Tests for services/monitoring/data_drift.py - compute_drift (pure logic, no DB/MLflow)."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Stub out mlflow and psycopg2 so the module loads without a running server
for _mod in ("mlflow", "mlflow.MlflowClient", "psycopg2", "psycopg2.extras"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from data_drift import FEATURE_COLUMNS, compute_drift


def _make_feature_df(seed: int, shift: float = 0.0, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(20, 70, n),
        "annual_income": rng.uniform(30_000, 200_000, n) + shift * 20_000,
        "credit_score": rng.integers(400, 820, n) - int(shift * 100),
        "loan_amount": rng.uniform(5_000, 80_000, n),
        "loan_term_months": rng.choice([24, 36, 48, 60], n).astype(float),
        "employment_length_years": rng.uniform(0.5, 20.0, n),
        "home_ownership_encoded": rng.integers(0, 4, n).astype(float),
        "debt_to_income_ratio": rng.uniform(0.1, 2.5, n),
        "num_credit_lines": rng.integers(2, 20, n).astype(float),
        "payment_history_score": rng.uniform(20.0, 98.0, n),
    })


class TestComputeDrift:
    def test_returns_expected_keys(self):
        ref = _make_feature_df(0)
        cur = _make_feature_df(1)
        result = compute_drift(cur, ref)
        assert "drift_detected" in result
        assert "drift_score" in result
        assert "num_drifted_features" in result
        assert "drifted_feature_names" in result

    def test_identical_distributions_low_drift(self):
        ref = _make_feature_df(42, n=1000)
        cur = _make_feature_df(42, n=1000)
        result = compute_drift(cur, ref)
        assert result["drift_score"] < 0.30

    def test_heavily_shifted_distribution_detected(self, monkeypatch):
        monkeypatch.setenv("MAX_DRIFT_FEATURE_FRACTION", "0.10")
        ref = _make_feature_df(0, shift=0.0, n=1000)
        cur = _make_feature_df(1, shift=5.0, n=1000)
        result = compute_drift(cur, ref)
        assert result["drift_score"] > 0.10

    def test_drift_score_is_float(self):
        ref = _make_feature_df(0)
        cur = _make_feature_df(1)
        result = compute_drift(cur, ref)
        assert isinstance(result["drift_score"], float)

    def test_drift_detected_is_bool(self):
        ref = _make_feature_df(0)
        cur = _make_feature_df(1)
        result = compute_drift(cur, ref)
        assert isinstance(result["drift_detected"], bool)

    def test_drifted_feature_names_is_list(self):
        ref = _make_feature_df(0)
        cur = _make_feature_df(1)
        result = compute_drift(cur, ref)
        assert isinstance(result["drifted_feature_names"], list)

    def test_num_drifted_features_matches_names_length(self):
        ref = _make_feature_df(0, n=1000)
        cur = _make_feature_df(1, shift=3.0, n=1000)
        result = compute_drift(cur, ref)
        assert result["num_drifted_features"] == len(result["drifted_feature_names"])

    def test_drift_score_bounded_0_to_1(self):
        ref = _make_feature_df(0)
        cur = _make_feature_df(1)
        result = compute_drift(cur, ref)
        assert 0.0 <= result["drift_score"] <= 1.0

    def test_only_feature_columns_used(self):
        ref = _make_feature_df(0)
        cur = _make_feature_df(1)
        # Extra columns should be ignored without error
        ref["extra_col"] = 999
        cur["extra_col"] = 0
        result = compute_drift(cur, ref)
        assert "extra_col" not in result.get("drifted_feature_names", [])

    def test_drifted_names_subset_of_feature_columns(self):
        ref = _make_feature_df(0, n=1000)
        cur = _make_feature_df(1, shift=2.0, n=1000)
        result = compute_drift(cur, ref)
        for name in result["drifted_feature_names"]:
            assert name in FEATURE_COLUMNS
