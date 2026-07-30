"""Tests for services/monitoring/hard_metrics.py - compute_metrics/should_retrain (pure logic, no DB/MLflow)."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Stub out mlflow and psycopg2 so the module loads without a running server
for _mod in ("mlflow", "mlflow.MlflowClient", "psycopg2", "psycopg2.extras"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from hard_metrics import compute_metrics, should_retrain
from metrics import compute_all_metrics


def _make_eval_df(seed: int, n: int = 400, threshold: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    y_score = np.clip(y_true * 0.6 + rng.normal(0.2, 0.2, n), 0.0, 1.0)
    y_pred = (y_score >= threshold).astype(int)
    return pd.DataFrame({
        "run_index": 1,
        "batch_run_id": "batch-1",
        "default_probability": y_score,
        "default_flag_predicted": y_pred,
        "actual_default_flag": y_true,
        "model_version": "1",
        "decision_threshold": threshold,
    })


class TestComputeMetrics:
    def test_returns_expected_keys(self):
        df = _make_eval_df(0)
        metrics, primary_metric, raw_metrics, run = compute_metrics(df)
        for key in ("roc_auc", "pr_auc", "f1_score", "precision_score", "recall_score", "accuracy", "n_records"):
            assert key in metrics
        assert isinstance(primary_metric, str)
        assert "roc_auc" in raw_metrics
        assert run is not None

    def test_n_records_matches_input(self):
        df = _make_eval_df(1, n=250)
        metrics, *_ = compute_metrics(df)
        assert metrics["n_records"] == 250

    def test_metrics_bounded_0_to_1(self):
        df = _make_eval_df(2)
        metrics, *_ = compute_metrics(df)
        for key in ("roc_auc", "pr_auc", "f1_score", "precision_score", "recall_score", "accuracy"):
            assert 0.0 <= metrics[key] <= 1.0

    def test_matches_sklearn_baseline_at_default_threshold(self):
        """Evidently-derived accuracy/precision/recall/f1/roc_auc should match the
        shared services/metrics.py baseline when the decision threshold is 0.5."""
        df = _make_eval_df(3, threshold=0.5)
        metrics, *_ = compute_metrics(df)

        y_true = df["actual_default_flag"].values.astype(int)
        y_score = df["default_probability"].values.astype(float)
        y_pred = df["default_flag_predicted"].values.astype(int)
        baseline = compute_all_metrics(y_true, y_score, y_pred=y_pred)

        assert metrics["roc_auc"] == pytest.approx(baseline["roc_auc"], abs=1e-3)
        assert metrics["f1_score"] == pytest.approx(baseline["f1"], abs=1e-3)
        assert metrics["precision_score"] == pytest.approx(baseline["precision"], abs=1e-3)
        assert metrics["recall_score"] == pytest.approx(baseline["recall"], abs=1e-3)
        assert metrics["accuracy"] == pytest.approx(baseline["accuracy"], abs=1e-3)

    def test_run_is_returned(self):
        df = _make_eval_df(4)
        *_, run = compute_metrics(df)
        assert run is not None
        assert hasattr(run, "save_html")


class TestShouldRetrain:
    def test_below_threshold_triggers_retrain(self, monkeypatch):
        monkeypatch.setenv("MIN_ROC_AUC", "0.70")
        triggered, reason = should_retrain({"roc_auc": 0.5}, "roc_auc")
        assert triggered is True
        assert "0.70" in reason or "0.7" in reason

    def test_above_threshold_no_retrain(self, monkeypatch):
        monkeypatch.setenv("MIN_ROC_AUC", "0.70")
        triggered, reason = should_retrain({"roc_auc": 0.95}, "roc_auc")
        assert triggered is False

    def test_uses_primary_metric_key_mapping(self, monkeypatch):
        monkeypatch.setenv("MIN_ROC_AUC", "0.70")
        triggered, _ = should_retrain({"f1_score": 0.95, "roc_auc": 0.1}, "f1")
        assert triggered is False
