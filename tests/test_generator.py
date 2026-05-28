"""Tests for services/data_generator/generator.py - synthetic data generation."""

import numpy as np
import pandas as pd
import pytest

from config import STABLE_PARAMS
from generator import _apply_drift, _sample_feature, generate_records

EXPECTED_COLUMNS = {
    "record_id", "customer_id", "age", "annual_income", "credit_score",
    "loan_amount", "loan_term_months", "employment_length_years",
    "home_ownership", "debt_to_income_ratio", "num_credit_lines",
    "payment_history_score", "default_flag", "drift_applied", "drift_factor",
    "created_at", "ground_truth_available_at", "ingested_at",
}


class TestGenerateRecords:
    def test_returns_dataframe(self):
        df = generate_records(50, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = generate_records(200, seed=1)
        assert len(df) == 200

    def test_all_expected_columns_present(self):
        df = generate_records(10, seed=2)
        assert EXPECTED_COLUMNS.issubset(set(df.columns))

    def test_stable_mode_drift_applied_false(self):
        df = generate_records(50, drift_factor=0.0, seed=3)
        assert df["drift_applied"].all() == False
        assert (df["drift_factor"] == 0.0).all()

    def test_drift_mode_drift_applied_true(self):
        df = generate_records(50, drift_factor=0.5, seed=4)
        assert df["drift_applied"].all()
        assert (df["drift_factor"] == 0.5).all()

    def test_default_flag_binary(self):
        df = generate_records(500, seed=5)
        assert set(df["default_flag"].unique()).issubset({0, 1})

    def test_default_rate_plausible(self):
        df = generate_records(1000, seed=6)
        rate = df["default_flag"].mean()
        assert 0.05 < rate < 0.60, f"Implausible default rate: {rate:.3f}"

    def test_age_within_bounds(self):
        df = generate_records(500, seed=7)
        assert df["age"].between(18, 75).all()

    def test_credit_score_within_bounds(self):
        df = generate_records(500, seed=8)
        assert df["credit_score"].between(300, 850).all()

    def test_payment_history_score_within_bounds(self):
        df = generate_records(500, seed=9)
        assert df["payment_history_score"].between(0.0, 100.0).all()

    def test_debt_to_income_within_bounds(self):
        df = generate_records(500, seed=10)
        assert df["debt_to_income_ratio"].between(0.0, 5.0).all()

    def test_record_ids_are_unique(self):
        df = generate_records(200, seed=11)
        assert df["record_id"].nunique() == 200

    def test_customer_ids_are_uuid_format(self):
        df = generate_records(10, seed=12)
        import re
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        assert df["customer_id"].apply(lambda x: bool(uuid_pattern.match(x))).all()

    def test_same_seed_same_numeric_output(self):
        # record_id uses uuid4 (truly random) so we compare numeric columns only
        df1 = generate_records(100, seed=42)
        df2 = generate_records(100, seed=42)
        numeric_cols = ["age", "annual_income", "credit_score", "loan_amount",
                        "loan_term_months", "employment_length_years",
                        "debt_to_income_ratio", "num_credit_lines",
                        "payment_history_score", "default_flag"]
        pd.testing.assert_frame_equal(df1[numeric_cols], df2[numeric_cols])

    def test_different_seeds_differ(self):
        df1 = generate_records(100, seed=1)
        df2 = generate_records(100, seed=2)
        assert not df1["credit_score"].equals(df2["credit_score"])

    def test_drift_shifts_credit_score_down(self):
        df_stable = generate_records(2000, drift_factor=0.0, seed=0)
        df_drift = generate_records(2000, drift_factor=1.0, seed=0)
        assert df_drift["credit_score"].mean() < df_stable["credit_score"].mean()

    def test_home_ownership_valid_categories(self):
        df = generate_records(500, seed=13)
        assert set(df["home_ownership"].unique()).issubset({"RENT", "MORTGAGE", "OWN", "OTHER"})


class TestApplyDrift:
    def test_zero_drift_returns_unchanged_params(self):
        params = STABLE_PARAMS["credit_score"].copy()
        result = _apply_drift(params, "credit_score", 0.0)
        assert result["mean"] == params["mean"]

    def test_full_drift_shifts_mean_down_for_credit_score(self):
        params = STABLE_PARAMS["credit_score"].copy()
        result = _apply_drift(params, "credit_score", 1.0)
        assert result["mean"] < params["mean"]

    def test_partial_drift_is_between_zero_and_full(self):
        params = STABLE_PARAMS["credit_score"].copy()
        base_mean = params["mean"]
        full = _apply_drift(params, "credit_score", 1.0)["mean"]
        half = _apply_drift(params, "credit_score", 0.5)["mean"]
        assert full < half < base_mean

    def test_unknown_feature_returns_unchanged(self):
        params = STABLE_PARAMS["age"].copy()
        result = _apply_drift(params, "age", 1.0)
        assert result == params

    def test_does_not_mutate_original_params(self):
        params = STABLE_PARAMS["credit_score"].copy()
        original_mean = params["mean"]
        _apply_drift(params, "credit_score", 1.0)
        assert params["mean"] == original_mean

    def test_std_never_drops_below_one(self):
        params = STABLE_PARAMS["credit_score"].copy()
        result = _apply_drift(params, "credit_score", 100.0)
        assert result["std"] >= 1.0


class TestSampleFeature:
    def _make_rng(self, seed=0):
        return np.random.default_rng(seed)

    def test_normal_returns_correct_length(self):
        params = STABLE_PARAMS["credit_score"]
        rng = self._make_rng()
        values = _sample_feature(rng, params, 100)
        assert len(values) == 100

    def test_normal_within_clip_bounds(self):
        params = STABLE_PARAMS["credit_score"]
        rng = self._make_rng()
        values = _sample_feature(rng, params, 1000)
        assert values.min() >= params["clip_min"]
        assert values.max() <= params["clip_max"]

    def test_choice_returns_only_valid_categories(self):
        params = STABLE_PARAMS["home_ownership"]
        rng = self._make_rng()
        values = _sample_feature(rng, params, 500)
        assert set(values).issubset(set(params["choices"]))

    def test_lognormal_within_clip_bounds(self):
        params = STABLE_PARAMS["annual_income"]
        rng = self._make_rng()
        values = _sample_feature(rng, params, 500)
        assert values.min() >= params["clip_min"]
        assert values.max() <= params["clip_max"]

    def test_gamma_within_clip_bounds(self):
        params = STABLE_PARAMS["employment_length_years"]
        rng = self._make_rng()
        values = _sample_feature(rng, params, 500)
        assert values.min() >= params["clip_min"]
        assert values.max() <= params["clip_max"]

    def test_beta_within_clip_bounds(self):
        params = STABLE_PARAMS["payment_history_score"]
        rng = self._make_rng()
        values = _sample_feature(rng, params, 500)
        assert values.min() >= params["clip_min"]
        assert values.max() <= params["clip_max"]

    def test_poisson_within_clip_bounds(self):
        params = STABLE_PARAMS["num_credit_lines"]
        rng = self._make_rng()
        values = _sample_feature(rng, params, 500)
        assert values.min() >= params["clip_min"]
        assert values.max() <= params["clip_max"]

    def test_unknown_dist_raises(self):
        params = {"dist": "unicorn", "clip_min": 0, "clip_max": 10}
        rng = self._make_rng()
        with pytest.raises(ValueError, match="Unknown distribution"):
            _sample_feature(rng, params, 10)

    def test_int_dtype_applied(self):
        params = STABLE_PARAMS["age"]
        rng = self._make_rng()
        values = _sample_feature(rng, params, 50)
        assert np.issubdtype(values.dtype, np.integer)
