"""Tests for services/processing_pipeline/clean.py - data cleaning pipeline."""

import numpy as np
import pandas as pd
import pytest

from clean import (
    HOME_OWNERSHIP_MAP,
    NUMERIC_FEATURES,
    _encode_home_ownership,
    _impute_categorical,
    _impute_numeric,
    _remove_outliers_iqr,
    _validate,
    clean,
)


class TestClean:
    def test_returns_dataframe(self, raw_loan_df):
        result = clean(raw_loan_df)
        assert isinstance(result, pd.DataFrame)

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame()
        result = clean(empty)
        assert result.empty

    def test_output_has_expected_columns(self, raw_loan_df):
        result = clean(raw_loan_df)
        expected = {
            "record_id", "customer_id", "age", "annual_income", "credit_score",
            "loan_amount", "loan_term_months", "employment_length_years",
            "home_ownership_encoded", "debt_to_income_ratio", "num_credit_lines",
            "payment_history_score", "default_flag",
            "created_at", "ground_truth_available_at",
        }
        assert expected.issubset(set(result.columns))

    def test_no_nulls_after_clean(self, raw_loan_df):
        result = clean(raw_loan_df)
        null_counts = result[NUMERIC_FEATURES + ["home_ownership_encoded"]].isnull().sum()
        assert null_counts.sum() == 0

    def test_home_ownership_column_dropped(self, raw_loan_df):
        result = clean(raw_loan_df)
        assert "home_ownership" not in result.columns

    def test_record_count_does_not_increase(self, raw_loan_df):
        result = clean(raw_loan_df)
        assert len(result) <= len(raw_loan_df)

    def test_source_parquet_file_preserved_if_present(self, raw_loan_df):
        raw_loan_df["source_parquet_file"] = "test.parquet"
        result = clean(raw_loan_df)
        assert "source_parquet_file" in result.columns


class TestImpute:
    def test_impute_numeric_fills_nulls(self, raw_loan_df):
        raw_loan_df.loc[0:4, "age"] = np.nan
        result = _impute_numeric(raw_loan_df.copy())
        assert result["age"].isna().sum() == 0

    def test_impute_numeric_uses_median(self, raw_loan_df):
        raw_loan_df.loc[0, "age"] = np.nan
        expected_median = raw_loan_df["age"].median()  # computed after introducing the null
        result = _impute_numeric(raw_loan_df.copy())
        assert result.loc[0, "age"] == expected_median

    def test_impute_numeric_skips_missing_column(self, raw_loan_df):
        df = raw_loan_df.drop(columns=["age"])
        result = _impute_numeric(df)  # should not raise
        assert "age" not in result.columns

    def test_impute_categorical_fills_nulls(self, raw_loan_df):
        raw_loan_df.loc[0:3, "home_ownership"] = np.nan
        result = _impute_categorical(raw_loan_df.copy())
        assert result["home_ownership"].isna().sum() == 0

    def test_impute_categorical_uses_mode(self, raw_loan_df):
        raw_loan_df["home_ownership"] = "RENT"
        raw_loan_df.loc[0, "home_ownership"] = np.nan
        result = _impute_categorical(raw_loan_df.copy())
        assert result.loc[0, "home_ownership"] == "RENT"


class TestRemoveOutliers:
    def test_removes_extreme_outlier(self):
        df = pd.DataFrame({
            "loan_amount": [10_000.0] * 99 + [9_999_999.0],
            "annual_income": [50_000.0] * 100,
        })
        result = _remove_outliers_iqr(df, ["loan_amount"], k=3.0)
        assert len(result) < 100

    def test_no_removal_when_no_outliers(self, raw_loan_df):
        before = len(raw_loan_df)
        result = _remove_outliers_iqr(raw_loan_df.copy(), ["loan_amount", "annual_income"])
        assert len(result) <= before

    def test_returns_copy(self, raw_loan_df):
        result = _remove_outliers_iqr(raw_loan_df.copy(), ["loan_amount"])
        assert result is not raw_loan_df

    def test_skips_missing_column(self, raw_loan_df):
        result = _remove_outliers_iqr(raw_loan_df.copy(), ["nonexistent_col"])
        assert len(result) == len(raw_loan_df)


class TestEncodeHomeOwnership:
    def test_known_values_encoded_correctly(self):
        df = pd.DataFrame({"home_ownership": ["RENT", "MORTGAGE", "OWN", "OTHER"]})
        result = _encode_home_ownership(df)
        assert list(result["home_ownership_encoded"]) == [0, 1, 2, 3]

    def test_lowercase_input_normalized(self):
        df = pd.DataFrame({"home_ownership": ["rent", "mortgage"]})
        result = _encode_home_ownership(df)
        assert result["home_ownership_encoded"].iloc[0] == HOME_OWNERSHIP_MAP["RENT"]

    def test_unknown_value_maps_to_other(self):
        df = pd.DataFrame({"home_ownership": ["UNKNOWN_TYPE"]})
        result = _encode_home_ownership(df)
        assert result["home_ownership_encoded"].iloc[0] == HOME_OWNERSHIP_MAP["OTHER"]

    def test_output_dtype_is_int(self):
        df = pd.DataFrame({"home_ownership": ["RENT", "OWN"]})
        result = _encode_home_ownership(df)
        assert np.issubdtype(result["home_ownership_encoded"].dtype, np.integer)


class TestValidate:
    def test_raises_on_nulls(self, raw_loan_df):
        raw_loan_df = _encode_home_ownership(raw_loan_df)
        raw_loan_df.loc[0, "age"] = np.nan
        with pytest.raises(ValueError, match="Null values remain"):
            _validate(raw_loan_df)

    def test_passes_on_clean_data(self, raw_loan_df):
        raw_loan_df = _encode_home_ownership(raw_loan_df)
        _validate(raw_loan_df)  # should not raise

    def test_range_violation_does_not_raise(self, raw_loan_df, caplog):
        import logging
        raw_loan_df = _encode_home_ownership(raw_loan_df)
        raw_loan_df.loc[0, "credit_score"] = 999  # out of [300, 850]
        with caplog.at_level(logging.WARNING):
            _validate(raw_loan_df)  # warns but does not raise
        assert "Range violation" in caplog.text
