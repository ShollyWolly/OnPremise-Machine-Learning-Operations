"""Integration tests: data flows through clean.py → schema validation end-to-end."""

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError
import pytest

from clean import clean
from platform_config import FEATURE_COLUMNS
from schema import CLEAN_SCHEMA, INFERENCE_FEATURE_SCHEMA, RAW_SCHEMA


class TestRawToCleanFlow:
    def test_clean_output_passes_clean_schema(self, raw_loan_df):
        """clean() output validates against CLEAN_SCHEMA without errors."""
        clean_df = clean(raw_loan_df)
        CLEAN_SCHEMA.validate(clean_df)

    def test_clean_output_feature_columns_pass_inference_schema(self, raw_loan_df):
        """Feature columns extracted from clean data pass INFERENCE_FEATURE_SCHEMA."""
        clean_df = clean(raw_loan_df)
        INFERENCE_FEATURE_SCHEMA.validate(clean_df[FEATURE_COLUMNS])

    def test_raw_fixture_passes_raw_schema(self, raw_loan_df):
        """Raw fixture data validates against RAW_SCHEMA."""
        RAW_SCHEMA.validate(raw_loan_df)

    def test_clean_removes_home_ownership_column(self, raw_loan_df):
        clean_df = clean(raw_loan_df)
        assert "home_ownership" not in clean_df.columns
        assert "home_ownership_encoded" in clean_df.columns

    def test_clean_output_has_no_nulls_in_features(self, raw_loan_df):
        clean_df = clean(raw_loan_df)
        null_counts = clean_df[FEATURE_COLUMNS].isnull().sum()
        assert null_counts.sum() == 0, f"Nulls found: {null_counts[null_counts > 0].to_dict()}"

    def test_clean_handles_nulls_in_input(self, raw_loan_df):
        """Nulls in raw input are imputed; clean output still passes schema."""
        raw_loan_df.loc[0:5, "age"] = np.nan
        raw_loan_df.loc[3:8, "annual_income"] = np.nan
        clean_df = clean(raw_loan_df)
        CLEAN_SCHEMA.validate(clean_df)

    def test_clean_handles_unknown_home_ownership(self, raw_loan_df):
        """Unknown home_ownership values get mapped to OTHER (encoded=3)."""
        raw_loan_df.loc[0, "home_ownership"] = "UNKNOWN_TYPE"
        clean_df = clean(raw_loan_df)
        assert clean_df.loc[clean_df.index[0], "home_ownership_encoded"] == 3

    def test_full_pipeline_record_ids_preserved(self, raw_loan_df):
        """record_id column survives cleaning unchanged."""
        original_ids = set(raw_loan_df["record_id"].astype(str))
        clean_df = clean(raw_loan_df)
        clean_ids = set(clean_df["record_id"].astype(str))
        # After outlier removal some records may be dropped, but no new IDs appear
        assert clean_ids.issubset(original_ids)

    def test_pipeline_output_column_order_stable(self, raw_loan_df):
        """FEATURE_COLUMNS are all present in clean output."""
        clean_df = clean(raw_loan_df)
        for col in FEATURE_COLUMNS:
            assert col in clean_df.columns, f"Missing column: {col}"


class TestSchemaRejectsInvalidData:
    def test_clean_schema_rejects_null_age(self):
        df = pd.DataFrame([{
            "record_id": "r1", "customer_id": "c1",
            "age": np.nan, "annual_income": 65000.0,
            "credit_score": 720.0, "loan_amount": 15000.0,
            "loan_term_months": 36.0, "employment_length_years": 5.0,
            "home_ownership_encoded": 1, "debt_to_income_ratio": 0.23,
            "num_credit_lines": 8.0, "payment_history_score": 85.0,
            "default_flag": 0,
        }])
        with pytest.raises(SchemaError):
            CLEAN_SCHEMA.validate(df)

    def test_inference_schema_rejects_missing_column(self):
        record = {col: 1.0 for col in FEATURE_COLUMNS}
        record.pop("credit_score")
        with pytest.raises(SchemaError):
            INFERENCE_FEATURE_SCHEMA.validate(pd.DataFrame([record]))

    def test_inference_schema_rejects_null_feature(self):
        record = {col: 1.0 for col in FEATURE_COLUMNS}
        record["age"] = np.nan
        with pytest.raises(SchemaError):
            INFERENCE_FEATURE_SCHEMA.validate(pd.DataFrame([record]))

    def test_raw_schema_rejects_negative_income(self):
        df = pd.DataFrame([{
            "record_id": "r1", "customer_id": "c1",
            "age": 30.0, "annual_income": -5000.0,
            "credit_score": 700.0, "loan_amount": 10000.0,
            "loan_term_months": 24.0, "employment_length_years": 2.0,
            "home_ownership": "RENT", "debt_to_income_ratio": 0.3,
            "num_credit_lines": 5.0, "payment_history_score": 70.0,
            "default_flag": 0,
        }])
        with pytest.raises(SchemaError):
            RAW_SCHEMA.validate(df)


class TestSchemaWithParquet:
    def test_clean_output_round_trips_through_parquet(self, raw_loan_df, tmp_path):
        """Data saved as parquet and re-loaded still passes CLEAN_SCHEMA."""
        clean_df = clean(raw_loan_df)
        path = tmp_path / "clean.parquet"
        clean_df.to_parquet(path, index=False, engine="pyarrow")
        reloaded = pd.read_parquet(path, engine="pyarrow")
        CLEAN_SCHEMA.validate(reloaded)

    def test_feature_columns_round_trip_inference_schema(self, raw_loan_df, tmp_path):
        """Feature columns saved as parquet and reloaded pass INFERENCE_FEATURE_SCHEMA."""
        clean_df = clean(raw_loan_df)
        path = tmp_path / "features.parquet"
        clean_df[FEATURE_COLUMNS].to_parquet(path, index=False, engine="pyarrow")
        reloaded = pd.read_parquet(path, engine="pyarrow")
        INFERENCE_FEATURE_SCHEMA.validate(reloaded)
