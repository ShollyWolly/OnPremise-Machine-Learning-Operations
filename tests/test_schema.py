"""Tests for services/schema.py — Pandera schema validation."""

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError
import pytest

from platform_config import FEATURE_COLUMNS
from schema import CLEAN_SCHEMA, INFERENCE_FEATURE_SCHEMA, RAW_SCHEMA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_raw_record() -> dict:
    return {
        "record_id": "r1",
        "customer_id": "c1",
        "age": 35.0,
        "annual_income": 65000.0,
        "credit_score": 720.0,
        "loan_amount": 15000.0,
        "loan_term_months": 36.0,
        "employment_length_years": 5.0,
        "home_ownership": "RENT",
        "debt_to_income_ratio": 0.23,
        "num_credit_lines": 8.0,
        "payment_history_score": 85.0,
        "default_flag": 0,
    }


def _base_clean_record() -> dict:
    return {
        "record_id": "r1",
        "customer_id": "c1",
        "age": 35.0,
        "annual_income": 65000.0,
        "credit_score": 720.0,
        "loan_amount": 15000.0,
        "loan_term_months": 36.0,
        "employment_length_years": 5.0,
        "home_ownership_encoded": 1,
        "debt_to_income_ratio": 0.23,
        "num_credit_lines": 8.0,
        "payment_history_score": 85.0,
        "default_flag": 0,
    }


def _base_inference_record() -> dict:
    return {
        "age": 35.0,
        "annual_income": 65000.0,
        "credit_score": 720.0,
        "loan_amount": 15000.0,
        "loan_term_months": 36.0,
        "employment_length_years": 5.0,
        "home_ownership_encoded": 1.0,
        "debt_to_income_ratio": 0.23,
        "num_credit_lines": 8.0,
        "payment_history_score": 85.0,
    }


# ---------------------------------------------------------------------------
# RAW_SCHEMA
# ---------------------------------------------------------------------------

class TestRawSchema:
    def test_valid_record_passes(self):
        df = pd.DataFrame([_base_raw_record()])
        RAW_SCHEMA.validate(df)

    def test_nullable_fields_allowed(self):
        record = _base_raw_record()
        record["age"] = None
        df = pd.DataFrame([record])
        RAW_SCHEMA.validate(df)

    def test_extra_columns_allowed(self):
        record = _base_raw_record()
        record["extra_col"] = "some_value"
        df = pd.DataFrame([record])
        RAW_SCHEMA.validate(df)

    def test_negative_annual_income_fails(self):
        record = _base_raw_record()
        record["annual_income"] = -100.0
        df = pd.DataFrame([record])
        with pytest.raises(SchemaError):
            RAW_SCHEMA.validate(df)

    def test_credit_score_below_300_fails(self):
        record = _base_raw_record()
        record["credit_score"] = 200.0
        df = pd.DataFrame([record])
        with pytest.raises(SchemaError):
            RAW_SCHEMA.validate(df)

    def test_credit_score_above_900_fails(self):
        record = _base_raw_record()
        record["credit_score"] = 950.0
        df = pd.DataFrame([record])
        with pytest.raises(SchemaError):
            RAW_SCHEMA.validate(df)

    def test_payment_history_above_100_fails(self):
        record = _base_raw_record()
        record["payment_history_score"] = 110.0
        df = pd.DataFrame([record])
        with pytest.raises(SchemaError):
            RAW_SCHEMA.validate(df)

    def test_invalid_default_flag_fails(self):
        record = _base_raw_record()
        record["default_flag"] = 5
        df = pd.DataFrame([record])
        with pytest.raises(SchemaError):
            RAW_SCHEMA.validate(df)

    def test_coerces_string_numbers(self):
        record = _base_raw_record()
        record["age"] = "35"
        df = pd.DataFrame([record])
        validated = RAW_SCHEMA.validate(df)
        assert validated["age"].iloc[0] == 35.0

    def test_multiple_valid_records(self):
        records = [_base_raw_record() for _ in range(10)]
        df = pd.DataFrame(records)
        RAW_SCHEMA.validate(df)


# ---------------------------------------------------------------------------
# CLEAN_SCHEMA
# ---------------------------------------------------------------------------

class TestCleanSchema:
    def test_valid_record_passes(self):
        df = pd.DataFrame([_base_clean_record()])
        CLEAN_SCHEMA.validate(df)

    def test_null_age_fails(self):
        record = _base_clean_record()
        record["age"] = None
        df = pd.DataFrame([record])
        with pytest.raises(SchemaError):
            CLEAN_SCHEMA.validate(df)

    def test_null_credit_score_fails(self):
        record = _base_clean_record()
        record["credit_score"] = np.nan
        df = pd.DataFrame([record])
        with pytest.raises(SchemaError):
            CLEAN_SCHEMA.validate(df)

    def test_invalid_home_ownership_encoded_fails(self):
        record = _base_clean_record()
        record["home_ownership_encoded"] = 9
        df = pd.DataFrame([record])
        with pytest.raises(SchemaError):
            CLEAN_SCHEMA.validate(df)

    def test_all_valid_home_ownership_codes_pass(self):
        records = []
        for code in [0, 1, 2, 3]:
            r = _base_clean_record()
            r["home_ownership_encoded"] = code
            records.append(r)
        CLEAN_SCHEMA.validate(pd.DataFrame(records))

    def test_null_default_flag_allowed(self):
        record = _base_clean_record()
        record["default_flag"] = None
        df = pd.DataFrame([record])
        CLEAN_SCHEMA.validate(df)

    def test_negative_loan_amount_fails(self):
        record = _base_clean_record()
        record["loan_amount"] = -500.0
        df = pd.DataFrame([record])
        with pytest.raises(SchemaError):
            CLEAN_SCHEMA.validate(df)

    def test_extra_columns_allowed(self):
        record = _base_clean_record()
        record["source_parquet_file"] = "test.parquet"
        df = pd.DataFrame([record])
        CLEAN_SCHEMA.validate(df)


# ---------------------------------------------------------------------------
# INFERENCE_FEATURE_SCHEMA
# ---------------------------------------------------------------------------

class TestInferenceFeatureSchema:
    def test_valid_record_passes(self):
        df = pd.DataFrame([_base_inference_record()])
        INFERENCE_FEATURE_SCHEMA.validate(df)

    def test_all_feature_columns_required(self):
        for col in FEATURE_COLUMNS:
            record = _base_inference_record()
            record.pop(col)
            df = pd.DataFrame([record])
            with pytest.raises(SchemaError):
                INFERENCE_FEATURE_SCHEMA.validate(df)

    def test_null_in_any_feature_fails(self):
        record = _base_inference_record()
        record["age"] = np.nan
        df = pd.DataFrame([record])
        with pytest.raises(SchemaError):
            INFERENCE_FEATURE_SCHEMA.validate(df)

    def test_coerces_int_to_float(self):
        record = {col: int(v) for col, v in _base_inference_record().items()}
        df = pd.DataFrame([record])
        validated = INFERENCE_FEATURE_SCHEMA.validate(df)
        assert validated["age"].dtype == float

    def test_batch_of_100_valid_records(self):
        records = [_base_inference_record() for _ in range(100)]
        INFERENCE_FEATURE_SCHEMA.validate(pd.DataFrame(records))

    def test_extra_columns_ignored(self):
        record = _base_inference_record()
        record["customer_id"] = "c1"
        record["record_id"] = "r1"
        df = pd.DataFrame([record])
        INFERENCE_FEATURE_SCHEMA.validate(df)
