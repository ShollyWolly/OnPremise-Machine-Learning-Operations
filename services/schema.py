"""Pandera schema definitions for the credit-risk MLOps platform.

Three validation gates in the data pipeline:
  RAW_SCHEMA              — records from generator / dwh_raw, before cleaning
  CLEAN_SCHEMA            — processed records entering dwh_clean and training
  INFERENCE_FEATURE_SCHEMA — feature vectors submitted to the /predict endpoint
"""

import os
import sys

_SERVICES_ROOT = os.path.dirname(os.path.abspath(__file__))
if _SERVICES_ROOT not in sys.path:
    sys.path.insert(0, _SERVICES_ROOT)

import pandera.pandas as pa
from pandera.pandas import Check, Column

from platform_config import FEATURE_COLUMNS

RAW_SCHEMA = pa.DataFrameSchema(
    {
        "record_id": Column(str, nullable=False),
        "customer_id": Column(str, nullable=False),
        "age": Column(float, [Check.ge(0), Check.le(120)], nullable=True, coerce=True),
        "annual_income": Column(float, Check.gt(0), nullable=True, coerce=True),
        "credit_score": Column(float, [Check.ge(300), Check.le(900)], nullable=True, coerce=True),
        "loan_amount": Column(float, Check.gt(0), nullable=True, coerce=True),
        "loan_term_months": Column(float, Check.gt(0), nullable=True, coerce=True),
        "employment_length_years": Column(float, Check.ge(0), nullable=True, coerce=True),
        "debt_to_income_ratio": Column(float, Check.ge(0), nullable=True, coerce=True),
        "num_credit_lines": Column(float, Check.ge(0), nullable=True, coerce=True),
        "payment_history_score": Column(
            float, [Check.ge(0), Check.le(100)], nullable=True, coerce=True
        ),
        "default_flag": Column(float, Check.isin([0.0, 1.0]), nullable=True, coerce=True),
    },
    coerce=True,
    strict=False,
    name="raw_loan_applications",
)

CLEAN_SCHEMA = pa.DataFrameSchema(
    {
        "record_id": Column(str, nullable=False),
        "customer_id": Column(str, nullable=False),
        "age": Column(float, [Check.ge(0), Check.le(120)], nullable=False, coerce=True),
        "annual_income": Column(float, Check.gt(0), nullable=False, coerce=True),
        "credit_score": Column(float, [Check.ge(300), Check.le(900)], nullable=False, coerce=True),
        "loan_amount": Column(float, Check.gt(0), nullable=False, coerce=True),
        "loan_term_months": Column(float, Check.gt(0), nullable=False, coerce=True),
        "employment_length_years": Column(float, Check.ge(0), nullable=False, coerce=True),
        "home_ownership_encoded": Column(
            int, Check.isin([0, 1, 2, 3]), nullable=False, coerce=True
        ),
        "debt_to_income_ratio": Column(float, Check.ge(0), nullable=False, coerce=True),
        "num_credit_lines": Column(float, Check.ge(0), nullable=False, coerce=True),
        "payment_history_score": Column(
            float, [Check.ge(0), Check.le(100)], nullable=False, coerce=True
        ),
        "default_flag": Column(float, Check.isin([0.0, 1.0]), nullable=True, coerce=True),
    },
    coerce=True,
    strict=False,
    name="cleaned_features",
)

INFERENCE_FEATURE_SCHEMA = pa.DataFrameSchema(
    {col: Column(float, nullable=False, coerce=True) for col in FEATURE_COLUMNS},
    coerce=True,
    strict=False,
    name="inference_features",
)
