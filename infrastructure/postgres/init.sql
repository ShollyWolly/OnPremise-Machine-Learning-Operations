-- =============================================================================
-- MLOps Platform - PostgreSQL Initialization Script
-- Creates database, schemas, and tables for all platform components.
-- =============================================================================

-- Create the main database (run as superuser; skip if already exists)
-- This file is executed by the postgres Docker image on first start via
-- the POSTGRES_DB env var creating the DB, so we only need schemas here.

-- ---------------------------------------------------------------------------
-- SCHEMA: mlflow
-- Managed entirely by the MLflow tracking server. Tables created automatically
-- by MLflow on first startup. We only create the schema here.
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS mlflow;

-- ---------------------------------------------------------------------------
-- SCHEMA: dwh_raw
-- Raw data as produced by the data generator. One row per loan application.
-- Timestamps are set 1 day in the past to simulate historical ingestion.
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS dwh_raw;

CREATE TABLE IF NOT EXISTS dwh_raw.raw_loan_applications (
    record_id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id             VARCHAR(36)     NOT NULL,
    age                     INT             NOT NULL CHECK (age BETWEEN 18 AND 75),
    annual_income           FLOAT           NOT NULL CHECK (annual_income > 0),
    credit_score            INT             NOT NULL CHECK (credit_score BETWEEN 300 AND 850),
    loan_amount             FLOAT           NOT NULL CHECK (loan_amount > 0),
    loan_term_months        INT             NOT NULL CHECK (loan_term_months > 0),
    employment_length_years FLOAT           NOT NULL CHECK (employment_length_years >= 0),
    home_ownership          VARCHAR(20)     NOT NULL,
    debt_to_income_ratio    FLOAT           NOT NULL,
    num_credit_lines        INT             NOT NULL CHECK (num_credit_lines >= 0),
    payment_history_score   FLOAT           NOT NULL CHECK (payment_history_score BETWEEN 0 AND 100),
    -- Target variable: 1 = defaulted, 0 = repaid
    default_flag            INT             NOT NULL CHECK (default_flag IN (0, 1)),
    -- Metadata about generation
    drift_applied           BOOLEAN         NOT NULL DEFAULT FALSE,
    drift_factor            FLOAT           NOT NULL DEFAULT 0.0,
    -- Timestamps
    -- created_at is set 1 day in the past to simulate historical data arrival
    created_at              TIMESTAMP       NOT NULL,
    -- ground truth becomes available 1 day after creation (simulates loan outcome delay)
    ground_truth_available_at TIMESTAMP     NOT NULL,
    ingested_at             TIMESTAMP       NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_raw_created_at
    ON dwh_raw.raw_loan_applications (created_at);

CREATE INDEX IF NOT EXISTS idx_raw_ground_truth_available_at
    ON dwh_raw.raw_loan_applications (ground_truth_available_at);

-- ---------------------------------------------------------------------------
-- SCHEMA: dwh_clean
-- Cleaned and feature-engineered data ready for model training and inference.
-- Home ownership is ordinally encoded; outliers removed; nulls imputed.
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS dwh_clean;

CREATE TABLE IF NOT EXISTS dwh_clean.cleaned_features (
    record_id               UUID PRIMARY KEY,
    customer_id             VARCHAR(36)     NOT NULL,
    age                     FLOAT           NOT NULL,
    annual_income           FLOAT           NOT NULL,
    credit_score            FLOAT           NOT NULL,
    loan_amount             FLOAT           NOT NULL,
    loan_term_months        FLOAT           NOT NULL,
    employment_length_years FLOAT           NOT NULL,
    -- Ordinally encoded: RENT=0, MORTGAGE=1, OWN=2, OTHER=3
    home_ownership_encoded  INT             NOT NULL,
    debt_to_income_ratio    FLOAT           NOT NULL,
    num_credit_lines        FLOAT           NOT NULL,
    payment_history_score   FLOAT           NOT NULL,
    default_flag            INT             NOT NULL,
    created_at              TIMESTAMP       NOT NULL,
    ground_truth_available_at TIMESTAMP     NOT NULL,
    processed_at            TIMESTAMP       NOT NULL DEFAULT NOW(),
    -- Source tracking
    source_parquet_file     VARCHAR(255),
    FOREIGN KEY (record_id) REFERENCES dwh_raw.raw_loan_applications(record_id)
);

CREATE INDEX IF NOT EXISTS idx_clean_created_at
    ON dwh_clean.cleaned_features (created_at);

-- ---------------------------------------------------------------------------
-- SCHEMA: dwh_predictions
-- Batch inference results. Each row is a model prediction for one application.
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS dwh_predictions;

CREATE TABLE IF NOT EXISTS dwh_predictions.batch_predictions (
    prediction_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    record_id               UUID,
    customer_id             VARCHAR(36)     NOT NULL,
    default_probability     FLOAT           NOT NULL CHECK (default_probability BETWEEN 0 AND 1),
    default_flag_predicted  INT             NOT NULL CHECK (default_flag_predicted IN (0, 1)),
    -- Threshold used for binary decision (configurable)
    decision_threshold      FLOAT           NOT NULL DEFAULT 0.5,
    model_name              VARCHAR(100)    NOT NULL,
    model_version           VARCHAR(20)     NOT NULL,
    batch_run_id            VARCHAR(100)    NOT NULL,
    predicted_at            TIMESTAMP       NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pred_batch_run_id
    ON dwh_predictions.batch_predictions (batch_run_id);

CREATE INDEX IF NOT EXISTS idx_pred_customer_id
    ON dwh_predictions.batch_predictions (customer_id);

-- ---------------------------------------------------------------------------
-- SCHEMA: dwh_monitoring
-- One row per monitoring pipeline run. Stores hard metrics, drift scores,
-- and flags whether retraining was triggered.
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS dwh_monitoring;

CREATE TABLE IF NOT EXISTS dwh_monitoring.monitoring_results (
    run_id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_date              DATE            NOT NULL,
    batch_run_id            VARCHAR(100),
    -- Hard metrics vs ground truth
    num_records_evaluated   INT,
    accuracy                FLOAT,
    f1_score                FLOAT,
    precision_score         FLOAT,
    recall_score            FLOAT,
    roc_auc                 FLOAT,
    -- Data drift
    drift_score             FLOAT,
    drift_detected          BOOLEAN,
    num_drifted_features    INT,
    drifted_feature_names   TEXT,           -- comma-separated list
    -- Explainability
    shap_logged             BOOLEAN         NOT NULL DEFAULT FALSE,
    mlflow_run_id           VARCHAR(100),
    -- Retraining decision
    retraining_triggered    BOOLEAN         NOT NULL DEFAULT FALSE,
    retraining_reason       TEXT,
    monitored_at            TIMESTAMP       NOT NULL DEFAULT NOW()
);

-- Track retraining events separately for audit trail
CREATE TABLE IF NOT EXISTS dwh_monitoring.retraining_log (
    log_id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    monitoring_run_id       UUID REFERENCES dwh_monitoring.monitoring_results(run_id),
    trigger_reason          TEXT            NOT NULL,
    old_model_version       VARCHAR(20),
    new_model_version       VARCHAR(20),
    new_model_roc_auc       FLOAT,
    promotion_succeeded     BOOLEAN,
    retrained_at            TIMESTAMP       NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- SCHEMA: dwh_history
-- Sequential inference run registry + combined prediction/ground-truth history.
-- run_registry uses SERIAL for sequential, index-based demo runs.
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS dwh_history;

CREATE TABLE IF NOT EXISTS dwh_history.run_registry (
    run_index               SERIAL PRIMARY KEY,
    batch_run_id            VARCHAR(36),
    model_name              VARCHAR(100),
    model_version           VARCHAR(20),
    n_records               INT,
    n_records_requested     INT,
    drift_factor            FLOAT           NOT NULL DEFAULT 0.0,
    status                  VARCHAR(20)     NOT NULL DEFAULT 'pending',
    created_at              TIMESTAMP       NOT NULL DEFAULT NOW(),
    completed_at            TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dwh_history.prediction_ground_truth (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_index               INT             NOT NULL REFERENCES dwh_history.run_registry(run_index),
    batch_run_id            VARCHAR(36)     NOT NULL,
    record_id               UUID,
    customer_id             VARCHAR(36),
    -- Features
    age                     FLOAT,
    annual_income           FLOAT,
    credit_score            FLOAT,
    loan_amount             FLOAT,
    loan_term_months        FLOAT,
    employment_length_years FLOAT,
    home_ownership_encoded  INT,
    debt_to_income_ratio    FLOAT,
    num_credit_lines        FLOAT,
    payment_history_score   FLOAT,
    -- Prediction output
    default_probability     FLOAT,
    default_flag_predicted  INT,
    decision_threshold      FLOAT,
    model_name              VARCHAR(100),
    model_version           VARCHAR(20),
    -- Ground truth
    actual_default_flag     INT,
    predicted_at            TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pgt_run_index
    ON dwh_history.prediction_ground_truth (run_index);

CREATE INDEX IF NOT EXISTS idx_pgt_batch_run_id
    ON dwh_history.prediction_ground_truth (batch_run_id);

-- Audit trail for retraining events
CREATE TABLE IF NOT EXISTS dwh_history.retraining_log (
    log_id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trigger_reason          TEXT            NOT NULL,
    trigger_roc_auc         FLOAT,
    new_model_version       VARCHAR(20),
    new_model_roc_auc       FLOAT,
    promotion_succeeded     BOOLEAN,
    retrained_at            TIMESTAMP       NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- SCHEMA: dwh_monitoring_hard
-- Hard classification metrics per inference run.
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS dwh_monitoring_hard;

CREATE TABLE IF NOT EXISTS dwh_monitoring_hard.results (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_index               INT             NOT NULL UNIQUE,
    batch_run_id            VARCHAR(36),
    n_records               INT,
    accuracy                FLOAT,
    f1_score                FLOAT,
    precision_score         FLOAT,
    recall_score            FLOAT,
    roc_auc                 FLOAT,
    pr_auc                  FLOAT,
    primary_metric          VARCHAR(20)     NOT NULL DEFAULT 'roc_auc',
    retraining_triggered    BOOLEAN         NOT NULL DEFAULT FALSE,
    retraining_reason       TEXT,
    mlflow_run_id           VARCHAR(100),
    parquet_path            TEXT,
    evaluated_at            TIMESTAMP       NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hard_run_index
    ON dwh_monitoring_hard.results (run_index);

-- ---------------------------------------------------------------------------
-- SCHEMA: dwh_monitoring_drift
-- Evidently covariate drift results per inference run.
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS dwh_monitoring_drift;

CREATE TABLE IF NOT EXISTS dwh_monitoring_drift.results (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_index               INT             NOT NULL UNIQUE,
    batch_run_id            VARCHAR(36),
    drift_detected          BOOLEAN         NOT NULL DEFAULT FALSE,
    drift_score             FLOAT,
    num_drifted_features    INT,
    drifted_feature_names   TEXT,
    reference_run_id        VARCHAR(100),
    n_records               INT,
    parquet_path            TEXT,
    evaluated_at            TIMESTAMP       NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drift_run_index
    ON dwh_monitoring_drift.results (run_index);

-- ---------------------------------------------------------------------------
-- SCHEMA: dwh_monitoring_shap
-- SHAP feature importance results per inference run.
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS dwh_monitoring_shap;

CREATE TABLE IF NOT EXISTS dwh_monitoring_shap.results (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_index               INT             NOT NULL UNIQUE,
    batch_run_id            VARCHAR(36),
    explainer_type          VARCHAR(50),
    top_feature             VARCHAR(100),
    feature_importances     JSONB,
    n_records               INT,
    mlflow_run_id           VARCHAR(100),
    parquet_path            TEXT,
    evaluated_at            TIMESTAMP       NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_shap_run_index
    ON dwh_monitoring_shap.results (run_index);

CREATE TABLE IF NOT EXISTS dwh_monitoring_shap.customer_shap_values (
    id                      SERIAL PRIMARY KEY,
    run_index               INT             NOT NULL,
    batch_run_id            VARCHAR(36),
    customer_id             VARCHAR(36)     NOT NULL,
    record_id               UUID,
    shap_values             JSONB           NOT NULL,
    base_value              FLOAT,
    predicted_probability   FLOAT,
    computed_at             TIMESTAMP       NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cshap_customer_id
    ON dwh_monitoring_shap.customer_shap_values (customer_id);

CREATE INDEX IF NOT EXISTS idx_cshap_run_index
    ON dwh_monitoring_shap.customer_shap_values (run_index);

-- ---------------------------------------------------------------------------
-- GRANT permissions so the app user can access all schemas
-- mlflow schema: mlops_user needs CREATE so MLflow can create its own tables
-- ---------------------------------------------------------------------------
GRANT ALL PRIVILEGES ON SCHEMA mlflow             TO mlops_user;
GRANT CREATE ON SCHEMA public                      TO mlops_user;
GRANT ALL PRIVILEGES ON SCHEMA dwh_raw             TO mlops_user;
GRANT ALL PRIVILEGES ON SCHEMA dwh_clean           TO mlops_user;
GRANT ALL PRIVILEGES ON SCHEMA dwh_predictions     TO mlops_user;
GRANT ALL PRIVILEGES ON SCHEMA dwh_monitoring      TO mlops_user;
GRANT ALL PRIVILEGES ON SCHEMA dwh_history         TO mlops_user;
GRANT ALL PRIVILEGES ON SCHEMA dwh_monitoring_hard  TO mlops_user;
GRANT ALL PRIVILEGES ON SCHEMA dwh_monitoring_drift TO mlops_user;
GRANT ALL PRIVILEGES ON SCHEMA dwh_monitoring_shap  TO mlops_user;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA dwh_raw              TO mlops_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA dwh_clean            TO mlops_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA dwh_predictions      TO mlops_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA dwh_monitoring       TO mlops_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA dwh_history          TO mlops_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA dwh_monitoring_hard  TO mlops_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA dwh_monitoring_drift TO mlops_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA dwh_monitoring_shap  TO mlops_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA dwh_raw
    GRANT ALL ON TABLES TO mlops_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA dwh_clean
    GRANT ALL ON TABLES TO mlops_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA dwh_predictions
    GRANT ALL ON TABLES TO mlops_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA dwh_monitoring
    GRANT ALL ON TABLES TO mlops_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA dwh_history
    GRANT ALL ON TABLES TO mlops_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA dwh_monitoring_hard
    GRANT ALL ON TABLES TO mlops_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA dwh_monitoring_drift
    GRANT ALL ON TABLES TO mlops_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA dwh_monitoring_shap
    GRANT ALL ON TABLES TO mlops_user;

-- Grant sequence usage for SERIAL columns
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA dwh_history TO mlops_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA dwh_monitoring_shap TO mlops_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA dwh_history
    GRANT USAGE, SELECT ON SEQUENCES TO mlops_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA dwh_monitoring_shap
    GRANT USAGE, SELECT ON SEQUENCES TO mlops_user;

-- ---------------------------------------------------------------------------
-- SCHEMA: dwh_challenger
-- Tracks challenger model comparison results.
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS dwh_challenger;

CREATE TABLE IF NOT EXISTS dwh_challenger.comparison_log (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    challenger_run_id        TEXT            NOT NULL,
    primary_metric           VARCHAR(20)     NOT NULL DEFAULT 'roc_auc',
    -- CV averages (mean across folds)
    prod_roc_auc             FLOAT,
    challenger_roc_auc       FLOAT,
    prod_primary_score       FLOAT,
    challenger_primary_score FLOAT,
    -- CV statistics
    prod_cv_std              FLOAT,
    challenger_cv_std        FLOAT,
    cv_folds                 INT,
    cv_margin                FLOAT,
    -- Decision
    challenger_wins          BOOLEAN,
    force_deploy             BOOLEAN         NOT NULL DEFAULT FALSE,
    promoted                 BOOLEAN         NOT NULL DEFAULT FALSE,
    eval_records             INT,
    compared_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

GRANT ALL PRIVILEGES ON SCHEMA dwh_challenger TO mlops_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA dwh_challenger TO mlops_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA dwh_challenger
    GRANT ALL ON TABLES TO mlops_user;
