# Data Dictionary

## Schema: `dwh_raw`

### Table: `raw_loan_applications`

One row per generated loan application. Timestamps are set 1 day in the past to simulate historical ingestion.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `record_id` | UUID | NO | Primary key. Auto-generated UUID. |
| `customer_id` | VARCHAR(36) | NO | Deterministic UUID5, one of 10,000 synthetic customer identifiers, stable across runs. |
| `age` | INT | NO | Applicant age in years. Range: [18, 75]. |
| `annual_income` | FLOAT | NO | Annual income in USD. LogNormal(μ=11, σ=0.5) → ~$60k median. |
| `credit_score` | INT | NO | FICO-style credit score. Range: [300, 850]. Normal(680, 50) baseline. |
| `loan_amount` | FLOAT | NO | Requested loan amount in USD. LogNormal(μ=10, σ=0.6) → ~$22k median. |
| `loan_term_months` | INT | NO | Loan term. Choices: [12, 24, 36, 48, 60, 72, 84]. |
| `employment_length_years` | FLOAT | NO | Years at current employer. Gamma(shape=2, scale=3). |
| `home_ownership` | VARCHAR(20) | NO | One of: RENT, MORTGAGE, OWN, OTHER. |
| `debt_to_income_ratio` | FLOAT | NO | `loan_amount / annual_income`. Derived. Range: [0, 5]. |
| `num_credit_lines` | INT | NO | Number of open credit lines. Poisson(λ=8). Range: [1, 40]. |
| `payment_history_score` | FLOAT | NO | Payment reliability score 0–100. Beta(a=5, b=2)×100 → right-skewed toward 80+. |
| `default_flag` | INT | NO | **Target variable.** 1 = defaulted, 0 = repaid. Derived from logistic formula. |
| `drift_applied` | BOOLEAN | NO | True if generated with drift mode enabled. |
| `drift_factor` | FLOAT | NO | Drift magnitude used (0.0 = stable, 1.0 = max shift). |
| `created_at` | TIMESTAMP | NO | Simulated data timestamp = `NOW() - 1 day`. |
| `ground_truth_available_at` | TIMESTAMP | NO | When default outcome is known = `created_at + 1 day`. |
| `ingested_at` | TIMESTAMP | NO | Actual insert time. |

---

## Schema: `dwh_clean`

### Table: `cleaned_features`

Output of the processing pipeline. References `dwh_raw.raw_loan_applications.record_id`. These are the features used as model inputs.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `record_id` | UUID | NO | FK → dwh_raw.raw_loan_applications.record_id |
| `customer_id` | VARCHAR(36) | NO | Carried from raw. |
| `age` | FLOAT | NO | Imputed with median if null. |
| `annual_income` | FLOAT | NO | Outliers removed (IQR × 3). |
| `credit_score` | FLOAT | NO | Imputed with median if null. |
| `loan_amount` | FLOAT | NO | Outliers removed (IQR × 3). |
| `loan_term_months` | FLOAT | NO | Cast to float for uniform ML pipeline input. |
| `employment_length_years` | FLOAT | NO | Imputed with median if null. |
| `home_ownership_encoded` | INT | NO | RENT=0, MORTGAGE=1, OWN=2, OTHER=3. |
| `debt_to_income_ratio` | FLOAT | NO | Clipped to [0, 5]. |
| `num_credit_lines` | FLOAT | NO | Imputed with median if null. |
| `payment_history_score` | FLOAT | NO | Validated in [0, 100]. |
| `default_flag` | INT | NO | Target variable, carried from raw. |
| `created_at` | TIMESTAMP | NO | Original simulated timestamp from raw. |
| `ground_truth_available_at` | TIMESTAMP | NO | Carried from raw. |
| `processed_at` | TIMESTAMP | NO | When this row was written by pipeline.py. |
| `source_parquet_file` | VARCHAR(255) | YES | Filename of the source parquet if loaded from file. |

---

## Schema: `dwh_history`

### Table: `run_registry`

One row per batch inference run. Provides the SERIAL `run_index` used across all monitoring and history tables.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `run_index` | SERIAL | NO | Primary key. Sequential integer, increments per batch run. |
| `batch_run_id` | VARCHAR(36) | NO | UUID shared by all records in this run. |
| `n_records` | INT | YES | Number of records scored in this run. |
| `drift_factor` | FLOAT | YES | Drift factor used for data generation. |
| `model_version` | VARCHAR(20) | YES | MLflow model version that scored this run. |
| `created_at` | TIMESTAMP | NO | When the run was registered. |

### Table: `prediction_ground_truth`

One row per record per batch run. Combines prediction output with actual labels. This is the primary source for monitoring and evaluation.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `id` | UUID | NO | Primary key. |
| `run_index` | INT | NO | FK → run_registry.run_index. |
| `batch_run_id` | VARCHAR(36) | NO | UUID of the batch run. |
| `record_id` | UUID | YES | FK → dwh_clean.cleaned_features.record_id. |
| `customer_id` | VARCHAR(36) | YES | Deterministic customer UUID5. |
| `age` | FLOAT | YES | Feature at inference time. |
| `annual_income` | FLOAT | YES | Feature at inference time. |
| `credit_score` | FLOAT | YES | Feature at inference time. |
| `loan_amount` | FLOAT | YES | Feature at inference time. |
| `loan_term_months` | FLOAT | YES | Feature at inference time. |
| `employment_length_years` | FLOAT | YES | Feature at inference time. |
| `home_ownership_encoded` | INT | YES | Feature at inference time. |
| `debt_to_income_ratio` | FLOAT | YES | Feature at inference time. |
| `num_credit_lines` | FLOAT | YES | Feature at inference time. |
| `payment_history_score` | FLOAT | YES | Feature at inference time. |
| `default_probability` | FLOAT | YES | Model output: P(default). Range [0, 1]. |
| `default_flag_predicted` | INT | YES | Binary decision at decision_threshold. |
| `decision_threshold` | FLOAT | YES | Threshold used for binary decision. |
| `model_name` | VARCHAR(100) | YES | MLflow registry model name. |
| `model_version` | VARCHAR(20) | YES | MLflow model version number. |
| `actual_default_flag` | INT | YES | Ground truth label. Populated from dwh_clean. NULL until labelled. |
| `predicted_at` | TIMESTAMP | YES | Inference execution time. |

### Table: `retraining_log`

Audit trail of every retraining event triggered by the pipeline.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `id` | UUID | NO | Primary key. |
| `trigger_reason` | TEXT | YES | Why retraining was triggered (e.g. "roc_auc=0.55 < 0.60"). |
| `old_model_version` | VARCHAR(20) | YES | Production version before retraining. |
| `new_model_version` | VARCHAR(20) | YES | New version if promotion succeeded. |
| `promotion_succeeded` | BOOLEAN | YES | True if new model passed the quality gate. |
| `retrained_at` | TIMESTAMP | NO | When retraining ran. |

---

## Schema: `dwh_monitoring_hard`

### Table: `results`

One row per batch run evaluation. Unique constraint on `run_index`, re-running monitoring for the same run_index overwrites via upsert.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `id` | UUID | NO | Primary key. |
| `run_index` | INT | NO | UNIQUE. The batch run being evaluated. |
| `batch_run_id` | VARCHAR(36) | YES | UUID of the batch run. |
| `n_records` | INT | YES | Records with ground truth available at evaluation time. |
| `accuracy` | FLOAT | YES | Overall correct prediction rate. |
| `f1_score` | FLOAT | YES | F1 score (harmonic mean of precision and recall). |
| `precision_score` | FLOAT | YES | Precision (TP / predicted positives). |
| `recall_score` | FLOAT | YES | Recall (TP / actual positives). |
| `roc_auc` | FLOAT | YES | Area under ROC curve. |
| `pr_auc` | FLOAT | YES | Area under precision-recall curve. |
| `primary_metric` | VARCHAR(20) | NO | Which metric was used for threshold check (default: `roc_auc`). |
| `retraining_triggered` | BOOLEAN | NO | True if monitoring triggered dag_05. |
| `retraining_reason` | TEXT | YES | Human-readable explanation of trigger decision. |
| `mlflow_run_id` | VARCHAR(100) | YES | Corresponding MLflow run ID in monitoring_hard experiment. |
| `parquet_path` | TEXT | YES | Path to local parquet file. |
| `evidently_snapshot_id` | VARCHAR(36) | YES | Snapshot id in the Evidently UI's `credit-risk-hard-metrics` project (deep link: `http://localhost:8000/projects/{project_id}/reports/{id}`). |
| `evaluated_at` | TIMESTAMP | NO | When monitoring ran. |

---

## Schema: `dwh_monitoring_drift`

### Table: `results`

One row per batch run drift evaluation. Unique on `run_index`.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `id` | UUID | NO | Primary key. |
| `run_index` | INT | NO | UNIQUE. The batch run being analysed. |
| `batch_run_id` | VARCHAR(36) | YES | UUID of the batch run. |
| `drift_detected` | BOOLEAN | NO | True if drift_score >= MAX_DRIFT_FEATURE_FRACTION (Evidently `DataDriftPreset(drift_share=...)`). |
| `drift_score` | FLOAT | YES | Fraction of features with significant drift [0, 1]. |
| `num_drifted_features` | INT | YES | Count of features that drifted. |
| `drifted_feature_names` | TEXT | YES | Comma-separated list of drifted feature names. |
| `reference_run_id` | VARCHAR(100) | YES | MLflow run ID of the reference model whose artifact was used. |
| `n_records` | INT | YES | Records in the current batch. |
| `parquet_path` | TEXT | YES | Path to local parquet file. |
| `evidently_snapshot_id` | VARCHAR(36) | YES | Snapshot id in the Evidently UI's `credit-risk-data-drift` project (deep link: `http://localhost:8000/projects/{project_id}/reports/{id}`). |
| `evaluated_at` | TIMESTAMP | NO | When drift analysis ran. |

---

## Schema: `dwh_monitoring_shap`

### Table: `results`

One row per batch run. Aggregate SHAP feature importances.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `id` | UUID | NO | Primary key. |
| `run_index` | INT | NO | UNIQUE. The batch run. |
| `batch_run_id` | VARCHAR(36) | YES | UUID of the batch run. |
| `explainer_type` | VARCHAR(50) | YES | `TreeExplainer` or `KernelExplainer`. |
| `top_feature` | VARCHAR(100) | YES | Feature with highest mean |SHAP| value. |
| `feature_importances` | JSONB | YES | `{"feature_name": mean_abs_shap, ...}` for all features. |
| `n_records` | INT | YES | Records SHAP was computed for. |
| `mlflow_run_id` | VARCHAR(100) | YES | Corresponding MLflow run ID in monitoring_shap experiment. |
| `parquet_path` | TEXT | YES | Path to aggregate parquet file. |
| `evaluated_at` | TIMESTAMP | NO | When SHAP computation ran. |

### Table: `customer_shap_values`

One row per record per batch run. Enables the Dashboard waterfall view.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `id` | UUID | NO | Primary key. |
| `run_index` | INT | NO | FK → dwh_monitoring_shap.results.run_index. |
| `customer_id` | VARCHAR(36) | YES | Deterministic UUID5, stable across runs for the same customer. |
| `record_id` | UUID | YES | FK → dwh_clean.cleaned_features.record_id. |
| `shap_values` | JSONB | YES | `{"feature_name": shap_value, ...}` for this record. |
| `base_value` | FLOAT | YES | Model's expected output (log-odds baseline). |
| `predicted_probability` | FLOAT | YES | Model's actual output P(default) for this record. |
| `computed_at` | TIMESTAMP | NO | When SHAP was computed. |

---

## Schema: `dwh_challenger`

### Table: `comparison_log`

One row per challenger comparison run.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `id` | UUID | NO | Primary key. |
| `challenger_run_id` | TEXT | NO | MLflow run ID of the challenger model. |
| `primary_metric` | VARCHAR(20) | NO | Metric used for comparison (from challenger's MLflow tag). |
| `prod_roc_auc` | FLOAT | YES | Production model ROC-AUC on eval data (informational). |
| `challenger_roc_auc` | FLOAT | YES | Challenger model ROC-AUC on eval data (informational). |
| `prod_primary_score` | FLOAT | YES | Production mean CV score on primary metric. |
| `challenger_primary_score` | FLOAT | YES | Challenger mean CV score on primary metric. |
| `prod_cv_std` | FLOAT | YES | Production CV standard deviation. |
| `challenger_cv_std` | FLOAT | YES | Challenger CV standard deviation. |
| `cv_folds` | INT | YES | Number of CV folds used. |
| `cv_margin` | FLOAT | YES | Required improvement margin (absolute). |
| `challenger_wins` | BOOLEAN | YES | True if challenger beat production by at least cv_margin. |
| `force_deploy` | BOOLEAN | NO | True if comparison was bypassed via force_deploy flag. |
| `promoted` | BOOLEAN | NO | True if challenger was promoted to production. |
| `eval_records` | INT | YES | Number of labelled records used for CV. |
| `compared_at` | TIMESTAMPTZ | NO | When comparison ran. |

---

## Schema: `dwh_predictions` (Legacy)

### Table: `batch_predictions`

Per-record prediction output. Written in parallel with `dwh_history.prediction_ground_truth`. Kept for backwards compatibility. Prefer `dwh_history.prediction_ground_truth` for analysis.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `prediction_id` | UUID | NO | Primary key. |
| `record_id` | UUID | YES | FK → dwh_clean.cleaned_features.record_id. |
| `customer_id` | VARCHAR(36) | NO | Customer identifier. |
| `default_probability` | FLOAT | NO | Model output: P(default). |
| `default_flag_predicted` | INT | NO | Binary decision at decision_threshold. |
| `decision_threshold` | FLOAT | NO | Threshold used (from DECISION_THRESHOLD env var). |
| `model_name` | VARCHAR(100) | NO | MLflow registry model name. |
| `model_version` | VARCHAR(20) | NO | MLflow model version number. |
| `batch_run_id` | VARCHAR(100) | NO | UUID shared by all rows in the same inference run. |
| `predicted_at` | TIMESTAMP | NO | Inference execution time. |

---

## Feature Engineering Notes

### `debt_to_income_ratio`
Derived as `loan_amount / annual_income`. High DTI (> 0.5) is the strongest predictor of default in the simulation. Clipped to [0, 5].

### `home_ownership_encoded`
Ordinal encoding: RENT=0, MORTGAGE=1, OWN=2, OTHER=3. Reflects housing stability as a crude ordinal, the model learns the true relationship from data.

### `payment_history_score`
Scored 0–100. Generated from Beta(5, 2) × 100, so most applicants cluster around 70–90. Lower values strongly predict default.

### `default_flag` (target)
Generated from a logistic function on: `credit_score` (−), `annual_income` (−), `loan_amount` (+), `debt_to_income_ratio` (+), `payment_history_score` (−), `employment_length_years` (−). Random noise added. Expected default rate ~25–30% in stable mode (class imbalance ~1:2.6 non-default to default).

### `customer_id`
`uuid.uuid5(NAMESPACE, pool_key)` where `pool_key` is drawn from a fixed pool of 10,000 synthetic IDs. This makes customer_id deterministic and stable, the same logical customer appears in multiple batch runs, enabling longitudinal SHAP analysis.
