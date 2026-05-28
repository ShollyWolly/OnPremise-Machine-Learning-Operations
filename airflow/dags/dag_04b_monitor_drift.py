"""
DAG 04b — Data Drift Monitoring
=================================
Computes covariate drift between the current batch and the Production model's
reference training dataset (stored as MLflow artifact reference_data.parquet).

Manually triggered from Airflow UI whenever drift analysis is needed.

Conf:
  run_index: int or null — if null, uses latest run in dwh_history

Tasks:
  run_data_drift → data_drift.py
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "email_on_failure": False,
}

DB_ENV = {
    "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "postgres"),
    "POSTGRES_PORT": os.getenv("POSTGRES_PORT", "5432"),
    "POSTGRES_DB": os.getenv("POSTGRES_DB", "mlops"),
    "POSTGRES_USER": os.getenv("POSTGRES_USER", "mlops_user"),
    "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    "MODEL_REGISTRY_NAME": os.getenv("MODEL_REGISTRY_NAME", "credit-risk-classifier"),
    "DATA_MONITORING_DRIFT_PATH": "/data/monitoring/drift",
}

with DAG(
    dag_id="dag_04b_monitor_drift",
    description="Data drift analysis using Evidently; compares batch features to training reference",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "monitoring"],
) as dag:

    run_drift = BashOperator(
        task_id="run_data_drift",
        bash_command=(
            "cd /opt/mlops/services/monitoring && python data_drift.py"
            "{% set conf = dag_run.conf or {} %}"
            "{% set ri = conf.get('run_index') %}"
            "{% if ri is not none %} --run-index {{ ri | int }}{% endif %}"
        ),
        env={**DB_ENV},
        do_xcom_push=True,
        doc_md=(
            "Runs Evidently DataDriftPreset comparing current batch features "
            "to the Production model's reference dataset. "
            "Writes results to dwh_monitoring_drift.results."
        ),
    )
