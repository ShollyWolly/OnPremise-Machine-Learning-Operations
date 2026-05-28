"""
DAG 04c — SHAP Explainability Monitoring
==========================================
Computes SHAP feature importances for the current batch using the Production
model. Logs summary plots to MLflow. Writes per-feature importances to parquet
and dwh_monitoring_shap.results.

Manually triggered from Airflow UI.

Conf:
  run_index: int or null — if null, uses latest run in dwh_history

Tasks:
  run_shap_explainability → shap_explainability.py
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
    "DATA_MONITORING_SHAP_PATH": "/data/monitoring/shap",
}

with DAG(
    dag_id="dag_04c_monitor_shap",
    description="SHAP explainability: feature importance plots logged to MLflow",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "monitoring"],
) as dag:

    run_shap = BashOperator(
        task_id="run_shap_explainability",
        bash_command=(
            "cd /opt/mlops/services/monitoring && python shap_explainability.py"
            "{% set conf = dag_run.conf or {} %}"
            "{% set ri = conf.get('run_index') %}"
            "{% if ri is not none %} --run-index {{ ri | int }}{% endif %}"
        ),
        env={**DB_ENV},
        do_xcom_push=True,
        doc_md=(
            "Loads Production model from MLflow registry, computes SHAP values "
            "(TreeExplainer or KernelExplainer fallback), logs bar chart + "
            "summary dot plot as MLflow artifacts."
        ),
    )
