"""
DAG 04b - Data Drift Monitoring
=================================
Computes covariate drift between the current batch and the Production model's
reference training dataset (stored as MLflow artifact reference_data.parquet).
Auto-cascades to retraining if the drift share exceeds MAX_DRIFT_FEATURE_FRACTION.

Manually triggered from Airflow UI whenever drift analysis is needed.
Also triggered by dag_05_retraining after a retrain (with skip_retrain_trigger=True).

Conf / Params:
  run_index:            int or null - if null, uses latest run in dwh_history
  skip_retrain_trigger: bool - if True, never trigger dag_05 (post-retrain check)

Tasks:
  run_data_drift     → data_drift.py
  parse_result       → extract drift_detected from JSON output
  decide_retrain     → BranchPythonOperator
  trigger_retraining → TriggerDagRunOperator → dag_05_retraining
  no_retraining      → EmptyOperator (end)
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

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
    "MAX_DRIFT_FEATURE_FRACTION": os.getenv("MAX_DRIFT_FEATURE_FRACTION", "0.30"),
    "EVIDENTLY_WORKSPACE_PATH": os.getenv("EVIDENTLY_WORKSPACE_PATH", "/data/monitoring/evidently_workspace"),
}


def _parse_drift_result(ti, **context):
    import json

    raw = ti.xcom_pull(task_ids="run_data_drift")
    if not raw:
        print("No output from data_drift.py - no retraining triggered")
        ti.xcom_push(key="drift_detected", value=False)
        ti.xcom_push(key="reason", value="no_output")
        return

    lines = raw.strip().splitlines()
    result = json.loads(lines[-1])

    drift_detected = result.get("drift_detected", False)
    drift_score = result.get("drift_score", 0.0)
    reason = f"drift_score={drift_score:.4f} drifted_features={result.get('drifted_feature_names', [])}"

    ti.xcom_push(key="drift_detected", value=drift_detected)
    ti.xcom_push(key="reason", value=reason)
    ti.xcom_push(key="drift_score", value=drift_score)
    ti.xcom_push(key="run_index", value=result.get("run_index"))

    print(f"Data drift: score={drift_score} detected={drift_detected} reason={reason}")


def _decide_retrain(ti, **context):
    conf = context["dag_run"].conf or {}
    skip = conf.get("skip_retrain_trigger", False)

    if skip:
        print("skip_retrain_trigger=True - skipping retraining cascade")
        return "no_retraining_needed"

    drift_detected = ti.xcom_pull(task_ids="parse_result", key="drift_detected")
    if drift_detected:
        return "trigger_retraining"
    return "no_retraining_needed"


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

    parse_result = PythonOperator(
        task_id="parse_result",
        python_callable=_parse_drift_result,
    )

    decide = BranchPythonOperator(
        task_id="decide_retrain",
        python_callable=_decide_retrain,
    )

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="dag_05_retraining",
        conf={
            "triggered_by": "dag_04b_monitor_drift",
            "drift_score": "{{ ti.xcom_pull(task_ids='parse_result', key='drift_score') }}",
            "reason": "{{ ti.xcom_pull(task_ids='parse_result', key='reason') }}",
        },
        doc_md="Triggers retraining DAG when drift share is at or above MAX_DRIFT_FEATURE_FRACTION.",
    )

    no_retrain = EmptyOperator(
        task_id="no_retraining_needed",
        doc_md="No drift detected or skip flag set - no retraining.",
    )

    run_drift >> parse_result >> decide >> [trigger_retraining, no_retrain]
