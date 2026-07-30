"""
DAG 04a - Hard Metrics Monitoring
====================================
Computes classification metrics (accuracy, F1, precision, recall, ROC-AUC)
for a specific inference run. Auto-cascades to retraining if ROC-AUC < threshold.

Triggered automatically by dag_03_batch_inference after each inference run.
Also triggered by dag_05_retraining after a retrain (with skip_retrain_trigger=True).
Can be triggered manually from Airflow UI.

Conf / Params:
  run_index:            int or null - if null, uses latest run in DB
  skip_retrain_trigger: bool - if True, never trigger dag_05 (post-retrain check)

Tasks:
  run_hard_metrics  → hard_metrics.py
  parse_result      → extract retraining_triggered from JSON output
  decide_retrain    → BranchPythonOperator
  trigger_retraining → TriggerDagRunOperator → dag_05_retraining
  no_retraining     → EmptyOperator (end)
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

MONITORING_PATH = "/opt/mlops/services/monitoring"

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
    "DATA_MONITORING_HARD_PATH": "/data/monitoring/hard",
    "MIN_ROC_AUC": os.getenv("MIN_ROC_AUC", "0.70"),
    "EVIDENTLY_WORKSPACE_PATH": os.getenv("EVIDENTLY_WORKSPACE_PATH", "/data/monitoring/evidently_workspace"),
}


def _build_metrics_command(**context):
    """Build the hard_metrics.py command with optional --run-index."""
    conf = context["dag_run"].conf or {}
    run_index = conf.get("run_index")

    if run_index is not None:
        return (
            f"cd {MONITORING_PATH} && python hard_metrics.py --run-index {int(run_index)}"
        )
    return f"cd {MONITORING_PATH} && python hard_metrics.py"


def _parse_hard_metrics(ti, **context):
    import json

    raw = ti.xcom_pull(task_ids="run_hard_metrics")
    if not raw:
        print("No output from hard_metrics.py - no retraining triggered")
        ti.xcom_push(key="retraining_triggered", value=False)
        ti.xcom_push(key="reason", value="no_output")
        return

    lines = raw.strip().splitlines()
    result = json.loads(lines[-1])

    ti.xcom_push(key="retraining_triggered", value=result.get("retraining_triggered", False))
    ti.xcom_push(key="reason", value=result.get("reason", ""))
    ti.xcom_push(key="roc_auc", value=result.get("roc_auc", 0.0))
    ti.xcom_push(key="run_index", value=result.get("run_index"))

    print(f"Hard metrics: ROC-AUC={result.get('roc_auc')} "
          f"retrain={result.get('retraining_triggered')} reason={result.get('reason')}")


def _decide_retrain(ti, **context):
    conf = context["dag_run"].conf or {}
    skip = conf.get("skip_retrain_trigger", False)

    if skip:
        print("skip_retrain_trigger=True - skipping retraining cascade")
        return "no_retraining_needed"

    retrain = ti.xcom_pull(task_ids="parse_result", key="retraining_triggered")
    if retrain:
        return "trigger_retraining"
    return "no_retraining_needed"


with DAG(
    dag_id="dag_04a_monitor_hard",
    description="Hard metrics monitoring (ROC, F1, …); auto-triggers retraining if below threshold",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "monitoring"],
) as dag:

    run_metrics = BashOperator(
        task_id="run_hard_metrics",
        bash_command=(
            "cd /opt/mlops/services/monitoring && python hard_metrics.py"
            "{% set conf = dag_run.conf or {} %}"
            "{% set ri = conf.get('run_index') %}"
            "{% if ri is not none %} --run-index {{ ri | int }}{% endif %}"
        ),
        env={**DB_ENV},
        do_xcom_push=True,
        doc_md="Computes accuracy, F1, precision, recall, ROC-AUC from dwh_history.",
    )

    parse_result = PythonOperator(
        task_id="parse_result",
        python_callable=_parse_hard_metrics,
    )

    decide = BranchPythonOperator(
        task_id="decide_retrain",
        python_callable=_decide_retrain,
    )

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="dag_05_retraining",
        conf={
            "triggered_by": "dag_04a_monitor_hard",
            "roc_auc": "{{ ti.xcom_pull(task_ids='parse_result', key='roc_auc') }}",
            "reason": "{{ ti.xcom_pull(task_ids='parse_result', key='reason') }}",
        },
        doc_md="Triggers retraining DAG when ROC-AUC is below threshold.",
    )

    no_retrain = EmptyOperator(
        task_id="no_retraining_needed",
        doc_md="Metrics within thresholds or skip flag set - no retraining.",
    )

    run_metrics >> parse_result >> decide >> [trigger_retraining, no_retrain]
