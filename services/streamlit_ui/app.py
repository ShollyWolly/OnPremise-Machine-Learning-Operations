"""
MLOps Platform - Control Panel
================================

Tabs:
  Deploy Model    - promote MLflow model version + reload Flask
  Batch Inference - full self-contained pipeline via dag_03_batch_inference
  Monitor         - trigger dag_04a/04b/04c for any run index
  DAG Logs        - view recent DAG run status and task logs
"""

import json
import os
import subprocess
import time

import mlflow
import numpy as np
import pandas as pd
import psycopg2
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FLASK_ENDPOINT = os.getenv("FLASK_ENDPOINT", "http://flask-api:5001")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_REGISTRY_NAME", "credit-risk-classifier")

POSTGRES_CONN = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname": os.getenv("POSTGRES_DB", "mlops"),
    "user": os.getenv("POSTGRES_USER", "mlops_user"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

GENERATOR_PATH = "/opt/mlops/services/data_generator"
PROCESSING_PATH = "/opt/mlops/services/processing_pipeline"
INFERENCE_PATH = "/opt/mlops/services/batch_inference"
MONITORING_PATH = "/opt/mlops/services/monitoring"

AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://airflow-webserver:8080/api/v1")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")

FEATURE_COLUMNS = [
    "age", "annual_income", "credit_score", "loan_amount",
    "loan_term_months", "employment_length_years", "home_ownership_encoded",
    "debt_to_income_ratio", "num_credit_lines", "payment_history_score",
]

ENV = {**os.environ}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def db_conn():
    return psycopg2.connect(**POSTGRES_CONN)


def run_script(cmd: str, cwd: str = None, placeholder=None) -> tuple:
    """Run shell command, stream stdout/stderr to a Streamlit placeholder."""
    lines = []
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
        env=ENV,
    )
    for line in process.stdout:
        lines.append(line.rstrip())
        if placeholder:
            placeholder.code("\n".join(lines[-60:]))
    process.wait()
    return process.returncode, "\n".join(lines)


def get_flask_info():
    try:
        r = requests.get(f"{FLASK_ENDPOINT}/model-info", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def get_mlflow_versions():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        return sorted(versions, key=lambda v: int(v.version), reverse=True)
    except Exception:
        return []


def get_recent_runs(limit: int = 10) -> pd.DataFrame:
    try:
        with db_conn() as conn:
            return pd.read_sql(
                """
                SELECT run_index, status, drift_factor, n_records_requested AS n_records,
                       model_version, created_at, completed_at
                FROM dwh_history.run_registry
                ORDER BY run_index DESC LIMIT %s
                """,
                conn,
                params=(limit,),
            )
    except Exception:
        return pd.DataFrame()


def get_available_run_indices() -> list:
    try:
        with db_conn() as conn:
            df = pd.read_sql(
                "SELECT DISTINCT run_index FROM dwh_history.prediction_ground_truth ORDER BY run_index DESC LIMIT 30",
                conn,
            )
            return df["run_index"].tolist()
    except Exception:
        return []


def register_run(drift_factor: float, n_records: int) -> int:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dwh_history.run_registry
                    (drift_factor, n_records_requested, status)
                VALUES (%s, %s, 'running')
                RETURNING run_index
                """,
                (drift_factor, n_records),
            )
            run_index = cur.fetchone()[0]
        conn.commit()
    return run_index


def get_dag_run_state(dag_id: str, dag_run_id: str) -> str:
    """Return state string: queued, running, success, failed."""
    try:
        data = airflow_request("GET", f"/dags/{dag_id}/dagRuns/{dag_run_id}")
        return data.get("state", "unknown")
    except Exception:
        return "unknown"


def airflow_request(method: str, path: str, json_body: dict = None):
    url = f"{AIRFLOW_API_URL}{path}"
    try:
        resp = requests.request(
            method, url,
            auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
            json=json_body,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}
    except requests.HTTPError as e:
        raise RuntimeError(f"Airflow API {method} {path} → {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        raise RuntimeError(f"Airflow API unreachable: {e}")


def get_airflow_dags() -> list:
    try:
        data = airflow_request("GET", "/dags?limit=50")
        return sorted(data.get("dags", []), key=lambda d: d["dag_id"])
    except Exception:
        return []


def get_dag_runs(dag_id: str, limit: int = 5) -> list:
    try:
        data = airflow_request("GET", f"/dags/{dag_id}/dagRuns?limit={limit}&order_by=-start_date")
        return data.get("dag_runs", [])
    except Exception:
        return []


def trigger_dag(dag_id: str, conf: dict = None) -> dict:
    body = {"conf": conf or {}}
    return airflow_request("POST", f"/dags/{dag_id}/dagRuns", json_body=body)


def unpause_dag(dag_id: str) -> None:
    airflow_request("PATCH", f"/dags/{dag_id}", json_body={"is_paused": False})


def parse_last_json(output: str) -> dict:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {}


import logging
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MLOps Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("MLOps Control Panel")

# ---------------------------------------------------------------------------
# Sidebar - platform status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Platform Status")

    flask_info = get_flask_info()
    if flask_info:
        st.success("Flask API ✓")
        st.caption(
            f"{flask_info.get('model_name')} "
            f"v{flask_info.get('model_version')} "
            f"(alias:{flask_info.get('model_alias', 'production')})"
        )
    else:
        st.error("Flask API ✗")

    try:
        requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=3)
        st.success("MLflow ✓")
    except Exception:
        st.error("MLflow ✗")

    try:
        airflow_request("GET", "/health")
        st.success("Airflow ✓")
    except Exception:
        st.error("Airflow ✗")

    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM dwh_history.run_registry")
                n_runs = cur.fetchone()[0]
        st.success(f"PostgreSQL ✓  ({n_runs} runs)")
    except Exception:
        st.error("PostgreSQL ✗")

    st.divider()
    st.subheader("Recent Runs")
    df_sidebar = get_recent_runs(5)
    if not df_sidebar.empty:
        st.dataframe(
            df_sidebar[["run_index", "status", "drift_factor", "n_records"]].set_index("run_index"),
            use_container_width=True,
        )
    else:
        st.caption("No runs yet")

    if st.button("↻ Refresh", use_container_width=True):
        st.rerun()

    st.divider()
    st.caption("Airflow UI: [localhost:8080](http://localhost:8080)")
    st.caption("MLflow UI: [localhost:5000](http://localhost:5000)")
    st.caption("Monitoring Dashboard: [localhost:8502](http://localhost:8502)")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_deploy, tab_infer, tab_monitor, tab_logs = st.tabs(
    ["🚀 Deploy Model", "⚡ Batch Inference", "📈 Monitor", "📋 DAG Logs"]
)


# ============================================================================
# Tab 1 - Deploy Model
# ============================================================================
with tab_deploy:
    st.header("Deploy Model to Production")
    st.caption("Select a model version from the MLflow registry, promote it, and reload the Flask API.")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Select Version")
        versions = get_mlflow_versions()
        if versions:
            labels = {
                f"v{v.version}  [tag:{v.tags.get('stage', 'unset')}]  -  run {v.run_id[:8]}": v.version
                for v in versions
            }
            selected_label = st.selectbox("Model version", list(labels.keys()))
            selected_version = labels[selected_label]
            selected_obj = next(v for v in versions if str(v.version) == str(selected_version))
        else:
            st.warning("No model versions found in registry. Train a model first.")
            selected_version = None
            selected_obj = None

        model_name_input = st.text_input("Model registry name", value=MODEL_NAME)

    with col_right:
        st.subheader("Current Production")
        if flask_info:
            st.metric("Serving version", f"v{flask_info.get('model_version', '-')}")
            st.metric("Alias", flask_info.get("model_alias", "production"))
            st.caption(f"Run: {flask_info.get('run_id', '-')}")
        else:
            st.warning("Flask API not reachable")

        if selected_obj:
            st.subheader("Selected Version Info")
            st.json(
                {
                    "version": selected_obj.version,
                    "stage_tag": selected_obj.tags.get("stage", "unset"),
                    "run_id": selected_obj.run_id,
                    "created": str(selected_obj.creation_timestamp),
                }
            )

    st.divider()
    col_btn, col_status = st.columns([1, 3])
    deploy_btn = col_btn.button(
        "Promote & Deploy",
        type="primary",
        disabled=not selected_version,
        use_container_width=True,
    )

    if deploy_btn and selected_version:
        out_ph = st.empty()
        lines = []
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.MlflowClient()

            try:
                current_prod = client.get_model_version_by_alias(model_name_input, "production")
                if str(current_prod.version) != str(selected_version):
                    client.set_model_version_tag(model_name_input, current_prod.version, "stage", "archived")
                    lines.append(f"Archived v{current_prod.version}")
                    out_ph.code("\n".join(lines))
            except Exception:
                pass

            client.set_registered_model_alias(model_name_input, "production", str(selected_version))
            client.set_model_version_tag(model_name_input, str(selected_version), "stage", "production")
            lines.append(f"Promoted v{selected_version} → alias:production")
            out_ph.code("\n".join(lines))

            resp = requests.post(f"{FLASK_ENDPOINT}/reload", timeout=60)
            if not resp.ok:
                try:
                    flask_err = resp.json().get("error", resp.text[:300])
                except Exception:
                    flask_err = resp.text[:300]
                raise RuntimeError(f"Flask /reload HTTP {resp.status_code}: {flask_err}")
            reload_data = resp.json()
            lines.append(f"Flask reloaded: now serving v{reload_data.get('model_version')}")
            out_ph.code("\n".join(lines))

            st.success(f"v{selected_version} is now live in Production!")
            time.sleep(1)
            st.rerun()

        except Exception as exc:
            st.error(f"Deployment failed: {exc}")
            st.info("Check `docker compose logs flask-api` for the full stack trace.")


# ============================================================================
# Tab 2 - Batch Inference
# ============================================================================
with tab_infer:
    st.header("Run Batch Inference")
    st.caption(
        "Full pipeline: generate data → clean & encode → inference → write to dwh_history. "
        "Each run gets a unique sequential run_index."
    )

    col1, col2 = st.columns(2)
    with col1:
        infer_drift = st.slider(
            "Drift Factor",
            0.0, 1.0, 0.0, 0.05,
            key="infer_drift",
            help="0 = stable distribution  |  1 = maximum covariate shift",
        )
        infer_n = st.number_input(
            "Number of Records", 100, 10000, 1000, 100, key="infer_n"
        )

    with col2:
        mode = "drift" if infer_drift > 0 else "stable"
        st.info(
            f"**Airflow DAG**: `dag_03_batch_inference`\n\n"
            f"1. Generate {infer_n:,} records (mode={mode}, drift={infer_drift:.2f})\n"
            f"2. Process (clean, encode, validate)\n"
            f"3. Inference → Flask API → dwh_history\n"
            f"4. Auto-triggers dag_04a hard metrics on completion\n"
        )

    run_btn = st.button(
        "▶  Trigger Batch Inference DAG",
        type="primary",
        key="infer_btn",
    )

    if run_btn:
        try:
            unpause_dag("dag_03_batch_inference")
            result = trigger_dag("dag_03_batch_inference", {
                "drift_factor": float(infer_drift),
                "n_records": int(infer_n),
            })
            st.session_state["infer_run_id"] = result.get("dag_run_id")
            st.session_state["infer_dag"] = "dag_03_batch_inference"
        except RuntimeError as e:
            st.error(str(e))

    # Status polling
    if st.session_state.get("infer_run_id"):
        run_id = st.session_state["infer_run_id"]
        dag_id = st.session_state.get("infer_dag", "dag_03_batch_inference")
        state = get_dag_run_state(dag_id, run_id)

        state_colors = {"success": "✅", "failed": "❌", "running": "⏳", "queued": "🕐"}
        icon = state_colors.get(state, "❓")
        st.info(f"{icon} DAG run `{run_id[:40]}…`  -  state: **{state}**")

        col_ref, col_clr = st.columns([1, 1])
        if col_ref.button("↻ Refresh", key="infer_refresh"):
            st.rerun()
        if col_clr.button("Clear", key="infer_clear"):
            del st.session_state["infer_run_id"]
            st.rerun()

        if state in ("queued", "running"):
            time.sleep(3)
            st.rerun()

        if state == "success":
            df_hist = get_recent_runs(1)
            if not df_hist.empty:
                latest = df_hist.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Run Index", int(latest["run_index"]))
                c2.metric("Records", f"{latest.get('n_records', '-'):,}" if latest.get('n_records') else "-")
                c3.metric("Model Version", f"v{latest.get('model_version', '-')}")

    st.divider()
    st.subheader("Inference Run History")
    df_hist = get_recent_runs(15)
    if not df_hist.empty:
        st.dataframe(df_hist, use_container_width=True, hide_index=True)


# ============================================================================
# Tab 3 - Monitor
# ============================================================================
with tab_monitor:
    st.header("Monitoring")
    st.caption(
        "Runs monitoring DAGs against the latest inference run. "
        "All results persist to DB (dwh_monitoring_hard/drift/shap). "
        "Data source: dwh_history.prediction_ground_truth (features + predictions + ground truth)."
    )

    run_indices = get_available_run_indices()
    latest_run = run_indices[0] if run_indices else None

    if latest_run:
        st.info(f"Target: **Run Index {latest_run}** (latest)")
    else:
        st.warning("No inference runs found. Run batch inference first.")

    col_ctrl, col_results = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Modules")
        do_hard = st.checkbox("Hard Metrics", value=True, help="dag_04a - ROC-AUC, F1, accuracy. Auto-triggers retraining if below threshold.")
        do_drift = st.checkbox("Data Drift", help="dag_04b - Evidently covariate drift vs training reference. ~30s")
        do_shap = st.checkbox("SHAP Explainability", help="dag_04c - SHAP TreeExplainer + per-customer values. ~60s")

        mon_btn = st.button(
            "Run Monitoring",
            type="primary",
            disabled=latest_run is None,
            use_container_width=True,
        )

    with col_results:
        if mon_btn and latest_run is not None:
            conf = {"run_index": int(latest_run)}

            if do_hard:
                try:
                    unpause_dag("dag_04a_monitor_hard")
                    res = trigger_dag("dag_04a_monitor_hard", conf)
                    st.session_state["mon_hard_run_id"] = res.get("dag_run_id")
                except RuntimeError as e:
                    st.error(f"Hard metrics DAG failed to trigger: {e}")

            if do_drift:
                try:
                    unpause_dag("dag_04b_monitor_drift")
                    res = trigger_dag("dag_04b_monitor_drift", conf)
                    st.session_state["mon_drift_run_id"] = res.get("dag_run_id")
                except RuntimeError as e:
                    st.error(f"Drift DAG failed to trigger: {e}")

            if do_shap:
                try:
                    unpause_dag("dag_04c_monitor_shap")
                    res = trigger_dag("dag_04c_monitor_shap", conf)
                    st.session_state["mon_shap_run_id"] = res.get("dag_run_id")
                except RuntimeError as e:
                    st.error(f"SHAP DAG failed to trigger: {e}")

        # --- DAG run status display ---
        state_icons = {"success": "✅", "failed": "❌", "running": "⏳", "queued": "🕐"}
        still_running = False

        for key, label, dag_id in [
            ("mon_hard_run_id", "Hard Metrics (dag_04a)", "dag_04a_monitor_hard"),
            ("mon_drift_run_id", "Data Drift (dag_04b)", "dag_04b_monitor_drift"),
            ("mon_shap_run_id", "SHAP (dag_04c)", "dag_04c_monitor_shap"),
        ]:
            run_id = st.session_state.get(key)
            if run_id:
                state = get_dag_run_state(dag_id, run_id)
                icon = state_icons.get(state, "❓")
                st.info(f"{icon} **{label}** - `{run_id[:40]}…` - **{state}**")
                if state in ("queued", "running"):
                    still_running = True

        if still_running:
            time.sleep(3)
            st.rerun()

        col_ref, col_clr = st.columns([1, 1])
        has_active = any(st.session_state.get(k) for k in ("mon_hard_run_id", "mon_drift_run_id", "mon_shap_run_id"))
        if has_active:
            if col_ref.button("↻ Refresh", key="mon_refresh"):
                st.rerun()
            if col_clr.button("Clear", key="mon_clear"):
                for k in ("mon_hard_run_id", "mon_drift_run_id", "mon_shap_run_id"):
                    st.session_state.pop(k, None)
                st.rerun()


# ============================================================================
# Tab 4 - DAG Logs
# ============================================================================
with tab_logs:
    st.header("DAG Run Status & Logs")
    st.caption("View recent DAG runs across all pipelines. Fetch task logs for any run.")

    try:
        all_dags = get_airflow_dags()
    except Exception:
        all_dags = []

    if not all_dags:
        st.warning("Could not reach Airflow API. Check that Airflow is running.")
    else:
        dag_ids = [d["dag_id"] for d in all_dags]
        sel_dag = st.selectbox("Select DAG", dag_ids, key="log_dag_sel")

        col_dag_ctrl, col_dag_status = st.columns([1, 2])

        with col_dag_ctrl:
            n_runs_show = st.number_input("Recent runs to show", 1, 20, 5, key="log_n_runs")
            if st.button("↻ Refresh runs", key="log_refresh_runs"):
                st.rerun()

        recent_runs = get_dag_runs(sel_dag, int(n_runs_show))

        if not recent_runs:
            st.info(f"No runs found for {sel_dag}.")
        else:
            run_table = []
            for r in recent_runs:
                run_table.append({
                    "dag_run_id": r.get("dag_run_id", "")[:50],
                    "state": r.get("state", "unknown"),
                    "start_date": r.get("start_date", ""),
                    "end_date": r.get("end_date", ""),
                    "run_type": r.get("run_type", ""),
                })
            st.dataframe(pd.DataFrame(run_table), use_container_width=True, hide_index=True)

            run_ids = [r.get("dag_run_id") for r in recent_runs if r.get("dag_run_id")]
            sel_run_id = st.selectbox(
                "Select run to inspect", run_ids, key="log_run_sel",
                format_func=lambda x: x[:60],
            )

            if sel_run_id:
                st.subheader("Task Instances")
                try:
                    tasks_data = airflow_request(
                        "GET", f"/dags/{sel_dag}/dagRuns/{sel_run_id}/taskInstances"
                    )
                    task_instances = tasks_data.get("task_instances", [])
                except Exception as e:
                    task_instances = []
                    st.error(f"Could not fetch task instances: {e}")

                if task_instances:
                    task_table = []
                    for ti in task_instances:
                        state = ti.get("state", "none")
                        icon = {"success": "✅", "failed": "❌", "running": "⏳",
                                "queued": "🕐", "skipped": "⏭"}.get(state, "❓")
                        task_table.append({
                            "task_id": ti.get("task_id"),
                            "state": f"{icon} {state}",
                            "start_date": ti.get("start_date", ""),
                            "end_date": ti.get("end_date", ""),
                            "duration": f"{ti.get('duration', 0):.1f}s" if ti.get("duration") else "-",
                            "try_number": ti.get("try_number", 1),
                        })
                    st.dataframe(pd.DataFrame(task_table), use_container_width=True, hide_index=True)

                    task_ids = [ti.get("task_id") for ti in task_instances]
                    sel_task = st.selectbox("Fetch logs for task", task_ids, key="log_task_sel")
                    sel_try = st.number_input("Try number", 1, 5, 1, key="log_try_num")

                    if st.button("📋 Fetch Logs", key="log_fetch_btn"):
                        try:
                            log_data = airflow_request(
                                "GET",
                                f"/dags/{sel_dag}/dagRuns/{sel_run_id}/taskInstances/{sel_task}/logs/{int(sel_try)}",
                            )
                            log_content = log_data.get("content", "") if isinstance(log_data, dict) else str(log_data)
                            # Show last 200 lines
                            lines = log_content.splitlines()
                            if len(lines) > 200:
                                st.caption(f"Showing last 200 of {len(lines)} lines")
                                lines = lines[-200:]
                            st.code("\n".join(lines), language="text")
                        except Exception as e:
                            st.error(f"Could not fetch logs: {e}")
                else:
                    st.info("No task instances found for this run.")

