"""
MLOps Control Panel
====================
Service health, pipeline triggers, inline DAG logs, model deployment.
Port 8501  |  Companion: Monitoring Dashboard on :8502
"""

import os
import sys
import time

import mlflow
import pandas as pd
import psycopg2
import requests
import streamlit as st
from sqlalchemy import create_engine, text as _sql

# Shared metrics catalogue
sys.path.insert(0, "/opt/mlops/services")
try:
    from metrics import DEFAULT_PRIMARY, REGISTRY
except ImportError:
    REGISTRY = {
        "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1",
        "precision": "Precision", "recall": "Recall", "accuracy": "Accuracy",
    }
    DEFAULT_PRIMARY = "roc_auc"

# ── Config ────────────────────────────────────────────────────────────────────

FLASK_ENDPOINT = os.getenv("FLASK_ENDPOINT",     "http://flask-api:5001")
MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI","http://mlflow:5000")
MODEL_NAME     = os.getenv("MODEL_REGISTRY_NAME","credit-risk-classifier")
AIRFLOW_URL    = os.getenv("AIRFLOW_API_URL",    "http://airflow-webserver:8080/api/v1")
AIRFLOW_USER   = os.getenv("AIRFLOW_USER",       "admin")
AIRFLOW_PASS   = os.getenv("AIRFLOW_PASSWORD",   "admin")
MIN_ROC_AUC    = float(os.getenv("MIN_ROC_AUC",  0.51))

DB = {
    "host":     os.getenv("POSTGRES_HOST",     "postgres"),
    "port":     int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname":   os.getenv("POSTGRES_DB",       "mlops"),
    "user":     os.getenv("POSTGRES_USER",     "mlops_user"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

STATE_ICON = {
    "success": "✅", "failed": "❌", "running": "⏳",
    "queued": "🕐", "skipped": "⏭", "unknown": "❓",
}

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MLOps Control Panel",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    div[data-testid="metric-container"] {
        background: #f8fafc !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        border: 1px solid #e2e8f0 !important;
    }
    div[data-testid="metric-container"] > label {
        font-size: 0.75rem !important;
        color: #475569 !important;
        font-weight: 600 !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #64748b !important;
    }

    .svc-name   { font-weight: 700; font-size: 0.9rem; color: inherit; }
    .svc-online { color: #22c55e !important; font-size: 0.82rem; font-weight: 600; }
    .svc-offline{ color: #ef4444 !important; font-size: 0.82rem; font-weight: 600; }
    .svc-detail { color: #94a3b8; font-size: 0.75rem; }

    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; margin-left: 4px;
    }
    .badge-success { background: #dcfce7; color: #166534; }
    .badge-failed  { background: #fee2e2; color: #991b1b; }
    .badge-running { background: #fef9c3; color: #854d0e; }
    .badge-queued  { background: #e0f2fe; color: #075985; }
    .badge-unknown { background: #f1f5f9; color: #475569; }

    .log-header {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.06em; color: #94a3b8; margin-bottom: 0.4rem;
    }
    .task-progress {
        font-size: 0.8rem; color: #475569; font-family: monospace;
        line-height: 1.6; margin-bottom: 0.4rem;
    }
    .log-idle {
        color: #cbd5e1; font-size: 0.82rem; font-style: italic;
        padding: 1.5rem 0; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── DB helpers ────────────────────────────────────────────────────────────────

def db_conn():
    return psycopg2.connect(**DB)


@st.cache_resource
def _get_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB['user']}:{DB['password']}@{DB['host']}:{DB['port']}/{DB['dbname']}",
        pool_pre_ping=True,
    )


def _query(sql: str, params: dict | None = None) -> pd.DataFrame:
    try:
        with _get_engine().connect() as conn:
            return pd.read_sql(_sql(sql), conn, params=params or {})
    except Exception:
        return pd.DataFrame()

# ── Airflow helpers ───────────────────────────────────────────────────────────

def _airflow(method: str, path: str, body: dict = None):
    r = requests.request(
        method, f"{AIRFLOW_URL}{path}",
        auth=(AIRFLOW_USER, AIRFLOW_PASS),
        json=body, timeout=10,
    )
    r.raise_for_status()
    return r.json() if r.content else {}

def trigger_dag(dag_id: str, conf: dict = None) -> str:
    _airflow("PATCH", f"/dags/{dag_id}", {"is_paused": False})
    res = _airflow("POST", f"/dags/{dag_id}/dagRuns", {"conf": conf or {}})
    return res.get("dag_run_id", "")

def dag_run_state(dag_id: str, run_id: str) -> str:
    try:
        return _airflow("GET", f"/dags/{dag_id}/dagRuns/{run_id}").get("state", "unknown")
    except Exception:
        return "unknown"

def get_dag_runs(dag_id: str, limit: int = 8) -> list:
    try:
        return _airflow("GET", f"/dags/{dag_id}/dagRuns?limit={limit}&order_by=-start_date").get("dag_runs", [])
    except Exception:
        return []

def get_task_instances(dag_id: str, run_id: str) -> list:
    try:
        return _airflow("GET", f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances").get("task_instances", [])
    except Exception:
        return []

def fetch_task_log(dag_id: str, run_id: str, task_id: str, try_num: int = 1) -> str:
    """Fetch Airflow task log as plain text (endpoint returns text/plain, not JSON)."""
    try:
        r = requests.get(
            f"{AIRFLOW_URL}/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs/{try_num}",
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
            timeout=15,
        )
        r.raise_for_status()
        return r.text
    except Exception as e:
        return f"[log fetch error: {e}]"

def _clean_log(raw: str, tail: int = 120) -> str:
    """Strip Airflow's hex-token header, return last N lines."""
    lines = raw.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("[") or ("INFO" in line and "{" in line):
            start = i
            break
    cleaned = lines[start:]
    if len(cleaned) > tail:
        cleaned = cleaned[-tail:]
    return "\n".join(cleaned)

def list_dags() -> list[str]:
    try:
        dags = _airflow("GET", "/dags?limit=50").get("dags", [])
        return sorted(d["dag_id"] for d in dags)
    except Exception:
        return []

# ── MLflow helpers ────────────────────────────────────────────────────────────

def get_model_versions():
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        return sorted(versions, key=lambda v: int(v.version), reverse=True)
    except Exception:
        return []


@st.cache_data(ttl=30)
def get_production_primary_metric() -> str:
    """Read primary_metric tag from the current production model version."""
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, "production")
        tag = mv.tags.get("primary_metric", DEFAULT_PRIMARY)
        return tag if tag in REGISTRY else DEFAULT_PRIMARY
    except Exception:
        return DEFAULT_PRIMARY


@st.cache_data(ttl=10)
def get_run_primary_metric(run_id: str) -> str:
    """Read primary_metric tag from a specific MLflow run. Falls back to env default."""
    _default = os.getenv("DEFAULT_PRIMARY_METRIC", DEFAULT_PRIMARY)
    if not run_id:
        return _default
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        tag = run.data.tags.get("primary_metric")
        if tag and tag in REGISTRY:
            return tag
    except Exception:
        pass
    return _default

# ── Service health ────────────────────────────────────────────────────────────

def check_services() -> dict[str, tuple[bool, str]]:
    out = {}
    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM dwh_history.run_registry")
                n = cur.fetchone()[0]
        out["PostgreSQL"] = (True, f"{n} runs recorded")
    except Exception as e:
        out["PostgreSQL"] = (False, str(e)[:60])

    try:
        requests.get(f"{MLFLOW_URI}/health", timeout=3).raise_for_status()
        out["MLflow"] = (True, "Tracking server ready")
    except Exception:
        out["MLflow"] = (False, "Unreachable")

    try:
        _airflow("GET", "/health")
        out["Airflow"] = (True, "Scheduler healthy")
    except Exception:
        out["Airflow"] = (False, "Unreachable")

    try:
        r = requests.get(f"{FLASK_ENDPOINT}/model-info", timeout=5)
        if r.ok:
            info = r.json()
            out["Flask API"] = (True, f"v{info.get('model_version')} · {info.get('model_alias','production')}")
        else:
            out["Flask API"] = (False, f"HTTP {r.status_code}")
    except Exception:
        out["Flask API"] = (False, "Unreachable")

    return out

# ── Platform stats ────────────────────────────────────────────────────────────

def get_platform_stats(primary_metric: str = DEFAULT_PRIMARY) -> dict:
    df = _query("""
        SELECT COUNT(*) AS total_runs, MAX(run_index) AS latest_run,
               SUM(n_records_requested) AS total_records,
               COUNT(*) FILTER (WHERE status='completed') AS completed
        FROM dwh_history.run_registry
    """)
    if df.empty:
        return {}
    r = df.iloc[0]
    stats = {
        "total_runs":    int(r["total_runs"]    or 0),
        "latest_run":    int(r["latest_run"]    or 0),
        "total_records": int(r["total_records"] or 0),
        "completed":     int(r["completed"]     or 0),
    }
    # Map metric key to DB column name
    _col = {
        "roc_auc": "roc_auc", "pr_auc": "pr_auc",
        "f1": "f1_score", "precision": "precision_score",
        "recall": "recall_score", "accuracy": "accuracy",
    }
    db_col = _col.get(primary_metric, "roc_auc")
    m = _query(f"SELECT {db_col} AS metric_val FROM dwh_monitoring_hard.results ORDER BY evaluated_at DESC LIMIT 1")
    stats["latest_primary_metric"]  = round(float(m.iloc[0]["metric_val"]), 4) if not m.empty and m.iloc[0]["metric_val"] is not None else None
    stats["primary_metric_key"]     = primary_metric
    stats["primary_metric_display"] = REGISTRY.get(primary_metric, primary_metric.upper())
    return stats

def get_run_history(limit: int = 20) -> pd.DataFrame:
    return _query("""
        SELECT run_index, status, drift_factor,
               n_records_requested AS records, model_version,
               created_at, completed_at
        FROM dwh_history.run_registry
        ORDER BY run_index DESC LIMIT :limit
    """, {"limit": limit})

def get_latest_run_index() -> int | None:
    df = _query("SELECT MAX(run_index) AS r FROM dwh_history.prediction_ground_truth")
    if df.empty or df.iloc[0]["r"] is None:
        return None
    return int(df.iloc[0]["r"])

def get_available_run_indices() -> list[int]:
    df = _query(
        "SELECT DISTINCT run_index FROM dwh_history.prediction_ground_truth ORDER BY run_index DESC"
    )
    if df.empty:
        return []
    return df["run_index"].astype(int).tolist()

def get_latest_comparison() -> dict:
    """Return the single most recent challenger comparison as a dict."""
    df = _query("""
        SELECT challenger_run_id, primary_metric,
               prod_primary_score, challenger_primary_score,
               prod_cv_std, challenger_cv_std,
               cv_folds, cv_margin,
               prod_roc_auc, challenger_roc_auc,
               challenger_wins, force_deploy, promoted, eval_records, compared_at
        FROM dwh_challenger.comparison_log
        ORDER BY compared_at DESC LIMIT 1
    """)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _badge(state: str) -> str:
    cls = {"success": "success", "failed": "failed",
           "running": "running", "queued": "queued"}.get(state, "unknown")
    return f'<span class="badge badge-{cls}">{STATE_ICON.get(state,"❓")} {state}</span>'


def _log_panel(dag_id: str, run_id: str, ss_key: str, refresh_key: str, clear_key: str):
    """
    Renders the right-hand log panel for a single DAG run.
    Manual refresh only - no auto-refresh.
    """
    if not run_id:
        st.markdown('<div class="log-idle">No active run</div>', unsafe_allow_html=True)
        return

    state = dag_run_state(dag_id, run_id)

    # Header: state badge + refresh + clear
    h1, h2, h3 = st.columns([3, 1, 1])
    h1.markdown(f'<div class="log-header">Logs</div>{_badge(state)}', unsafe_allow_html=True)
    if h2.button("↻", key=refresh_key, help="Refresh logs"):
        st.rerun()
    if h3.button("✕", key=clear_key, help="Clear"):
        st.session_state.pop(ss_key, None)
        st.rerun()

    tasks = get_task_instances(dag_id, run_id)
    if not tasks:
        st.caption("Waiting for tasks…")
        return

    # Task progress strip
    progress = "  →  ".join(
        f"{STATE_ICON.get(t.get('state', ''), '❓')} {t['task_id']}"
        for t in tasks
    )
    st.markdown(f'<div class="task-progress">{progress}</div>', unsafe_allow_html=True)

    # Pick most interesting task: running > queued > failed > last
    priority = {"running": 0, "queued": 1, "failed": 2, "success": 3}
    focus = sorted(tasks, key=lambda t: priority.get(t.get("state", ""), 4))[0]
    task_id = focus["task_id"]
    try_num = focus.get("try_number", 1)
    t_state = focus.get("state", "")

    raw = fetch_task_log(dag_id, run_id, task_id, try_num)
    log_text = _clean_log(raw, tail=100)

    lbl = f"`{task_id}` (try {try_num})"
    if t_state == "running":
        lbl += " - running"

    with st.expander(lbl, expanded=False):
        st.code(log_text, language="text")

    # Other completed tasks collapsed
    for ot in tasks:
        if ot["task_id"] == task_id or ot.get("state") != "success":
            continue
        with st.expander(f"`{ot['task_id']}` ✅", expanded=False):
            st.code(_clean_log(
                fetch_task_log(dag_id, run_id, ot["task_id"], ot.get("try_number", 1)),
                tail=60,
            ), language="text")


def _multi_log_panel(dag_configs: list[tuple[str, str, str]], refresh_key: str, clear_all_key: str):
    """
    Log panel for multiple DAGs (monitoring).
    dag_configs: list of (ss_key, dag_id, label)
    """
    active = [(ss_k, dag_id, lbl) for ss_k, dag_id, lbl in dag_configs
              if st.session_state.get(ss_k)]

    if not active:
        st.markdown('<div class="log-idle">No active run</div>', unsafe_allow_html=True)
        return

    h1, h2, h3 = st.columns([3, 1, 1])
    h1.markdown('<div class="log-header">Logs</div>', unsafe_allow_html=True)
    if h2.button("↻", key=refresh_key, help="Refresh logs"):
        st.rerun()
    if h3.button("✕ all", key=clear_all_key, help="Clear all"):
        for ss_k, _, _ in dag_configs:
            st.session_state.pop(ss_k, None)
        st.rerun()

    for ss_k, dag_id, label in active:
        run_id = st.session_state[ss_k]
        state  = dag_run_state(dag_id, run_id)

        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{label}** {_badge(state)}", unsafe_allow_html=True)
        if c2.button("✕", key=f"clr_{ss_k}_inline"):
            st.session_state.pop(ss_k, None)
            st.rerun()

        tasks = get_task_instances(dag_id, run_id)
        if not tasks:
            st.caption("Waiting for tasks…")
            continue

        progress = "  →  ".join(
            f"{STATE_ICON.get(t.get('state', ''), '❓')} {t['task_id']}"
            for t in tasks
        )
        st.markdown(f'<div class="task-progress">{progress}</div>', unsafe_allow_html=True)

        priority = {"running": 0, "queued": 1, "failed": 2, "success": 3}
        focus = sorted(tasks, key=lambda t: priority.get(t.get("state", ""), 4))[0]
        task_id = focus["task_id"]
        try_num = focus.get("try_number", 1)
        t_state = focus.get("state", "")

        lbl = f"`{task_id}` (try {try_num})"
        if t_state == "running":
            lbl += " - running"

        raw = fetch_task_log(dag_id, run_id, task_id, try_num)
        with st.expander(lbl, expanded=False):
            st.code(_clean_log(raw, tail=80), language="text")


def _render_dag_log_expander(dag_id: str, run_id: str, label: str = "View Logs", key_prefix: str = ""):
    """Historical log viewer used only in DAG Browser."""
    k = f"{key_prefix}__{dag_id}__{run_id}"
    with st.expander(f"📋 {label}", expanded=False):
        tasks = get_task_instances(dag_id, run_id)
        if not tasks:
            st.info("No task instances found.")
            return

        task_rows = []
        for t in tasks:
            s = t.get("state") or "none"
            dur = t.get("duration")
            task_rows.append({
                "task":     t.get("task_id"),
                "state":    f"{STATE_ICON.get(s, '❓')} {s}",
                "start":    (t.get("start_date") or "")[:19],
                "duration": f"{dur:.1f}s" if dur else "-",
                "try":      t.get("try_number", 1),
            })
        st.dataframe(pd.DataFrame(task_rows), width="stretch", hide_index=True)

        task_ids = [t.get("task_id") for t in tasks]
        lc1, lc2, lc3 = st.columns([3, 1, 1])
        sel_task = lc1.selectbox("Task", task_ids, key=f"lt__{k}")
        sel_try  = lc2.number_input("Try #", 1, 5, 1, key=f"ltr__{k}")
        if lc3.button("Fetch", key=f"lf__{k}", width="stretch"):
            with st.spinner("Fetching log…"):
                raw = fetch_task_log(dag_id, run_id, sel_task, int(sel_try))
            st.code(_clean_log(raw, tail=300), language="text")


# ═══════════════════════════════════════════════════════════════════
# PAGE - Header
# ═══════════════════════════════════════════════════════════════════

col_title, col_links = st.columns([4, 1])
with col_title:
    st.title("🎛️ MLOps Control Panel")
    st.caption("Credit Risk Classifier · Pipeline Orchestration & Deployment")
with col_links:
    st.caption("")
    st.caption("📊 [Monitoring Dashboard](http://localhost:8502)")
    st.caption("🔗 [Airflow](http://localhost:8080) · [MLflow](http://localhost:5000)")

# ── Service health ────────────────────────────────────────────────────────────

services = check_services()
svc_cols = st.columns(len(services))
for col, (name, (up, detail)) in zip(svc_cols, services.items()):
    with col:
        with st.container(border=True):
            icon = "🟢" if up else "🔴"
            st.markdown(f'<div class="svc-name">{icon} {name}</div>', unsafe_allow_html=True)
            if up:
                st.markdown('<div class="svc-online">● Online</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="svc-offline">● Offline</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="svc-detail">{detail}</div>', unsafe_allow_html=True)

all_up = all(up for up, _ in services.values())
if not all_up:
    down = [n for n, (up, _) in services.items() if not up]
    st.warning(f"Services offline: {', '.join(down)}")

# ── Platform stats ────────────────────────────────────────────────────────────

_prod_primary_metric = get_production_primary_metric()
stats = get_platform_stats(_prod_primary_metric)
if stats:
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Runs",        stats["total_runs"])
    s2.metric("Current Run Index", stats["latest_run"] or "-")
    s3.metric("Total Records",     f"{stats['total_records']:,}" if stats["total_records"] else "-")
    _pm_label = f"Latest {stats.get('primary_metric_display', 'ROC-AUC')}"
    _pm_val   = stats.get("latest_primary_metric")
    s4.metric(_pm_label, f"{_pm_val:.4f}" if _pm_val is not None else "-")

cr, _ = st.columns([1, 5])
if cr.button("↻ Refresh", width="stretch"):
    st.rerun()

st.divider()

# ═══════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════

tab_pipe, tab_deploy, tab_challenger = st.tabs(["🔄  Pipelines", "🚀  Deploy Model", "🧪  Challenger"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 - Pipelines
# ════════════════════════════════════════════════════════════════════════════
with tab_pipe:

    latest_run  = get_latest_run_index()
    run_history = get_run_history(20)

    # ── Batch Inference ──────────────────────────────────────────────────────
    with st.container(border=True):
        col_ctrl, col_log = st.columns([1, 1])

        with col_ctrl:
            st.subheader("⚡ Batch Inference  `dag_03`")
            st.caption("Generate → process → infer → write to history")

            ci1, ci2 = st.columns(2)
            drift  = ci1.slider("Drift factor", 0.0, 1.0, 0.0, 0.05, key="p_drift",
                                help="0 = stable distribution · 1 = maximum covariate shift")
            n_recs = ci2.number_input("Records", 100, 10_000, 1_000, 100, key="p_nrecs")

            if st.button("▶  Run Batch Inference", type="primary",
                         key="p_infer_btn", width="stretch"):
                try:
                    rid = trigger_dag("dag_03_batch_inference",
                                      {"drift_factor": float(drift), "n_records": int(n_recs)})
                    st.session_state["infer_run_id"] = rid
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to trigger DAG: {e}")

        with col_log:
            _log_panel(
                dag_id="dag_03_batch_inference",
                run_id=st.session_state.get("infer_run_id"),
                ss_key="infer_run_id",
                refresh_key="infer_ref",
                clear_key="infer_clr",
            )

    st.markdown("")

    # ── Monitoring ───────────────────────────────────────────────────────────
    with st.container(border=True):
        col_ctrl, col_log = st.columns([1, 1])

        with col_ctrl:
            st.subheader("📈 Monitoring  `dag_04a / 04b / 04c`")

            available_runs = get_available_run_indices()
            if not available_runs:
                st.warning("No inference runs yet - run Batch Inference first.")
                mon_run_index = None
            else:
                mon_run_index = st.selectbox(
                    "Run index to monitor",
                    options=available_runs,
                    format_func=lambda x: f"Run {x}",
                    key="mon_run_sel",
                )

            if st.button("▶  Run Monitoring", type="primary", key="p_mon_btn",
                         disabled=mon_run_index is None, width="stretch"):
                conf = {"run_index": int(mon_run_index)}
                for dag_key, dag_id in [
                    ("mon_hard_id",  "dag_04a_monitor_hard"),
                    ("mon_drift_id", "dag_04b_monitor_drift"),
                    ("mon_shap_id",  "dag_04c_monitor_shap"),
                ]:
                    try:
                        st.session_state[dag_key] = trigger_dag(dag_id, conf)
                    except Exception as e:
                        st.error(f"{dag_id}: {e}")
                st.rerun()

        with col_log:
            _multi_log_panel(
                dag_configs=[
                    ("mon_hard_id",  "dag_04a_monitor_hard",  "Hard Metrics"),
                    ("mon_drift_id", "dag_04b_monitor_drift", "Data Drift"),
                    ("mon_shap_id",  "dag_04c_monitor_shap",  "SHAP"),
                ],
                refresh_key="mon_ref",
                clear_all_key="mon_clr_all",
            )

    st.markdown("")

    # ── Manual Retraining ────────────────────────────────────────────────────
    with st.container(border=True):
        col_ctrl, col_log = st.columns([1, 1])

        with col_ctrl:
            st.subheader("🔁 Manual Retraining  `dag_05`")

            if st.button("▶  Trigger Retraining", type="primary",
                         key="p_retrain_btn", width="stretch"):
                try:
                    rid = trigger_dag("dag_05_retraining")
                    st.session_state["retrain_run_id"] = rid
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

        with col_log:
            _log_panel(
                dag_id="dag_05_retraining",
                run_id=st.session_state.get("retrain_run_id"),
                ss_key="retrain_run_id",
                refresh_key="retrain_ref",
                clear_key="retrain_clr",
            )

    st.divider()

    # ── Run History ──────────────────────────────────────────────────────────
    st.subheader("Run History")
    if not run_history.empty:
        st.dataframe(
            run_history.style.format({"drift_factor": "{:.2f}", "records": "{:,.0f}"},
                                     na_rep="-"),
            width="stretch", hide_index=True,
        )
    else:
        st.info("No runs recorded yet.")



# ════════════════════════════════════════════════════════════════════════════
# TAB 2 - Deploy Model
# ════════════════════════════════════════════════════════════════════════════
with tab_deploy:

    versions = get_model_versions()

    col_sel, col_curr = st.columns(2)

    with col_sel:
        with st.container(border=True):
            st.subheader("Select Version")
            if not versions:
                st.warning("No model versions in registry. Train a model first.")
                selected_version = None
                selected_obj     = None
                model_name_input = MODEL_NAME
            else:
                labels = {
                    f"v{v.version}  [{v.tags.get('stage','-')}]  run:{v.run_id[:8]}": v.version
                    for v in versions
                }
                selected_label   = st.selectbox("Model version", list(labels.keys()), key="dep_ver")
                selected_version = labels[selected_label]
                selected_obj     = next(v for v in versions if str(v.version) == str(selected_version))
                model_name_input = st.text_input("Registry name", value=MODEL_NAME, key="dep_name")

                st.json({
                    "version":   selected_obj.version,
                    "stage_tag": selected_obj.tags.get("stage", "-"),
                    "run_id":    selected_obj.run_id,
                    "created":   str(selected_obj.creation_timestamp),
                }, expanded=False)

    with col_curr:
        with st.container(border=True):
            st.subheader("Currently Serving")
            try:
                info = requests.get(f"{FLASK_ENDPOINT}/model-info", timeout=5).json()
                st.metric("Version", f"v{info.get('model_version', '-')}")
                st.metric("Alias",   info.get("model_alias", "production"))
                st.caption(f"Run ID: `{info.get('run_id','-')[:24]}…`")
            except Exception:
                st.error("Flask API unreachable")

    st.divider()

    btn_deploy = st.button(
        "🚀  Promote & Deploy to Production",
        type="primary", key="dep_btn",
        disabled=not (versions and selected_version),
        width="stretch",
    )

    if btn_deploy and selected_version:
        log_ph = st.empty()
        lines  = []
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            client = mlflow.MlflowClient()

            try:
                prev = client.get_model_version_by_alias(model_name_input, "production")
                if str(prev.version) != str(selected_version):
                    client.set_model_version_tag(model_name_input, prev.version, "stage", "archived")
                    lines.append(f"[1/3] Archived previous production v{prev.version}")
                    log_ph.code("\n".join(lines))
            except Exception:
                lines.append("[1/3] No previous production alias - first deploy")
                log_ph.code("\n".join(lines))

            client.set_registered_model_alias(model_name_input, "production", str(selected_version))
            client.set_model_version_tag(model_name_input, str(selected_version), "stage", "production")
            lines.append(f"[2/3] Alias 'production' → v{selected_version}")
            log_ph.code("\n".join(lines))

            resp = requests.post(f"{FLASK_ENDPOINT}/reload", timeout=60)
            if not resp.ok:
                try:
                    flask_err = resp.json().get("error", resp.text[:300])
                except Exception:
                    flask_err = resp.text[:300]
                raise RuntimeError(f"Flask /reload HTTP {resp.status_code}: {flask_err}")
            reload_info = resp.json()
            lines.append(f"[3/3] Flask reloaded - serving v{reload_info.get('model_version')}")
            log_ph.code("\n".join(lines))

            st.success(f"✅  v{selected_version} is live in Production!")
            time.sleep(1)
            st.rerun()

        except Exception as exc:
            st.error(f"Deployment failed: {exc}")
            st.info("Check `docker compose logs flask-api` for the full stack trace.")

    st.divider()
    st.subheader("All Registered Versions")
    if versions:
        tbl = [{
            "version":   v.version,
            "stage_tag": v.tags.get("stage", "-"),
            "run_id":    v.run_id[:20] + "…",
            "created":   str(v.creation_timestamp),
        } for v in versions]
        st.dataframe(pd.DataFrame(tbl), width="stretch", hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 - Challenger Model
# ════════════════════════════════════════════════════════════════════════════
with tab_challenger:

    with st.container(border=True):
        col_ctrl, col_log = st.columns([1, 1])

        with col_ctrl:
            st.subheader("🧪 Challenger Comparison  `dag_06`")

            challenger_run_id = st.text_input(
                "MLflow Run ID",
                placeholder="e.g. a1b2c3d4e5f6...",
                key="chal_run_id",
                help="Copy run_id from JupyterLab after mlflow.start_run()",
            )

            _detected_metric = get_run_primary_metric(challenger_run_id.strip())
            chal_primary_metric = _detected_metric
            _metric_display = REGISTRY.get(_detected_metric, _detected_metric.upper())
            if challenger_run_id.strip():
                st.info(f"Primary metric: **{_metric_display}** (from MLflow run tag)")
            else:
                st.caption(f"Primary metric: **{_metric_display}** (default - enter run ID to auto-detect)")

            force_deploy = st.checkbox(
                "Force Deploy (promote even if challenger loses)",
                key="chal_force",
                help="Comparison still runs and is logged; production is replaced regardless of outcome.",
            )

            if force_deploy:
                st.warning(f"⚠ Force Deploy enabled - challenger will be promoted regardless of {REGISTRY[chal_primary_metric]} comparison.")

            trigger_disabled = not challenger_run_id.strip()
            if st.button(
                "▶  Run Challenger Comparison",
                type="primary",
                key="chal_btn",
                disabled=trigger_disabled,
                width="stretch",
            ):
                try:
                    rid = trigger_dag("dag_06_challenger_comparison", {
                        "challenger_run_id": challenger_run_id.strip(),
                        "force_deploy":      force_deploy,
                        "primary_metric":    chal_primary_metric,
                    })
                    st.session_state["chal_dag_run_id"] = rid
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to trigger DAG: {e}")

        with col_log:
            _log_panel(
                dag_id="dag_06_challenger_comparison",
                run_id=st.session_state.get("chal_dag_run_id"),
                ss_key="chal_dag_run_id",
                refresh_key="chal_ref",
                clear_key="chal_clr",
            )

    st.markdown("")

    # ── Latest Comparison Result ─────────────────────────────────────────────
    st.subheader("Latest Comparison Result")
    cmp = get_latest_comparison()
    if cmp:
        metric_key   = cmp.get("primary_metric", DEFAULT_PRIMARY)
        metric_label = REGISTRY.get(metric_key, metric_key.upper())
        prod_avg     = cmp.get("prod_primary_score")
        chal_avg     = cmp.get("challenger_primary_score")
        prod_std     = cmp.get("prod_cv_std")
        chal_std     = cmp.get("challenger_cv_std")
        cv_folds     = cmp.get("cv_folds", 5)
        cv_margin    = cmp.get("cv_margin", 0.05)
        wins         = cmp.get("challenger_wins")
        promoted     = cmp.get("promoted")
        run_id_short = str(cmp.get("challenger_run_id", ""))[:16] + "…"
        compared_at  = str(cmp.get("compared_at", ""))[:19]

        delta = (chal_avg or 0) - (prod_avg or 0) if prod_avg is not None and chal_avg is not None else None

        with st.container(border=True):
            st.caption(
                f"**{metric_label}** · {cv_folds}-fold Stratified CV  "
                f"· margin required: **+{cv_margin:.0%}**  "
                f"· run `{run_id_short}`  · {compared_at}"
            )

            col_prod, col_vs, col_chal, col_verdict = st.columns([3, 1, 3, 2])

            with col_prod:
                st.markdown("#### 🏭 Production")
                if prod_avg is not None:
                    st.metric(f"Avg {metric_label}", f"{prod_avg:.4f}")
                    if prod_std is not None:
                        st.caption(f"std ± {prod_std:.4f}  ({cv_folds} folds)")
                else:
                    st.metric(f"Avg {metric_label}", "-")

            with col_vs:
                st.markdown("<br><br><div style='text-align:center;font-size:1.4rem'>vs</div>",
                            unsafe_allow_html=True)

            with col_chal:
                st.markdown("#### 🧪 Challenger")
                if chal_avg is not None:
                    delta_display = f"{delta:+.4f}" if delta is not None else None
                    st.metric(f"Avg {metric_label}", f"{chal_avg:.4f}", delta=delta_display)
                    if chal_std is not None:
                        st.caption(f"std ± {chal_std:.4f}  ({cv_folds} folds)")
                else:
                    st.metric(f"Avg {metric_label}", "-")

            with col_verdict:
                st.markdown("<br>", unsafe_allow_html=True)
                if wins is True:
                    beats_margin = delta is not None and delta > cv_margin
                    st.success(f"✅ Challenger wins\n\n+{delta:.4f} > margin +{cv_margin:.2f}")
                elif wins is False:
                    if delta is not None:
                        st.error(f"❌ Challenger loses\n\nΔ={delta:+.4f}, need >{cv_margin:.2f}")
                    else:
                        st.error("❌ Challenger loses")
                else:
                    st.info("Result unknown")

                if promoted:
                    st.success("🚀 Promoted to production")
                elif cmp.get("force_deploy"):
                    st.warning("⚡ Force-deployed")
    else:
        st.info("No challenger comparisons yet. Run your first comparison above.")
