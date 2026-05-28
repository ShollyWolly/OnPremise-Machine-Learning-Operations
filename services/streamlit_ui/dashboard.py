"""
MLOps Monitoring Dashboard
============================
Port 8502  |  Control Panel on :8501

Tabs:
  Data Drift      - feature distributions, KS stats, drift trends
  Hard Metrics    - confusion matrix, ROC, PR, calibration, metric trends
  Explainability  - SHAP beeswarm, bar, outcome-grouped impact, waterfall
"""

import json
import logging
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import requests
from sqlalchemy import create_engine, text as _sql
import shap
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

try:
    from sklearn.calibration import calibration_curve
    HAS_CALIBRATION = True
except ImportError:
    HAS_CALIBRATION = False

try:
    from scipy.stats import ks_2samp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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

# DB column name for each metric key
_METRIC_DB_COL = {
    "roc_auc":   "roc_auc",
    "pr_auc":    "pr_auc",
    "f1":        "f1_score",
    "precision": "precision_score",
    "recall":    "recall_score",
    "accuracy":  "accuracy",
}

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

FLASK_ENDPOINT    = os.getenv("FLASK_ENDPOINT",        "http://flask-api:5001")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME        = os.getenv("MODEL_REGISTRY_NAME",   "credit-risk-classifier")

MIN_ROC_AUC              = float(os.getenv("MIN_ROC_AUC",               0.51))
MAX_DRIFT_FEATURE_FRACTION = float(os.getenv("MAX_DRIFT_FEATURE_FRACTION", 0.50))

POSTGRES_CONN = {
    "host":     os.getenv("POSTGRES_HOST",     "postgres"),
    "port":     int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname":   os.getenv("POSTGRES_DB",       "mlops"),
    "user":     os.getenv("POSTGRES_USER",     "mlops_user"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

FEATURE_COLUMNS = [
    "age", "annual_income", "credit_score", "loan_amount",
    "loan_term_months", "employment_length_years", "home_ownership_encoded",
    "debt_to_income_ratio", "num_credit_lines", "payment_history_score",
]

# Color palette - consistent across all charts
PALETTE = {
    "primary":   "#3b82f6",
    "success":   "#22c55e",
    "warning":   "#f59e0b",
    "danger":    "#ef4444",
    "purple":    "#8b5cf6",
    "reference": "#64748b",
    "current":   "#f97316",
    "TP":        "#22c55e",
    "FP":        "#f97316",
    "TN":        "#64748b",
    "FN":        "#ef4444",
}

PLOTLY_TEMPLATE = "plotly_white"

# ── Page setup (must come first) ──────────────────────────────────────────────

st.set_page_config(
    page_title="MLOps Monitoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Main area */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #f8fafc;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        border: 1px solid #e2e8f0;
    }
    div[data-testid="metric-container"] > label { font-size: 0.75rem !important; color: #64748b !important; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.5rem !important; }

    /* Status indicators in sidebar */
    .svc-row { display:flex; align-items:center; gap:0.4rem;
               padding:0.35rem 0.5rem; border-radius:6px; margin-bottom:0.2rem; background:#f8fafc; }
    .svc-dot-up   { width:8px;height:8px;border-radius:50%;background:#22c55e;flex-shrink:0; }
    .svc-dot-down { width:8px;height:8px;border-radius:50%;background:#ef4444;flex-shrink:0; }
    .svc-name { font-size:0.82rem; font-weight:600; color:#1e293b; }
    .svc-detail { font-size:0.73rem; color:#64748b; }

    /* Drift detected badge */
    .badge-warn { background:#fef3c7;color:#92400e;padding:3px 10px;
                  border-radius:12px;font-size:0.8rem;font-weight:600; }
    .badge-ok   { background:#dcfce7;color:#166534;padding:3px 10px;
                  border-radius:12px;font-size:0.8rem;font-weight:600; }

    /* Section headers */
    .section-header { font-size:1rem;font-weight:700;color:#1e293b;
                      margin-bottom:0.25rem;margin-top:0.5rem; }

    /* Empty state */
    .empty-state { text-align:center;padding:2.5rem;color:#94a3b8;font-size:0.9rem; }

    /* Tab run selector bar */
    .run-bar { background:#f1f5f9;border-radius:8px;padding:0.5rem 0.75rem;
               margin-bottom:0.75rem;font-size:0.85rem;color:#475569; }
</style>
""", unsafe_allow_html=True)

# ── DB helpers ────────────────────────────────────────────────────────────────

def db_conn():
    return psycopg2.connect(**POSTGRES_CONN)


@st.cache_resource
def _get_engine():
    c = POSTGRES_CONN
    return create_engine(
        f"postgresql+psycopg2://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['dbname']}",
        pool_pre_ping=True,
    )


def _read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    with _get_engine().connect() as conn:
        return pd.read_sql(_sql(query), conn, params=params or {})


@st.cache_data(ttl=60)
def get_drift_runs() -> list:
    try:
        df = _read_sql("SELECT DISTINCT run_index FROM dwh_monitoring_drift.results ORDER BY run_index DESC LIMIT 50")
        return df["run_index"].tolist()
    except Exception:
        return []


@st.cache_data(ttl=60)
def get_hard_metric_runs() -> list:
    try:
        df = _read_sql("SELECT DISTINCT run_index FROM dwh_monitoring_hard.results ORDER BY run_index DESC LIMIT 50")
        return df["run_index"].tolist()
    except Exception:
        return []


@st.cache_data(ttl=60)
def get_shap_runs() -> list:
    try:
        df = _read_sql("SELECT DISTINCT run_index FROM dwh_monitoring_shap.results ORDER BY run_index DESC LIMIT 50")
        return df["run_index"].tolist()
    except Exception:
        return []


@st.cache_data(ttl=60)
def get_drift_result(run_index: int) -> dict:
    try:
        df = _read_sql(
            "SELECT * FROM dwh_monitoring_drift.results WHERE run_index = :run_index ORDER BY evaluated_at DESC LIMIT 1",
            {"run_index": run_index},
        )
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception:
        return {}


@st.cache_data(ttl=60)
def get_hard_metrics_row(run_index: int) -> dict:
    try:
        df = _read_sql(
            "SELECT * FROM dwh_monitoring_hard.results WHERE run_index = :run_index ORDER BY evaluated_at DESC LIMIT 1",
            {"run_index": run_index},
        )
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception:
        return {}


@st.cache_data(ttl=60)
def get_hard_metrics_history(limit: int = 50) -> pd.DataFrame:
    try:
        return _read_sql(
            """SELECT run_index, roc_auc, f1_score, accuracy, precision_score,
                      recall_score, n_records, retraining_triggered, evaluated_at
               FROM dwh_monitoring_hard.results
               ORDER BY run_index ASC LIMIT :limit""",
            {"limit": limit},
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_drift_history(limit: int = 50) -> pd.DataFrame:
    try:
        return _read_sql(
            """SELECT run_index, drift_score, drift_detected, num_drifted_features,
                      drifted_feature_names, evaluated_at
               FROM dwh_monitoring_drift.results
               ORDER BY run_index ASC LIMIT :limit""",
            {"limit": limit},
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_raw_predictions(run_index: int) -> pd.DataFrame:
    try:
        return _read_sql(
            """SELECT default_probability, default_flag_predicted, actual_default_flag
               FROM dwh_history.prediction_ground_truth
               WHERE run_index = :run_index AND actual_default_flag IS NOT NULL""",
            {"run_index": run_index},
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_current_features(run_index: int) -> pd.DataFrame:
    cols = ", ".join(FEATURE_COLUMNS)
    try:
        return _read_sql(
            f"SELECT {cols} FROM dwh_history.prediction_ground_truth WHERE run_index = :run_index",
            {"run_index": run_index},
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_reference_features() -> pd.DataFrame | None:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        version = client.get_model_version_by_alias(MODEL_NAME, "production")
        with tempfile.TemporaryDirectory() as tmp:
            local = mlflow.artifacts.download_artifacts(
                run_id=version.run_id, artifact_path="reference_data.parquet", dst_path=tmp
            )
            # Old runs logged reference_data.parquet as a subdirectory (MLflow artifact_path bug).
            # New runs log it as a file at the root. Handle both.
            if os.path.isdir(local):
                parquets = [f for f in os.listdir(local) if f.endswith(".parquet")]
                if not parquets:
                    return None
                local = os.path.join(local, parquets[0])
            return pd.read_parquet(local, engine="pyarrow")[FEATURE_COLUMNS]
    except Exception:
        return None


@st.cache_data(ttl=60)
def get_shap_aggregate(run_index: int) -> dict:
    try:
        df = _read_sql(
            """SELECT feature_importances, top_feature, explainer_type, n_records
               FROM dwh_monitoring_shap.results WHERE run_index = :run_index LIMIT 1""",
            {"run_index": run_index},
        )
        if df.empty:
            return {}
        row = df.iloc[0]
        fi = row["feature_importances"]
        return {
            "feature_importances": fi if isinstance(fi, dict) else json.loads(fi),
            "top_feature":   row["top_feature"],
            "explainer_type": row["explainer_type"],
            "n_records":     row["n_records"],
        }
    except Exception:
        return {}


@st.cache_data(ttl=60)
def get_shap_with_outcomes(run_index: int) -> pd.DataFrame:
    try:
        return _read_sql(
            """SELECT s.record_id, s.shap_values,
                      p.default_flag_predicted, p.actual_default_flag
               FROM dwh_monitoring_shap.customer_shap_values s
               JOIN dwh_history.prediction_ground_truth p
                 ON s.run_index = p.run_index AND s.record_id = p.record_id
               WHERE s.run_index = :run_index AND p.actual_default_flag IS NOT NULL""",
            {"run_index": run_index},
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_beeswarm_data(run_index: int) -> pd.DataFrame:
    try:
        return _read_sql(
            "SELECT shap_values FROM dwh_monitoring_shap.customer_shap_values WHERE run_index = :run_index",
            {"run_index": run_index},
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_customer_ids_for_run(run_index: int) -> list:
    try:
        df = _read_sql(
            "SELECT DISTINCT customer_id FROM dwh_monitoring_shap.customer_shap_values WHERE run_index = :run_index ORDER BY customer_id",
            {"run_index": run_index},
        )
        return df["customer_id"].tolist()
    except Exception:
        return []


@st.cache_data(ttl=60)
def get_customer_shap(run_index: int, customer_id: str) -> pd.DataFrame:
    try:
        return _read_sql(
            """SELECT customer_id, record_id, shap_values, base_value, predicted_probability
               FROM dwh_monitoring_shap.customer_shap_values
               WHERE run_index = :run_index AND customer_id = :customer_id
               ORDER BY computed_at DESC LIMIT 1""",
            {"run_index": run_index, "customer_id": customer_id},
        )
    except Exception:
        return pd.DataFrame()


def get_flask_info() -> dict | None:
    try:
        r = requests.get(f"{FLASK_ENDPOINT}/model-info", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


@st.cache_data(ttl=30)
def get_production_primary_metric() -> str:
    """Read primary_metric tag from the current production model version in MLflow."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, "production")
        tag = mv.tags.get("primary_metric", DEFAULT_PRIMARY)
        return tag if tag in REGISTRY else DEFAULT_PRIMARY
    except Exception:
        return DEFAULT_PRIMARY


@st.cache_data(ttl=60)
def get_latest_summary(primary_metric: str = DEFAULT_PRIMARY) -> dict:
    """Quick summary of latest monitoring results for the overview section."""
    out = {}
    db_col = _METRIC_DB_COL.get(primary_metric, "roc_auc")
    try:
        h = _read_sql(
            f"SELECT roc_auc, {db_col} AS primary_value, primary_metric, f1_score, run_index "
            f"FROM dwh_monitoring_hard.results ORDER BY evaluated_at DESC LIMIT 1"
        )
        if not h.empty:
            out["hard"] = h.iloc[0].to_dict()
        d = _read_sql("SELECT drift_score, drift_detected, num_drifted_features, run_index FROM dwh_monitoring_drift.results ORDER BY evaluated_at DESC LIMIT 1")
        if not d.empty:
            out["drift"] = d.iloc[0].to_dict()
        s = _read_sql("SELECT top_feature, n_records, run_index FROM dwh_monitoring_shap.results ORDER BY evaluated_at DESC LIMIT 1")
        if not s.empty:
            out["shap"] = s.iloc[0].to_dict()
    except Exception:
        pass
    return out


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _plotly_base(height: int = 320) -> dict:
    return dict(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=30, r=20, t=65, b=55),
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        legend=dict(orientation="h", y=-0.22, x=0),
    )


_HOME_OWNERSHIP_LABELS = {0: "RENT", 1: "MORTGAGE", 2: "OWN", 3: "OTHER"}

def make_distribution_figure(feature: str, ref_series: pd.Series | None, curr_series: pd.Series) -> go.Figure:
    ks_stat = None
    if HAS_SCIPY and ref_series is not None and len(ref_series) > 0:
        ks_stat, _ = ks_2samp(ref_series.dropna(), curr_series.dropna())

    ks_label = f" · KS={ks_stat:.3f}" if ks_stat is not None else ""
    drift_color = PALETTE["danger"] if (ks_stat or 0) > 0.2 else PALETTE["success"]

    all_vals = pd.concat([s for s in [ref_series, curr_series] if s is not None and len(s) > 0])
    is_discrete = all_vals.nunique() <= 15

    fig = go.Figure()

    if is_discrete:
        all_cats = sorted(all_vals.dropna().unique())
        if feature == "home_ownership_encoded":
            labels = [_HOME_OWNERSHIP_LABELS.get(int(v), str(v)) for v in all_cats]
        else:
            labels = [str(v) for v in all_cats]

        def _norm_counts(s):
            vc = s.value_counts()
            total = len(s)
            return [(vc.get(v, 0) / total) for v in all_cats]

        if ref_series is not None and len(ref_series) > 0:
            fig.add_trace(go.Bar(
                x=labels, y=_norm_counts(ref_series),
                name="Reference", opacity=0.7,
                marker_color=PALETTE["reference"],
            ))
        fig.add_trace(go.Bar(
            x=labels, y=_norm_counts(curr_series),
            name="Current", opacity=0.85,
            marker_color=PALETTE["current"],
        ))
        fig.update_layout(
            **_plotly_base(270),
            title=dict(
                text=f"<b>{feature}</b><span style='color:{drift_color};font-size:11px'>{ks_label}</span>",
                font=dict(size=13),
            ),
            barmode="group",
            xaxis_title=feature,
            yaxis_title="Proportion",
            xaxis=dict(type="category"),
        )
    else:
        if ref_series is not None and len(ref_series) > 0:
            fig.add_trace(go.Histogram(
                x=ref_series, name="Reference", histnorm="probability density",
                opacity=0.55, marker_color=PALETTE["reference"], nbinsx=30,
            ))
        fig.add_trace(go.Histogram(
            x=curr_series, name="Current", histnorm="probability density",
            opacity=0.65, marker_color=PALETTE["current"], nbinsx=30,
        ))
        fig.update_layout(
            **_plotly_base(270),
            title=dict(
                text=f"<b>{feature}</b><span style='color:{drift_color};font-size:11px'>{ks_label}</span>",
                font=dict(size=13),
            ),
            barmode="overlay",
            xaxis_title=feature,
            yaxis_title="Density",
        )

    return fig


def make_confusion_matrix_fig(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred)
    labels = ["No Default", "Default"]
    total  = cm.sum()
    text   = [[f"{cm[i,j]}<br>({cm[i,j]/total*100:.1f}%)" for j in range(2)] for i in range(2)]

    # Heatmap via plotly for consistent styling
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Pred: {l}" for l in labels],
        y=[f"Actual: {l}" for l in labels],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=15, color="white"),
        colorscale=[[0, "#dbeafe"], [1, "#1d4ed8"]],
        showscale=False,
    ))
    fig.update_layout(
        **_plotly_base(340),
        title=dict(text="<b>Confusion Matrix</b>", font=dict(size=14)),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def make_roc_fig(y_true: np.ndarray, y_score: np.ndarray) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc_val:.4f})",
        line=dict(color=PALETTE["danger"], width=2.5),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="No skill",
        line=dict(color="#94a3b8", dash="dash", width=1.5),
    ))
    fig.update_layout(
        **_plotly_base(340),
        title=dict(text=f"<b>ROC Curve</b>  ·  AUC = {auc_val:.4f}", font=dict(size=14)),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    return fig


def make_pr_fig(y_true: np.ndarray, y_score: np.ndarray) -> go.Figure:
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap       = average_precision_score(y_true, y_score)
    baseline = float(y_true.mean())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rec, y=prec, mode="lines", name=f"PR (AP = {ap:.4f})",
        line=dict(color=PALETTE["success"], width=2.5),
        fill="tozeroy", fillcolor="rgba(34,197,94,0.08)",
    ))
    fig.add_hline(y=baseline, line_dash="dash", line_color="#94a3b8",
                  annotation_text=f"Baseline ({baseline:.3f})")
    fig.update_layout(
        **_plotly_base(340),
        title=dict(text=f"<b>Precision-Recall Curve</b>  ·  AP = {ap:.4f}", font=dict(size=14)),
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    return fig


def make_calibration_fig(y_true: np.ndarray, y_score: np.ndarray) -> go.Figure:
    fig = go.Figure()
    if HAS_CALIBRATION:
        frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10)
        fig.add_trace(go.Scatter(
            x=mean_pred, y=frac_pos, mode="lines+markers", name="Model",
            line=dict(color=PALETTE["purple"], width=2.5),
            marker=dict(size=8, symbol="square"),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
        line=dict(color="#94a3b8", dash="dash", width=1.5),
    ))
    fig.update_layout(
        **_plotly_base(340),
        title=dict(text="<b>Calibration Plot</b>", font=dict(size=14)),
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    return fig


def make_metric_trend_fig(hist: pd.DataFrame) -> go.Figure:
    retrain_runs = hist[hist.get("retraining_triggered", False) == True]["run_index"].tolist() \
        if "retraining_triggered" in hist.columns else []

    fig = go.Figure()
    series = [
        ("roc_auc",         "ROC-AUC",   PALETTE["danger"]),
        ("f1_score",        "F1 Score",  PALETTE["success"]),
        ("accuracy",        "Accuracy",  PALETTE["primary"]),
        ("precision_score", "Precision", PALETTE["purple"]),
        ("recall_score",    "Recall",    PALETTE["warning"]),
    ]
    for col, name, color in series:
        if col in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist["run_index"], y=hist[col],
                mode="lines+markers", name=name,
                line=dict(color=color, width=2),
                marker=dict(size=6),
            ))

    for ri in retrain_runs:
        fig.add_vline(x=ri, line_dash="dot", line_color="#94a3b8",
                      annotation_text="retrain", annotation_position="top right",
                      annotation_font_size=10)

    fig.add_hline(y=MIN_ROC_AUC, line_dash="dash", line_color=PALETTE["danger"],
                  annotation_text=f"Retrain threshold ({MIN_ROC_AUC:.2f})", annotation_font_size=10)

    fig.update_layout(
        **_plotly_base(360),
        title=dict(text="<b>Metric Trends Across Runs</b>", font=dict(size=14)),
        xaxis_title="Run Index",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def make_drift_trend_fig(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["run_index"], y=hist["drift_score"],
        mode="lines+markers", name="Drift Score",
        line=dict(color=PALETTE["warning"], width=2),
        marker=dict(size=6),
    ))
    fig.update_layout(
        **_plotly_base(320),
        title=dict(text="<b>Drift Score Over Runs (Mean KS)</b>", font=dict(size=14)),
        xaxis_title="Run Index",
        yaxis_title="Mean KS Statistic",
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def make_shap_bar_fig(fi: dict) -> go.Figure:
    fi_df = pd.DataFrame.from_dict(fi, orient="index", columns=["mean_abs_shap"])
    fi_df = fi_df.sort_values("mean_abs_shap")

    colors = [
        PALETTE["primary"] if v > fi_df["mean_abs_shap"].median() else "#93c5fd"
        for v in fi_df["mean_abs_shap"]
    ]

    fig = go.Figure(go.Bar(
        x=fi_df["mean_abs_shap"],
        y=fi_df.index,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in fi_df["mean_abs_shap"]],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        **_plotly_base(380),
        title=dict(text="<b>Mean |SHAP| - Feature Importance</b>", font=dict(size=14)),
        xaxis_title="Mean |SHAP value|",
        yaxis=dict(autorange=True),
    )
    fig.update_layout(margin=dict(l=140, r=60, t=50, b=30))
    return fig


def make_shap_outcome_fig(outcome_df: pd.DataFrame) -> go.Figure:
    df = outcome_df.copy()
    df["outcome"] = df.apply(
        lambda r: (
            "TP" if r["default_flag_predicted"] == 1 and r["actual_default_flag"] == 1
            else "FP" if r["default_flag_predicted"] == 1 and r["actual_default_flag"] == 0
            else "TN" if r["default_flag_predicted"] == 0 and r["actual_default_flag"] == 0
            else "FN"
        ), axis=1,
    )

    rows = []
    for _, row in df.iterrows():
        sv = row["shap_values"] if isinstance(row["shap_values"], dict) else json.loads(row["shap_values"])
        for feat in FEATURE_COLUMNS:
            rows.append({"feature": feat, "outcome": row["outcome"], "abs_shap": abs(sv.get(feat, 0.0))})

    if not rows:
        return go.Figure()

    agg = pd.DataFrame(rows).groupby(["outcome", "feature"])["abs_shap"].mean().reset_index()
    counts = df["outcome"].value_counts().to_dict()

    fig = go.Figure()
    for outcome in ["TP", "FP", "TN", "FN"]:
        sub = agg[agg["outcome"] == outcome].sort_values("feature")
        if sub.empty:
            continue
        n = counts.get(outcome, 0)
        fig.add_trace(go.Bar(
            name=f"{outcome} (n={n})",
            x=sub["feature"],
            y=sub["abs_shap"],
            marker_color=PALETTE[outcome],
        ))

    fig.update_layout(
        **_plotly_base(400),
        title=dict(text="<b>Mean |SHAP| by Prediction Outcome</b>", font=dict(size=14)),
        barmode="group",
        xaxis_title="Feature",
        yaxis_title="Mean |SHAP|",
    )
    return fig


def make_beeswarm_figure(run_index: int) -> plt.Figure | None:
    df = get_beeswarm_data(run_index)
    if df.empty:
        return None
    rows = []
    for sv in df["shap_values"]:
        d = sv if isinstance(sv, dict) else json.loads(sv)
        rows.append([d.get(f, 0.0) for f in FEATURE_COLUMNS])
    shap_matrix = np.array(rows)
    X_df = pd.DataFrame(shap_matrix, columns=FEATURE_COLUMNS)
    plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_matrix, X_df, feature_names=FEATURE_COLUMNS, show=False, max_display=10)
    plt.title(f"SHAP Summary - Run {run_index}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return plt.gcf()


def make_waterfall_figure(row: pd.Series) -> go.Figure:
    shap_dict = row["shap_values"] if isinstance(row["shap_values"], dict) else json.loads(row["shap_values"])
    base = float(row["base_value"] or 0.0)
    pred = float(row["predicted_probability"]) if row["predicted_probability"] is not None else None

    features = list(shap_dict.keys())
    values   = np.array([shap_dict[f] for f in features])
    idx      = np.argsort(np.abs(values))[::-1]
    features = [features[i] for i in idx]
    values   = values[idx]

    cumulative = np.concatenate([[base], base + np.cumsum(values)])
    colors     = [PALETTE["danger"] if v > 0 else PALETTE["primary"] for v in values]

    fig = go.Figure()
    for i, (feat, val, col) in enumerate(zip(features, values, colors)):
        left = cumulative[i]
        sign = "+" if val >= 0 else ""
        fig.add_trace(go.Bar(
            x=[val], y=[feat],
            base=[left],
            orientation="h",
            marker_color=col,
            name=f"{sign}{val:.4f}",
            showlegend=False,
            text=[f"{sign}{val:.4f}"],
            textposition="outside",
            textfont=dict(size=10),
        ))

    if pred is not None:
        fig.add_vline(x=pred, line_dash="dot", line_color="#1e293b",
                      annotation_text=f"Prediction: {pred:.4f}",
                      annotation_position="top")
    fig.add_vline(x=base, line_dash="dash", line_color="#94a3b8",
                  annotation_text=f"Base: {base:.4f}",
                  annotation_position="bottom")

    cid = str(row.get("customer_id", ""))[:20]
    fig.update_layout(
        **_plotly_base(400),
        title=dict(text=f"<b>SHAP Waterfall</b>  ·  {cid}…", font=dict(size=14)),
        xaxis_title="SHAP contribution (probability space)",
        yaxis=dict(autorange=True),
    )
    fig.update_layout(margin=dict(l=150, r=60, t=60, b=30))
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📊 Monitoring Dashboard")
    st.divider()

    st.markdown("**Service Status**")

    # Flask
    flask_info = get_flask_info()
    if flask_info:
        st.markdown(
            f'<div class="svc-row"><div class="svc-dot-up"></div>'
            f'<span class="svc-name">Flask API</span>'
            f'<span class="svc-detail"> · v{flask_info.get("model_version","?")} · {flask_info.get("model_alias","production")}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="svc-row"><div class="svc-dot-down"></div>'
            '<span class="svc-name">Flask API</span>'
            '<span class="svc-detail"> · unreachable</span></div>',
            unsafe_allow_html=True,
        )

    # MLflow
    try:
        requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=3).raise_for_status()
        mlflow_ok = True
    except Exception:
        mlflow_ok = False
    st.markdown(
        f'<div class="svc-row"><div class="svc-dot-{"up" if mlflow_ok else "down"}"></div>'
        f'<span class="svc-name">MLflow</span>'
        f'<span class="svc-detail"> · {"ready" if mlflow_ok else "unreachable"}</span></div>',
        unsafe_allow_html=True,
    )

    # PostgreSQL
    try:
        with db_conn() as _c:
            with _c.cursor() as _cur:
                _cur.execute("SELECT COUNT(*) FROM dwh_history.run_registry")
                _n = _cur.fetchone()[0]
        pg_ok, pg_detail = True, f"{_n} runs"
    except Exception:
        pg_ok, pg_detail = False, "unreachable"
    st.markdown(
        f'<div class="svc-row"><div class="svc-dot-{"up" if pg_ok else "down"}"></div>'
        f'<span class="svc-name">PostgreSQL</span>'
        f'<span class="svc-detail"> · {pg_detail}</span></div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("🎛️ [Control Panel](http://localhost:8501)")
    st.caption("🔗 [MLflow UI](http://localhost:5000)")
    st.caption("🔗 [Airflow UI](http://localhost:8080)")

    st.divider()
    if st.button("↻ Refresh All", width="stretch"):
        st.cache_data.clear()
        for _k in ["drift_run_sel", "hard_run_sel", "shap_run_sel"]:
            st.session_state.pop(_k, None)
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────

col_h, col_sub = st.columns([3, 1])
with col_h:
    st.title("📊 MLOps Monitoring Dashboard")
with col_sub:
    st.markdown("")
    if flask_info:
        st.caption(f"Serving model **v{flask_info.get('model_version','?')}** · alias `{flask_info.get('model_alias','production')}`")

# ── Quick Overview ────────────────────────────────────────────────────────────

_prod_primary_metric = get_production_primary_metric()
_prod_primary_label  = REGISTRY.get(_prod_primary_metric, _prod_primary_metric.upper())
summary = get_latest_summary(_prod_primary_metric)
if summary:
    st.markdown("##### Latest Monitoring Results")
    ov1, ov2, ov3 = st.columns(3)
    with ov1:
        with st.container(border=True):
            h = summary.get("hard", {})
            primary_val = h.get("primary_value") or h.get("roc_auc")
            st.markdown("📈 **Hard Metrics**")
            st.metric(_prod_primary_label, f"{primary_val:.4f}" if primary_val else "-")
            st.caption(f"F1: {h.get('f1_score', 0):.4f}  ·  Run {h.get('run_index','-')}" if h else "No data")
    with ov2:
        with st.container(border=True):
            d = summary.get("drift", {})
            ds = d.get("drift_score")
            detected = d.get("drift_detected", False)
            badge = '<span class="badge-warn">⚠ Drift Detected</span>' if detected else '<span class="badge-ok">✓ No Drift</span>'
            st.markdown("📊 **Data Drift**")
            st.metric("Drift Score", f"{ds:.4f}" if ds is not None else "-")
            st.markdown(badge, unsafe_allow_html=True)
            st.caption(f"Run {d.get('run_index','-')}" if d else "No data")
    with ov3:
        with st.container(border=True):
            s = summary.get("shap", {})
            st.markdown("🔍 **Explainability**")
            st.metric("Top Feature", s.get("top_feature", "-") if s else "-")
            st.caption(f"{s.get('n_records','-')} records  ·  Run {s.get('run_index','-')}" if s else "No data")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_drift, tab_hard, tab_shap = st.tabs(
    ["📊  Data Drift", "📈  Hard Metrics", "🔍  Explainability"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 - Data Drift
# ════════════════════════════════════════════════════════════════════════════
with tab_drift:
    drift_runs = get_drift_runs()

    if not drift_runs:
        st.markdown(
            '<div class="empty-state">No drift results yet.<br>'
            'Run <b>Data Drift monitoring (dag_04b)</b> on a batch inference run from the Control Panel.</div>',
            unsafe_allow_html=True,
        )
    else:
        sel_drift_run = st.selectbox(
            "Run index", drift_runs, key="drift_run_sel",
            format_func=lambda x: f"Run {x}",
        )

        result = get_drift_result(sel_drift_run)

        # ── Summary metrics ──────────────────────────────────────────────────────
        if result:
            drift_det  = result.get("drift_detected", False)
            drift_scr  = result.get("drift_score", 0)
            num_drifted = int(result.get("num_drifted_features", 0))
            n_records  = int(result.get("n_records", 0))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Drift Score", f"{drift_scr:.4f}",
                      delta=f"threshold {MAX_DRIFT_FEATURE_FRACTION:.2f}", delta_color="off")
            m2.metric("Drifted Features", num_drifted,
                      delta=f"of {len(FEATURE_COLUMNS)}", delta_color="off")
            m3.metric("Status",
                      "Drift Detected ⚠" if drift_det else "No Drift ✓",
                      delta_color="off")
            m4.metric("Records in Run", f"{n_records:,}")

            drifted = result.get("drifted_feature_names", "")
            if drifted:
                names = drifted if isinstance(drifted, list) else [s.strip() for s in str(drifted).split(",") if s.strip()]
                if names:
                    if drift_det:
                        st.warning(f"⚠ Drifted features: **{', '.join(names)}**")
                    else:
                        st.info(f"ℹ {len(names)} feature(s) show individual drift but below overall threshold: **{', '.join(names)}**")
        else:
            st.warning("Could not load drift result for this run.")

        # ── Feature distributions ────────────────────────────────────────────────
        with st.spinner("Loading feature distributions…"):
            curr_df = get_current_features(sel_drift_run)
            ref_df  = get_reference_features()

        if ref_df is None:
            st.info("Reference dataset not available from MLflow - showing current distribution only.")

        if not curr_df.empty:
            with st.expander("📋 Dataset Statistics", expanded=False):
                rows = []
                for feat in FEATURE_COLUMNS:
                    row = {"Feature": feat}
                    if ref_df is not None and feat in ref_df.columns:
                        s = ref_df[feat].describe()
                        row.update({"Ref Mean": f"{s['mean']:.3f}", "Ref Std": f"{s['std']:.3f}",
                                    "Ref Min": f"{s['min']:.3f}", "Ref Max": f"{s['max']:.3f}"})
                    else:
                        row.update({"Ref Mean": "-", "Ref Std": "-", "Ref Min": "-", "Ref Max": "-"})
                    if feat in curr_df.columns:
                        s = curr_df[feat].describe()
                        row.update({"Curr Mean": f"{s['mean']:.3f}", "Curr Std": f"{s['std']:.3f}",
                                    "Curr Min": f"{s['min']:.3f}", "Curr Max": f"{s['max']:.3f}"})
                    else:
                        row.update({"Curr Mean": "-", "Curr Std": "-", "Curr Min": "-", "Curr Max": "-"})
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows).set_index("Feature"), width="stretch")

            st.markdown("##### Feature Distribution Comparison")
            st.caption(
                f"Grey = Reference (training data, {len(ref_df):,} records)  ·  "
                f"Orange = Current run (run {sel_drift_run}, {len(curr_df):,} records)  ·  "
                "Normalized to probability density  ·  KS statistic shown in title"
                if ref_df is not None else
                f"Orange = Current run (run {sel_drift_run}, {len(curr_df):,} records)"
            )
            feat_pairs = [FEATURE_COLUMNS[i:i+2] for i in range(0, len(FEATURE_COLUMNS), 2)]
            for pair in feat_pairs:
                cols = st.columns(2)
                for ci, feat in enumerate(pair):
                    if feat not in curr_df.columns:
                        continue
                    ref_s = ref_df[feat] if (ref_df is not None and feat in ref_df.columns) else None
                    cols[ci].plotly_chart(
                        make_distribution_figure(feat, ref_s, curr_df[feat]),
                        width="stretch",
                    )
        else:
            st.markdown(
                '<div class="empty-state">No inference data for this run.<br>Run batch inference first.</div>',
                unsafe_allow_html=True,
            )

        # ── Drift trend ──────────────────────────────────────────────────────────
        st.markdown("##### Drift Trend")
        drift_hist = get_drift_history(50)
        if len(drift_hist) >= 2:
            st.plotly_chart(make_drift_trend_fig(drift_hist), width="stretch")
        else:
            st.info("Need at least 2 runs to show a trend.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 - Hard Metrics
# ════════════════════════════════════════════════════════════════════════════
with tab_hard:
    hard_runs = get_hard_metric_runs()

    if not hard_runs:
        st.markdown(
            '<div class="empty-state">No hard metrics yet.<br>'
            'Run <b>Hard Metrics monitoring (dag_04a)</b> on a batch inference run from the Control Panel.</div>',
            unsafe_allow_html=True,
        )
    else:
        sel_hard_run = st.selectbox(
            "Run index", hard_runs, key="hard_run_sel",
            format_func=lambda x: f"Run {x}",
        )

        m    = get_hard_metrics_row(sel_hard_run)
        hist = get_hard_metrics_history(50)

        # ── Metric cards ─────────────────────────────────────────────────────────
        if m:
            prev = {}
            if not hist.empty:
                earlier = hist[hist["run_index"] < sel_hard_run]
                if not earlier.empty:
                    prev = earlier.iloc[-1].to_dict()

            def _delta(key):
                if prev and key in prev:
                    try:
                        return round(float(m[key]) - float(prev[key]), 4)
                    except (TypeError, ValueError):
                        return None
                return None

            # Let user interactively choose the starred metric; default to run's stored value
            _stored_primary = m.get("primary_metric") or _prod_primary_metric
            _metric_keys = list(REGISTRY.keys())
            run_primary = st.selectbox(
                "★ Primary metric",
                options=_metric_keys,
                index=_metric_keys.index(_stored_primary) if _stored_primary in _metric_keys else 0,
                format_func=lambda k: REGISTRY[k],
                key=f"hard_primary_sel_{sel_hard_run}",
                help="Controls which metric card gets the ★ star.",
            )

            # All metrics with their DB column names and labels
            _all_metrics = [
                ("ROC-AUC",   "roc_auc"),
                ("PR-AUC",    "pr_auc"),
                ("F1",        "f1_score"),
                ("Precision", "precision_score"),
                ("Recall",    "recall_score"),
                ("Accuracy",  "accuracy"),
            ]

            cols = st.columns(len(_all_metrics))
            for col, (label, db_key) in zip(cols, _all_metrics):
                val = m.get(db_key)
                is_primary = (_METRIC_DB_COL.get(run_primary, "roc_auc") == db_key)
                display_label = f"★ {label}" if is_primary else label
                col.metric(
                    display_label,
                    f"{val:.4f}" if val is not None else "-",
                    delta=_delta(db_key),
                )

            if m.get("retraining_triggered"):
                st.warning(f"⚠ Retraining was triggered for this run: {m.get('retraining_reason', '')}")
        else:
            st.warning("Could not load metrics for this run.")

        # ── Charts ───────────────────────────────────────────────────────────────
        raw_df = get_raw_predictions(sel_hard_run)
        if raw_df.empty:
            st.info(
                "No raw prediction data for this run. "
                "Ground truth is populated during batch inference - ensure the inference run completed."
            )
        else:
            y_true  = raw_df["actual_default_flag"].values.astype(int)
            y_pred  = raw_df["default_flag_predicted"].values.astype(int)
            y_score = raw_df["default_probability"].values.astype(float)

            # Row 1: Confusion Matrix | ROC
            col_l, col_r = st.columns(2)
            with col_l:
                st.plotly_chart(make_confusion_matrix_fig(y_true, y_pred), width="stretch")
            with col_r:
                st.plotly_chart(make_roc_fig(y_true, y_score), width="stretch")

            # Row 2: PR | Calibration
            col_l2, col_r2 = st.columns(2)
            with col_l2:
                st.plotly_chart(make_pr_fig(y_true, y_score), width="stretch")
            with col_r2:
                st.plotly_chart(make_calibration_fig(y_true, y_score), width="stretch")

        # ── Metric trends ─────────────────────────────────────────────────────────
        st.markdown("##### Metric Trends")
        if not hist.empty and len(hist) > 1:
            st.plotly_chart(make_metric_trend_fig(hist), width="stretch")
        elif not hist.empty:
            st.info("Need at least 2 runs to show a trend.")
        else:
            st.info("No metric history available.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 - Explainability
# ════════════════════════════════════════════════════════════════════════════
with tab_shap:
    shap_runs = get_shap_runs()

    if not shap_runs:
        st.markdown(
            '<div class="empty-state">No SHAP results yet.<br>'
            'Run <b>SHAP Explainability monitoring (dag_04c)</b> on a batch inference run from the Control Panel.</div>',
            unsafe_allow_html=True,
        )
    else:
        sel_shap_run = st.selectbox(
            "Run index", shap_runs, key="shap_run_sel",
            format_func=lambda x: f"Run {x}",
        )

        agg = get_shap_aggregate(sel_shap_run)

        if agg:
            ai1, ai2, ai3 = st.columns(3)
            ai1.metric("Top Feature",   agg.get("top_feature", "-"))
            ai2.metric("Explainer",     agg.get("explainer_type", "-"))
            ai3.metric("Records",       f"{agg.get('n_records', 0):,}")

        # ── Beeswarm + Bar ────────────────────────────────────────────────────────
        col_bee, col_bar = st.columns(2)

        with col_bee:
            st.markdown("##### SHAP Beeswarm")
            st.caption("Each dot = one prediction. Red = high feature value, blue = low.")
            with st.spinner("Building beeswarm…"):
                bee_fig = make_beeswarm_figure(sel_shap_run)
            if bee_fig:
                st.pyplot(bee_fig, width="stretch")
                plt.close(bee_fig)
            else:
                st.markdown(
                    '<div class="empty-state">No beeswarm data for this run.</div>',
                    unsafe_allow_html=True,
                )

        with col_bar:
            st.markdown("##### Mean |SHAP| Importance")
            st.caption("Higher = feature has larger average impact on predictions.")
            if agg.get("feature_importances"):
                st.plotly_chart(make_shap_bar_fig(agg["feature_importances"]), width="stretch")
            else:
                st.markdown(
                    '<div class="empty-state">No aggregate SHAP data for this run.</div>',
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Outcome-grouped impact ────────────────────────────────────────────────
        st.markdown("##### Feature Impact by Prediction Outcome")
        st.caption(
            "Mean |SHAP| per feature, grouped by TP / FP / TN / FN. "
            "Requires both SHAP (dag_04c) and Hard Metrics (dag_04a) on the same run."
        )
        with st.spinner("Loading outcome-grouped SHAP…"):
            outcome_df = get_shap_with_outcomes(sel_shap_run)

        if not outcome_df.empty:
            outcome_df["outcome"] = outcome_df.apply(
                lambda r: (
                    "TP" if r["default_flag_predicted"] == 1 and r["actual_default_flag"] == 1
                    else "FP" if r["default_flag_predicted"] == 1 and r["actual_default_flag"] == 0
                    else "TN" if r["default_flag_predicted"] == 0 and r["actual_default_flag"] == 0
                    else "FN"
                ), axis=1,
            )
            counts = outcome_df["outcome"].value_counts().to_dict()

            # Outcome count pills
            oc1, oc2, oc3, oc4 = st.columns(4)
            for col, label, color in [
                (oc1, "TP - Correct Default",    PALETTE["TP"]),
                (oc2, "FP - False Alarm",        PALETTE["FP"]),
                (oc3, "TN - Correct Non-Default", PALETTE["TN"]),
                (oc4, "FN - Missed Default",     PALETTE["FN"]),
            ]:
                short = label.split(" - ")[0]
                n = counts.get(short, 0)
                col.markdown(
                    f'<div style="text-align:center;padding:0.5rem;background:{color}22;'
                    f'border-left:4px solid {color};border-radius:6px;">'
                    f'<div style="font-size:1.4rem;font-weight:700;color:{color}">{n}</div>'
                    f'<div style="font-size:0.72rem;color:#64748b">{label}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")
            st.plotly_chart(make_shap_outcome_fig(outcome_df), width="stretch")
        else:
            st.info(
                "No outcome-grouped data found. "
                "SHAP (dag_04c) and Hard Metrics (dag_04a) must both be run on the same run_index."
            )

        st.divider()

        # ── Per-customer waterfall ────────────────────────────────────────────────
        with st.expander("🔎 Per-Customer SHAP Waterfall", expanded=False):
            st.caption("Shows how each feature pushed a specific customer's default probability up or down.")
            customer_ids = get_customer_ids_for_run(sel_shap_run)
            if customer_ids:
                sel_cust = st.selectbox(
                    "Customer ID", customer_ids, key="shap_customer",
                    format_func=lambda x: x[:32] + "…" if len(str(x)) > 32 else str(x),
                )
                cust_df = get_customer_shap(sel_shap_run, sel_cust)
                if not cust_df.empty:
                    row = cust_df.iloc[0]
                    st.plotly_chart(make_waterfall_figure(row), width="stretch")
                    pred = row["predicted_probability"]
                    if pred is not None:
                        risk_color = PALETTE["danger"] if float(pred) > 0.5 else PALETTE["success"]
                        st.markdown(
                            f'<span style="color:{risk_color};font-weight:700;font-size:1.1rem">'
                            f'Default Probability: {float(pred):.3f}</span>  '
                            f'<span style="color:#64748b;font-size:0.8rem">record: {str(row["record_id"])[:24]}…</span>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.warning("No SHAP data for this customer in this run.")
            else:
                st.info("No per-customer SHAP values stored for this run.")
