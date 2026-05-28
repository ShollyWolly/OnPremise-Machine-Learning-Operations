#!/bin/bash
# =============================================================================
# Platform Init Script
# =============================================================================
# Runs once at startup (via platform-init container).
# Idempotent — safe to run on repeated docker compose up.
#
# Steps:
#   1. Migrate Airflow DB
#   2. Create Airflow admin user (skip if exists)
#   3. Fix data volume permissions
#   4. Wait for MLflow to be ready
#   5. Bootstrap: generate data + process + train initial model
#      (skipped if a Production model already exists)
# =============================================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║         MLOps Platform Initialization        ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ---------------------------------------------------------------------------
# 1. Airflow DB migration
# ---------------------------------------------------------------------------
echo "[1/5] Migrating Airflow database..."
airflow db migrate
echo "      Done."

echo "[1b] Creating Airflow pools..."
airflow pools set training_pool 1 "Dedicated slot for model retraining — prevents resource contention" \
    2>&1 || echo "      Pool already exists — continuing."
echo "      Done."

echo "[1c] Deduplicating monitoring tables (keep latest row per run_index)..."
python - <<'PYEOF'
import os, psycopg2

conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST", "postgres"),
    port=int(os.getenv("POSTGRES_PORT", 5432)),
    dbname=os.getenv("POSTGRES_DB", "mlops"),
    user=os.getenv("POSTGRES_USER", "mlops_user"),
    password=os.getenv("POSTGRES_PASSWORD", ""),
)
tables = [
    "dwh_monitoring_hard.results",
    "dwh_monitoring_drift.results",
    "dwh_monitoring_shap.results",
]
with conn:
    with conn.cursor() as cur:
        for tbl in tables:
            cur.execute(f"""
                DELETE FROM {tbl}
                WHERE id NOT IN (
                    SELECT DISTINCT ON (run_index) id
                    FROM {tbl}
                    ORDER BY run_index, evaluated_at DESC NULLS LAST
                )
            """)
            print(f"  {tbl}: {cur.rowcount} duplicate(s) removed")
conn.close()
PYEOF
echo "      Done."

# ---------------------------------------------------------------------------
# 2. Create Airflow admin user (idempotent)
# ---------------------------------------------------------------------------
echo "[2/5] Creating Airflow admin user..."
airflow users create \
    --username "${AIRFLOW_ADMIN_USER:-admin}" \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email "${AIRFLOW_ADMIN_EMAIL:-admin@mlops.local}" \
    --password "${AIRFLOW_ADMIN_PASSWORD:-admin}" \
    2>&1 || echo "      User may already exist — continuing."

# ---------------------------------------------------------------------------
# 3. Volume permissions — handled by volume-init service (runs as root)
# ---------------------------------------------------------------------------
echo "[3/5] Volume permissions pre-set by volume-init — skipping."

# ---------------------------------------------------------------------------
# 4. Wait for MLflow
# ---------------------------------------------------------------------------
echo "[4/5] Waiting for MLflow tracking server..."
ATTEMPTS=0
until python -c "import urllib.request; urllib.request.urlopen('http://mlflow:5000/health')" 2>/dev/null; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if [ $ATTEMPTS -ge 30 ]; then
        echo "ERROR: MLflow did not become ready after 150s"
        exit 1
    fi
    echo "      Not ready yet (attempt ${ATTEMPTS}/30), retrying in 5s..."
    sleep 5
done
echo "      MLflow is ready."

# ---------------------------------------------------------------------------
# 5. Bootstrap: initial data + model (idempotent)
# ---------------------------------------------------------------------------
echo "[5/5] Checking model registry..."

HAS_MODEL=$(python -c "
import mlflow, sys
try:
    mlflow.set_tracking_uri('http://mlflow:5000')
    client = mlflow.MlflowClient()
    client.get_model_version_by_alias('${MODEL_REGISTRY_NAME:-credit-risk-classifier}', 'production')
    print('yes')
except Exception:
    print('no')
" 2>/dev/null)

if [ "$HAS_MODEL" = "yes" ]; then
    echo "      Production model found — skipping bootstrap."
else
    echo "      No Production model found — running bootstrap..."
    echo ""

    echo "  [5a] Generating initial training data (3000 stable records)..."
    cd /opt/mlops/services/data_generator && \
        python generator.py \
            --mode stable \
            --n-records 3000 \
            --seed 42 \
            --output-dir "${DATA_RAW_PATH:-/data/raw}"
    echo "       Done."

    echo "  [5b] Processing pipeline (clean + encode + validate)..."
    cd /opt/mlops/services/processing_pipeline && \
        python pipeline.py
    echo "       Done."

    echo "  [5c] Training initial model and promoting to Production..."
    cd /opt/mlops/training && \
        python train.py --promote
    echo "       Done."

    echo ""
    echo "  Bootstrap complete."
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║         Platform Init Complete ✓             ║"
echo "║                                              ║"
echo "║  Airflow UI  : http://localhost:8080         ║"
echo "║  MLflow UI   : http://localhost:5000         ║"
echo "║  Control Panel: http://localhost:8501        ║"
echo "║  Dashboard   : http://localhost:8502         ║"
echo "║  Flask API   : http://localhost:5001         ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
