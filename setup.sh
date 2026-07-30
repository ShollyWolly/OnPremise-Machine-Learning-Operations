#!/bin/bash
# =============================================================================
# MLOps Platform Setup
# =============================================================================
# Run once before the first `docker compose up -d` (safe to re-run any time -
# every step below is idempotent). Replaces the old volume-init/platform-init
# one-shot containers with an explicit host-side script:
#
#   1. Prep bind-mounted data dirs (was: volume-init)
#   2. Bring up postgres + mlflow, wait for healthy
#   3. Migrate Airflow DB, create admin user, bootstrap initial model
#      via `docker compose run` against airflow/init.sh (was: platform-init)
#   4. Bring up everything else
# =============================================================================

set -e

# On Windows Git Bash/MSYS, absolute-looking args like /opt/mlops/init.sh get silently
# rewritten to a Windows path before reaching `docker`; harmless no-op elsewhere.
export MSYS_NO_PATHCONV=1

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f .env ]; then
    echo "ERROR: .env not found. Run: cp .env.example .env   (then edit it)"
    exit 1
fi

echo ""
echo "[1/4] Preparing data directories..."
mkdir -p \
    data/raw data/processed data/predictions \
    data/monitoring/hard data/monitoring/drift data/monitoring/shap \
    volumes/mlflow-artifacts volumes/airflow-logs volumes/postgres-data
# Best-effort - required on Linux/Mac so the uid-50000 containers can write
# to these dirs; a no-op on Windows/NTFS bind mounts.
chmod -R 775 data volumes/mlflow-artifacts volumes/airflow-logs 2>/dev/null || true
echo "      Done."

echo ""
echo "[2/4] Starting postgres + mlflow..."
docker compose up -d --build postgres mlflow

wait_for_healthy() {
    local service="$1"
    local container="mlops-${service}"
    local attempts=0
    until [ "$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null)" = "healthy" ]; do
        attempts=$((attempts + 1))
        if [ "$attempts" -ge 30 ]; then
            echo "ERROR: $service did not become healthy after 150s"
            exit 1
        fi
        echo "      Waiting for $service (attempt ${attempts}/30)..."
        sleep 5
    done
    echo "      $service is healthy."
}

wait_for_healthy postgres
wait_for_healthy mlflow

echo ""
echo "[3/4] Running platform bootstrap (Airflow DB migration, admin user, initial model)..."
docker compose build airflow-webserver
docker compose run --rm --no-deps airflow-webserver bash /opt/mlops/init.sh

echo ""
echo "[4/4] Starting the rest of the stack..."
docker compose up -d --build

echo ""
echo "Setup complete. Run 'docker compose ps' to check status."
