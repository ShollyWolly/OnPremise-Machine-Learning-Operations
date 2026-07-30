# Setup Guide

## Prerequisites

- Docker Desktop >= 4.20 with Docker Compose v2
- 8 GB RAM available for Docker (Airflow + PostgreSQL + MLflow are the heavy consumers)
- Ports free: 5432, 5000, 5001, 8000, 8080, 8501, 8502, 8888

## Startup Sequence

There's no init-container in the compose file - bootstrap is an explicit script step, run once before the stack comes up:

```
./setup.sh
        │
        ├─ 1. mkdir -p data/*, volumes/* + chmod (best-effort; no-op on Windows)
        │
        ├─ 2. docker compose up -d postgres mlflow
        │      wait for both healthy
        │
        ├─ 3. docker compose run --rm airflow-webserver bash /opt/mlops/init.sh
        │      a. airflow db migrate
        │      b. create admin user (idempotent)
        │      c. wait for MLflow
        │      d. bootstrap: generate data + train initial model
        │         (skipped if a 'production' alias already exists)
        │
        └─ 4. docker compose up -d
               brings up airflow-webserver, airflow-scheduler, flask-api,
               streamlit-ui, streamlit-dashboard, evidently-ui, jupyter

  Ready when: `docker compose ps` shows all services healthy/running (~60–120 s on first run)
```

Every step is idempotent - `./setup.sh` is safe to re-run any time, and on subsequent restarts (no schema/model changes) you can just use `docker compose up -d` directly.

---

## Quick Start

### 1. Configure

```bash
cp .env.example .env
```

Edit `.env`:
- Set `POSTGRES_PASSWORD` and `POSTGRES_SUPERUSER_PASSWORD` (change from defaults in any shared environment)
- Generate `AIRFLOW__CORE__FERNET_KEY`:
  ```bash
  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
  ```
- Set `AIRFLOW__WEBSERVER__SECRET_KEY` to any random string

### 2. Start the stack

```bash
./setup.sh
```

This builds the images, migrates the Airflow DB, creates the admin user, and trains an initial
model (~60–120 seconds on first run) before bringing up the rest of the stack. Output streams
to the terminal directly - no separate container logs to follow.

All services healthy:
```bash
docker compose ps
```

### 3. Verify infrastructure

```bash
# Check postgres schemas exist
docker exec mlops-postgres psql -U mlops_user -d mlops -c "\dn"
# Expected schemas: dwh_raw, dwh_clean, dwh_history, dwh_monitoring_hard,
#                   dwh_monitoring_drift, dwh_monitoring_shap, dwh_challenger,
#                   dwh_predictions, mlflow, airflow

# Check Flask API is live
curl http://localhost:5001/health
# → {"status":"ok"}

# Check a model is loaded
curl http://localhost:5001/model-info
# → {"model_name":"credit-risk-classifier","model_version":"1","model_alias":"production","run_id":"..."}
```

### 4. Open the Control Panel

Go to `http://localhost:8501`

**Pipelines tab → Batch Inference → Run**, this generates data, processes it, runs inference, and automatically triggers hard metric monitoring.

### 5. Open the Monitoring Dashboard

Go to `http://localhost:8502`

After the first batch inference run + monitoring run, all three dashboard tabs will have data.
Drift and Hard Metrics tabs embed reports from the self-hosted Evidently UI (`http://localhost:8000`),
which you can also browse directly.

---

## Running the Full Pipeline Manually

### Batch Inference (dag_03)

From Control Panel → Pipelines tab:
- Set `Drift Factor` to 0.0 for stable data, 0.5–1.0 to simulate distribution shift
- Set `Records` count (default 1000)
- Click **Run Batch Inference**

Or from Airflow UI (`http://localhost:8080`):
- Trigger `dag_03_batch_inference` with optional conf: `{"n_records": 1000, "drift_factor": 0.0}`

### Monitoring (dag_04a / 04b / 04c)

From Control Panel → Monitor tab, trigger any combination of the three monitoring DAGs.

Or from Airflow UI, trigger individually with `{"run_index": N}` (omit to use latest run).

### Manual Retraining (dag_05)

From Control Panel → Monitor tab → Manual Retraining button.

Or from Airflow UI, trigger `dag_05_retraining` with no conf.

### Challenger Workflow (dag_06)

1. Open JupyterLab at `http://localhost:8888`
2. Open `notebooks/01_challenger_template.ipynb`
3. Train a model and log it to MLflow (follow the notebook cells)
4. Copy the `run_id` printed at the end
5. Open Control Panel → **Challenger** tab
6. Paste the `run_id`, primary metric auto-detects from the MLflow tag
7. Optionally enable Force Deploy
8. Click **Run Challenger Comparison**

---

## Accessing Services

| Service | URL | Credentials |
|---|---|---|
| Control Panel | http://localhost:8501 | none |
| Monitoring Dashboard | http://localhost:8502 | none |
| Evidently UI | http://localhost:8000 | none |
| MLflow | http://localhost:5000 | none |
| Airflow | http://localhost:8080 | admin / admin (from AIRFLOW_ADMIN_USER/PASSWORD in .env) |
| JupyterLab | http://localhost:8888 | none (no token) |
| Flask API health | http://localhost:5001/health | none |
| Flask API model info | http://localhost:5001/model-info | none |
| PostgreSQL | localhost:5432 / db: mlops | POSTGRES_USER / POSTGRES_PASSWORD from .env |

---

## Stopping the Stack

```bash
# Stop all services (data and volumes preserved)
docker compose down

# Full reset - removes all volumes (data lost)
docker compose down -v && ./setup.sh
```

---

## Common Issues

**`./setup.sh` fails waiting for postgres/mlflow**: One of them didn't reach `healthy` within 150s. Check `docker compose logs postgres` / `docker compose logs mlflow` and re-run `./setup.sh` once they're healthy - every step is idempotent.

**Flask API stuck in `unhealthy`**: No `production` alias in MLflow registry. This should not happen after `./setup.sh` completes - re-run it and check its output for bootstrap errors.

**JupyterLab "File Load Error"**: The `.ipynb_checkpoints` directory may have wrong permissions if notebooks were executed via `docker exec`. Fix:
```bash
docker exec -u root mlops-jupyter chmod 777 /home/jupyter/notebooks/.ipynb_checkpoints
```

**Airflow DAG not showing**: Scheduler may still be parsing. Wait ~30 seconds and refresh the Airflow UI.

**MLflow can't connect to postgres**: Check that `MLFLOW_BACKEND_STORE_URI` in `.env` uses the correct password matching `POSTGRES_PASSWORD`.

**SHAP monitoring runs very slowly**: The fallback `KernelExplainer` is slow for non-tree models. For RandomForest/XGBoost, `TreeExplainer` runs in seconds. If you see KernelExplainer being used for a tree model, check that `sklearn` and the model type are compatible.
