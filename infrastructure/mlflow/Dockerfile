FROM python:3.11-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir \
    mlflow==2.14.3 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.0

# Match Airflow's UID so shared bind-mount artifacts are writable by both
RUN useradd -u 50000 -g 0 -m mlflow-user
USER 50000

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "${MLFLOW_BACKEND_STORE_URI}", \
     "--default-artifact-root", "${MLFLOW_ARTIFACT_ROOT}", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
