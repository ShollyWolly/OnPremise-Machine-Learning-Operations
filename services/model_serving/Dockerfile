FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Match Airflow/MLflow UID so shared bind-mount artifacts are writable by all services
RUN useradd -u 50000 -g 0 -m flask-user
USER 50000

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--timeout", "120", "--preload", "app:app"]
