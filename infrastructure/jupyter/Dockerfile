FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev git curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    jupyterlab==4.* \
    pandas pyarrow \
    scikit-learn xgboost \
    "mlflow==2.14.3" psycopg2-binary sqlalchemy \
    matplotlib seaborn plotly \
    shap scipy \
    python-dotenv ipywidgets \
    nbformat

RUN useradd -u 50000 -g 0 -m jupyter-user \
    && mkdir -p /home/jupyter/notebooks \
    && chown -R 50000:0 /home/jupyter \
    && chmod -R 775 /home/jupyter

WORKDIR /home/jupyter

USER 50000

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''", \
     "--notebook-dir=/home/jupyter"]
