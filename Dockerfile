FROM python:3.12-slim AS application

WORKDIR /project
ENV PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY src/ ./src/

FROM apache/airflow:2.11.0-python3.12 AS airflow
RUN pip uninstall -y apache-airflow-providers-google apache-airflow-providers-snowflake
RUN pip install --no-cache-dir pandas==3.1.1 numpy==2.5.4 scikit-learn==1.9.1 joblib==1.6.0 matplotlib==3.11.2
COPY src/ /project/src/
COPY dag/ /project/dag/
WORKDIR /project
ENV PYTHONPATH=/project \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1
