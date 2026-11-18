import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.data import prepare_data
from src.preprocessing import prepare_preprocessing
from src.train import train_models
from src.evaluate import evaluate_model

with DAG(
    dag_id='student_mental_health_pipeline',
    start_date=pendulum.datetime(2026, 11, 9, tz='Africa/Casablanca'),
    schedule='@daily',
    catchup=False,
) as dag:
    prepare = PythonOperator(task_id='prepare_data', python_callable=prepare_data)
    preprocess = PythonOperator(task_id='preprocess', python_callable=prepare_preprocessing)
    train = PythonOperator(task_id='train_and_optimize', python_callable=train_models)
    evaluate = PythonOperator(task_id='evaluate', python_callable=evaluate_model)

    prepare >> preprocess >> train >> evaluate
