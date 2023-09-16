import io
from airflow.models import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models.param import Param
from datetime import timedelta
import pandas as pd
from networkx.algorithms import dag
from src import lab2

# Create PythonOperator tasks for each step
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=lab2.load_model(),
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing',
    python_callable=lab2.data_preprocessing(),
    provide_context=True,
    dag=dag,
)

build_model_save_task = PythonOperator(
    task_id='build_save_model',
    python_callable=lab2.build_save_model(),
    provide_context=True,
    dag=dag,
)

load_predict_task = PythonOperator(
    task_id='load_model',
    python_callable=lab2.load_model(),
    provide_context=True,
    dag=dag,
)

# Set task dependencies
load_and_process_data_task >> build_save_model_task >> load_model_and_predict_task

if _name_ == "__main__":
    dag.cli()