from __future__ import annotations

from datetime import UTC, datetime

from airflow.operators.bash import BashOperator

from airflow import DAG

with DAG(
    dag_id="forgepulse_training_pipeline",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "manufacturing"],
) as dag:
    validate = BashOperator(task_id="validate_data", bash_command="python -m forgepulse.validate")
    train = BashOperator(task_id="train_model", bash_command="python -m forgepulse.train")
    evaluate = BashOperator(task_id="evaluate_model", bash_command="python -m forgepulse.evaluate")
    validate >> train >> evaluate
