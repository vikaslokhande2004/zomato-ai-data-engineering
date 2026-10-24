"""
Zomato batch pipeline (local, PostgreSQL):
reload_raw -> dbt_build_core -> enrich_reviews -> dbt_build_ai
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator                          # Airflow 2 import

PY = "/opt/airflow/dbt_venv/bin/python"
DBT = "/opt/airflow/dbt_venv/bin/dbt"
DBT_PROJECT = "/opt/airflow/dbt/zomato"
DBT_PROFILES = f"{DBT_PROJECT}/profiles"

# Flip to True once enrichment/enrich_reviews.py exists and ai.review_enriched is populated.
ENABLE_AI = False

default_args = {
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="zomato_batch",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["zomato", "dbt", "postgres"],
    doc_md=__doc__,
) as dag:

    reload_raw = BashOperator(
        task_id="reload_raw",
        bash_command=f"{PY} /opt/airflow/ingestion/load_raw_from_minio.py",
    )

    dbt_build_core = BashOperator(
        task_id="dbt_build_core",
        bash_command=(
            f"{DBT} build --exclude tag:ai "
            f"--project-dir {DBT_PROJECT} --profiles-dir {DBT_PROFILES}"
        ),
    )

    enrich_reviews = BashOperator(
            task_id="enrich_reviews",
            bash_command=f"{PY} /opt/airflow/ai/enrich_reviews.py",
    )

    dbt_build_ai = BashOperator(
            task_id="dbt_build_ai",
            bash_command=(
                f"{DBT} build --select tag:ai "
                f"--project-dir {DBT_PROJECT} --profiles-dir {DBT_PROFILES}"
            ),
        )



    reload_raw >> dbt_build_core >> enrich_reviews >> dbt_build_ai