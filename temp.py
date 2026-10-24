from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

# Virtual environment python and dbt inside the Airflow container
PYTHON = "/opt/airflow/dbt_venv/bin/python"
DBT = "/opt/airflow/dbt_venv/bin/dbt"
DBT_PROJECT = "/opt/airflow/dbt/zomato"

with DAG(
    dag_id="zomato_batch",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["zomato", "dbt", "postgres", "minio"],
    doc_md=__doc__,
) as dag:

    # 1. Load raw data from MinIO into PostgreSQL raw schema
    reload_raw = BashOperator(
        task_id="reload_raw",
        bash_command=f"{PYTHON} /opt/airflow/ingestion/load_raw_from_minio.py",
    )

    # 2. Run dbt staging and core marts (excluding LLM-dependent models)
    dbt_build_core = BashOperator(
        task_id="dbt_build_core",
        bash_command=f"{DBT} build --exclude tag:ai --project-dir {DBT_PROJECT} --profiles-dir {DBT_PROJECT}",
    )

    # 3. LLM enrichment for customer reviews
    enrich_reviews = BashOperator(
        task_id="enrich_reviews",
        bash_command=f"{PYTHON} /opt/airflow/ai/enrich_reviews.py",
    )

    # 4. Run AI marts that depend on enriched reviews
    dbt_build_ai = BashOperator(
        task_id="dbt_build_ai",
        bash_command=f"{DBT} build --select tag:ai --project-dir {DBT_PROJECT} --profiles-dir {DBT_PROJECT}",
    )

    reload_raw >> dbt_build_core >> enrich_reviews >> dbt_build_ai