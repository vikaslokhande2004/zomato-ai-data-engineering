# Airflow 3.x image + Postgres provider + FAB auth (for admin/admin login) + dbt venv.
FROM apache/airflow:3.3.2

# Providers: postgres (SQL tasks), fab (classic username/password login).
# Pinning apache-airflow keeps pip from upgrading Airflow itself.
RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    apache-airflow-providers-postgres \
    apache-airflow-providers-fab

# dbt + ingestion deps in their own venv so they never clash with Airflow's deps.
# DAG calls /opt/airflow/dbt_venv/bin/dbt and /opt/airflow/dbt_venv/bin/python.
# Pin dbt-postgres to your local version (check with: dbt --version).
RUN python -m venv /opt/airflow/dbt_venv && \
    /opt/airflow/dbt_venv/bin/pip install --no-cache-dir \
    dbt-postgres \
    boto3 \
    "psycopg[binary]" \
    python-dotenv \
    google-genai