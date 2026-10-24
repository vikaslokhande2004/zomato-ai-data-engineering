import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg import sql

load_dotenv()

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"

ROLE_PASSWORDS = {
    "loader": os.environ["LOADER_PASSWORD"],
    "dbt_user": os.environ["DBT_PASSWORD"],
    "app_readonly": os.environ["APP_RO_PASSWORD"],
}


def run_file(cur, name: str) -> None:
    print(f"Running {name} ...")
    cur.execute((SQL_DIR / name).read_text(encoding="utf-8"))


def main() -> None:
    with psycopg.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5433"),
        dbname=os.environ["PG_DB"],
        user=os.environ["PG_ADMIN_USER"],
        password=os.environ["PG_ADMIN_PASSWORD"],
        autocommit=True,
    ) as conn, conn.cursor() as cur:
        run_file(cur, "01_setup.sql")

        for role, password in ROLE_PASSWORDS.items():
            cur.execute(
                sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(
                    sql.Identifier(role), sql.Literal(password)
                )
            )

        run_file(cur, "02_raw_tables.sql")

    print("setup complete")


if __name__ == "__main__":
    main()