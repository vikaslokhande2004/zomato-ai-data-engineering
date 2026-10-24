import os

import boto3
import psycopg
from botocore.client import Config
from dotenv import load_dotenv
from psycopg import sql

load_dotenv()

BUCKET = os.getenv("MINIO_BUCKET", "zomato-raw")

# MinIO folder -> raw table
DATASETS = {
    "restaurant":  "restaurants",
    "users":       "users",
    "food":        "food",
    "menu":        "menu",
    "orders":      "orders",
    "order_items": "order_items",
    "reviews":     "reviews",
}

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
    aws_access_key_id=os.environ["MINIO_ROOT_USER"],
    aws_secret_access_key=os.environ["MINIO_ROOT_PASSWORD"],
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)


def find_csv_key(folder: str) -> str | None:
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{folder}/")
    for obj in resp.get("Contents", []):
        if obj["Key"].lower().endswith(".csv"):
            return obj["Key"]
    return None


def log(conn, dataset, key, rows, status, error=None):
    conn.execute(
        "INSERT INTO raw.ingestion_log "
        "(dataset, source_key, rows_loaded, status, error) VALUES (%s, %s, %s, %s, %s)",
        (dataset, key, rows, status, error),
    )


def load_table(conn, table: str, key: str) -> int:
    tbl = sql.Identifier("raw", table)
    with conn.transaction():
        conn.execute(sql.SQL("TRUNCATE {}").format(tbl))
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"]
        copy_sql = sql.SQL("COPY {} FROM STDIN WITH (FORMAT csv, HEADER true)").format(tbl)
        with conn.cursor() as cur, cur.copy(copy_sql) as copy:
            for chunk in body.iter_chunks(chunk_size=1024 * 1024):
                copy.write(chunk)
        rows = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(tbl)).fetchone()[0]
    conn.execute(sql.SQL("ANALYZE {}").format(tbl))
    return rows


def main() -> None:
    with psycopg.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5433"),
        dbname=os.environ["PG_DB"],
        user="loader",
        password=os.environ["LOADER_PASSWORD"],
        autocommit=True,
        client_encoding="UTF8",
    ) as conn:
        for folder, table in DATASETS.items():
            key = find_csv_key(folder)
            if key is None:
                print(f"MISSING  no CSV under {BUCKET}/{folder}/")
                log(conn, table, None, None, "failed", "file not found in MinIO")
                continue

            print(f"LOAD  {key} -> raw.{table} ...", flush=True)
            try:
                rows = load_table(conn, table, key)
                log(conn, table, key, rows, "success")
                print(f"  OK  {rows:,} rows")
            except Exception as e:
                log(conn, table, key, None, "failed", str(e)[:1000])
                print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()