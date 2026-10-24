import os
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")
BUCKET = os.getenv("MINIO_BUCKET", "zomato-raw")
SOURCE_DIR = Path(os.getenv("SOURCE_DIR", "data"))

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)


def ensure_bucket(name: str) -> None:
    try:
        s3.head_bucket(Bucket=name)
    except ClientError:
        s3.create_bucket(Bucket=name)
        print(f"Created bucket: {name}")


def already_uploaded(key: str, size: int) -> bool:
    try:
        head = s3.head_object(Bucket=BUCKET, Key=key)
        return head["ContentLength"] == size
    except ClientError:
        return False


def main() -> None:
    if not ACCESS_KEY or not SECRET_KEY:
        raise SystemExit("Missing MINIO_ROOT_USER / MINIO_ROOT_PASSWORD in .env")

    files = sorted(SOURCE_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {SOURCE_DIR.resolve()}")

    ensure_bucket(BUCKET)

    for path in files:
        key = f"{path.stem}/{path.name}"   # e.g. orders/orders.csv
        size = path.stat().st_size
        if already_uploaded(key, size):
            print(f"SKIP   {key} (already uploaded)")
            continue
        print(f"UPLOAD {path.name} -> s3://{BUCKET}/{key} ({size / 1_048_576:.1f} MB)")
        s3.upload_file(str(path), BUCKET, key)

    print("\nObjects in bucket:")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET):
        for obj in page.get("Contents", []):
            print(f"  {obj['Key']}  ({obj['Size']:,} bytes)")


if __name__ == "__main__":
    main()