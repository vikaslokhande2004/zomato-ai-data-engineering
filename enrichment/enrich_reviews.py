"""
Review enrichment pipeline for Zomato using NVIDIA Nemotron 3.5 Lightning & Google Gemini AI.

This module reads raw customer reviews from PostgreSQL (raw.reviews),
classifies sentiment, topic, and key issues using NVIDIA Nemotron 3.5 Lightning
(nvidia/nemotron-3.5-lightning-30b-a3b) or Google Gemini (gemini-3.5-flash-lite),
and writes the enriched results to PostgreSQL (ai.review_enriched).
Supports parallel execution via ThreadPoolExecutor for high-throughput batch processing.
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal, Optional

import psycopg
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Try importing google.genai, optional if only using NVIDIA NIM
try:
    from google import genai
    from google.genai import types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

# Load environment variables from ai/.env, project root .env, and cwd
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

# Configuration
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

# Default model defaults to Nemotron 3.5 Lightning
DEFAULT_MODEL = (
    os.getenv("AI_MODEL")
    or os.getenv("NVIDIA_MODEL")
    or os.getenv("GEMINI_MODEL", DEFAULT_NVIDIA_MODEL)
)
DEFAULT_LIMIT = int(os.getenv("ENRICH_LIMIT", "50"))
DEFAULT_WORKERS = int(os.getenv("ENRICH_WORKERS", "5"))
TOPICS = ["food quality", "delivery", "pricing", "service", "packaging", "other"]

SYSTEM_PROMPT = f"""
You are an expert customer feedback classification model for a food delivery platform.
Analyze the given customer review and output a strictly valid JSON object with the following fields:

1. sentiment_label: exactly one of "positive", "negative", or "neutral"
2. sentiment_score: a float between -1.0 (extremely negative) and 1.0 (extremely positive)
3. topic: exactly one of {TOPICS}
4. key_issue: a concise phrase (6 words or fewer) summarizing the core problem if any; return null if there is no complaint or issue.

Return ONLY raw JSON in this exact structure, with no markdown code blocks, preamble, or commentary:
{{
    "sentiment_label": "<positive|negative|neutral>",
    "sentiment_score": 0.0,
    "topic": "<food quality|delivery|pricing|service|packaging|other>",
    "key_issue": null
}}
"""


class ReviewClassification(BaseModel):
    sentiment_label: Literal["positive", "negative", "neutral"]
    sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment score from -1.0 (very negative) to 1.0 (very positive)",
    )
    topic: Literal["food quality", "delivery", "pricing", "service", "packaging", "other"]
    key_issue: Optional[str] = Field(
        default=None,
        description="Short phrase of 6 words or less describing main complaint, or null if positive/neutral",
    )

    @field_validator("sentiment_label", mode="before")
    @classmethod
    def normalize_sentiment(cls, v: Any) -> str:
        if not isinstance(v, str):
            return "neutral"
        v_clean = v.strip().lower()
        if "pos" in v_clean:
            return "positive"
        if "neg" in v_clean:
            return "negative"
        return "neutral"

    @field_validator("topic", mode="before")
    @classmethod
    def normalize_topic(cls, v: Any) -> str:
        if not isinstance(v, str):
            return "other"
        v_clean = v.strip().lower()
        if v_clean in ["food quality", "delivery", "pricing", "service", "packaging", "other"]:
            return v_clean
        if "deliver" in v_clean:
            return "delivery"
        if "food" in v_clean or "taste" in v_clean or "stale" in v_clean or "qualit" in v_clean or "cold" in v_clean or "oily" in v_clean:
            return "food quality"
        if "price" in v_clean or "cost" in v_clean or "expensive" in v_clean or "charg" in v_clean:
            return "pricing"
        if "pack" in v_clean or "spill" in v_clean:
            return "packaging"
        if "service" in v_clean or "wait" in v_clean or "staff" in v_clean or "rude" in v_clean:
            return "service"
        return "other"


def extract_json_from_text(text: str) -> dict:
    """Robustly extract and parse JSON object from model output."""
    cleaned = text.strip()

    # Strip markdown code fences if present
    if "```" in cleaned:
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

    # Try raw_decode starting from first '{'
    start = cleaned.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Extract outermost curly braces as fallback
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        sanitized = re.sub(r",\s*([\]}])", r"\1", cleaned)
        sanitized = sanitized.replace("'", '"')
        return json.loads(sanitized)


def get_model_provider(model_name: str, provider_override: Optional[str] = None) -> str:
    """Determine whether to use NVIDIA NIM or Google Gemini."""
    if provider_override and provider_override.lower() in ("nvidia", "gemini"):
        return provider_override.lower()

    model_lower = model_name.lower()
    if model_lower.startswith("nvidia/") or "nemotron" in model_lower:
        return "nvidia"
    if model_lower.startswith("gemini"):
        return "gemini"

    explicit = os.getenv("AI_PROVIDER")
    if explicit:
        return explicit.lower()
    if os.getenv("NVIDIA_API_KEY"):
        return "nvidia"
    return "nvidia" if "nemotron" in DEFAULT_MODEL else "gemini"


def get_ai_client(
    model_name: str,
    provider_override: Optional[str] = None,
) -> tuple[Any, str, str]:
    """
    Initialize and return (client_instance, model_name, provider).
    For NVIDIA NIM, returns (None, model_name, 'nvidia') after validating API key.
    For Gemini, returns (genai.Client, model_name, 'gemini').
    """
    provider = get_model_provider(model_name, provider_override)

    if provider == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key or api_key.strip() in ("", "your_nvidia_api_key_here"):
            raise ValueError(
                "Missing NVIDIA API key! Please set NVIDIA_API_KEY in your .env file.\n"
                "Get a free API key with 1,000 credits at: "
                "https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b"
            )
        return None, model_name, "nvidia"

    elif provider == "gemini":
        if not HAS_GOOGLE_GENAI:
            raise ImportError(
                "google-genai package is not installed. Run `pip install google-genai` to use Gemini."
            )
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key.strip() in ("", "your_gemini_api_key_here"):
            raise ValueError(
                "Missing Gemini API key! Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file."
            )
        client = genai.Client(api_key=api_key.strip())
        return client, model_name, "gemini"

    raise ValueError(f"Unknown AI provider: {provider}")


def get_connection():
    """Establish connection to PostgreSQL warehouse using project configuration."""
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5433")
    dbname = os.getenv("PG_DB", "zomato")

    # Try credentials candidates in priority order to pair user and password accurately
    candidates = []

    # 1. Explicitly set PG_USER and PG_PASSWORD
    if os.getenv("PG_USER") and os.getenv("PG_PASSWORD"):
        candidates.append((os.getenv("PG_USER"), os.getenv("PG_PASSWORD")))

    # 2. dbt_user with DBT_PASSWORD (used inside Airflow container and dbt)
    if os.getenv("DBT_PASSWORD"):
        candidates.append((os.getenv("DBT_USER", "dbt_user"), os.getenv("DBT_PASSWORD")))

    # 3. Postgres admin credentials
    if os.getenv("PG_ADMIN_PASSWORD") or os.getenv("PG_PASSWORD"):
        candidates.append(
            (
                os.getenv("PG_ADMIN_USER", "postgres"),
                os.getenv("PG_ADMIN_PASSWORD") or os.getenv("PG_PASSWORD", "change_me_admin"),
            )
        )

    # 4. Standard defaults
    candidates.append(("dbt_user", os.getenv("DBT_PASSWORD", "change_me_dbt")))
    candidates.append(("postgres", os.getenv("PG_ADMIN_PASSWORD", "change_me_admin")))

    last_error = None
    for user, password in candidates:
        try:
            return psycopg.connect(
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password,
                autocommit=False,
                connect_timeout=5,
            )
        except Exception as e:
            last_error = e
            continue

    raise last_error


def create_output_table(cursor) -> None:
    """Ensure the target AI schema and enriched reviews table exist in PostgreSQL."""
    try:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS ai;")
    except Exception:
        pass

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ai.review_enriched (
            review_id BIGINT PRIMARY KEY,
            sentiment_label TEXT NOT NULL,
            sentiment_score NUMERIC(4, 2) NOT NULL,
            topic TEXT NOT NULL,
            key_issue TEXT,
            model TEXT NOT NULL,
            enriched_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    try:
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_review_enriched_topic 
            ON ai.review_enriched (topic);
            """
        )
    except Exception:
        pass

    try:
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_review_enriched_sentiment 
            ON ai.review_enriched (sentiment_label);
            """
        )
    except Exception:
        pass

    try:
        cursor.execute("GRANT ALL ON TABLE ai.review_enriched TO dbt_user;")
    except Exception:
        pass
    try:
        cursor.execute("GRANT SELECT ON TABLE ai.review_enriched TO app_readonly;")
    except Exception:
        pass


def get_reviews_to_enrich(cursor, limit: int = DEFAULT_LIMIT) -> list[tuple[int, str]]:
    """Fetch unprocessed reviews with valid comments from raw.reviews."""
    cursor.execute(
        """
        SELECT review_id, comment
        FROM raw.reviews
        WHERE comment IS NOT NULL
          AND trim(comment) != ''
          AND review_id NOT IN (SELECT review_id FROM ai.review_enriched)
        ORDER BY review_id ASC
        LIMIT %s;
        """,
        (limit,),
    )
    return cursor.fetchall()


def classify_review_nvidia(comment: str, model: str) -> dict:
    """Classify a single review using NVIDIA NIM Nemotron."""
    key = os.getenv("NVIDIA_API_KEY", "").strip()
    base_url = (
        os.getenv("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_BASE_URL
    ).rstrip("/")
    endpoint = f"{base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Customer review text to classify:\n\"{comment}\""},
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
    }

    for attempt in range(2):
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=180)
            if resp.status_code != 200:
                raise RuntimeError(f"NVIDIA NIM error ({resp.status_code}): {resp.text}")

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise ValueError(f"Empty choices in NVIDIA response: {data}")

            message = choices[0]["message"]
            raw_text = message.get("content") or message.get("reasoning_content")
            if not raw_text:
                raise ValueError(f"Empty message content in response: {message}")

            parsed_json = extract_json_from_text(raw_text)
            validated = ReviewClassification.model_validate(parsed_json)
            return validated.model_dump()
        except requests.exceptions.Timeout as e:
            if attempt == 1:
                raise e
            continue


def classify_review_gemini(client: Any, model: str, comment: str) -> dict:
    """Classify a single review text using Google Gemini with structured output."""
    response = client.models.generate_content(
        model=model,
        contents=comment,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ReviewClassification,
            temperature=0.0,
        ),
    )

    if response.parsed:
        if isinstance(response.parsed, ReviewClassification):
            return response.parsed.model_dump()
        return dict(response.parsed)

    parsed_json = extract_json_from_text(response.text)
    validated = ReviewClassification.model_validate(parsed_json)
    return validated.model_dump()


def classify_review(
    client: Optional[Any],
    model: str,
    provider: str,
    comment: str,
) -> dict:
    """Dispatch review classification to the appropriate AI provider."""
    if provider == "nvidia":
        return classify_review_nvidia(comment, model=model)
    else:
        return classify_review_gemini(client, model=model, comment=comment)


def save_results(cursor, results: list[tuple]) -> None:
    """Insert or update enriched reviews in PostgreSQL in a single batch."""
    if not results:
        return

    print(f"Saving {len(results)} enriched reviews to PostgreSQL ai.review_enriched...")
    cursor.executemany(
        """
        INSERT INTO ai.review_enriched (
            review_id,
            sentiment_label,
            sentiment_score,
            topic,
            key_issue,
            model,
            enriched_at
        ) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (review_id) DO UPDATE SET
            sentiment_label = EXCLUDED.sentiment_label,
            sentiment_score = EXCLUDED.sentiment_score,
            topic = EXCLUDED.topic,
            key_issue = EXCLUDED.key_issue,
            model = EXCLUDED.model,
            enriched_at = CURRENT_TIMESTAMP;
        """,
        results,
    )


def process_review_worker(
    client: Optional[Any],
    model: str,
    provider: str,
    review_id: int,
    comment: str,
) -> tuple[int, Optional[dict], Optional[str]]:
    """Helper worker to classify a review concurrently."""
    try:
        labels = classify_review(client, model, provider, comment)
        return review_id, labels, None
    except Exception as e:
        return review_id, None, str(e)


def enrich_pipeline(
    limit: int = DEFAULT_LIMIT,
    model_name: Optional[str] = None,
    provider_override: Optional[str] = None,
    workers: int = DEFAULT_WORKERS,
    dry_run: bool = False,
) -> int:
    """Run the review enrichment pipeline with parallel processing."""
    effective_model = model_name or DEFAULT_MODEL
    print("=" * 65)
    print("Starting Zomato Review Enrichment Pipeline")
    print("=" * 65)

    try:
        client, model, provider = get_ai_client(effective_model, provider_override)
        print(f"AI Provider : {provider.upper()}")
        print(f"Model ID    : {model}")
        print(f"Workers     : {workers}")
        print(f"Batch Limit : {limit}")
        print("=" * 65)
    except (ValueError, ImportError) as e:
        print(f"\n[ERROR] {e}\n", file=sys.stderr)
        return 1

    try:
        conn = get_connection()
    except Exception as e:
        print(f"\n[ERROR] Failed to connect to PostgreSQL: {e}", file=sys.stderr)
        return 1

    with conn, conn.cursor() as cursor:
        create_output_table(cursor)
        conn.commit()

        reviews = get_reviews_to_enrich(cursor, limit=limit)
        total_found = len(reviews)

        if total_found == 0:
            print("No new unprocessed reviews found in raw.reviews.")
            return 0

        print(f"Found {total_found} reviews to enrich.\n")

        results = []
        completed = 0

        # Execute concurrently with ThreadPoolExecutor for high throughput
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_to_review = {
                executor.submit(
                    process_review_worker, client, model, provider, rid, comment
                ): (rid, comment)
                for rid, comment in reviews
            }

            for future in as_completed(future_to_review):
                completed += 1
                rid, comment = future_to_review[future]
                review_id, labels, error = future.result()

                short_comment = (
                    (comment[:55] + "...") if len(comment) > 55 else comment
                )

                if error:
                    print(
                        f"[{completed}/{total_found}] Review {review_id}: [FAILED] {error}"
                    )
                else:
                    score = labels["sentiment_score"]
                    print(
                        f"[{completed}/{total_found}] Review {review_id}: \"{short_comment}\" "
                        f"-> {labels['sentiment_label']} ({score:+.2f}) | {labels['topic']}"
                    )
                    results.append(
                        (
                            review_id,
                            labels["sentiment_label"],
                            float(score),
                            labels["topic"],
                            labels["key_issue"],
                            model,
                        )
                    )

        if dry_run:
            print(
                f"\n[DRY RUN] Classified {len(results)}/{total_found} reviews. No DB writes committed."
            )
            return 0

        if results:
            save_results(cursor, results)
            conn.commit()
            print(
                f"\nSuccessfully committed {len(results)} enriched rows to ai.review_enriched using {model}."
            )
        else:
            print("\nNo reviews were successfully classified.")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Enrich customer reviews using NVIDIA Nemotron or Google Gemini AI"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["auto", "nvidia", "gemini"],
        default="auto",
        help="AI provider override ('auto', 'nvidia', or 'gemini')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of reviews to enrich in this run (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of concurrent threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run classifications without writing results to the database",
    )
    args = parser.parse_args()

    provider_override = None if args.provider == "auto" else args.provider
    sys.exit(
        enrich_pipeline(
            limit=args.limit,
            model_name=args.model,
            provider_override=provider_override,
            workers=args.workers,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
