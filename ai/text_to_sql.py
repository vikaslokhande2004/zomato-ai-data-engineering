"""
Text-to-SQL Streamlit Application for Zomato Data Warehouse.
Powered by NVIDIA Nemotron 3.5 Lightning (30B-A3B) and PostgreSQL.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import psycopg
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from local folder, parent folder, and cwd
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

# Constants & Models
DEFAULT_MODEL = os.getenv("AI_MODEL", os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"))
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

AVAILABLE_MODELS = [
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "gpt-4o-mini",
    "gemini-3.5-flash-lite",
]

FORBIDDEN_WORDS = [
    "drop", "delete", "truncate", "alter", "update", "insert",
    "create", "replace", "grant", "revoke", "copy", "execute"
]

EXAMPLE_QUESTIONS = [
    "Top 10 cities by total GMV",
    "Which cuisine has the most delivered orders?",
    "Average delivery time by city, worst first",
    "Order cancel rate and total orders by city",
    "Top 10 restaurants by revenue and average rating",
    "Delivery SLA: p50 and p90 delivery time by city",
    "Customer sentiment by topic and city"
]

# Accurate PostgreSQL schema matching the dbt marts models in zomato/models/marts
SCHEMA = """
Available PostgreSQL tables (schema: marts and ai).
Search path is set to marts and ai, so use bare table names:

1. fct_orders:
   - order_id (BIGINT, primary key)
   - order_timestamp (TIMESTAMP)
   - order_date (DATE)
   - customer_id (BIGINT)
   - restaurant_id (BIGINT)
   - city (TEXT)
   - cuisine (TEXT)
   - payment_method (TEXT: 'UPI', 'Credit Card', 'Cash on Delivery', etc.)
   - order_status (TEXT: 'Delivered', 'Cancelled', 'Refunded')
   - is_delivered (BOOLEAN)
   - items_count (INT)
   - sales_qty (INT)
   - subtotal (NUMERIC)
   - discount (NUMERIC)
   - delivery_fee (NUMERIC)
   - gst (NUMERIC)
   - sales_amount (NUMERIC)
   - customer_rating (NUMERIC: 1 to 5)
   - delivery_time_min (NUMERIC)

2. dim_restaurants:
   - restaurant_id (BIGINT, primary key)
   - restaurant_name (TEXT)
   - city (TEXT)
   - cuisine (TEXT)
   - rating (NUMERIC)
   - rating_count (INT)
   - cost_for_two (NUMERIC)

3. dim_customer:
   - customer_id (BIGINT, primary key)
   - customer_name (TEXT)
   - email (TEXT)
   - age (INT)
   - age_segment (TEXT: 'Gen Z', 'Millennial', 'Gen X', 'Boomer')
   - gender (TEXT)
   - marital_status (TEXT)
   - occupation (TEXT)
   - income_band (TEXT)
   - education (TEXT)
   - family_size (INT)

4. fact_order_items:
   - order_item_id (BIGINT)
   - order_id (BIGINT)
   - restaurant_id (BIGINT)
   - f_id (TEXT: food item id)
   - order_ts (TIMESTAMP)
   - order_date (DATE)
   - city (TEXT)
   - price (NUMERIC)
   - quantity (INT)
   - line_amount (NUMERIC)

5. dim_food:
   - f_id (TEXT, primary key)
   - food_name (TEXT)
   - veg_or_non_veg (TEXT)

6. mart_daily_city_revenune:
   - order_date (DATE)
   - city (TEXT)
   - orders (BIGINT: total orders)
   - delivered_orders (BIGINT)
   - cancel_rate (NUMERIC: cancelled / total)
   - gmv (NUMERIC: sum of delivered sales_amount)
   - aov (NUMERIC: average order value of delivered orders)

7. mart_restaurant_performance:
   - restaurant_id (BIGINT)
   - restaurant_name (TEXT)
   - city (TEXT)
   - cuisine (TEXT)
   - orders (BIGINT)
   - revenue (NUMERIC: delivered sales_amount)
   - avg_customer_rating (NUMERIC)
   - avg_delivery_min (NUMERIC)

8. mart_delivery_sla:
   - city (TEXT)
   - order_hour (INT: 0 to 23)
   - delivered_orders (BIGINT)
   - p50 (NUMERIC: 50th percentile delivery minutes)
   - p90 (NUMERIC: 90th percentile delivery minutes)

9. mart_review_insights:
   - city (TEXT)
   - topic (TEXT: 'food quality', 'delivery', 'pricing', 'service', 'packaging', 'other')
   - sentiment_label (TEXT: 'positive', 'negative', 'neutral')
   - reviews (BIGINT: count of reviews)
   - avg_sentiment_score (NUMERIC: -1.0 to 1.0)
   - avg_star_rating (NUMERIC: 1.0 to 5.0)
   - flagged_issues (BIGINT: reviews with a complaint)

Notes:
- Use mart_daily_city_revenune for city GMV, AOV, or cancel rate aggregations.
- Always cast numeric aggregations appropriately (e.g. ROUND(AVG(...)::numeric, 2)).
- Prefer standard PostgreSQL syntax.
"""

SYSTEM_PROMPT = f"""You are an expert PostgreSQL data engineer for Zomato.
Write ONE clean, executable PostgreSQL SELECT query that accurately answers the user's question.

Rules:
1. SELECT queries ONLY. Never use DROP, DELETE, INSERT, UPDATE, ALTER, TRUNCATE, or CREATE.
2. Use bare table names (e.g., mart_daily_city_revenune, not zomato.marts.mart_daily_city_revenune).
3. Always include LIMIT 100 or less, unless the user explicitly requests a single total/aggregate.
4. Reply strictly as a valid JSON object in this exact format with NO markdown fences, preamble, or commentary:
{{"sql": "your SELECT query here"}}

{SCHEMA}
"""


def extract_sql_from_response(text: str) -> str:
    """Robustly extract SQL string from JSON response or markdown fences."""
    cleaned = text.strip()

    # Strip code block fences
    if "```" in cleaned:
        match = re.search(r"```(?:json|sql)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()

    # Extract JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if "sql" in parsed:
                cleaned = parsed["sql"]
        except Exception:
            pass

    # Clean up trailing semicolons and schema prefixes
    cleaned = cleaned.replace("zomato.marts.", "").replace("ZOMATO.MARTS.", "")
    cleaned = cleaned.replace("marts.", "").replace("MARTS.", "")
    return cleaned.strip().rstrip(";")


def generate_sql_nvidia(question: str, model: str = DEFAULT_MODEL) -> str:
    """Generate SQL using NVIDIA NIM (e.g. nvidia/nemotron-3.5-lightning-30b-a3b)."""
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing NVIDIA_API_KEY! Please set it in .env or the sidebar.")

    endpoint = f"{(os.getenv('NVIDIA_BASE_URL') or DEFAULT_NVIDIA_BASE_URL).rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"NVIDIA NIM error ({resp.status_code}): {resp.text}")

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("Empty response from NVIDIA NIM.")

    message = choices[0]["message"]
    raw_content = message.get("content") or message.get("reasoning_content", "")
    return extract_sql_from_response(raw_content)


def generate_sql_openai(question: str, model: str = "gpt-4o-mini") -> str:
    """Generate SQL using OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return extract_sql_from_response(response.choices[0].message.content)


def generate_sql(question: str, model: str = DEFAULT_MODEL) -> str:
    """Dispatch SQL generation based on the selected model."""
    if model.startswith("nvidia/") or "nemotron" in model:
        return generate_sql_nvidia(question, model=model)
    elif "gpt" in model:
        return generate_sql_openai(question, model=model)
    else:
        return generate_sql_nvidia(question, model=DEFAULT_MODEL)


def is_safe(sql: str) -> tuple[bool, Optional[str]]:
    """Validate query safety: read-only SELECT/WITH queries without forbidden keywords."""
    lowered = sql.lower().strip()

    # Must start with SELECT or WITH
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False, "Query must start with SELECT or WITH."

    # Prevent multiple statements separated by semicolon
    if ";" in lowered:
        return False, "Multiple statements are not permitted."

    # Check forbidden DDL/DML keywords
    for word in FORBIDDEN_WORDS:
        # Match as whole word
        if re.search(rf"\b{word}\b", lowered):
            return False, f"Query contains forbidden keyword: '{word}'"

    return True, None


@st.cache_resource
def get_connection():
    """Establish connection to PostgreSQL warehouse using app_readonly or postgres user."""
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5433")
    dbname = os.getenv("PG_DB", "zomato")

    # Prefer app_readonly role for secure read-only execution with 30s timeout
    user = os.getenv("PG_USER") or "app_readonly"
    password = (
        os.getenv("APP_RO_PASSWORD")
        or os.getenv("PG_PASSWORD")
        or os.getenv("PG_ADMIN_PASSWORD")
        or "change_me_readonly"
    )

    try:
        conn = psycopg.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            connect_timeout=5,
            options="-c search_path=marts,ai,public",
        )
        return conn
    except Exception:
        # Fallback to postgres admin user if app_readonly role hasn't been set up yet
        admin_user = os.getenv("PG_ADMIN_USER", "postgres")
        admin_pass = os.getenv("PG_ADMIN_PASSWORD", "change_me_admin")
        return psycopg.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=admin_user,
            password=admin_pass,
            connect_timeout=5,
            options="-c search_path=marts,ai,public",
        )


def run_query(sql: str) -> pd.DataFrame:
    """Execute SQL query on PostgreSQL and return results as a pandas DataFrame."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(sql)
        if cur.description is None:
            return pd.DataFrame()
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)


# Streamlit Page Setup
st.set_page_config(page_title="Zomato Text-to-SQL Assistant", page_icon="📊", layout="wide")

st.title("📊 Chat with your Zomato Data (Text-to-SQL)")
st.caption(f"Ask business questions in natural language. Powered by **{DEFAULT_MODEL}** & **PostgreSQL**.")

with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "SQL Generation Model:",
        options=AVAILABLE_MODELS,
        index=0,
        help="Select the AI model to generate PostgreSQL queries."
    )

    if selected_model.startswith("nvidia/"):
        nvidia_key = os.getenv("NVIDIA_API_KEY", "")
        if nvidia_key:
            st.success("🟢 NVIDIA API Key Active")
        else:
            st.warning("⚠️ Missing NVIDIA_API_KEY in `.env`")
    elif "gpt" in selected_model:
        if os.getenv("OPENAI_API_KEY"):
            st.success("🟢 OpenAI API Key Active")
        else:
            st.warning("⚠️ Missing OPENAI_API_KEY in `.env`")

    st.markdown("---")
    st.header("💡 Example Questions")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"btn_{q}"):
            st.session_state["user_question"] = q

question = st.text_input(
    "Enter your question here:",
    value=st.session_state.get("user_question", ""),
    placeholder="e.g. Top 10 cities by total GMV",
)

if question:
    with st.spinner(f"Generating PostgreSQL query with {selected_model}..."):
        try:
            sql = generate_sql(question, model=selected_model)
            st.subheader("📝 Generated SQL")
            st.code(sql, language="sql")

            safe, reason = is_safe(sql)
            if not safe:
                st.error(f"⚠️ Safety validation rejected this query: {reason}")
            else:
                try:
                    with st.spinner("Executing query on PostgreSQL warehouse..."):
                        df = run_query(sql)

                    st.success(f"✅ Returned {len(df)} row(s)")
                    st.dataframe(df, hide_index=True, use_container_width=True)

                    # Auto-charting: if 2 columns and second is numeric
                    if len(df.columns) == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                        st.subheader("📈 Visualization")
                        st.bar_chart(df, x=df.columns[0], y=df.columns[1])

                except Exception as e:
                    st.error(f"❌ Error executing PostgreSQL query: {e}")
                    st.info(
                        "Make sure PostgreSQL is running on port 5433 (`docker compose up -d`)."
                    )

        except Exception as e:
            st.error(f"❌ Error generating SQL: {e}")