"""
Streamlit RAG Chat application for Zomato Customer Reviews.
Powered by NVIDIA Nemotron 3.5 Lightning (30B-A3B) & Google Gemini AI.
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import psycopg
import requests
import streamlit as st
from dotenv import load_dotenv

# Optional import for Gemini
try:
    from google import genai
    from google.genai import errors, types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

# Load environment variables from local folder, parent folder, and cwd
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

# Configuration
DEFAULT_NVIDIA_CHAT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
DEFAULT_NVIDIA_EMBED_MODEL = "nvidia/nemotron-3.5-embed-1b"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

DEFAULT_GEMINI_CHAT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GEMINI_EMBED_MODEL = "gemini-embedding-001"

AVAILABLE_CHAT_MODELS = [
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
]

DEFAULT_CHAT_MODEL = (
    os.getenv("AI_MODEL")
    or os.getenv("NVIDIA_MODEL")
    or DEFAULT_NVIDIA_CHAT_MODEL
)
DEFAULT_REVIEWS_COUNT = int(os.getenv("RAG_REVIEWS_COUNT", "500"))
DEFAULT_TOP_K = 5
CACHE_FILE = Path(__file__).resolve().parent / "review_embeddings.parquet"

# Streamlit Page Setup
st.set_page_config(
    page_title="Zomato Reviews RAG Assistant",
    page_icon="🍔",
    layout="wide",
)


def get_gemini_client() -> Optional[Any]:
    """Initialize and return the Google Gemini client if available."""
    if not HAS_GOOGLE_GENAI:
        return None
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip() in ("", "your_gemini_api_key_here"):
        return None
    try:
        return genai.Client(api_key=api_key.strip())
    except Exception:
        return None


def get_db_connection():
    """Establish connection to the PostgreSQL database."""
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5433")
    dbname = os.getenv("PG_DB", "zomato")

    user = os.getenv("PG_USER") or os.getenv("PG_ADMIN_USER") or "postgres"
    password = (
        os.getenv("PG_PASSWORD")
        or os.getenv("PG_ADMIN_PASSWORD")
        or os.getenv("APP_RO_PASSWORD")
        or "change_me_admin"
    )

    return psycopg.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        connect_timeout=5,
    )


def fetch_reviews_from_postgres(limit: int = DEFAULT_REVIEWS_COUNT) -> pd.DataFrame:
    """Fetch customer reviews from PostgreSQL staging.stg_reviews."""
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            query = """
                SELECT 
                    r.review_id,
                    COALESCE(r.city, 'Unknown') AS city,
                    COALESCE(r.rating, 3) AS rating,
                    r.comment
                FROM staging.stg_reviews r
                WHERE r.comment IS NOT NULL 
                  AND trim(r.comment) != ''
                ORDER BY r.review_id ASC
                LIMIT %s;
            """
            cur.execute(query, (limit,))
            rows = cur.fetchall()

        if not rows:
            return pd.DataFrame(columns=["review_id", "city", "rating", "comment"])

        return pd.DataFrame(rows, columns=["review_id", "city", "rating", "comment"])
    except Exception as e:
        st.warning(f"ℹ️ Could not connect to live PostgreSQL warehouse ({e}). Using cached reviews.")
        return pd.DataFrame(columns=["review_id", "city", "rating", "comment"])


def embed_texts_nvidia(
    texts: list[str],
    input_type: str = "passage",
    api_key: Optional[str] = None,
    batch_size: int = 25,
) -> list[list[float]]:
    """Embed texts using NVIDIA NIM (nvidia/nemotron-3.5-embed-1b)."""
    key = api_key or os.getenv("NVIDIA_API_KEY")
    if not key or key.strip() in ("", "your_nvidia_api_key_here"):
        raise ValueError("Missing NVIDIA API key! Please configure NVIDIA_API_KEY in .env or the sidebar.")

    base_url = (os.getenv("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_BASE_URL).rstrip("/")
    endpoint = f"{base_url}/embeddings"

    headers = {
        "Authorization": f"Bearer {key.strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    all_embeddings: list[list[float]] = []
    total = len(texts)

    progress_bar = None
    if st.runtime.exists() and total > 1:
        progress_bar = st.progress(0.0, text="Generating embeddings with NVIDIA Nemotron...")

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        payload = {
            "model": DEFAULT_NVIDIA_EMBED_MODEL,
            "input": batch,
            "input_type": input_type,
        }
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"NVIDIA Embedding API Error ({resp.status_code}): {resp.text}")

        data = resp.json()
        for item in data.get("data", []):
            all_embeddings.append(item["embedding"])

        if progress_bar is not None:
            pct = min(1.0, (i + len(batch)) / total)
            progress_bar.progress(pct, text=f"Embedding reviews ({min(i + len(batch), total)}/{total})...")

    if progress_bar is not None:
        progress_bar.empty()

    return all_embeddings


def embed_texts_gemini(client: Any, texts: list[str], batch_size: int = 50) -> list[list[float]]:
    """Embed texts using Google Gemini embedding model."""
    all_embeddings: list[list[float]] = []
    total = len(texts)

    progress_bar = None
    if st.runtime.exists() and total > 1:
        progress_bar = st.progress(0.0, text="Generating embeddings with Google Gemini...")

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        response = client.models.embed_content(
            model=DEFAULT_GEMINI_EMBED_MODEL,
            contents=batch,
        )
        for emb in response.embeddings:
            all_embeddings.append(emb.values)

        if progress_bar is not None:
            pct = min(1.0, (i + len(batch)) / total)
            progress_bar.progress(pct, text=f"Embedding reviews ({min(i + len(batch), total)}/{total})...")

    if progress_bar is not None:
        progress_bar.empty()
    return all_embeddings


@st.cache_data(show_spinner=False)
def load_and_embed_reviews(
    provider: str,
    _gemini_client: Optional[Any],
    nvidia_key: Optional[str],
    count: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load reviews from local parquet cache or fetch and embed from PostgreSQL."""
    expected_dim = 2048 if provider == "nvidia" else 3072

    if not force_refresh and CACHE_FILE.exists():
        try:
            df = pd.read_parquet(CACHE_FILE)
            if not df.empty and "embedding" in df.columns:
                sample_emb = df["embedding"].iloc[0]
                if hasattr(sample_emb, "__len__") and len(sample_emb) == expected_dim:
                    return df.head(count)
        except Exception:
            pass

    # If force refresh or dimension mismatch, try fetching fresh from PostgreSQL
    df = fetch_reviews_from_postgres(limit=count)

    # If DB has no new data or is offline, fallback to existing parquet cache
    if df.empty and CACHE_FILE.exists():
        try:
            fallback_df = pd.read_parquet(CACHE_FILE)
            if not fallback_df.empty and "embedding" in fallback_df.columns:
                sample_emb = fallback_df["embedding"].iloc[0]
                if hasattr(sample_emb, "__len__") and len(sample_emb) == expected_dim:
                    return fallback_df.head(count)
                else:
                    # Re-embed the cached comments using the active provider
                    comments = fallback_df["comment"].tolist()
                    with st.spinner(f"Re-indexing cached reviews using {provider.upper()} embeddings..."):
                        if provider == "nvidia":
                            fallback_df["embedding"] = embed_texts_nvidia(
                                comments, input_type="passage", api_key=nvidia_key
                            )
                        else:
                            if not _gemini_client:
                                st.error("⚠️ Gemini API key required for Gemini embeddings.")
                                return fallback_df
                            fallback_df["embedding"] = embed_texts_gemini(_gemini_client, comments)
                        fallback_df.to_parquet(CACHE_FILE, index=False)
                        return fallback_df.head(count)
        except Exception as e:
            st.error(f"Failed to read parquet cache: {e}")
            return pd.DataFrame()

    if df.empty:
        st.warning("No reviews found in PostgreSQL or local cache.")
        return df

    # Generate embeddings with active provider
    with st.spinner(f"Generating review embeddings using {provider.upper()}..."):
        if provider == "nvidia":
            embeddings = embed_texts_nvidia(
                df["comment"].tolist(), input_type="passage", api_key=nvidia_key
            )
        else:
            if _gemini_client is None:
                st.error("⚠️ Missing Gemini API key for embeddings.")
                return df
            embeddings = embed_texts_gemini(_gemini_client, df["comment"].tolist())

        df["embedding"] = embeddings

    try:
        df.to_parquet(CACHE_FILE, index=False)
    except Exception as e:
        st.warning(f"Failed to cache embeddings to parquet: {e}")

    return df


def find_similar_reviews(
    question: str,
    df: pd.DataFrame,
    provider: str = "nvidia",
    gemini_client: Optional[Any] = None,
    nvidia_key: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Vectorized cosine similarity search to retrieve top-K relevant reviews."""
    if df.empty:
        return df

    # Generate query embedding
    q_vector = None
    if provider == "nvidia":
        try:
            q_embeddings = embed_texts_nvidia(
                [question], input_type="query", api_key=nvidia_key
            )
            q_vector = np.array(q_embeddings[0], dtype=np.float32)
        except Exception as e:
            st.warning(f"NVIDIA embedding failed: {e}. Falling back to keyword search.")
    else:
        if gemini_client is not None:
            try:
                q_response = gemini_client.models.embed_content(
                    model=DEFAULT_GEMINI_EMBED_MODEL,
                    contents=[question],
                )
                q_vector = np.array(q_response.embeddings[0].values, dtype=np.float32)
            except Exception as e:
                st.warning(f"Gemini embedding failed: {e}. Falling back to keyword search.")

    # Fallback to lexical search if embedding failed
    if q_vector is None:
        query_words = set(question.lower().split())
        scores = []
        for c in df["comment"].astype(str):
            words = set(c.lower().split())
            scores.append(len(query_words & words) / max(1, len(query_words)))
        result_df = df.copy()
        result_df["similarity_score"] = scores
        return result_df.nlargest(top_k, "similarity_score")

    # Cosine similarity
    q_norm = q_vector / (np.linalg.norm(q_vector) + 1e-10)
    matrix = np.vstack(df["embedding"].values).astype(np.float32)
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    norm_matrix = matrix / matrix_norms

    scores = np.dot(norm_matrix, q_norm)

    result_df = df.copy()
    result_df["similarity_score"] = scores
    return result_df.nlargest(top_k, "similarity_score")


def ask_nvidia_nim(
    question: str,
    top_reviews: pd.DataFrame,
    model: str = DEFAULT_NVIDIA_CHAT_MODEL,
    api_key: Optional[str] = None,
) -> str:
    """Generate answer using NVIDIA NIM (e.g. nvidia/nemotron-3.5-lightning-30b-a3b)."""
    key = api_key or os.getenv("NVIDIA_API_KEY")
    if not key or key.strip() in ("", "your_nvidia_api_key_here"):
        raise ValueError(
            "Missing NVIDIA API key! Please enter it in the sidebar or set `NVIDIA_API_KEY` in `.env`. "
            "Get a free API key with 1,000 credits at https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b"
        )

    base_url = (os.getenv("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_BASE_URL).rstrip("/")
    endpoint = f"{base_url}/chat/completions"

    context_chunks = []
    for _, row in top_reviews.iterrows():
        context_chunks.append(
            f"• [City: {row['city']} | Rating: {row['rating']}/5 | ID: {row['review_id']}] \"{row['comment']}\""
        )
    context_str = "\n".join(context_chunks)

    system_instruction = (
        "You are an expert AI data assistant for Zomato food delivery. "
        "Answer the user's question accurately, concisely, and objectively using ONLY the provided customer reviews. "
        "If the reviews do not contain enough relevant information, state that clearly and honestly. "
        "Highlight common themes, recurring feedback, and mention specific cities or ratings where relevant."
    )

    user_prompt = f"User Question: {question}\n\nRelevant Customer Reviews:\n{context_str}"

    headers = {
        "Authorization": f"Bearer {key.strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"NVIDIA NIM error ({resp.status_code}): {resp.text}")

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"Empty response from NVIDIA NIM: {data}")

    message = choices[0]["message"]
    content = message.get("content") or message.get("reasoning_content")
    if not content:
        raise ValueError(f"Empty content from NVIDIA NIM: {data}")

    return content.strip()


def ask_gemini(
    client: Any,
    question: str,
    top_reviews: pd.DataFrame,
    chat_model: str = DEFAULT_GEMINI_CHAT_MODEL,
) -> tuple[str, str]:
    """Generate answer using Gemini grounded in customer reviews with auto-fallback."""
    if client is None:
        raise ValueError("Gemini client is not initialized. Please verify your GEMINI_API_KEY.")

    context_chunks = []
    for _, row in top_reviews.iterrows():
        context_chunks.append(
            f"• [City: {row['city']} | Rating: {row['rating']}/5 | ID: {row['review_id']}] \"{row['comment']}\""
        )
    context_str = "\n".join(context_chunks)

    system_instruction = (
        "You are an expert AI data assistant for Zomato. "
        "Answer the user's question accurately and objectively using ONLY the provided customer reviews. "
        "If the reviews do not contain enough relevant information, state that honestly. "
        "Highlight common themes, sentiments, and reference cities or ratings where appropriate. "
        "Be concise, clear, and professional."
    )

    user_prompt = f"User Question: {question}\n\nRelevant Customer Reviews:\n{context_str}"

    models_to_try = [
        chat_model,
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-flash-latest",
    ]
    models_to_try = list(dict.fromkeys(models_to_try))

    last_error = None
    for candidate in models_to_try:
        try:
            response = client.models.generate_content(
                model=candidate,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                ),
            )
            return response.text.strip(), candidate
        except Exception as e:
            last_error = e
            err_str = str(e)
            if any(code in err_str for code in ["503", "UNAVAILABLE", "429", "404"]):
                continue
            raise e

    raise last_error


def ask_assistant(
    question: str,
    top_reviews: pd.DataFrame,
    model: str,
    gemini_client: Optional[Any],
    nvidia_api_key: Optional[str] = None,
) -> tuple[str, str]:
    """Unified dispatch to either NVIDIA Nemotron or Gemini."""
    model_lower = model.lower()
    if model_lower.startswith("nvidia/") or "nemotron" in model_lower:
        answer = ask_nvidia_nim(
            question=question,
            top_reviews=top_reviews,
            model=model,
            api_key=nvidia_api_key,
        )
        return answer, model
    else:
        return ask_gemini(
            client=gemini_client,
            question=question,
            top_reviews=top_reviews,
            chat_model=model,
        )


def main():
    gemini_client = get_gemini_client()

    # Sidebar configuration
    st.sidebar.title("⚙️ Model & RAG Settings")

    # Select Chat Model
    default_idx = (
        AVAILABLE_CHAT_MODELS.index(DEFAULT_CHAT_MODEL)
        if DEFAULT_CHAT_MODEL in AVAILABLE_CHAT_MODELS
        else 0
    )
    selected_model = st.sidebar.selectbox(
        "AI Chat Model:",
        options=AVAILABLE_CHAT_MODELS,
        index=default_idx,
        help="Select the AI model for RAG response generation.",
    )

    is_nvidia = selected_model.startswith("nvidia/") or "nemotron" in selected_model
    active_provider = "nvidia" if is_nvidia else "gemini"

    # Provider-specific settings in Sidebar
    nvidia_key_input = None
    if is_nvidia:
        st.sidebar.markdown("### ⚡ NVIDIA NIM Config")
        existing_nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
        if existing_nvidia_key and existing_nvidia_key != "your_nvidia_api_key_here":
            st.sidebar.success("🟢 NVIDIA API Key Active")
            nvidia_key_input = existing_nvidia_key
        else:
            nvidia_key_input = st.sidebar.text_input(
                "NVIDIA API Key:",
                type="password",
                help="Paste your nvapi-... key here or set NVIDIA_API_KEY in .env",
            )
            if not nvidia_key_input:
                st.sidebar.warning("⚠️ NVIDIA API Key needed for Nemotron")
                st.sidebar.markdown(
                    "[👉 Get free NVIDIA API Key](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b)"
                )
    else:
        st.sidebar.markdown("### 🔷 Google Gemini Config")
        if gemini_client:
            st.sidebar.success("🟢 Gemini API Key configured")
        else:
            st.sidebar.error("⚠️ Invalid or missing Gemini API key in `.env`")

    st.sidebar.markdown("---")
    active_embed_model = (
        DEFAULT_NVIDIA_EMBED_MODEL if is_nvidia else DEFAULT_GEMINI_EMBED_MODEL
    )
    st.sidebar.markdown(f"**Embedding Model:** `{active_embed_model}`")

    sample_size = st.sidebar.slider(
        "Reviews to Search Across:",
        min_value=25,
        max_value=2000,
        value=min(DEFAULT_REVIEWS_COUNT, 500),
        step=25,
    )
    top_k = st.sidebar.slider(
        "Top Reviews to Retrieve (Top-K):",
        min_value=3,
        max_value=15,
        value=DEFAULT_TOP_K,
    )

    force_refresh = st.sidebar.button("🔄 Refresh / Re-index Database")
    if force_refresh and CACHE_FILE.exists():
        CACHE_FILE.unlink(missing_ok=True)
        st.cache_data.clear()

    # Load Reviews
    review_df = load_and_embed_reviews(
        provider=active_provider,
        _gemini_client=gemini_client,
        nvidia_key=nvidia_key_input,
        count=sample_size,
        force_refresh=force_refresh,
    )

    # Main UI Header
    st.title("🍔 Zomato Customer Reviews — RAG Assistant")
    st.caption(
        f"Semantic search across **{len(review_df):,}** reviews "
        f"| Powered by **{selected_model}** & **{active_embed_model}**"
    )

    # Quick prompt buttons
    st.markdown("##### 💡 Suggested Questions")
    col1, col2, col3 = st.columns(3)
    preset_question = None
    with col1:
        if st.button("📦 Packaging complaints"):
            preset_question = "What are the common packaging complaints and spills?"
    with col2:
        if st.button("⏱️ Delivery speed & delays"):
            preset_question = "How do customers feel about delivery times and lateness?"
    with col3:
        if st.button("🍕 Food taste and freshness"):
            preset_question = "What do positive reviews say about food quality and taste?"

    # Question Input
    question = st.text_input(
        "Ask anything about customer reviews:",
        value=preset_question or "",
        placeholder="e.g. What are the most common complaints about delivery packaging?",
    )

    if question:
        with st.spinner(f"Retrieving top reviews and generating answer with {selected_model}..."):
            top_reviews = find_similar_reviews(
                question=question,
                df=review_df,
                provider=active_provider,
                gemini_client=gemini_client,
                nvidia_key=nvidia_key_input,
                top_k=top_k,
            )

            try:
                answer, used_model = ask_assistant(
                    question=question,
                    top_reviews=top_reviews,
                    model=selected_model,
                    gemini_client=gemini_client,
                    nvidia_api_key=nvidia_key_input,
                )
                st.subheader("💡 Answer")

                # Badge
                if "nemotron" in used_model.lower():
                    st.info(f"⚡ Generated by **NVIDIA Nemotron 3.5 Lightning (30B-A3B)** via NVIDIA NIM")
                else:
                    st.info(f"🔷 Generated by **Google Gemini** ({used_model})")

                st.markdown(answer)
            except Exception as e:
                st.error(f"⚠️ Unable to generate answer: {e}")

        st.markdown("---")
        with st.expander(
            f"🔍 Top {len(top_reviews)} Reviews Used as Context", expanded=False
        ):
            display_df = top_reviews[
                ["review_id", "city", "rating", "comment", "similarity_score"]
            ].copy()
            display_df["similarity_score"] = display_df["similarity_score"].round(3)
            st.dataframe(
                display_df,
                column_config={
                    "similarity_score": st.column_config.ProgressColumn(
                        "Similarity", min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                    "rating": st.column_config.NumberColumn("Stars ⭐", format="%d ⭐"),
                },
                hide_index=True,
                use_container_width=True,
            )


if __name__ == "__main__":
    main()