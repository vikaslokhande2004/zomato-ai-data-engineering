"""
Unified LLM Provider for Zomato AI Data Engineering.
Supports:
1. NVIDIA NIM (nvidia/nemotron-3.5-lightning-30b-a3b) via OpenAI-compatible API
2. Google Gemini (gemini-3.5-flash-lite, etc.) via Google GenAI SDK
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load environment variables
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

# Constants
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
DEFAULT_NVIDIA_EMBED_MODEL = "nvidia/nemotron-3.5-embed-1b"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GEMINI_EMBED_MODEL = "gemini-embedding-001"

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

    # 1. Strip markdown code fences if present
    if "```" in cleaned:
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

    # 2. Try raw_decode starting from first '{'
    start = cleaned.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3. Extract outermost curly braces as fallback
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as err:
        sanitized = re.sub(r",\s*([\]}])", r"\1", cleaned)
        sanitized = sanitized.replace("'", '"')
        return json.loads(sanitized)


def get_model_provider(model_name: str) -> str:
    """Determine provider based on model identifier or environment settings."""
    model_lower = model_name.lower()
    if model_lower.startswith("nvidia/") or "nemotron" in model_lower:
        return "nvidia"
    if model_lower.startswith("gemini"):
        return "gemini"

    # Fallback to configured provider or presence of API keys
    explicit = os.getenv("AI_PROVIDER")
    if explicit:
        return explicit.lower()
    if os.getenv("NVIDIA_API_KEY"):
        return "nvidia"
    return "gemini"


def call_nvidia_nim_chat(
    messages: list[dict[str, str]],
    model: str = DEFAULT_NVIDIA_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """Call NVIDIA NIM API (OpenAI-compatible endpoint) using requests."""
    key = api_key or os.getenv("NVIDIA_API_KEY")
    if not key or key.strip() in ("", "your_nvidia_api_key_here"):
        raise ValueError(
            "Missing NVIDIA API key! Please set NVIDIA_API_KEY in your .env file or environment variables. "
            "Get a free API key at: https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b"
        )

    endpoint_base = (
        base_url or os.getenv("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_BASE_URL
    ).rstrip("/")
    endpoint = f"{endpoint_base}/chat/completions"

    headers = {
        "Authorization": f"Bearer {key.strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"NVIDIA NIM API Error ({resp.status_code}): {resp.text}"
        )

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"Empty choices received from NVIDIA NIM: {data}")

    message = choices[0]["message"]
    content = message.get("content")
    if not content:
        # Fallback to reasoning_content if token budget split between reasoning and final answer
        content = message.get("reasoning_content")

    if not content:
        raise ValueError(f"No content returned in message: {message}")

    return content.strip()


def classify_with_nemotron(
    comment: str,
    model: str = DEFAULT_NVIDIA_MODEL,
    api_key: Optional[str] = None,
) -> dict:
    """Classify a customer review using NVIDIA Nemotron 3.5 Lightning."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Customer review text to classify:\n\"{comment}\""},
    ]

    raw_text = call_nvidia_nim_chat(
        messages=messages,
        model=model,
        temperature=0.0,
        max_tokens=4096,
        api_key=api_key,
    )

    parsed_json = extract_json_from_text(raw_text)

    # Validate with Pydantic schema
    validated = ReviewClassification.model_validate(parsed_json)
    return validated.model_dump()


def classify_with_gemini(
    gemini_client: Any,
    model: str,
    comment: str,
) -> dict:
    """Classify a customer review using Google Gemini with structured schema."""
    from google.genai import types

    response = gemini_client.models.generate_content(
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


def classify_review_unified(
    comment: str,
    model: str,
    gemini_client: Optional[Any] = None,
    nvidia_api_key: Optional[str] = None,
) -> dict:
    """Unified classification dispatcher supporting Nemotron & Gemini."""
    provider = get_model_provider(model)

    if provider == "nvidia":
        return classify_with_nemotron(comment, model=model, api_key=nvidia_api_key)
    else:
        if gemini_client is None:
            from google import genai
            gkey = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            gemini_client = genai.Client(api_key=gkey)
        return classify_with_gemini(gemini_client, model=model, comment=comment)


def generate_rag_response(
    question: str,
    context_reviews: list[dict],
    model: str = DEFAULT_NVIDIA_MODEL,
    gemini_client: Optional[Any] = None,
    nvidia_api_key: Optional[str] = None,
) -> tuple[str, str]:
    """Generate grounded RAG answer using NVIDIA Nemotron or Google Gemini."""
    context_chunks = []
    for row in context_reviews:
        city = row.get("city", "Unknown")
        rating = row.get("rating", 3)
        rid = row.get("review_id", "N/A")
        comment = row.get("comment", "")
        context_chunks.append(
            f"• [City: {city} | Rating: {rating}/5 | ID: {rid}] \"{comment}\""
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

    provider = get_model_provider(model)

    if provider == "nvidia":
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ]
        answer = call_nvidia_nim_chat(
            messages=messages,
            model=model,
            temperature=0.2,
            max_tokens=1024,
            api_key=nvidia_api_key,
        )
        return answer, model
    else:
        from google.genai import types

        if gemini_client is None:
            from google import genai
            gkey = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            gemini_client = genai.Client(api_key=gkey)

        response = gemini_client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )
        return response.text.strip(), model
