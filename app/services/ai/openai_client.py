"""
OpenAI API client.
"""
from typing import Optional, Tuple

import httpx

from .config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_API_BASE


def extract_text_from_response(payload: dict) -> str:
    """Extract text content from OpenAI API response."""
    if isinstance(payload, dict):
        if payload.get("output_text"):
            return payload["output_text"]
        output = payload.get("output", [])
        for item in output:
            if item.get("type") == "message":
                content = item.get("content", [])
                parts = []
                for c in content:
                    if c.get("type") in {"output_text", "text"} and c.get("text"):
                        parts.append(c.get("text", ""))
                if parts:
                    return "\n".join(parts).strip()
    return ""


def call_openai(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Call OpenAI API with the given prompt.

    Returns:
        Tuple of (response_text, error_message)
    """
    # Import dynamically to get current settings
    from .config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_API_BASE

    if not OPENAI_API_KEY:
        return None, "missing_api_key"

    url = f"{OPENAI_API_BASE}/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": OPENAI_MODEL,
        "input": prompt,
        "text": {"format": {"type": "text"}},
        "max_output_tokens": 400,
        "temperature": 0.8,
    }

    with httpx.Client(timeout=30) as client:
        response = client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            try:
                data = response.json()
                msg = data.get("error", {}).get("message", "")
            except Exception:
                msg = response.text[:200]
            return None, f"{response.status_code}: {msg}"
        return extract_text_from_response(response.json()), None
