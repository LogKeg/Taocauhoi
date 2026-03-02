"""
Google Gemini API client.
"""
from typing import Optional, Tuple

import httpx

from .config import GEMINI_API_KEY, GEMINI_MODEL


def call_gemini(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Call Gemini API with the given prompt.

    Returns:
        Tuple of (response_text, error_message)
    """
    # Import dynamically to get current settings
    from .config import GEMINI_API_KEY, GEMINI_MODEL

    if not GEMINI_API_KEY:
        return None, "missing_gemini_api_key"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 400,
            "temperature": 0.8,
        },
    }

    with httpx.Client(timeout=30) as client:
        response = client.post(url, json=payload)
        if response.status_code != 200:
            try:
                data = response.json()
                msg = data.get("error", {}).get("message", "")
            except Exception:
                msg = response.text[:200]
            return None, f"{response.status_code}: {msg}"

        data = response.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text.strip(), None
        except (KeyError, IndexError):
            return None, "Không parse được phản hồi Gemini"
