"""
Anthropic Claude API client.
"""
from typing import Optional, Tuple

import httpx

from .config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL


def call_anthropic(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Call Anthropic Claude API with the given prompt.

    Returns:
        Tuple of (response_text, error_message)
    """
    # Import dynamically to get current settings
    from .config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

    if not ANTHROPIC_API_KEY:
        return None, "missing_anthropic_api_key"

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }

    with httpx.Client(timeout=60) as client:
        response = client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            try:
                data = response.json()
                msg = data.get("error", {}).get("message", "")
            except Exception:
                msg = response.text[:200]
            return None, f"{response.status_code}: {msg}"

        data = response.json()
        try:
            # Extract text from content blocks
            content = data.get("content", [])
            texts = [block.get("text", "") for block in content if block.get("type") == "text"]
            return "\n".join(texts).strip(), None
        except (KeyError, IndexError):
            return None, "Không parse được phản hồi Anthropic"
