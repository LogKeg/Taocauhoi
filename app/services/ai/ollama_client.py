"""
Ollama API client.
"""
from typing import Optional, Tuple

import httpx

from .config import OLLAMA_BASE, OLLAMA_MODEL


def call_ollama(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Call Ollama API with the given prompt.

    Returns:
        Tuple of (response_text, error_message)
    """
    # Import dynamically to get current settings
    from .config import OLLAMA_BASE, OLLAMA_MODEL

    url = f"{OLLAMA_BASE}/api/generate"

    # Check if model is Qwen 3/3.5 (has thinking mode)
    is_qwen3 = "qwen3" in OLLAMA_MODEL.lower()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 4096,
            "temperature": 0.7,
        },
    }

    # Disable thinking mode for Qwen 3/3.5 models
    if is_qwen3:
        payload["think"] = False

    try:
        with httpx.Client(timeout=300) as client:  # 5 minutes for first load
            response = client.post(url, json=payload)
            if response.status_code != 200:
                try:
                    data = response.json()
                    msg = data.get("error", response.text[:200])
                except Exception:
                    msg = response.text[:200]
                return None, f"{response.status_code}: {msg}"

            data = response.json()
            text = data.get("response", "")
            return text.strip() if text else None, None
    except httpx.ConnectError:
        return None, "Không kết nối được Ollama. Hãy chắc chắn Ollama đang chạy (ollama serve)."
