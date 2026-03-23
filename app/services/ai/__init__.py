"""
AI service clients for OpenAI, Gemini, Ollama, and Anthropic Claude.
"""
from typing import Optional, Tuple

from .openai_client import call_openai, extract_text_from_response
from .gemini_client import call_gemini
from .ollama_client import call_ollama
from .anthropic_client import call_anthropic
from .config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_API_BASE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OLLAMA_BASE,
    OLLAMA_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    load_saved_settings,
    save_settings_to_file,
    reload_settings,
    SETTINGS_FILE,
    BASE_DIR,
)


def call_ai(prompt: str, engine: str = "openai") -> Tuple[Optional[str], Optional[str]]:
    """Call AI engine with the given prompt.

    Args:
        prompt: The prompt to send
        engine: One of "openai", "gemini", "ollama", "anthropic"

    Returns:
        Tuple of (response_text, error_message)
    """
    if engine == "gemini":
        return call_gemini(prompt)
    if engine == "ollama":
        return call_ollama(prompt)
    if engine == "anthropic":
        return call_anthropic(prompt)
    return call_openai(prompt)


__all__ = [
    # Clients
    "call_ai",
    "call_openai",
    "call_gemini",
    "call_ollama",
    "call_anthropic",
    "extract_text_from_response",
    # Config
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_API_BASE",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "OLLAMA_BASE",
    "OLLAMA_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "load_saved_settings",
    "save_settings_to_file",
    "reload_settings",
    "SETTINGS_FILE",
    "BASE_DIR",
]
