"""
AI service configuration and settings management.
"""
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SETTINGS_FILE = BASE_DIR / "ai_settings.json"


def load_saved_settings() -> dict:
    """Load AI settings from file."""
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_settings_to_file(data: dict) -> None:
    """Save AI settings to file."""
    try:
        SETTINGS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


# Load settings on import
_saved = load_saved_settings()

# OpenAI settings
OPENAI_API_KEY = _saved.get("openai_key") or os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = _saved.get("openai_model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_BASE = _saved.get("openai_base") or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

# Gemini settings
GEMINI_API_KEY = _saved.get("gemini_key") or os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = _saved.get("gemini_model") or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Ollama settings
OLLAMA_BASE = _saved.get("ollama_base") or os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = _saved.get("ollama_model") or os.getenv("OLLAMA_MODEL", "qwen3.5:4b")

# Anthropic Claude settings
ANTHROPIC_API_KEY = _saved.get("anthropic_key") or os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = _saved.get("anthropic_model") or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")


def reload_settings():
    """Reload settings from file and update module variables."""
    global OPENAI_API_KEY, OPENAI_MODEL, OPENAI_API_BASE
    global GEMINI_API_KEY, GEMINI_MODEL
    global OLLAMA_BASE, OLLAMA_MODEL
    global ANTHROPIC_API_KEY, ANTHROPIC_MODEL

    _saved = load_saved_settings()

    OPENAI_API_KEY = _saved.get("openai_key") or os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = _saved.get("openai_model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_API_BASE = _saved.get("openai_base") or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    GEMINI_API_KEY = _saved.get("gemini_key") or os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = _saved.get("gemini_model") or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    OLLAMA_BASE = _saved.get("ollama_base") or os.getenv("OLLAMA_BASE", "http://localhost:11434")
    OLLAMA_MODEL = _saved.get("ollama_model") or os.getenv("OLLAMA_MODEL", "qwen3.5:4b")

    ANTHROPIC_API_KEY = _saved.get("anthropic_key") or os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = _saved.get("anthropic_model") or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
