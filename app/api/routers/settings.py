"""
Settings and configuration API endpoints.
"""
import os
import httpx
from fastapi import APIRouter, Request

from app.core import TOPICS, TEMPLATES, SUBJECTS, SUBJECT_TOPICS, QUESTION_TYPES
from app.services.ai import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_API_BASE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OLLAMA_BASE,
    OLLAMA_MODEL,
    save_settings_to_file,
    reload_settings,
)

router = APIRouter(tags=["settings"])


@router.post("/ai-settings")
async def update_ai_settings(request: Request) -> dict:
    """Update AI settings (API keys, models, etc.)."""
    # Need to reload module-level variables
    import app.services.ai.config as ai_config

    data = await request.json()
    if "openai_key" in data:
        ai_config.OPENAI_API_KEY = data["openai_key"].strip()
    if "openai_model" in data:
        ai_config.OPENAI_MODEL = data["openai_model"].strip() or "gpt-4o-mini"
    if "openai_base" in data:
        ai_config.OPENAI_API_BASE = data["openai_base"].strip() or "https://api.openai.com/v1"
    if "gemini_key" in data:
        ai_config.GEMINI_API_KEY = data["gemini_key"].strip()
    if "gemini_model" in data:
        ai_config.GEMINI_MODEL = data["gemini_model"].strip() or "gemini-2.0-flash"
    if "ollama_base" in data:
        ai_config.OLLAMA_BASE = data["ollama_base"].strip() or "http://localhost:11434"
    if "ollama_model" in data:
        ai_config.OLLAMA_MODEL = data["ollama_model"].strip() or "llama3.2:latest"
    if "anthropic_key" in data:
        ai_config.ANTHROPIC_API_KEY = data["anthropic_key"].strip()
    if "anthropic_model" in data:
        ai_config.ANTHROPIC_MODEL = data["anthropic_model"].strip() or "claude-sonnet-4-20250514"
    if "anthropic_base" in data:
        ai_config.ANTHROPIC_API_BASE = data["anthropic_base"].strip() or "https://api.anthropic.com"

    save_settings_to_file({
        "openai_key": ai_config.OPENAI_API_KEY,
        "openai_model": ai_config.OPENAI_MODEL,
        "openai_base": ai_config.OPENAI_API_BASE,
        "gemini_key": ai_config.GEMINI_API_KEY,
        "gemini_model": ai_config.GEMINI_MODEL,
        "ollama_base": ai_config.OLLAMA_BASE,
        "ollama_model": ai_config.OLLAMA_MODEL,
        "anthropic_key": ai_config.ANTHROPIC_API_KEY,
        "anthropic_model": ai_config.ANTHROPIC_MODEL,
        "anthropic_base": ai_config.ANTHROPIC_API_BASE,
    })
    return {"ok": True}


@router.get("/ai-settings")
def get_ai_settings() -> dict:
    """Get current AI settings (keys are masked)."""
    import app.services.ai.config as ai_config
    return {
        "openai_key": ai_config.OPENAI_API_KEY[:4] + "****" if len(ai_config.OPENAI_API_KEY) > 4 else "",
        "openai_model": ai_config.OPENAI_MODEL,
        "openai_base": ai_config.OPENAI_API_BASE,
        "gemini_key": ai_config.GEMINI_API_KEY[:4] + "****" if len(ai_config.GEMINI_API_KEY) > 4 else "",
        "gemini_model": ai_config.GEMINI_MODEL,
        "ollama_base": ai_config.OLLAMA_BASE,
        "ollama_model": ai_config.OLLAMA_MODEL,
        "anthropic_key": ai_config.ANTHROPIC_API_KEY[:4] + "****" if len(ai_config.ANTHROPIC_API_KEY) > 4 else "",
        "anthropic_model": ai_config.ANTHROPIC_MODEL,
        "anthropic_base": ai_config.ANTHROPIC_API_BASE,
    }


@router.get("/ai-engines")
def list_ai_engines() -> dict:
    """List available AI engines and their status."""
    import app.services.ai.config as ai_config

    engines = []
    if ai_config.OPENAI_API_KEY:
        engines.append({"key": "openai", "label": f"OpenAI ({ai_config.OPENAI_MODEL})", "available": True})
    else:
        engines.append({"key": "openai", "label": "OpenAI (chưa có API key)", "available": False})

    if ai_config.GEMINI_API_KEY:
        engines.append({"key": "gemini", "label": f"Gemini ({ai_config.GEMINI_MODEL})", "available": True})
    else:
        engines.append({"key": "gemini", "label": "Gemini (chưa có API key)", "available": False})

    # Ollama: try a quick check
    ollama_ok = False
    try:
        with httpx.Client(timeout=2) as client:
            r = client.get(f"{ai_config.OLLAMA_BASE}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        pass

    if ollama_ok:
        engines.append({"key": "ollama", "label": f"Ollama ({ai_config.OLLAMA_MODEL})", "available": True})
    else:
        engines.append({"key": "ollama", "label": "Ollama (không kết nối được)", "available": False})

    # Anthropic Claude
    if ai_config.ANTHROPIC_API_KEY:
        engines.append({"key": "anthropic", "label": f"Claude ({ai_config.ANTHROPIC_MODEL})", "available": True})
    else:
        engines.append({"key": "anthropic", "label": "Claude (chưa có API key)", "available": False})

    return {"engines": engines}


@router.get("/topics")
def list_topics() -> dict:
    """List available topics for question generation."""
    return {
        "topics": [
            {"key": key, "label": value["label"], "keywords": value["keywords"]}
            for key, value in TOPICS.items()
        ]
    }


@router.get("/subjects")
def list_subjects() -> dict:
    """List available subjects, question types, and grades."""
    return {
        "subjects": [
            {"key": key, "label": value["label"], "lang": value.get("lang", "vi")}
            for key, value in SUBJECTS.items()
        ],
        "question_types": [
            {"key": key, "label": value} for key, value in QUESTION_TYPES.items()
        ],
        "grades": list(range(1, 13)),
        "topics": {key: topics for key, topics in SUBJECT_TOPICS.items()},
    }


@router.get("/templates")
def list_templates() -> dict:
    """List available question templates."""
    return {
        "templates": [
            {
                "key": key,
                "label": TOPICS.get(key, {}).get("label", key),
                "items": TEMPLATES.get(key, []),
            }
            for key in TEMPLATES.keys()
        ]
    }


@router.get("/version")
def version() -> dict:
    """Get application version info."""
    return {
        "commit": os.getenv("VERCEL_GIT_COMMIT_SHA", ""),
        "time": os.getenv("VERCEL_GIT_COMMIT_MESSAGE", ""),
    }
