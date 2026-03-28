"""
Helper functions for generation endpoints.

Shared utilities for question generation, PDF font handling, and text wrapping.
"""
import os
from pathlib import Path
from typing import List

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from app.services.ai import BASE_DIR
from app.services.text import normalize_name


def _resolve_sample_dir() -> Path:
    """Resolve the sample directory path."""
    env_dir = os.getenv("SAMPLE_DIR")
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
        if path.exists() and path.is_dir():
            return path

    target = normalize_name("đề mẫu")
    for entry in BASE_DIR.iterdir():
        if entry.is_dir() and normalize_name(entry.name) == target:
            return entry
    return None


SAMPLE_DIR = _resolve_sample_dir()


def _is_engine_available(engine: str) -> bool:
    """Check if an AI engine is available."""
    from app.services.ai import OPENAI_API_KEY, GEMINI_API_KEY
    if engine == "gemini":
        return bool(GEMINI_API_KEY)
    if engine == "ollama":
        return True
    return bool(OPENAI_API_KEY)


def _get_pdf_font():
    """Get a font that supports Vietnamese characters."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            font_name = "VietnameseFont"
            if font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, path))
            return font_name, path
    return "Helvetica", None


def _wrap_text(text: str, max_width: float, font_name: str, font_size: int) -> List[str]:
    """Wrap text to fit within max_width."""
    words = text.split()
    if not words:
        return [text]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _save_text_questions_to_bank(
    questions: List[str],
    subject: str = "general",
    source: str = "generated",
    difficulty: str = "medium",
    explanations: List[str] = None,
) -> int:
    """Save generated text questions to the question bank."""
    from app.database import SessionLocal, QuestionCRUD

    saved = 0
    db = SessionLocal()
    try:
        for i, q in enumerate(questions):
            if not q.strip():
                continue
            explanation = explanations[i] if explanations and i < len(explanations) else None
            QuestionCRUD.create(
                db,
                content=q.strip(),
                subject=subject,
                source=source,
                difficulty=difficulty,
                explanation=explanation,
            )
            saved += 1
    finally:
        db.close()
    return saved


def _parse_explanations(explanations_text: str) -> List[str]:
    """Parse explanations text into list of individual explanations."""
    import re
    if not explanations_text:
        return []

    explanations = []
    lines = explanations_text.strip().split('\n')
    current_expl = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check if line starts with a number (new explanation)
        match = re.match(r'^(?:Câu\s*)?(\d+)[\.\):\s]+(.+)', line, re.IGNORECASE)
        if match:
            if current_expl:
                explanations.append(' '.join(current_expl))
            current_expl = [match.group(2).strip()]
        else:
            current_expl.append(line)

    if current_expl:
        explanations.append(' '.join(current_expl))

    return explanations
