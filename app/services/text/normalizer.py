"""
Text normalization utilities.
"""
import re
import unicodedata

from app.core.constants import LEADING_NUM_RE

# Question prefix pattern
_QUESTION_PREFIX_RE = re.compile(
    r"^\s*(Câu(\s+hỏi)?\s*\d*\s*[:.)]\s*|Question\s*\d*\s*[:.)]\s*)",
    re.IGNORECASE,
)


def normalize_name(value: str) -> str:
    """Remove accents and normalize text for comparison."""
    text = unicodedata.normalize("NFD", value)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\\s+", " ", text).strip().lower()


def strip_leading_numbering(text: str) -> str:
    """Remove leading numbers and question prefixes from text."""
    text = _QUESTION_PREFIX_RE.sub("", text)
    return LEADING_NUM_RE.sub("", text).strip()


def normalize_question(text: str) -> str:
    """Normalize question text for comparison."""
    return strip_leading_numbering(text).strip().lower()
