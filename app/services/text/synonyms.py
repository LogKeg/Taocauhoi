"""
Synonym replacement and context application utilities.
"""
import random
from typing import List

from app.core.constants import TOPICS, SYNONYMS


def apply_synonyms(text: str) -> str:
    """Replace words with their synonyms."""
    out = text
    for src, options in SYNONYMS.items():
        if src in out:
            out = out.replace(src, random.choice(options))
    return out


def apply_context(text: str, topic_key: str, custom_keywords: List[str]) -> str:
    """Apply topic-specific context replacements."""
    out = text
    topic = TOPICS.get(topic_key, {})
    for src, dst in topic.get("context", {}).items():
        out = out.replace(src, dst)
    if custom_keywords:
        for kw in custom_keywords:
            if kw.strip():
                out = out.replace("...", kw.strip(), 1)
    return out
