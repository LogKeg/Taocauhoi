"""
AI response normalization for question generation.
"""
import re
from typing import List

from app.core import MCQ_OPTION_RE, LEADING_NUM_RE
from app.services.text import strip_leading_numbering


# Label pattern for question headers like "Câu 1:", "Question 2:", etc.
LABEL_RE = re.compile(r"^(Câu(\s+hỏi)?|Question)\s*\d*\s*:?\s*$", re.IGNORECASE)


def normalize_ai_lines(text: str) -> List[str]:
    """Normalize AI response into individual question lines."""
    lines = []
    for raw in text.splitlines():
        cleaned = re.sub(r"^\s*\d+[\).\-:]\s*", "", raw).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def normalize_ai_blocks(text: str) -> List[str]:
    """Normalize AI response into question blocks (for MCQ with options)."""
    normalized = text.replace("\r\n", "\n").strip()
    blocks = [b.strip() for b in re.split(r"\n\s*\n", normalized) if b.strip()]

    # First pass: merge blocks that are MCQ options or question labels
    merged: List[str] = []
    for block in blocks:
        first_line = block.splitlines()[0].strip() if block.strip() else ""
        is_option_block = bool(MCQ_OPTION_RE.match(first_line))
        is_label = bool(LABEL_RE.match(block.strip()))
        if is_option_block and merged:
            # Attach options to previous question
            merged[-1] = merged[-1] + "\n" + block
        elif is_label:
            # "Câu 1:", "Câu hỏi 2:" — start a new entry that will absorb
            # the next block as the actual question content
            merged.append(block)
        elif merged and LABEL_RE.match(merged[-1].splitlines()[0].strip()):
            # Previous block was just a label; merge this content into it
            merged[-1] = merged[-1] + "\n" + block
        else:
            merged.append(block)

    # Second pass: clean up numbering and drop empty / separator blocks
    cleaned: List[str] = []
    for block in merged:
        lines = block.splitlines()
        # Remove leading label lines like "Câu 1:"
        while lines and LABEL_RE.match(lines[0].strip()):
            lines.pop(0)
        if lines:
            lines[0] = strip_leading_numbering(lines[0])
        result = "\n".join(lines).strip()
        if not result or re.match(r"^-{2,}$", result):
            continue
        # Skip intro/filler lines that aren't actual questions
        # (no "?" and no MCQ options means it's likely just a preamble)
        has_question_mark = "?" in result
        has_options = any(MCQ_OPTION_RE.match(ln.strip()) for ln in result.splitlines())
        has_blank = "..." in result
        if not has_question_mark and not has_options and not has_blank:
            continue
        cleaned.append(result)
    return cleaned
