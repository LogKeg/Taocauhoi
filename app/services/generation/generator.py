"""
Question generation core logic.
"""
import os
import re
from pathlib import Path
from typing import List

from docx import Document

from app.core import GenerateRequest
from app.services.ai import call_ai, OPENAI_API_KEY, GEMINI_API_KEY, BASE_DIR
from app.services.text import (
    normalize_name,
    normalize_question,
    strip_leading_numbering,
    apply_synonyms,
    apply_context,
    replace_numbers,
)
from app.parsers.docx import extract_docx_content

from .prompt_builder import build_ai_prompt
from .normalizer import normalize_ai_lines


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


def _read_sample_file(path: Path) -> str:
    """Read content from a sample file."""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        doc = Document(str(path))
        return extract_docx_content(doc)
    return path.read_text(encoding="utf-8", errors="ignore")


def split_questions(text: str) -> List[str]:
    """Split text into individual questions."""
    normalized = text.replace("\r\n", "\n")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", normalized) if b.strip()]
    questions: List[str] = []

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        # Keep multiple-choice options attached to the preceding question line
        merged_lines: List[str] = []
        option_re = re.compile(r"^[A-H][).\-:]\s+", re.IGNORECASE)
        has_options = False
        for line in lines:
            if option_re.match(line) and merged_lines:
                merged_lines[-1] = f"{merged_lines[-1]}\n{line}"
                has_options = True
            else:
                merged_lines.append(line)
        lines = merged_lines

        # Remove consecutive duplicate lines
        deduped_lines: List[str] = []
        for line in lines:
            if not deduped_lines or deduped_lines[-1] != line:
                deduped_lines.append(line)
        lines = deduped_lines

        if has_options:
            questions.append("\n".join(lines).strip())
            continue

        numbered_lines = [ln for ln in lines if re.match(r"^\d{1,3}[).\-:]\s+", ln)]
        if numbered_lines:
            buffer: List[str] = []
            for line in lines:
                if re.match(r"^\d{1,3}[).\-:]\s+", line):
                    if buffer:
                        questions.append(" ".join(buffer).strip())
                        buffer = []
                    line = re.sub(r"^\d{1,3}[).\-:]\s+", "", line)
                buffer.append(line)
            if buffer:
                questions.append(" ".join(buffer).strip())
            continue

        joined = " ".join(lines).strip()
        parts = [p.strip() for p in re.split(r"(?<=[?？])\s+", joined) if p.strip()]
        if len(parts) > 1:
            questions.extend(parts)
        else:
            questions.append(joined)

    # Final de-duplication while preserving order
    seen = set()
    unique_questions: List[str] = []
    for q in questions:
        key = q.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique_questions.append(q)
    return unique_questions


def load_questions_from_subject(subject: str) -> List[str]:
    """Load sample questions from a subject directory."""
    if SAMPLE_DIR is None:
        return []
    subject_dir = (SAMPLE_DIR / subject).resolve()
    if not SAMPLE_DIR.exists() or SAMPLE_DIR not in subject_dir.parents:
        return []
    if not subject_dir.exists() or not subject_dir.is_dir():
        return []
    questions: List[str] = []
    for path in subject_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".txt", ".docx", ".md"}:
            content = _read_sample_file(path)
            questions.extend(split_questions(content))
    return [q for q in questions if q]


def is_engine_available(engine: str) -> bool:
    """Check if an AI engine is available."""
    if engine == "gemini":
        return bool(GEMINI_API_KEY)
    if engine == "ollama":
        return True  # Ollama runs locally, availability checked at call time
    return bool(OPENAI_API_KEY)


def _rewrite_mcq_block(block: str) -> str:
    """Rewrite MCQ block to vary the structure."""
    from app.core import MCQ_OPTION_RE

    lines = block.splitlines()
    if len(lines) < 2:
        return block
    if not MCQ_OPTION_RE.match(lines[1]):
        return block
    question = lines[0]
    prefix = ""
    if ":" in question:
        head, tail = question.split(":", 1)
        if len(head.split()) <= 5:
            prefix = head.strip()
            question = tail.strip()
    options = [ln for ln in lines[1:] if MCQ_OPTION_RE.match(ln)]
    if not options:
        return block
    # Rebuild with rewritten question
    new_lines = [f"{prefix}: {question}" if prefix else question]
    new_lines.extend(options)
    return "\n".join(new_lines)


def _force_variation(text: str) -> str:
    """Force variation when AI returns similar content."""
    from app.core import MCQ_OPTION_RE

    stripped = text.strip()
    if not stripped:
        return stripped

    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    is_mcq = len(lines) > 1 and any(MCQ_OPTION_RE.match(ln) for ln in lines[1:])

    if is_mcq:
        return _rewrite_mcq_block(stripped)

    # Heuristic: if mostly ASCII, use English prefix; otherwise Vietnamese
    ascii_ratio = sum(1 for ch in stripped if ord(ch) < 128) / max(1, len(stripped))
    if ascii_ratio > 0.9:
        return f"Choose the correct option: {stripped}"
    return f"Hãy cho biết: {stripped}"


def generate_variants(req: GenerateRequest) -> List[str]:
    """Generate question variants based on request parameters."""
    if req.use_ai and is_engine_available(req.ai_engine):
        generated: List[str] = []
        samples = req.samples or [None]
        for sample in samples:
            if sample:
                sample = strip_leading_numbering(sample)
            prompt = build_ai_prompt(
                sample,
                req.topic,
                req.custom_keywords,
                req.paraphrase,
                req.change_numbers,
                req.change_context,
                req.variants_per_question,
            )
            attempts = 0
            while attempts < 2:
                attempts += 1
                text, err = call_ai(prompt, req.ai_engine)
                if text:
                    lines = [strip_leading_numbering(line) for line in normalize_ai_lines(text)]
                    if sample:
                        lines = [ln for ln in lines if ln.strip().lower() != sample.strip().lower()]
                    if lines:
                        for ln in lines[: req.variants_per_question]:
                            generated.append(_rewrite_mcq_block(ln))
                        break
                prompt = prompt + "\nNếu câu trả về trùng câu gốc, hãy viết lại hoàn toàn khác.\n"
        if generated:
            # De-duplicate while preserving order
            seen = set()
            unique = []
            for q in generated:
                if q and q not in seen:
                    seen.add(q)
                    unique.append(q)
            return unique

    results: List[str] = []
    for sample in req.samples:
        sample = sample.strip()
        if not sample:
            continue
        sample = strip_leading_numbering(sample)
        for _ in range(req.variants_per_question):
            variant = sample
            if req.paraphrase:
                variant = apply_synonyms(variant)
            if req.change_numbers:
                variant = replace_numbers(variant)
            if req.change_context:
                variant = apply_context(variant, req.topic, req.custom_keywords)
            variant = strip_leading_numbering(variant)
            variant = _rewrite_mcq_block(variant)
            if normalize_question(variant) == normalize_question(sample):
                variant = _force_variation(variant)
            results.append(variant)
    return results
