import io
import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from fastapi import FastAPI, Form, Request, UploadFile, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# Import from core module
from app.core import (
    TOPICS,
    TEMPLATES,
    TOPIC_AI_GUIDE,
    SUBJECTS,
    SUBJECT_TOPICS,
    QUESTION_TYPES,
    SYNONYMS,
    NUMBER_RE,
    LEADING_NUM_RE,
    MCQ_OPTION_RE,
    ANSWER_TEMPLATES,
    GenerateRequest,
    ParseSamplesRequest,
    QuestionCreate,
    QuestionUpdate,
    ExamCreate,
    ExamUpdate,
    BulkSaveRequest,
    AIAnalyzeRequest,
    AISuggestRequest,
    AIReviewRequest,
)

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include API routers (will be added after all imports)
# app.include_router(settings_router)  # Settings routes moved to router

# Import AI services
from app.services.ai import (
    call_ai as _call_ai,
    call_openai as _call_openai,
    call_gemini as _call_gemini,
    call_ollama as _call_ollama,
    extract_text_from_response as _extract_text_from_response,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_API_BASE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OLLAMA_BASE,
    OLLAMA_MODEL,
    load_saved_settings as _load_saved_settings,
    save_settings_to_file as _save_settings_to_file,
    SETTINGS_FILE,
    BASE_DIR,
)

# Import text services
from app.services.text import (
    normalize_name as _normalize_name,
    normalize_question as _normalize_question,
    strip_leading_numbering as _strip_leading_numbering,
    apply_synonyms as _apply_synonyms,
    apply_context as _apply_context,
    replace_numbers as _replace_numbers,
)

# Import math services
from app.services.math import (
    omml_to_latex,
    omml_children_to_latex,
)

# Import DOCX parsers
from app.parsers.docx import (
    extract_paragraph_with_math,
    extract_cell_with_math,
    extract_docx_content,
    extract_docx_lines,
    parse_cell_based_questions,
    extract_docx_lines_with_options,
    parse_bilingual_questions,
)

# Import API routers
from app.api.routers import (
    settings_router,
    questions_router,
    exams_router,
    import_export_router,
    ai_features_router,
    generation_router,
    grading_router,
    parsing_router,
    crawler_router,
    storage_router,
    curriculum_router,
    system_router,
    ai_tools_router,
)


def _resolve_sample_dir() -> Optional[Path]:
    env_dir = os.getenv("SAMPLE_DIR")
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
        if path.exists() and path.is_dir():
            return path

    target = _normalize_name("đề mẫu")
    for entry in BASE_DIR.iterdir():
        if entry.is_dir() and _normalize_name(entry.name) == target:
            return entry
    return None


SAMPLE_DIR = _resolve_sample_dir()


def _is_mcq_block(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return len(lines) > 1 and any(MCQ_OPTION_RE.match(ln) for ln in lines[1:])


def _force_variation(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    if _is_mcq_block(stripped):
        return _rewrite_mcq_block(stripped)
    # Heuristic: if mostly ASCII, use English prefix; otherwise Vietnamese.
    ascii_ratio = sum(1 for ch in stripped if ord(ch) < 128) / max(1, len(stripped))
    if ascii_ratio > 0.9:
        return f"Choose the correct option: {stripped}"
    return f"Hãy cho biết: {stripped}"


def _rewrite_english_question(question: str) -> str:
    q = question.strip()
    if not q:
        return q
    pattern = re.compile(
        r"^The\s+\.\.\.\s+you\s+try,\s+the\s+more\s+likely\s+you\s+are\s+to\s+be\s+successful\.?$",
        re.IGNORECASE,
    )
    if pattern.match(q):
        return "The more ... you attempt, the higher your chance of success."

    pattern2 = re.compile(
        r"^All the players did their best apart from Johnson\. Johnson was \.\.\. his best\.?$",
        re.IGNORECASE,
    )
    if pattern2.match(q):
        variants = [
            "Only Johnson failed to give maximum effort. Complete the sentence: Johnson did not ... his best.",
            "Everyone except Johnson gave their best; Johnson did not ... his best.",
            "Johnson was the only player who didn't perform to his best. Choose the correct completion.",
            "All the other players performed at their peak. Johnson, however, did not ... his best.",
            "Unlike the rest of the team, Johnson didn't ... his best. Select the best completion.",
            "The entire squad tried their hardest, but Johnson did not ... his best. Choose the correct option.",
        ]
        return random.choice(variants)

    candidates = [
        (r"\bapart from\b", "except for"),
        (r"\bdid their best\b", "performed to the best of their ability"),
        (r"\bwas \.\.\. his best\b", "did not ... his best"),
        (r"\bmore likely\b", "more probable"),
        (r"\byou are to be successful\b", "you will succeed"),
        (r"\bthe more\b", "the greater"),
        (r"\btry\b", "attempt"),
    ]
    rewritten = q
    for src, dst in candidates:
        rewritten = re.sub(src, dst, rewritten, flags=re.IGNORECASE)
    if _normalize_question(rewritten) != _normalize_question(q):
        return rewritten
    return q


def _rewrite_mcq_block(block: str) -> str:
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
    ascii_ratio = sum(1 for ch in question if ord(ch) < 128) / max(1, len(question))
    if ascii_ratio > 0.9:
        rewritten = _rewrite_english_question(question)
    else:
        rewritten = question
    if _normalize_question(rewritten) == _normalize_question(question):
        rewritten = f"Complete the sentence: {question}"
    # Context swap for English MCQ to force stronger change.
    if ascii_ratio > 0.9 and re.search(r"players|team|squad", rewritten, re.IGNORECASE):
        rewritten = re.sub(r"players", "athletes", rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"team|squad", "group", rewritten, flags=re.IGNORECASE)
    new_prefix = "Select the best completion"
    if re.match(r"^(Choose|Select|Complete)\\b", rewritten, re.IGNORECASE):
        qline = rewritten
    else:
        qline = f"{new_prefix}: {rewritten}"
    return "\n".join([qline] + lines[1:])


# Include API routers
app.include_router(settings_router)
app.include_router(questions_router)
app.include_router(exams_router)
app.include_router(import_export_router)
app.include_router(ai_features_router)
app.include_router(generation_router)
app.include_router(grading_router)
app.include_router(parsing_router)
app.include_router(crawler_router)
app.include_router(storage_router)
app.include_router(curriculum_router)
app.include_router(system_router)
app.include_router(ai_tools_router)


@app.get("/")
def index() -> HTMLResponse:
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/upload-sample")
def upload_sample(subject: str = Form(...), file: UploadFile = Form(...)) -> dict:
    if not subject.strip():
        return {"ok": False, "message": "Thiếu tên môn"}
    if SAMPLE_DIR is None:
        return {"ok": False, "message": "Không tìm thấy thư mục đề mẫu"}
    if not SAMPLE_DIR.exists():
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    subject_dir = (SAMPLE_DIR / subject.strip()).resolve()
    if SAMPLE_DIR not in subject_dir.parents:
        return {"ok": False, "message": "Đường dẫn không hợp lệ"}
    subject_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename or "").name
    if not filename:
        return {"ok": False, "message": "Tên file không hợp lệ"}
    if Path(filename).suffix.lower() not in {".txt", ".docx", ".md"}:
        return {"ok": False, "message": "Chỉ hỗ trợ .txt, .docx, .md"}
    target = subject_dir / filename
    with target.open("wb") as f:
        f.write(file.file.read())
    return {"ok": True, "message": "Đã tải lên", "filename": filename, "subject": subject.strip()}



# Question bank, exams, history, import/export APIs moved to:
# - app.api.routers.questions
# - app.api.routers.exams
# - app.api.routers.import_export


# ============================================================================
# MATH EXAM PARSING
# ============================================================================

