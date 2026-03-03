"""
Crawler API endpoints for importing questions from external sources.
"""
import io
import json
import re
from typing import List, Optional

from docx import Document
from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from PyPDF2 import PdfReader

from app.database import SessionLocal, QuestionCRUD
from app.services.crawler import fetch_questions_from_url, crawl_multiple_urls

router = APIRouter(prefix="/api/crawler", tags=["crawler"])


# ============================================================================
# URL Crawling
# ============================================================================

@router.post("/fetch-url")
def fetch_from_url(
    url: str = Form(...),
    subject: str = Form(""),
    difficulty: str = Form("medium"),
    save_to_db: bool = Form(True),
) -> dict:
    """
    Fetch and parse questions from a single URL.
    """
    result = fetch_questions_from_url(url, subject, difficulty)

    if result["error"]:
        return {
            "success": False,
            "error": result["error"],
            "questions": [],
            "count": 0,
        }

    saved_count = 0
    if save_to_db and result["questions"]:
        db = SessionLocal()
        try:
            for q in result["questions"]:
                QuestionCRUD.create(
                    db,
                    content=q["content"],
                    options=q.get("options"),
                    answer=q.get("answer", ""),
                    subject=q.get("subject", subject) or "general",
                    source=q.get("source", url),
                    difficulty=q.get("difficulty", difficulty),
                    question_type=q.get("question_type", "mcq"),
                )
                saved_count += 1
        finally:
            db.close()

    return {
        "success": True,
        "questions": result["questions"],
        "count": result["count"],
        "saved_count": saved_count,
        "source": url,
    }


@router.post("/fetch-multiple")
def fetch_from_multiple_urls(
    urls: str = Form(...),  # Newline-separated URLs
    subject: str = Form(""),
    difficulty: str = Form("medium"),
    save_to_db: bool = Form(True),
) -> dict:
    """
    Fetch and parse questions from multiple URLs.
    """
    url_list = [u.strip() for u in urls.split("\n") if u.strip()]

    if not url_list:
        return {
            "success": False,
            "error": "Không có URL hợp lệ",
            "total_questions": 0,
            "saved_count": 0,
        }

    result = crawl_multiple_urls(url_list, subject, difficulty, save_to_db)

    return {
        "success": True,
        "total_questions": result["total_questions"],
        "saved_count": result["saved_count"],
        "results": result["results"],
    }


# ============================================================================
# File Upload
# ============================================================================

def _parse_docx_questions(content: bytes, subject: str, difficulty: str) -> List[dict]:
    """Parse questions from a DOCX file."""
    from app.parsers.docx import extract_docx_content
    from app.services.generation import split_questions

    doc = Document(io.BytesIO(content))
    text = extract_docx_content(doc)
    raw_questions = split_questions(text)

    questions = []
    for q in raw_questions:
        if not q.strip():
            continue

        # Check if MCQ (has A), B), C), D) options)
        lines = q.split("\n")
        content = lines[0] if lines else q
        options = []

        for line in lines[1:]:
            line = line.strip()
            if re.match(r"^[A-Da-d][.)]\s*", line):
                opt = re.sub(r"^[A-Da-d][.)]\s*", "", line)
                options.append(opt)

        questions.append({
            "content": content,
            "options": json.dumps(options, ensure_ascii=False) if options else None,
            "answer": "",
            "subject": subject,
            "difficulty": difficulty,
            "question_type": "mcq" if options else "essay",
            "source": "uploaded-docx",
        })

    return questions


def _parse_pdf_questions(content: bytes, subject: str, difficulty: str) -> List[dict]:
    """Parse questions from a PDF file."""
    from app.services.generation import split_questions

    reader = PdfReader(io.BytesIO(content))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    raw_questions = split_questions(text)

    questions = []
    for q in raw_questions:
        if not q.strip():
            continue

        lines = q.split("\n")
        content = lines[0] if lines else q
        options = []

        for line in lines[1:]:
            line = line.strip()
            if re.match(r"^[A-Da-d][.)]\s*", line):
                opt = re.sub(r"^[A-Da-d][.)]\s*", "", line)
                options.append(opt)

        questions.append({
            "content": content,
            "options": json.dumps(options, ensure_ascii=False) if options else None,
            "answer": "",
            "subject": subject,
            "difficulty": difficulty,
            "question_type": "mcq" if options else "essay",
            "source": "uploaded-pdf",
        })

    return questions


def _parse_txt_questions(content: bytes, subject: str, difficulty: str) -> List[dict]:
    """Parse questions from a TXT file."""
    from app.services.generation import split_questions

    text = content.decode("utf-8", errors="ignore")
    raw_questions = split_questions(text)

    questions = []
    for q in raw_questions:
        if not q.strip():
            continue

        lines = q.split("\n")
        content = lines[0] if lines else q
        options = []

        for line in lines[1:]:
            line = line.strip()
            if re.match(r"^[A-Da-d][.)]\s*", line):
                opt = re.sub(r"^[A-Da-d][.)]\s*", "", line)
                options.append(opt)

        questions.append({
            "content": content,
            "options": json.dumps(options, ensure_ascii=False) if options else None,
            "answer": "",
            "subject": subject,
            "difficulty": difficulty,
            "question_type": "mcq" if options else "essay",
            "source": "uploaded-txt",
        })

    return questions


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    subject: str = Form(""),
    difficulty: str = Form("medium"),
    save_to_db: bool = Form(True),
) -> dict:
    """
    Upload and parse a file (DOCX, PDF, TXT) containing questions.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Không có file")

    content = await file.read()
    filename = file.filename.lower()

    questions = []
    if filename.endswith(".docx"):
        questions = _parse_docx_questions(content, subject, difficulty)
    elif filename.endswith(".pdf"):
        questions = _parse_pdf_questions(content, subject, difficulty)
    elif filename.endswith(".txt"):
        questions = _parse_txt_questions(content, subject, difficulty)
    else:
        raise HTTPException(
            status_code=400,
            detail="Định dạng file không hỗ trợ. Chỉ hỗ trợ DOCX, PDF, TXT.",
        )

    saved_count = 0
    if save_to_db and questions:
        db = SessionLocal()
        try:
            for q in questions:
                QuestionCRUD.create(
                    db,
                    content=q["content"],
                    options=q.get("options"),
                    answer=q.get("answer", ""),
                    subject=q.get("subject", subject) or "general",
                    source=q.get("source", "uploaded"),
                    difficulty=q.get("difficulty", difficulty),
                    question_type=q.get("question_type", "mcq"),
                )
                saved_count += 1
        finally:
            db.close()

    return {
        "success": True,
        "filename": file.filename,
        "questions": questions,
        "count": len(questions),
        "saved_count": saved_count,
    }


# ============================================================================
# Suggested Sources
# ============================================================================

@router.get("/suggested-sources")
def get_suggested_sources() -> dict:
    """
    Get list of suggested education websites for crawling.
    """
    sources = [
        {
            "name": "VietJack",
            "url": "https://vietjack.com",
            "description": "Đề thi, bài tập các môn từ lớp 1-12",
            "subjects": ["toan", "ngu_van", "tieng_anh", "vat_ly", "hoa_hoc", "sinh_hoc"],
        },
        {
            "name": "Hoc247",
            "url": "https://hoc247.net",
            "description": "Đề thi thử, đề kiểm tra các cấp",
            "subjects": ["toan", "ngu_van", "tieng_anh", "vat_ly", "hoa_hoc"],
        },
        {
            "name": "Lời Giải Hay",
            "url": "https://loigiaihay.com",
            "description": "Lời giải bài tập SGK, đề thi",
            "subjects": ["toan", "ngu_van", "tieng_anh"],
        },
    ]

    return {"sources": sources}
