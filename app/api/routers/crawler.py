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
from app.services.crawler.thuvienhoclieu import (
    scrape_category,
    scrape_single_quiz,
    QUIZ_CATEGORIES,
)

router = APIRouter(prefix="/api/crawler", tags=["crawler"])


def _save_questions(db, questions: list, default_subject: str = "", default_difficulty: str = "medium", default_source: str = "") -> dict:
    """Save questions to DB with duplicate detection. Returns {saved, skipped}."""
    saved = 0
    skipped = 0
    for q in questions:
        content = q.get("content", "").strip()
        if not content:
            continue
        if QuestionCRUD.exists_by_content(db, content):
            skipped += 1
            continue
        QuestionCRUD.create(
            db,
            content=content,
            options=q.get("options"),
            answer=q.get("answer", ""),
            subject=q.get("subject", default_subject) or "general",
            grade=q.get("grade", ""),
            source=q.get("source", default_source),
            difficulty=q.get("difficulty", default_difficulty),
            question_type=q.get("question_type", "mcq"),
        )
        saved += 1
    return {"saved": saved, "skipped": skipped}


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
    skipped_count = 0
    if save_to_db and result["questions"]:
        db = SessionLocal()
        try:
            r = _save_questions(db, result["questions"], subject, difficulty, url)
            saved_count = r["saved"]
            skipped_count = r["skipped"]
        finally:
            db.close()

    return {
        "success": True,
        "questions": result["questions"],
        "count": result["count"],
        "saved_count": saved_count,
        "skipped_count": skipped_count,
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
    skipped_count = 0
    if save_to_db and questions:
        db = SessionLocal()
        try:
            r = _save_questions(db, questions, subject, difficulty, "uploaded")
            saved_count = r["saved"]
            skipped_count = r["skipped"]
        finally:
            db.close()

    return {
        "success": True,
        "filename": file.filename,
        "questions": questions,
        "count": len(questions),
        "saved_count": saved_count,
        "skipped_count": skipped_count,
    }


# ============================================================================
# ThuVienHocLieu.com Auto-Scraper
# ============================================================================

@router.get("/thuvienhoclieu/categories")
def get_thuvienhoclieu_categories() -> dict:
    """Get available quiz categories from thuvienhoclieu.com."""
    return {"categories": QUIZ_CATEGORIES}


@router.post("/thuvienhoclieu/scrape")
def scrape_thuvienhoclieu(
    category_url: str = Form(""),
    custom_url: str = Form(""),
    max_quizzes: int = Form(50),
    save_to_db: bool = Form(True),
) -> dict:
    """
    Scrape questions from thuvienhoclieu.com.

    Either provide a category_url to auto-discover quizzes,
    or custom_url for a single quiz page.
    """
    url = custom_url.strip() or category_url.strip()
    if not url:
        return {"success": False, "error": "Vui lòng chọn danh mục hoặc nhập URL"}

    # Detect if it's a single quiz page or a category
    is_single = any(p in url for p in [
        "de-trac-nghiem-online-",
        "kiem-tra-",
        "trac-nghiem-truc-tuyen-",
        "de-thi-thu-",
    ]) and "/trac-nghiem-online/" not in url.split("de-trac-nghiem")[0].split("kiem-tra")[0]

    if is_single or custom_url.strip():
        # Try single quiz first
        result = scrape_single_quiz(url)
        if result["count"] > 0:
            saved_count = 0
            skipped_count = 0
            if save_to_db:
                db = SessionLocal()
                try:
                    r = _save_questions(db, result["questions"], "", "medium", url)
                    saved_count = r["saved"]
                    skipped_count = r["skipped"]
                finally:
                    db.close()
            return {
                "success": True,
                "question_count": result["count"],
                "quiz_count": 1,
                "saved_count": saved_count,
                "skipped_count": skipped_count,
                "scanned_pages": 1,
                "error_count": 0,
                "errors": [],
            }
        elif not category_url.strip():
            # Not a quiz page and no category fallback
            return {
                "success": False,
                "error": result.get("error") or "Không tìm thấy câu hỏi trên trang này",
            }

    # Category scrape
    result = scrape_category(url, max_quizzes=max_quizzes)

    saved_count = 0
    skipped_count = 0
    if save_to_db and result["questions"]:
        db = SessionLocal()
        try:
            r = _save_questions(db, result["questions"], "", "medium", url)
            saved_count = r["saved"]
            skipped_count = r["skipped"]
        finally:
            db.close()

    return {
        "success": True,
        "question_count": result["question_count"],
        "quiz_count": result["quiz_count"],
        "saved_count": saved_count,
        "skipped_count": skipped_count,
        "scanned_pages": result["scanned_pages"],
        "error_count": result["error_count"],
        "errors": result["errors"],
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
            "name": "Thư Viện Học Liệu",
            "url": "https://thuvienhoclieu.com",
            "description": "Trắc nghiệm online, đề thi các môn lớp 10-12 (hỗ trợ auto-scrape)",
            "subjects": ["toan", "vat_ly", "hoa_hoc", "sinh_hoc", "tieng_anh", "lich_su", "gdcd"],
        },
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


# ============================================================================
# VietJack.com Scraper
# ============================================================================

# Load VietJack scraper module
import importlib.util
import os
_crawler_dir = os.path.join(os.path.dirname(__file__), "..", "..", "services", "crawler")
_vj_path = os.path.join(_crawler_dir, "vietjack-exam-scraper.py")
_vj_spec = importlib.util.spec_from_file_location("vietjack_scraper", _vj_path)
_vietjack = importlib.util.module_from_spec(_vj_spec)
_vj_spec.loader.exec_module(_vietjack)


VIETJACK_CATEGORIES = [
    # Toán 12
    {"name": "Toán 12 - Giữa kì 1 (KNTT)", "url": "https://vietjack.com/de-kiem-tra-lop-12/de-thi-toan-12-giua-ki-1-ket-noi-tri-thuc.jsp"},
    {"name": "Toán 12 - Bộ đề giữa kì 1", "url": "https://vietjack.com/de-kiem-tra-lop-12/bo-de-thi-toan-lop-12-giua-hoc-ki-1.jsp"},
    {"name": "Toán 12 - Học kì 1", "url": "https://vietjack.com/de-kiem-tra-lop-12/de-thi-toan-12-hoc-ki-1.jsp"},
    # Địa lí 12
    {"name": "Địa lí 12 - Giữa kì 1 (KNTT)", "url": "https://vietjack.com/de-kiem-tra-lop-12/de-thi-dia-li-12-giua-ki-1-ket-noi-tri-thuc.jsp"},
    {"name": "Địa lí 12 - Giữa kì 1 (CTST)", "url": "https://vietjack.com/de-kiem-tra-lop-12/de-thi-dia-li-12-giua-ki-1-chan-troi-sang-tao.jsp"},
    {"name": "Địa lí 12 - Giữa kì 1 (Cánh Diều)", "url": "https://vietjack.com/de-kiem-tra-lop-12/de-thi-dia-li-12-giua-ki-1-canh-dieu.jsp"},
    # Lịch sử 12
    {"name": "Lịch sử 12 - Giữa kì 1 (KNTT)", "url": "https://vietjack.com/de-kiem-tra-lop-12/de-thi-lich-su-12-giua-ki-1-ket-noi-tri-thuc.jsp"},
    {"name": "Lịch sử 12 - Giữa kì 1 (CTST)", "url": "https://vietjack.com/de-kiem-tra-lop-12/de-thi-lich-su-12-giua-ki-1-chan-troi-sang-tao.jsp"},
    # Công nghệ 12
    {"name": "Công nghệ 12 - Giữa kì 1 (KNTT)", "url": "https://vietjack.com/de-kiem-tra-lop-12/de-thi-cong-nghe-12-giua-ki-1-ket-noi-tri-thuc.jsp"},
    {"name": "Công nghệ 12 - Giữa kì 1 (Cánh Diều)", "url": "https://vietjack.com/de-kiem-tra-lop-12/de-thi-cong-nghe-12-giua-ki-1-canh-dieu.jsp"},
    # Tiếng Anh 12
    {"name": "Tiếng Anh 12 - Bright", "url": "https://vietjack.com/de-kiem-tra-lop-12/bo-de-thi-tieng-anh-12-bright.jsp"},
    {"name": "Tiếng Anh 12 - English Discovery", "url": "https://vietjack.com/de-kiem-tra-lop-12/bo-de-thi-tieng-anh-12-english-discovery.jsp"},
    # Tổng hợp
    {"name": "Tổng hợp đề thi cuối kì 1 lớp 12", "url": "https://vietjack.com/de-kiem-tra-lop-12/de-thi-cuoi-ki-1-lop-12.jsp"},
    {"name": "Tổng hợp đề thi cuối kì 2 lớp 12", "url": "https://vietjack.com/de-kiem-tra-lop-12/de-thi-cuoi-ki-2-lop-12.jsp"},
]


@router.get("/vietjack/categories")
def get_vietjack_categories() -> dict:
    """Get available exam categories from vietjack.com."""
    return {"categories": VIETJACK_CATEGORIES}


@router.post("/vietjack/scrape")
def scrape_vietjack(
    category_url: str = Form(""),
    custom_url: str = Form(""),
    max_pages: int = Form(5),
    save_to_db: bool = Form(True),
) -> dict:
    """
    Scrape questions from vietjack.com.

    Provide a URL to a specific exam page or a category to discover pages.
    """
    url = custom_url.strip() or category_url.strip()
    if not url:
        return {"success": False, "error": "Vui lòng nhập URL trang đề thi"}

    if not url.startswith("https://vietjack.com/"):
        return {"success": False, "error": "URL không hợp lệ. Chỉ hỗ trợ vietjack.com"}

    # Scrape the page
    questions, error = _vietjack.scrape_exam(url)

    if error:
        return {"success": False, "error": error, "questions": [], "count": 0}

    # Convert questions format for saving
    formatted_questions = []
    for q in questions:
        formatted_questions.append({
            "content": q["question"],
            "options": json.dumps(q["options"], ensure_ascii=False) if q["options"] else None,
            "answer": q.get("answer", ""),
            "subject": q.get("subject", "general"),
            "grade": q.get("grade", ""),
            "source": q.get("source", url),
            "difficulty": "medium",
            "question_type": "mcq",
        })

    saved_count = 0
    skipped_count = 0
    if save_to_db and formatted_questions:
        db = SessionLocal()
        try:
            r = _save_questions(db, formatted_questions, "", "medium", url)
            saved_count = r["saved"]
            skipped_count = r["skipped"]
        finally:
            db.close()

    return {
        "success": True,
        "question_count": len(questions),
        "saved_count": saved_count,
        "skipped_count": skipped_count,
    }


# ============================================================================
# Hoc247.net Scraper
# ============================================================================

# Load Hoc247 scraper module
_hoc247_path = os.path.join(_crawler_dir, "hoc247-exam-scraper.py")
_hoc247_spec = importlib.util.spec_from_file_location("hoc247_scraper", _hoc247_path)
_hoc247 = importlib.util.module_from_spec(_hoc247_spec)
_hoc247_spec.loader.exec_module(_hoc247)


HOC247_CATEGORIES = [
    # === TOÁN ===
    {"name": "Toán 12", "url": "https://hoc247.net/trac-nghiem-toan-12-index.html"},
    {"name": "Toán 11", "url": "https://hoc247.net/trac-nghiem-toan-11-index.html"},
    {"name": "Toán 10", "url": "https://hoc247.net/trac-nghiem-toan-10-index.html"},
    {"name": "Toán 9", "url": "https://hoc247.net/trac-nghiem-toan-9-index.html"},
    {"name": "Toán 7", "url": "https://hoc247.net/trac-nghiem-toan-7-index.html"},
    {"name": "Toán 6", "url": "https://hoc247.net/trac-nghiem-toan-6-index.html"},
    # === VẬT LÝ ===
    {"name": "Vật lý 12", "url": "https://hoc247.net/trac-nghiem-vat-ly-12-index.html"},
    {"name": "Vật lý 11", "url": "https://hoc247.net/trac-nghiem-vat-ly-11-index.html"},
    {"name": "Vật lý 10", "url": "https://hoc247.net/trac-nghiem-vat-ly-10-index.html"},
    {"name": "Vật lý 9", "url": "https://hoc247.net/trac-nghiem-vat-ly-9-index.html"},
    # === HÓA HỌC ===
    {"name": "Hóa học 12", "url": "https://hoc247.net/trac-nghiem-hoa-hoc-12-index.html"},
    {"name": "Hóa học 10", "url": "https://hoc247.net/trac-nghiem-hoa-hoc-10-index.html"},
    {"name": "Hóa học 9", "url": "https://hoc247.net/trac-nghiem-hoa-hoc-9-index.html"},
    # === SINH HỌC ===
    {"name": "Sinh học 12", "url": "https://hoc247.net/trac-nghiem-sinh-12-index.html"},
    {"name": "Sinh học 11", "url": "https://hoc247.net/trac-nghiem-sinh-hoc-11-index.html"},
    {"name": "Sinh học 10", "url": "https://hoc247.net/trac-nghiem-sinh-hoc-10-index.html"},
    {"name": "Sinh học 9", "url": "https://hoc247.net/trac-nghiem-sinh-9-index.html"},
    # === TIẾNG ANH ===
    {"name": "Tiếng Anh 12", "url": "https://hoc247.net/trac-nghiem-tieng-anh-12-index.html"},
    {"name": "Tiếng Anh 11 KNTT", "url": "https://hoc247.net/trac-nghiem-tieng-anh-11-ket-noi-tri-thuc-index.html"},
    {"name": "Tiếng Anh 11 CTST", "url": "https://hoc247.net/trac-nghiem-tieng-anh-11-chan-troi-sang-tao-index.html"},
    {"name": "Tiếng Anh 10 KNTT", "url": "https://hoc247.net/trac-nghiem-tieng-anh-10-ket-noi-tri-thuc-index.html"},
    {"name": "Tiếng Anh 10 CTST", "url": "https://hoc247.net/trac-nghiem-tieng-anh-10-chan-troi-sang-tao-index.html"},
    {"name": "Tiếng Anh 10 CD", "url": "https://hoc247.net/trac-nghiem-tieng-anh-10-canh-dieu-index.html"},
    {"name": "Tiếng Anh 9", "url": "https://hoc247.net/trac-nghiem-tieng-anh-9-index.html"},
    {"name": "Tiếng Anh 7 KNTT", "url": "https://hoc247.net/trac-nghiem-tieng-anh-7-ket-noi-tri-thuc-index.html"},
    {"name": "Tiếng Anh 6 KNTT", "url": "https://hoc247.net/trac-nghiem-tieng-anh-6-ket-noi-tri-thuc-index.html"},
    # === LỊCH SỬ ===
    {"name": "Lịch sử 12", "url": "https://hoc247.net/trac-nghiem-lich-su-12-index.html"},
    {"name": "Lịch sử 11", "url": "https://hoc247.net/trac-nghiem-lich-su-11-index.html"},
    {"name": "Lịch sử 10", "url": "https://hoc247.net/trac-nghiem-lich-su-10-index.html"},
    {"name": "Lịch sử 9", "url": "https://hoc247.net/trac-nghiem-lich-su-9-index.html"},
    # === ĐỊA LÝ ===
    {"name": "Địa lý 12", "url": "https://hoc247.net/trac-nghiem-dia-12-index.html"},
    {"name": "Địa lý 11", "url": "https://hoc247.net/trac-nghiem-dia-li-11-index.html"},
    {"name": "Địa lý 10", "url": "https://hoc247.net/trac-nghiem-dia-10-index.html"},
    {"name": "Địa lý 9", "url": "https://hoc247.net/trac-nghiem-dia-9-index.html"},
    # === GDCD / KTPL ===
    {"name": "GDCD 12", "url": "https://hoc247.net/trac-nghiem-gdcd-12-index.html"},
    {"name": "GDCD 9", "url": "https://hoc247.net/trac-nghiem-gdcd-9-index.html"},
    {"name": "KTPL 11", "url": "https://hoc247.net/trac-nghiem-giao-duc-kinh-te-va-phap-luat-11-index.html"},
    {"name": "KTPL 10", "url": "https://hoc247.net/trac-nghiem-giao-duc-kinh-te-va-phap-luat-10-index.html"},
    # === TIN HỌC ===
    {"name": "Tin học 12", "url": "https://hoc247.net/trac-nghiem-tin-hoc-12-index.html"},
    {"name": "Tin học 11", "url": "https://hoc247.net/trac-nghiem-tin-hoc-11-index.html"},
    {"name": "Tin học 10", "url": "https://hoc247.net/trac-nghiem-tin-hoc-10-index.html"},
    # === CÔNG NGHỆ ===
    {"name": "Công nghệ 12", "url": "https://hoc247.net/trac-nghiem-cong-nghe-12-index.html"},
    {"name": "Công nghệ 11", "url": "https://hoc247.net/trac-nghiem-cong-nghe-11-index.html"},
    {"name": "Công nghệ 10", "url": "https://hoc247.net/trac-nghiem-cong-nghe-10-index.html"},
    # === KHTN (THCS) ===
    {"name": "KHTN 8", "url": "https://hoc247.net/trac-nghiem-khoa-hoc-tu-nhien-8-index.html"},
    {"name": "KHTN 7", "url": "https://hoc247.net/trac-nghiem-khoa-hoc-tu-nhien-7-index.html"},
    {"name": "KHTN 6", "url": "https://hoc247.net/trac-nghiem-khoa-hoc-tu-nhien-6-index.html"},
    # === ĐẠI HỌC ===
    {"name": "ĐH: Triết học", "url": "https://hoc247.net/trac-nghiem-on-thi-mon-triet-hoc-index.html"},
    {"name": "ĐH: Tư tưởng HCM", "url": "https://hoc247.net/trac-nghiem-on-thi-mon-tu-tuong-ho-chi-minh-index.html"},
    {"name": "ĐH: Pháp luật đại cương", "url": "https://hoc247.net/trac-nghiem-on-thi-mon-phap-luat-dai-cuong-index.html"},
    {"name": "ĐH: Kinh tế vi mô", "url": "https://hoc247.net/trac-nghiem-on-thi-mon-kinh-te-vi-mo-index.html"},
    {"name": "ĐH: Toán cao cấp", "url": "https://hoc247.net/trac-nghiem-on-thi-mon-toan-cao-cap-index.html"},
]


@router.get("/hoc247/categories")
def get_hoc247_categories() -> dict:
    """Get available quiz categories from hoc247.net."""
    return {"categories": HOC247_CATEGORIES}


@router.post("/hoc247/scrape")
def scrape_hoc247(
    category_url: str = Form(""),
    custom_url: str = Form(""),
    max_pages: int = Form(10),
    save_to_db: bool = Form(True),
) -> dict:
    """
    Scrape questions from hoc247.net.

    Provide a category URL to discover quiz pages, or a custom URL for a specific quiz.
    """
    url = custom_url.strip() or category_url.strip()
    if not url:
        return {"success": False, "error": "Vui lòng chọn danh mục hoặc nhập URL"}

    if not url.startswith("https://hoc247.net/"):
        return {"success": False, "error": "URL không hợp lệ. Chỉ hỗ trợ hoc247.net"}

    # Check if it's an index page (category) or a specific quiz page
    is_category = "-index.html" in url

    if is_category:
        # Scrape category - discover and scrape multiple quizzes
        questions, errors = _hoc247.scrape_category(url, max_pages=max_pages)
    else:
        # Scrape single quiz
        questions, error = _hoc247.scrape_quiz(url)
        errors = [error] if error else []

    if not questions and errors:
        return {"success": False, "error": errors[0] if errors else "Không tìm thấy câu hỏi"}

    # Convert questions format for saving
    formatted_questions = []
    for q in questions:
        formatted_questions.append({
            "content": q["question"],
            "options": json.dumps(q["options"], ensure_ascii=False) if q["options"] else None,
            "answer": q.get("answer", ""),
            "subject": q.get("subject", "general"),
            "grade": q.get("grade", ""),
            "source": q.get("source", url),
            "difficulty": "medium",
            "question_type": "mcq",
        })

    saved_count = 0
    skipped_count = 0
    if save_to_db and formatted_questions:
        db = SessionLocal()
        try:
            r = _save_questions(db, formatted_questions, "", "medium", url)
            saved_count = r["saved"]
            skipped_count = r["skipped"]
        finally:
            db.close()

    return {
        "success": True,
        "question_count": len(questions),
        "saved_count": saved_count,
        "skipped_count": skipped_count,
        "errors": errors if errors else [],
    }
