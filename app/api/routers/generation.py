"""
Question generation API endpoints.
"""
import io
import os
import random
import re
from pathlib import Path
from typing import List

import httpx
from docx import Document
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import StreamingResponse
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from app.core import GenerateRequest, ParseSamplesRequest
from app.services.ai import call_ai, BASE_DIR
from app.services.text import normalize_name, normalize_question
from app.services.generation import (
    generate_variants,
    split_questions,
    load_questions_from_subject,
    build_topic_prompt,
    normalize_ai_blocks,
)
from app.parsers.docx import extract_docx_content

router = APIRouter(tags=["generation"])


# ============================================================================
# Helper functions
# ============================================================================

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


def _read_sample_url(url: str) -> str:
    """Read content from a URL."""
    suffix = Path(url.split("?")[0]).suffix.lower()
    with httpx.Client(timeout=30) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.content
    if suffix == ".docx":
        doc = Document(io.BytesIO(data))
        lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(lines)
    return data.decode("utf-8", errors="ignore")


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
) -> int:
    """Save generated text questions to the question bank."""
    from app.database import SessionLocal, QuestionCRUD

    saved = 0
    db = SessionLocal()
    try:
        for q in questions:
            if not q.strip():
                continue
            QuestionCRUD.create(
                db,
                content=q.strip(),
                subject=subject,
                source=source,
                difficulty=difficulty,
            )
            saved += 1
    finally:
        db.close()
    return saved


# ============================================================================
# Generation endpoints
# ============================================================================

@router.post("/generate")
def generate(payload: GenerateRequest) -> dict:
    """Generate question variants from samples."""
    engine = payload.ai_engine
    if payload.use_ai and not _is_engine_available(engine):
        engine_names = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "ollama": "Ollama"}
        questions = generate_variants(payload)
        saved = _save_text_questions_to_bank(
            questions,
            subject=payload.topic or "general",
            source="generated",
        )
        return {
            "questions": questions,
            "message": f"Chưa cấu hình {engine_names.get(engine, engine)} nên AI không được dùng.",
            "saved_to_bank": saved,
        }

    questions = generate_variants(payload)
    saved = _save_text_questions_to_bank(
        questions,
        subject=payload.topic or "general",
        source="generated-ai" if payload.use_ai else "generated",
    )

    if payload.use_ai and _is_engine_available(engine):
        src = {normalize_question(s) for s in payload.samples if s.strip()}
        out = {normalize_question(q) for q in questions if q.strip()}
        if out and out.issubset(src):
            return {
                "questions": questions,
                "message": "AI đang trả về câu gần giống câu gốc. Hệ thống đã thêm biến thể đơn giản.",
                "saved_to_bank": saved,
            }
    return {"questions": questions, "saved_to_bank": saved}


@router.post("/generate-topic")
def generate_topic(
    subject: str = Form(...),
    grade: int = Form(1),
    qtype: str = Form("mcq"),
    count: int = Form(10),
    ai_engine: str = Form("ollama"),
    topic: str = Form(""),
    difficulty: str = Form("medium"),
) -> dict:
    """Generate questions by topic using AI."""
    if not _is_engine_available(ai_engine):
        engine_names = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "ollama": "Ollama"}
        return {"questions": [], "message": f"Chưa cấu hình {engine_names.get(ai_engine, ai_engine)} nên AI không được dùng."}

    count = max(1, min(50, count))
    grade = max(1, min(12, grade))
    prompt = build_topic_prompt(subject, grade, qtype, count, topic, difficulty)
    text, err = call_ai(prompt, ai_engine)

    if not text:
        msg = f"Không nhận được phản hồi từ AI. {err}" if err else "Không nhận được phản hồi từ AI."
        return {"questions": [], "answers": "", "message": msg}

    # Split answers section from questions
    answers = ""
    answer_pattern = re.compile(
        r"\n\s*-{0,3}\s*(?:ĐÁP ÁN|Đáp án|đáp án|ANSWERS|Answers|Answer Key|answer key)\s*-{0,3}\s*:?\s*\n",
        re.IGNORECASE,
    )
    match = answer_pattern.search(text)
    if match:
        answers = text[match.end():].strip()
        text = text[:match.start()]

    questions = normalize_ai_blocks(text)
    questions = [q for q in questions if q.strip()]
    final_questions = questions[:count]

    saved = _save_text_questions_to_bank(
        final_questions,
        subject=subject,
        source="generated-topic",
        difficulty=difficulty,
    )

    return {"questions": final_questions, "answers": answers, "saved_to_bank": saved}


@router.post("/auto-generate")
def auto_generate(
    subject: str = Form(...),
    count: int = Form(10),
    ai_ratio: int = Form(30),
    topic: str = Form(""),
    custom_keywords: str = Form(""),
    paraphrase: bool = Form(True),
    change_numbers: bool = Form(True),
    change_context: bool = Form(True),
    use_ai: bool = Form(False),
    samples_text: str = Form(""),
    ai_engine: str = Form("openai"),
) -> dict:
    """Auto-generate questions with configurable AI ratio."""
    if samples_text.strip():
        samples = split_questions(samples_text)
    else:
        samples = load_questions_from_subject(subject)

    if not samples:
        return {"questions": [], "message": "Không tìm thấy câu hỏi mẫu."}

    count = max(1, min(200, count))
    ai_ratio = max(0, min(100, ai_ratio))
    ai_count = int(round(count * (ai_ratio / 100)))
    rule_count = count - ai_count

    random.shuffle(samples)
    selected = samples[: min(len(samples), max(1, rule_count))]
    if len(selected) < rule_count:
        selected = selected * (rule_count // max(1, len(selected)) + 1)
        selected = selected[:rule_count]

    req = GenerateRequest(
        samples=selected,
        topic=topic or subject,
        custom_keywords=[s for s in custom_keywords.split(",") if s.strip()],
        paraphrase=paraphrase,
        change_numbers=change_numbers,
        change_context=change_context,
        variants_per_question=1,
        use_ai=False,
    )
    questions = generate_variants(req)

    if use_ai and ai_count > 0 and not _is_engine_available(ai_engine):
        engine_names = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "ollama": "Ollama"}
        return {
            "questions": questions[:count],
            "message": f"Chưa cấu hình {engine_names.get(ai_engine, ai_engine)} nên AI không được dùng.",
        }

    if use_ai and ai_count > 0:
        ai_req = GenerateRequest(
            samples=[random.choice(samples)] if samples else [],
            topic=topic or subject,
            custom_keywords=[s for s in custom_keywords.split(",") if s.strip()],
            paraphrase=paraphrase,
            change_numbers=change_numbers,
            change_context=change_context,
            variants_per_question=max(1, ai_count),
            use_ai=True,
            ai_engine=ai_engine,
        )
        ai_questions = generate_variants(ai_req)
        if ai_questions:
            questions.extend(ai_questions[:ai_count])

    random.shuffle(questions)
    final_questions = questions[:count]

    saved = _save_text_questions_to_bank(
        final_questions,
        subject=subject,
        source="auto-generated",
    )

    result = {"questions": final_questions, "saved_to_bank": saved}
    if use_ai and _is_engine_available(ai_engine):
        src = {normalize_question(s) for s in samples if s.strip()}
        out = {normalize_question(q) for q in result["questions"] if q.strip()}
        if out and out.issubset(src):
            result["message"] = "AI đang trả về câu gần giống câu gốc. Hệ thống đã thêm biến thể đơn giản."
    return result


@router.post("/export")
def export(
    samples: str = Form(...),
    topic: str = Form(...),
    custom_keywords: str = Form(""),
    paraphrase: bool = Form(False),
    change_numbers: bool = Form(False),
    change_context: bool = Form(False),
    variants_per_question: int = Form(1),
    fmt: str = Form("txt"),
    use_ai: bool = Form(False),
    ai_engine: str = Form("openai"),
):
    """Export generated questions to various formats."""
    req = GenerateRequest(
        samples=[s for s in samples.split("\n") if s.strip()],
        topic=topic,
        custom_keywords=[s for s in custom_keywords.split(",") if s.strip()],
        paraphrase=paraphrase,
        change_numbers=change_numbers,
        change_context=change_context,
        variants_per_question=variants_per_question,
        use_ai=use_ai,
        ai_engine=ai_engine,
    )
    questions = generate_variants(req)

    if fmt == "csv":
        content = "id,question\n" + "\n".join(
            f"{i+1},\"{q.replace('\"', '\"\"')}\"" for i, q in enumerate(questions)
        )
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=questions.csv"},
        )

    if fmt == "docx":
        doc = Document()
        doc.add_heading("Danh sách câu hỏi", level=1)
        for i, q in enumerate(questions, 1):
            doc.add_paragraph(f"{i}. {q}")
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": "attachment; filename=questions.docx"},
        )

    if fmt == "pdf":
        font_name, _ = _get_pdf_font()
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = height - 60
        font_size = 12
        c.setFont(font_name, font_size)
        c.drawString(50, y, "Danh sách câu hỏi")
        y -= 30
        for i, q in enumerate(questions, 1):
            line = f"{i}. {q}"
            wrapped = _wrap_text(line, width - 100, font_name, font_size)
            for part in wrapped:
                if y < 60:
                    c.showPage()
                    c.setFont(font_name, font_size)
                    y = height - 60
                c.drawString(50, y, part)
                y -= 18
        c.save()
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=questions.pdf"},
        )

    content = "\n".join(questions)
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=questions.txt"},
    )


@router.post("/api/export-exam")
async def export_exam(request: Request):
    """Export exam questions with full content and options."""
    data = await request.json()
    questions = data.get("questions", [])
    fmt = data.get("format", "docx")
    title = data.get("title", "Đề thi")

    if not questions:
        raise HTTPException(status_code=400, detail="Không có câu hỏi để xuất")

    if fmt == "docx":
        doc = Document()
        doc.add_heading(title, level=1)

        for i, q in enumerate(questions, 1):
            content = q.get("content", "")
            options = q.get("options", [])

            p = doc.add_paragraph()
            p.add_run(f"Câu {i}. ").bold = True
            p.add_run(content)

            if options:
                labels = ['A', 'B', 'C', 'D', 'E']
                for j, opt in enumerate(options[:5]):
                    opt_clean = re.sub(r'^[A-Ea-e]\s*[.)]\s*', '', str(opt).strip())
                    doc.add_paragraph(f"    {labels[j]}) {opt_clean}")

            doc.add_paragraph("")

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename=de_thi.docx"},
        )

    if fmt == "pdf":
        font_name, _ = _get_pdf_font()
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = height - 60
        font_size = 12

        c.setFont(font_name, 16)
        c.drawString(50, y, title)
        y -= 40
        c.setFont(font_name, font_size)

        for i, q in enumerate(questions, 1):
            content = q.get("content", "")
            options = q.get("options", [])

            line = f"Câu {i}. {content}"
            wrapped = _wrap_text(line, width - 100, font_name, font_size)
            for part in wrapped:
                if y < 80:
                    c.showPage()
                    c.setFont(font_name, font_size)
                    y = height - 60
                c.drawString(50, y, part)
                y -= 18

            if options:
                labels = ['A', 'B', 'C', 'D', 'E']
                for j, opt in enumerate(options[:5]):
                    opt_clean = re.sub(r'^[A-Ea-e]\s*[.)]\s*', '', str(opt).strip())
                    opt_line = f"    {labels[j]}) {opt_clean}"
                    opt_wrapped = _wrap_text(opt_line, width - 120, font_name, font_size)
                    for part in opt_wrapped:
                        if y < 80:
                            c.showPage()
                            c.setFont(font_name, font_size)
                            y = height - 60
                        c.drawString(70, y, part)
                        y -= 16

            y -= 10

        c.save()
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=de_thi.pdf"},
        )

    raise HTTPException(status_code=400, detail="Format không hỗ trợ")


# ============================================================================
# Sample file endpoints
# ============================================================================

@router.get("/sample-folders")
def list_sample_folders() -> dict:
    """List available sample folders."""
    if SAMPLE_DIR is None or not SAMPLE_DIR.exists():
        return {"folders": []}
    folders = [p.name for p in SAMPLE_DIR.iterdir() if p.is_dir()]
    folders.sort()
    return {"folders": folders}


@router.get("/sample-files")
def list_sample_files(subject: str) -> dict:
    """List sample files in a subject folder."""
    if SAMPLE_DIR is None or not subject or not SAMPLE_DIR.exists():
        return {"files": []}
    subject_dir = (SAMPLE_DIR / subject).resolve()
    if SAMPLE_DIR not in subject_dir.parents or not subject_dir.exists():
        return {"files": []}
    files = [
        p.name
        for p in subject_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".txt", ".docx", ".md"}
    ]
    files.sort()
    return {"files": files}


@router.get("/sample-content")
def sample_content(subject: str, filename: str) -> dict:
    """Get content of a sample file."""
    if not subject or not filename:
        return {"content": ""}
    if SAMPLE_DIR is None:
        return {"content": ""}
    subject_dir = (SAMPLE_DIR / subject).resolve()
    if not SAMPLE_DIR.exists() or SAMPLE_DIR not in subject_dir.parents:
        return {"content": ""}
    target = (subject_dir / filename).resolve()
    if subject_dir not in target.parents or not target.exists() or not target.is_file():
        return {"content": ""}
    content = _read_sample_file(target)
    return {"content": content}


@router.post("/parse-sample-urls")
def parse_sample_urls(payload: ParseSamplesRequest) -> dict:
    """Parse sample questions from URLs."""
    contents: List[str] = []
    for url in payload.urls:
        if not url:
            continue
        try:
            contents.append(_read_sample_url(url))
        except Exception:
            continue
    merged = "\n".join(contents)
    return {"content": merged, "samples": split_questions(merged)}
