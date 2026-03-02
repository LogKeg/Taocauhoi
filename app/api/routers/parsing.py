"""
Word document parsing API endpoints.
"""
import io
import json
import re
from pathlib import Path
from typing import List

from docx import Document
from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.core import ANSWER_TEMPLATES

# Import parsers directly
from app.parsers.docx import (
    parse_cell_based_questions,
    extract_docx_lines_with_options,
    extract_docx_content,
    parse_bilingual_questions,
)

router = APIRouter(tags=["parsing"])


def _save_questions_to_bank(
    questions: List[dict],
    subject: str = "general",
    source: str = "imported",
    question_type: str = "mcq",
) -> int:
    """Save parsed questions to the question bank."""
    from app.database import SessionLocal, QuestionCRUD

    saved = 0
    db = SessionLocal()
    try:
        for q in questions:
            content = q.get('question', '')
            if not content.strip():
                continue

            # Convert options list to JSON string
            options = q.get('options', [])
            options_json = json.dumps(options) if options else None
            answer = q.get('answer', q.get('correct_answer', ''))

            QuestionCRUD.create(
                db,
                content=content.strip(),
                subject=subject,
                source=source,
                difficulty="medium",
                question_type=question_type,
                options=options_json,
                answer=answer,
            )
            saved += 1
    finally:
        db.close()
    return saved


def _get_parsing_functions():
    """Lazy import of parsing functions from main module."""
    from app import main
    from app.services.ai import call_ai, load_saved_settings
    return {
        'parse_cell_based_questions': parse_cell_based_questions,
        'extract_docx_lines': extract_docx_lines_with_options,
        'extract_docx_content': extract_docx_content,
        'parse_math_exam_questions': main._parse_math_exam_questions,
        'parse_english_exam_questions': main._parse_english_exam_questions,
        'parse_envie_questions': main._parse_envie_questions,
        'parse_bilingual_questions': parse_bilingual_questions,
        'save_questions_to_bank': _save_questions_to_bank,
        'call_ai': call_ai,
        'load_ai_settings': load_saved_settings,
    }


@router.post("/api/parse-exam")
async def parse_exam_file(file: UploadFile):
    """Parse questions from uploaded Word document."""
    funcs = _get_parsing_functions()

    if not file.filename.endswith('.docx'):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .docx")

    content = await file.read()
    doc = Document(io.BytesIO(content))

    # Try cell-based parser first (for ASMO Science format)
    questions = funcs['parse_cell_based_questions'](doc)
    if questions:
        return {
            "ok": True,
            "filename": file.filename,
            "total_lines": len(questions),
            "questions": questions,
            "count": len(questions),
        }

    # Extract lines from document
    lines, table_options = funcs['extract_docx_lines'](doc)

    # Detect format and use appropriate parser
    is_math_format = any(re.match(r'^Question\s+\d+', line, re.IGNORECASE) for line in lines[:20])
    is_english_level_format = (
        any('Section A' in line or 'Section B' in line for line in lines[:15]) and
        file.filename and 'LEVEL' in file.filename.upper()
    )
    is_envie_format = file.filename and 'EN-VIE' in file.filename.upper()

    if is_math_format:
        questions = funcs['parse_math_exam_questions'](lines)
    elif is_english_level_format:
        questions = funcs['parse_english_exam_questions'](doc)
    elif is_envie_format:
        questions = funcs['parse_envie_questions'](doc)
    else:
        questions = funcs['parse_bilingual_questions'](lines, table_options)

    return {
        "ok": True,
        "filename": file.filename,
        "total_lines": len(lines),
        "questions": questions,
        "count": len(questions),
    }


@router.post("/convert-word-to-excel")
def convert_word_to_excel(
    file: UploadFile = Form(...),
    use_latex: str = Form("0"),
    subject: str = Form("general")
) -> StreamingResponse:
    """Convert a Word file containing questions to Excel format."""
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from docx.oxml.ns import qn as docx_qn

    funcs = _get_parsing_functions()

    if not file.filename or not file.filename.lower().endswith('.docx'):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .docx")

    latex_enabled = use_latex == "1"

    try:
        content = file.file.read()
        doc = Document(io.BytesIO(content))

        # Try cell-based parser first
        questions = funcs['parse_cell_based_questions'](doc)

        is_math_format = False
        is_english_level_format = False
        is_envie_format = False

        if not questions:
            lines, table_options = funcs['extract_docx_lines'](doc, include_textboxes=True, use_latex=latex_enabled)

            is_math_format = any(re.match(r'^Question\s+\d+', line, re.IGNORECASE) for line in lines[:20])
            is_english_level_format = (
                any('Section A' in line or 'Section B' in line for line in lines[:15]) and
                file.filename and 'LEVEL' in file.filename.upper()
            )
            is_envie_format = file.filename and 'EN-VIE' in file.filename.upper()

            # Check for Word numbered list format
            is_word_numbered_format = False
            if not is_envie_format:
                body = doc._element.body
                num_level_0 = 0
                num_level_1 = 0
                for child in body:
                    tag = child.tag.split('}')[-1]
                    if tag == 'p':
                        pPr = child.find(docx_qn('w:pPr'))
                        if pPr is not None:
                            numPr = pPr.find(docx_qn('w:numPr'))
                            if numPr is not None:
                                ilvl_elem = numPr.find(docx_qn('w:ilvl'))
                                if ilvl_elem is not None:
                                    ilvl = int(ilvl_elem.get(docx_qn('w:val')) or '0')
                                    if ilvl == 0:
                                        num_level_0 += 1
                                    elif ilvl == 1:
                                        num_level_1 += 1
                if num_level_0 > 10 and num_level_0 == num_level_1:
                    is_word_numbered_format = True

            if is_math_format:
                questions = funcs['parse_math_exam_questions'](lines)
            elif is_english_level_format:
                questions = funcs['parse_english_exam_questions'](doc)
            elif is_envie_format or is_word_numbered_format:
                questions = funcs['parse_envie_questions'](doc)
            else:
                questions = funcs['parse_bilingual_questions'](lines, table_options)

        # Save to question bank
        detected_subject = subject if subject in ('english', 'math', 'science', 'general') else 'general'
        funcs['save_questions_to_bank'](
            questions,
            subject=detected_subject,
            source=f"word-import:{file.filename}",
            question_type='mcq' if questions and questions[0].get('options') else 'blank',
        )

        # Create Excel workbook
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Questions"

        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        highlight_fill = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")
        center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
        left_align = Alignment(horizontal="left", vertical="center", wrap_text=True)
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )

        # Determine max options
        max_options = 5
        for q in questions:
            opts = q.get('options', [])
            if len(opts) > max_options:
                max_options = len(opts)

        headers = ["#", "Question", "Answer"]
        for i in range(max_options):
            headers.append(f"Option {chr(65+i)}")

        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_align
            cell.border = thin_border

        for row_idx, q in enumerate(questions, 2):
            question_text = q.get('question', '')
            options = q.get('options', [])
            answer = q.get('answer', '')

            # Row number
            ws.cell(row=row_idx, column=1, value=row_idx - 1).alignment = center_align

            # Question text
            cell = ws.cell(row=row_idx, column=2, value=question_text)
            cell.alignment = left_align
            cell.border = thin_border

            # Answer
            answer_cell = ws.cell(row=row_idx, column=3, value=answer)
            answer_cell.alignment = center_align
            answer_cell.border = thin_border

            # Options
            for opt_idx, opt in enumerate(options):
                cell = ws.cell(row=row_idx, column=4 + opt_idx, value=opt)
                cell.alignment = left_align
                cell.border = thin_border

                # Highlight correct answer
                if answer and isinstance(answer, str):
                    answer_upper = answer.upper().strip()
                    opt_letter = chr(65 + opt_idx)
                    if (answer_upper == opt_letter or
                        answer_upper == opt.strip() or
                        (len(answer_upper) == 1 and ord(answer_upper) - ord('A') == opt_idx)):
                        cell.fill = highlight_fill

        # Adjust column widths
        ws.column_dimensions["A"].width = 5
        ws.column_dimensions["B"].width = 60
        ws.column_dimensions["C"].width = 10

        for i in range(max_options):
            col_letter = chr(68 + i)  # D, E, F, ...
            ws.column_dimensions[col_letter].width = 25

        output = io.BytesIO()
        wb.save(output)
        output.seek(0)

        base_name = Path(file.filename).stem
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={base_name}_questions.xlsx"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý file: {str(e)}")


@router.post("/api/analyze-exam")
async def analyze_exam_with_ai(
    file: UploadFile,
    ai_engine: str = Form("openai"),
):
    """Analyze an exam file using AI to extract and categorize questions."""
    funcs = _get_parsing_functions()

    if not file.filename.endswith('.docx'):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .docx")

    content = await file.read()
    doc = Document(io.BytesIO(content))

    text_content = funcs['extract_docx_content'](doc)
    settings = funcs['load_ai_settings']()

    prompt = f"""Phân tích đề thi sau và trích xuất các câu hỏi. Trả về JSON array với format:
[
    {{
        "number": 1,
        "content": "nội dung câu hỏi",
        "options": ["A", "B", "C", "D"],
        "correct_answer": "A",
        "topic": "chủ đề",
        "difficulty": "easy|medium|hard"
    }}
]

Nội dung đề thi:
{text_content[:8000]}

Chỉ trả về JSON array, không giải thích."""

    try:
        response, error = funcs['call_ai'](prompt, ai_engine)
        if error:
            return {"ok": False, "error": error}
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            questions = json.loads(json_match.group())
            return {
                "ok": True,
                "filename": file.filename,
                "questions": questions,
                "count": len(questions),
            }
        return {"ok": False, "error": "Không thể phân tích kết quả AI"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/api/generate-similar-exam")
async def generate_similar_exam(
    file: UploadFile,
    difficulty: str = Form("same"),
    subject: str = Form("auto"),
    bilingual: str = Form("auto"),
    ai_engine: str = Form("openai"),
):
    """Generate similar questions based on an exam file."""
    funcs = _get_parsing_functions()

    if not file.filename.endswith('.docx'):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .docx")

    content = await file.read()
    doc = Document(io.BytesIO(content))

    # Try cell-based parser first
    sample_questions = funcs['parse_cell_based_questions'](doc)

    if not sample_questions:
        lines, table_options = funcs['extract_docx_lines'](doc)

        is_math_format = any(re.match(r'^Question\s+\d+', line, re.IGNORECASE) for line in lines[:20])
        is_english_level_format = (
            any('Section A' in line or 'Section B' in line for line in lines[:15]) and
            file.filename and 'LEVEL' in file.filename.upper()
        )
        is_envie_format = file.filename and 'EN-VIE' in file.filename.upper()

        if is_math_format:
            sample_questions = funcs['parse_math_exam_questions'](lines)
        elif is_english_level_format:
            sample_questions = funcs['parse_english_exam_questions'](doc)
        elif is_envie_format:
            sample_questions = funcs['parse_envie_questions'](doc)
        else:
            sample_questions = funcs['parse_bilingual_questions'](lines, table_options)

    if not sample_questions:
        return {"ok": False, "error": "Không tìm thấy câu hỏi trong file"}

    # Build difficulty instruction
    difficulty_text = {
        "same": "tương đương về độ khó",
        "easier": "DỄ HƠN (đơn giản hơn)",
        "harder": "KHÓ HƠN (phức tạp hơn)"
    }.get(difficulty, "tương đương về độ khó")

    settings = funcs['load_ai_settings']()

    # Process in batches
    BATCH_SIZE = 5
    all_generated = []

    subject_names = {
        "science": "Science/Khoa học",
        "math": "Math/Toán học",
        "history": "History/Lịch sử",
        "geography": "Geography/Địa lý",
        "english": "English/Tiếng Anh",
        "general": "General",
        "auto": "General"
    }

    # Auto-detect subject if needed
    if subject == "auto":
        all_sample_text = ""
        for q in sample_questions[:10]:
            all_sample_text += q.get('question', '') + " " + " ".join(q.get('options', []))
        all_sample_lower = all_sample_text.lower()

        detected_subject = "general"
        if any(kw in all_sample_lower for kw in ['science', 'khoa học', 'biology', 'chemistry', 'physics']):
            detected_subject = "science"
        elif any(kw in all_sample_lower for kw in ['math', 'toán', 'calculate', 'equation', 'number']):
            detected_subject = "math"
        elif any(kw in all_sample_lower for kw in ['history', 'lịch sử', 'war', 'dynasty']):
            detected_subject = "history"
        elif any(kw in all_sample_lower for kw in ['geography', 'địa lý', 'country', 'continent']):
            detected_subject = "geography"
        elif any(kw in all_sample_lower for kw in ['english', 'tiếng anh', 'vocabulary', 'grammar']):
            detected_subject = "english"
    else:
        detected_subject = subject

    for batch_start in range(0, len(sample_questions), BATCH_SIZE):
        batch = sample_questions[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1

        # Build prompt for batch
        batch_prompt = f"""Bạn là chuyên gia tạo đề thi môn {subject_names.get(detected_subject, 'General')}.

Yêu cầu:
- Tạo câu hỏi MỚI HOÀN TOÀN cho từng câu hỏi mẫu bên dưới
- Câu hỏi mới phải {difficulty_text}
- Nếu câu hỏi gốc song ngữ (Anh-Việt), câu hỏi mới PHẢI song ngữ
- KHÔNG được copy nguyên văn câu hỏi gốc
- Giữ nguyên số lượng đáp án và format

"""
        for i, q in enumerate(batch, 1):
            q_text = q.get('question', '')
            opts = q.get('options', [])
            batch_prompt += f"\n--- Câu mẫu {batch_start + i} ---\n{q_text}\n"
            if opts:
                for j, opt in enumerate(opts):
                    batch_prompt += f"{chr(65+j)}) {opt}\n"

        batch_prompt += f"""

Trả về JSON array với format:
[
    {{"question": "câu hỏi mới", "options": ["A", "B", "C", "D"], "answer": "A"}}
]

Tạo ĐÚNG {len(batch)} câu hỏi mới. Chỉ trả về JSON array."""

        try:
            response, error = funcs['call_ai'](batch_prompt, ai_engine)
            if not error and response:
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    batch_generated = json.loads(json_match.group())
                    all_generated.extend(batch_generated)
                    continue
            # Add placeholder for failed batch
            for _ in batch:
                all_generated.append({
                    "question": "Lỗi tạo câu hỏi",
                    "options": [],
                    "answer": ""
                })
        except Exception:
            # Add placeholder for failed batch
            for _ in batch:
                all_generated.append({
                    "question": "Lỗi tạo câu hỏi",
                    "options": [],
                    "answer": ""
                })

    return {
        "ok": True,
        "filename": file.filename,
        "original_count": len(sample_questions),
        "generated_count": len(all_generated),
        "questions": all_generated,
        "detected_subject": detected_subject
    }


@router.get("/api/answer-templates")
def get_answer_templates():
    """Get available answer templates for grading."""
    templates = []
    for key, value in ANSWER_TEMPLATES.items():
        templates.append({
            "key": key,
            "name": value.get("name", key),
            "questions": value.get("questions", 30),
            "options": value.get("options", 4),
            "layout": value.get("layout", "row")
        })
    return {"templates": templates}
