"""
Exam file parsing endpoints.

POST /api/parse-exam - Parse questions from uploaded Word document.
"""
import io
import re

from docx import Document
from fastapi import APIRouter, HTTPException, UploadFile

from app.api.routers.parsing.helpers import _detect_exam_info, _get_parsing_functions

router = APIRouter(tags=["parsing"])


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
        # Detect subject and language
        detected_subject, detected_language = _detect_exam_info(questions)
        return {
            "ok": True,
            "filename": file.filename,
            "total_lines": len(questions),
            "questions": questions,
            "count": len(questions),
            "detected_subject": detected_subject,
            "detected_language": detected_language,
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

    # Detect bilingual science (IKSC) format
    is_bilingual_science_format = False
    if not is_math_format and not is_english_level_format and not is_envie_format:
        bilingual_option_count = sum(
            1 for line in lines[:80]
            if re.match(r'^[A-E][.)]\s*.+\s*/\s*.+', line)
        )
        bilingual_pair_count = 0
        for li in range(len(lines) - 1):
            l1, l2 = lines[li], lines[li + 1]
            if (re.match(r'^\d+\.\s*', l1)
                    and not any(ord(c) > 127 for c in l1[:200])
                    and any(ord(c) > 127 for c in l2[:200])
                    and not re.match(r'^[A-E][.)]\s*', l2)):
                bilingual_pair_count += 1
                if bilingual_pair_count >= 3:
                    break
        if bilingual_option_count >= 3 or bilingual_pair_count >= 3:
            is_bilingual_science_format = True
            is_envie_format = True  # Use envie parser

    # Detect English reading/competition exam format
    is_english_reading_format = False
    if not is_math_format and not is_english_level_format and not is_envie_format:
        text_sample = ' '.join(lines[:50]).lower()
        has_instruction = any(
            kw in text_sample
            for kw in ['for each question', 'read the text', 'choose the correct answer', 'choose the best answer']
        )
        has_merged_options = any(
            re.search(r'[A-D]\)\s*\w', line) for line in lines[:80]
        )
        if has_instruction or has_merged_options:
            is_english_reading_format = True

    if is_math_format:
        questions = funcs['parse_math_exam_questions'](lines)
    elif is_english_level_format:
        questions = funcs['parse_english_exam_questions'](doc)
    elif is_envie_format or is_english_reading_format:
        questions = funcs['parse_envie_questions'](doc)
    else:
        questions = funcs['parse_bilingual_questions'](lines, table_options)

    # Detect subject and language
    detected_subject, detected_language = _detect_exam_info(questions)

    # For science subject, apply bilingual dedup (IKSC format: EN+VN pairs)
    if detected_subject == 'science' or is_bilingual_science_format:
        from app.parsers import _dedup_bilingual_science
        questions = _dedup_bilingual_science(questions)

    return {
        "ok": True,
        "filename": file.filename,
        "total_lines": len(lines),
        "questions": questions,
        "count": len(questions),
        "detected_subject": detected_subject,
        "detected_language": detected_language,
    }
