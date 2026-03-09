"""
AI-powered exam analysis and question generation endpoints.

POST /api/analyze-exam - Analyze exam file using AI to extract and categorize questions.
POST /api/generate-similar-exam - Generate similar questions based on exam file.
"""
import io
import re

from docx import Document
from fastapi import APIRouter, Form, HTTPException, UploadFile

from app.api.routers.parsing.helpers import (
    _extract_questions_from_json,
    _get_parsing_functions,
    _is_bilingual_question,
    _translate_to_bilingual,
)

router = APIRouter(tags=["parsing"])


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
            import json
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

    # Detect if questions are bilingual
    is_bilingual = bilingual in ("yes", "bilingual") or (bilingual == "auto" and any(
        _is_bilingual_question(q.get('question', ''))
        for q in sample_questions[:5]
    ))

    # Process in batches
    BATCH_SIZE = 5
    all_generated = []

    for batch_start in range(0, len(sample_questions), BATCH_SIZE):
        batch = sample_questions[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1

        # Build prompt for batch
        if is_bilingual:
            # Step 1: Create English questions first
            batch_prompt = f"""Create {len(batch)} {subject_names.get(detected_subject, 'science')} multiple choice questions in ENGLISH ONLY.

Requirements:
- Create NEW questions similar in style and topic to the samples below
- Difficulty: {difficulty_text}
- Each question has 5 options (A through E)
- Return a SINGLE JSON array with ALL {len(batch)} questions

SAMPLES:
"""
        else:
            batch_prompt = f"""Bạn là chuyên gia tạo đề thi môn {subject_names.get(detected_subject, 'General')}.

Yêu cầu:
- Tạo câu hỏi MỚI HOÀN TOÀN cho từng câu hỏi mẫu bên dưới
- Câu hỏi mới phải {difficulty_text}
- KHÔNG được copy nguyên văn câu hỏi gốc
- Giữ nguyên số lượng đáp án và format
"""
        for i, q in enumerate(batch, 1):
            q_text = q.get('question', '')
            opts = q.get('options', [])
            # For bilingual, show only English part of sample
            if is_bilingual and ' / ' in q_text:
                q_text = q_text.split(' / ')[0]
            batch_prompt += f"\n--- Sample {batch_start + i} ---\n{q_text}\n"
            if opts:
                for j, opt in enumerate(opts):
                    opt_text = opt.split(' / ')[0] if is_bilingual and ' / ' in opt else opt
                    batch_prompt += f"{chr(65+j)}) {opt_text}\n"

        if is_bilingual:
            batch_prompt += f"""

IMPORTANT: Return ONLY a single JSON array with exactly {len(batch)} question objects. No extra text.
Format: [{{"question": "...", "options": ["A text", "B text", "C text", "D text", "E text"], "answer": "C"}}]"""
        else:
            batch_prompt += f"""

Trả về JSON array với format:
[
    {{"question": "câu hỏi mới", "options": ["A", "B", "C", "D"], "answer": "A"}}
]

Tạo ĐÚNG {len(batch)} câu hỏi mới. Chỉ trả về JSON array."""

        try:
            # Try up to 2 times to get enough questions
            batch_generated = []
            for attempt in range(2):
                response, error = funcs['call_ai'](batch_prompt, ai_engine)
                if not error and response:
                    response = re.sub(r'<think>[\s\S]*?</think>', '', response).strip()
                    batch_generated = _extract_questions_from_json(response)
                if len(batch_generated) >= len(batch):
                    break
            if batch_generated:
                all_generated.extend(batch_generated[:len(batch)])
                continue
            # Add placeholder for failed batch
            for _ in batch:
                all_generated.append({
                    "question": "Lỗi tạo câu hỏi",
                    "options": [],
                    "answer": ""
                })
        except Exception:
            for _ in batch:
                all_generated.append({
                    "question": "Lỗi tạo câu hỏi",
                    "options": [],
                    "answer": ""
                })

    # Step 2: If bilingual, translate to Vietnamese
    if is_bilingual and all_generated:
        all_generated = _translate_to_bilingual(all_generated, funcs['call_ai'], ai_engine)

    return {
        "ok": True,
        "filename": file.filename,
        "original_count": len(sample_questions),
        "generated_count": len(all_generated),
        "questions": all_generated,
        "detected_subject": detected_subject,
    }
