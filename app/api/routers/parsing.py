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


def _clean_option_label(opt: str) -> str:
    """Remove leading option labels like 'A)', 'A.', 'Option A:' from option text."""
    return re.sub(r'^(?:Option\s+)?[A-E][).:]\s*', '', opt.strip())


def _extract_questions_from_json(text: str) -> list:
    """Extract question objects from AI response that may contain multiple JSON arrays."""
    # 1. Try parsing as single JSON array (greedy match)
    json_match = re.search(r'\[[\s\S]*\]', text)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if isinstance(result, list):
                # Clean option labels
                for q in result:
                    if 'options' in q:
                        q['options'] = [_clean_option_label(o) for o in q['options']]
                return result
        except json.JSONDecodeError:
            pass

    # 2. Find individual JSON objects with 'question' key
    questions = []
    for m in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text):
        try:
            obj = json.loads(m.group())
            if 'question' in obj:
                if 'options' in obj:
                    obj['options'] = [_clean_option_label(o) for o in obj['options']]
                questions.append(obj)
        except (json.JSONDecodeError, ValueError):
            pass
    return questions


def _is_bilingual_question(text: str) -> bool:
    """Check if a question text is bilingual (contains both English and Vietnamese)."""
    if not text:
        return False
    # Check for common bilingual separators
    if ' / ' in text or ' - ' in text:
        parts = re.split(r'\s*/\s*|\s*-\s*', text, maxsplit=1)
        if len(parts) >= 2:
            # Check if one part has Vietnamese characters and other has English
            has_viet = any('\u00c0' <= c <= '\u1ef9' for c in text)
            has_english = bool(re.search(r'[a-zA-Z]{3,}', text))
            return has_viet and has_english
    return False


def _has_vietnamese(text: str) -> bool:
    """Check if text contains Vietnamese characters."""
    return any('\u00c0' <= c <= '\u1ef9' for c in text)


def _has_english(text: str) -> bool:
    """Check if text contains English words."""
    return bool(re.search(r'[a-zA-Z]{3,}', text))


def _detect_exam_info(questions: list) -> tuple:
    """Detect subject and language from parsed questions.

    Returns:
        tuple: (detected_subject, detected_language)
    """
    if not questions:
        return ("general", "unknown")

    # Combine text from first 10 questions for analysis
    all_text = ""
    for q in questions[:10]:
        all_text += q.get('question', '') + " " + " ".join(q.get('options', []))
    all_text_lower = all_text.lower()

    # Detect subject
    detected_subject = "general"
    subject_keywords = {
        "science": ['science', 'khoa học', 'biology', 'sinh học', 'chemistry', 'hóa học',
                    'physics', 'vật lý', 'organism', 'cell', 'atom', 'molecule', 'energy'],
        "math": ['math', 'toán', 'calculate', 'tính', 'equation', 'phương trình',
                 'number', 'số', 'algebra', 'geometry', 'hình học'],
        "history": ['history', 'lịch sử', 'war', 'chiến tranh', 'dynasty', 'triều đại',
                    'king', 'vua', 'emperor', 'revolution', 'cách mạng'],
        "geography": ['geography', 'địa lý', 'country', 'quốc gia', 'continent', 'châu lục',
                      'river', 'sông', 'mountain', 'núi', 'capital', 'thủ đô'],
        "english": ['grammar', 'ngữ pháp', 'vocabulary', 'từ vựng', 'tense', 'thì',
                    'adjective', 'tính từ', 'verb', 'động từ', 'noun', 'danh từ'],
    }

    max_score = 0
    for subject, keywords in subject_keywords.items():
        score = sum(1 for kw in keywords if kw in all_text_lower)
        if score > max_score:
            max_score = score
            detected_subject = subject

    # Detect language
    has_viet = _has_vietnamese(all_text)
    has_eng = _has_english(all_text)
    has_bilingual_separator = ' / ' in all_text

    if has_bilingual_separator and has_viet and has_eng:
        detected_language = "bilingual"
    elif has_viet and has_eng:
        detected_language = "mixed"
    elif has_viet:
        detected_language = "vietnamese"
    elif has_eng:
        detected_language = "english"
    else:
        detected_language = "unknown"

    return (detected_subject, detected_language)


def _translate_to_bilingual(questions: list, call_ai_func, ai_engine: str) -> list:
    """Translate English questions to bilingual format (English / Vietnamese)."""
    result = []

    # Process in small batches of 3 for better translation accuracy
    TRANS_BATCH = 3
    for i in range(0, len(questions), TRANS_BATCH):
        batch = questions[i:i + TRANS_BATCH]

        # Build numbered translation prompt for better alignment
        items_to_translate = []
        for q in batch:
            items_to_translate.append(q.get('question', ''))
            for opt in q.get('options', []):
                items_to_translate.append(opt)

        prompt = f"Dịch {len(items_to_translate)} dòng sau sang tiếng Việt. CHỈ trả về bản dịch, giữ nguyên đánh số, KHÔNG thêm lời giới thiệu.\n\n"
        for idx, item in enumerate(items_to_translate, 1):
            prompt += f"{idx}. {item}\n"

        try:
            response, error = call_ai_func(prompt, ai_engine)
            if not error and response:
                # Remove thinking tags if present
                response = re.sub(r'<think>[\s\S]*?</think>', '', response).strip()
                # Build dict keyed by line number for reliable alignment
                trans_dict = {}
                for line in response.strip().split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    # Only accept lines starting with a number
                    m = re.match(r'^(\d+)[.):\s]+(.+)', line)
                    if m:
                        num = int(m.group(1))
                        trans_dict[num] = m.group(2).strip()

                # Convert to ordered list
                translations = []
                for idx in range(1, len(items_to_translate) + 1):
                    translations.append(trans_dict.get(idx, ''))

                idx = 0
                for q in batch:
                    eng_question = q.get('question', '')
                    eng_options = q.get('options', [])
                    answer = q.get('answer', 'A')

                    # Get Vietnamese translation for question
                    viet_question = translations[idx] if idx < len(translations) else eng_question
                    idx += 1
                    # Clean any leftover label prefix
                    viet_question = re.sub(r'^(?:Option\s+)?[A-E][).:]\s*', '', viet_question)

                    # Get Vietnamese translations for options
                    bilingual_options = []
                    for opt in eng_options:
                        viet_opt = translations[idx] if idx < len(translations) else opt
                        idx += 1
                        # Clean up any A), B) prefix from translation
                        viet_opt = re.sub(r'^(?:Option\s+)?[A-E][).:]\s*', '', viet_opt)
                        bilingual_options.append(f"{opt} / {viet_opt}")

                    result.append({
                        'question': f"{eng_question}\n{viet_question}",
                        'options': bilingual_options,
                        'answer': answer,
                    })
                continue
        except Exception:
            pass

        # Fallback: keep original if translation fails
        result.extend(batch)

    return result


def _ensure_bilingual_format(questions: list) -> list:
    """Post-process to ensure questions are in bilingual format."""
    result = []
    for q in questions:
        question_text = q.get('question', '')
        options = q.get('options', [])
        answer = q.get('answer', 'A')

        # Check if question already has both languages
        has_slash = ' / ' in question_text
        has_viet = _has_vietnamese(question_text)
        has_eng = _has_english(question_text)

        # If question is only in one language, try to detect and note it
        if not has_slash or not (has_viet and has_eng):
            # If only English, add Vietnamese placeholder
            if has_eng and not has_viet:
                question_text = question_text + " / [Cần dịch sang tiếng Việt]"
            # If only Vietnamese, add English placeholder
            elif has_viet and not has_eng:
                question_text = "[Need English translation] / " + question_text

        # Process options similarly
        processed_options = []
        for opt in options:
            opt_has_slash = ' / ' in opt
            opt_has_viet = _has_vietnamese(opt)
            opt_has_eng = _has_english(opt)

            if not opt_has_slash or not (opt_has_viet and opt_has_eng):
                if opt_has_eng and not opt_has_viet:
                    opt = opt + " / [Cần dịch]"
                elif opt_has_viet and not opt_has_eng:
                    opt = "[Translation] / " + opt
            processed_options.append(opt)

        result.append({
            'question': question_text,
            'options': processed_options,
            'answer': answer
        })

    return result


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

    if is_math_format:
        questions = funcs['parse_math_exam_questions'](lines)
    elif is_english_level_format:
        questions = funcs['parse_english_exam_questions'](doc)
    elif is_envie_format:
        questions = funcs['parse_envie_questions'](doc)
    else:
        questions = funcs['parse_bilingual_questions'](lines, table_options)

    # Detect subject and language
    detected_subject, detected_language = _detect_exam_info(questions)

    return {
        "ok": True,
        "filename": file.filename,
        "total_lines": len(lines),
        "questions": questions,
        "count": len(questions),
        "detected_subject": detected_subject,
        "detected_language": detected_language,
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
