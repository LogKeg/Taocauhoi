"""
Helper functions for parsing endpoints.

Shared utilities for exam parsing, format detection, and question processing.
"""
import json
import re
from typing import Dict, List, Tuple


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


def _detect_exam_info(questions: list) -> Tuple[str, str]:
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
    """Lazy import of parsing functions from modules."""
    from app.parsers.docx import (
        parse_cell_based_questions,
        extract_docx_lines_with_options,
        extract_docx_content,
        parse_bilingual_questions,
    )
    from app.parsers import (
        _parse_math_exam_questions,
        _parse_english_exam_questions,
        _parse_envie_questions,
    )
    from app.services.ai import call_ai, load_saved_settings
    return {
        'parse_cell_based_questions': parse_cell_based_questions,
        'extract_docx_lines': extract_docx_lines_with_options,
        'extract_docx_content': extract_docx_content,
        'parse_math_exam_questions': _parse_math_exam_questions,
        'parse_english_exam_questions': _parse_english_exam_questions,
        'parse_envie_questions': _parse_envie_questions,
        'parse_bilingual_questions': parse_bilingual_questions,
        'save_questions_to_bank': _save_questions_to_bank,
        'call_ai': call_ai,
        'load_ai_settings': load_saved_settings,
    }
