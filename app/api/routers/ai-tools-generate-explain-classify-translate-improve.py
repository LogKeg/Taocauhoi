"""
AI Tools API endpoints for question generation, explanation, classification, translation, and improvement.
"""
import json
from typing import Optional
from fastapi import APIRouter, Form
from sqlalchemy import text

from app.database import SessionLocal
from app.services.ai import call_ai

router = APIRouter(prefix="/ai-tools", tags=["ai-tools"])


def _get_question_by_id(question_id: int) -> Optional[dict]:
    """Get question from database by ID."""
    db = SessionLocal()
    try:
        result = db.execute(
            text("SELECT id, content, options, answer, explanation, subject, topic, grade, difficulty FROM questions WHERE id = :id"),
            {"id": question_id}
        ).fetchone()
        if result:
            return {
                "id": result[0],
                "content": result[1],
                "options": json.loads(result[2]) if result[2] else None,
                "answer": result[3],
                "explanation": result[4],
                "subject": result[5],
                "topic": result[6],
                "grade": result[7],
                "difficulty": result[8],
            }
        return None
    finally:
        db.close()


def _update_question_field(question_id: int, field: str, value: str) -> bool:
    """Update a single field of a question."""
    db = SessionLocal()
    try:
        db.execute(
            text(f"UPDATE questions SET {field} = :value WHERE id = :id"),
            {"value": value, "id": question_id}
        )
        db.commit()
        return True
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


@router.post("/generate")
def generate_questions(
    subject: str = Form(...),
    grade: str = Form("12"),
    topic: str = Form(""),
    difficulty: str = Form("medium"),
    count: int = Form(5),
    engine: str = Form("gemini"),
) -> dict:
    """Generate new questions using AI.

    Returns list of generated questions in JSON format.
    """
    prompt = f"""Bạn là giáo viên chuyên ra đề thi. Hãy tạo {count} câu hỏi trắc nghiệm môn {subject} lớp {grade}.
{"Chủ đề: " + topic if topic else ""}
Độ khó: {difficulty}

Yêu cầu:
- Mỗi câu có 4 đáp án A, B, C, D
- Chỉ có 1 đáp án đúng
- Câu hỏi phải rõ ràng, chính xác
- Phù hợp với chương trình học Việt Nam

Trả về JSON array với format:
[
  {{
    "content": "Nội dung câu hỏi",
    "options": ["Đáp án A", "Đáp án B", "Đáp án C", "Đáp án D"],
    "answer": "A",
    "explanation": "Giải thích ngắn gọn"
  }}
]

Chỉ trả về JSON, không có text khác."""

    response, error = call_ai(prompt, engine)
    if error:
        return {"success": False, "error": error}

    try:
        # Extract JSON from response
        text = response.strip()
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        questions = json.loads(text.strip())
        return {
            "success": True,
            "questions": questions,
            "count": len(questions)
        }
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {str(e)}", "raw": response[:500]}


@router.post("/explain")
def explain_answer(
    question_id: int = Form(None),
    content: str = Form(None),
    options: str = Form(None),
    answer: str = Form(None),
    engine: str = Form("gemini"),
) -> dict:
    """Generate explanation for why an answer is correct.

    Can use question_id to load from DB, or provide content/options/answer directly.
    """
    if question_id:
        q = _get_question_by_id(question_id)
        if not q:
            return {"success": False, "error": "Không tìm thấy câu hỏi"}
        content = q["content"]
        options = q["options"]
        answer = q["answer"]

    if not content or not answer:
        return {"success": False, "error": "Thiếu nội dung câu hỏi hoặc đáp án"}

    # Parse options if string
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except json.JSONDecodeError:
            options = None

    options_text = ""
    if options:
        labels = ["A", "B", "C", "D"]
        options_text = "\n".join([f"{labels[i]}. {opt}" for i, opt in enumerate(options) if i < len(labels)])

    prompt = f"""Câu hỏi: {content}

{options_text}

Đáp án đúng: {answer}

Hãy giải thích ngắn gọn (2-3 câu) tại sao đáp án {answer} là đúng và các đáp án khác sai.
Chỉ trả về phần giải thích, không lặp lại câu hỏi."""

    response, error = call_ai(prompt, engine)
    if error:
        return {"success": False, "error": error}

    # Save to DB if question_id provided
    if question_id:
        _update_question_field(question_id, "explanation", response.strip())

    return {
        "success": True,
        "explanation": response.strip(),
        "saved": question_id is not None
    }


@router.post("/classify")
def classify_question(
    question_id: int = Form(None),
    content: str = Form(None),
    engine: str = Form("gemini"),
) -> dict:
    """Classify question by subject, topic, grade, and difficulty.

    Can use question_id to load from DB, or provide content directly.
    """
    if question_id:
        q = _get_question_by_id(question_id)
        if not q:
            return {"success": False, "error": "Không tìm thấy câu hỏi"}
        content = q["content"]

    if not content:
        return {"success": False, "error": "Thiếu nội dung câu hỏi"}

    prompt = f"""Phân loại câu hỏi sau theo chương trình giáo dục Việt Nam:

"{content}"

Trả về JSON với format:
{{
  "subject": "toan|vat_ly|hoa_hoc|sinh_hoc|tieng_anh|ngu_van|lich_su|dia_ly|gdcd|tin_hoc",
  "topic": "Chủ đề cụ thể (VD: Đạo hàm, Động lực học, ...)",
  "grade": "10|11|12",
  "difficulty": "easy|medium|hard"
}}

Chỉ trả về JSON, không có text khác."""

    response, error = call_ai(prompt, engine)
    if error:
        return {"success": False, "error": error}

    try:
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        classification = json.loads(text.strip())

        # Save to DB if question_id provided
        if question_id:
            db = SessionLocal()
            try:
                db.execute(
                    text("UPDATE questions SET subject = :subject, topic = :topic, grade = :grade, difficulty = :difficulty WHERE id = :id"),
                    {
                        "subject": classification.get("subject", ""),
                        "topic": classification.get("topic", ""),
                        "grade": classification.get("grade", ""),
                        "difficulty": classification.get("difficulty", ""),
                        "id": question_id
                    }
                )
                db.commit()
            finally:
                db.close()

        return {
            "success": True,
            "classification": classification,
            "saved": question_id is not None
        }
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {str(e)}", "raw": response[:300]}


@router.post("/translate")
def translate_question(
    question_id: int = Form(None),
    content: str = Form(None),
    options: str = Form(None),
    target_lang: str = Form("vi"),  # vi or en
    engine: str = Form("gemini"),
) -> dict:
    """Translate question between Vietnamese and English.

    target_lang: 'vi' for Vietnamese, 'en' for English
    """
    if question_id:
        q = _get_question_by_id(question_id)
        if not q:
            return {"success": False, "error": "Không tìm thấy câu hỏi"}
        content = q["content"]
        options = q["options"]

    if not content:
        return {"success": False, "error": "Thiếu nội dung câu hỏi"}

    # Parse options if string
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except json.JSONDecodeError:
            options = None

    lang_name = "tiếng Việt" if target_lang == "vi" else "tiếng Anh"

    options_text = ""
    if options:
        labels = ["A", "B", "C", "D"]
        options_text = "\nCác đáp án:\n" + "\n".join([f"{labels[i]}. {opt}" for i, opt in enumerate(options) if i < len(labels)])

    prompt = f"""Dịch câu hỏi trắc nghiệm sau sang {lang_name}. Giữ nguyên ý nghĩa và format.

Câu hỏi: {content}{options_text}

Trả về JSON với format:
{{
  "content": "Nội dung đã dịch",
  "options": ["Đáp án A đã dịch", "Đáp án B đã dịch", "Đáp án C đã dịch", "Đáp án D đã dịch"]
}}

Chỉ trả về JSON, không có text khác."""

    response, error = call_ai(prompt, engine)
    if error:
        return {"success": False, "error": error}

    try:
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        translated = json.loads(text.strip())
        return {
            "success": True,
            "translated": translated,
            "target_lang": target_lang
        }
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {str(e)}", "raw": response[:300]}


@router.post("/improve")
def improve_question(
    question_id: int = Form(None),
    content: str = Form(None),
    options: str = Form(None),
    engine: str = Form("gemini"),
) -> dict:
    """Improve question clarity, grammar, and quality.

    Returns improved version of the question.
    """
    if question_id:
        q = _get_question_by_id(question_id)
        if not q:
            return {"success": False, "error": "Không tìm thấy câu hỏi"}
        content = q["content"]
        options = q["options"]

    if not content:
        return {"success": False, "error": "Thiếu nội dung câu hỏi"}

    # Parse options if string
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except json.JSONDecodeError:
            options = None

    options_text = ""
    if options:
        labels = ["A", "B", "C", "D"]
        options_text = "\nCác đáp án:\n" + "\n".join([f"{labels[i]}. {opt}" for i, opt in enumerate(options) if i < len(labels)])

    prompt = f"""Cải thiện câu hỏi trắc nghiệm sau:

Câu hỏi: {content}{options_text}

Yêu cầu cải thiện:
1. Sửa lỗi chính tả, ngữ pháp
2. Làm rõ ràng hơn nếu câu hỏi mơ hồ
3. Đảm bảo các đáp án có độ dài tương đương
4. Loại bỏ các đáp án quá dễ đoán
5. Giữ nguyên ý nghĩa và độ khó

Trả về JSON với format:
{{
  "content": "Nội dung đã cải thiện",
  "options": ["Đáp án A", "Đáp án B", "Đáp án C", "Đáp án D"],
  "changes": ["Mô tả thay đổi 1", "Mô tả thay đổi 2"]
}}

Chỉ trả về JSON, không có text khác."""

    response, error = call_ai(prompt, engine)
    if error:
        return {"success": False, "error": error}

    try:
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        improved = json.loads(text.strip())
        return {
            "success": True,
            "improved": improved,
            "original": {"content": content, "options": options}
        }
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {str(e)}", "raw": response[:300]}


@router.post("/batch-explain")
def batch_explain(
    question_ids: str = Form(...),  # Comma-separated IDs
    engine: str = Form("gemini"),
) -> dict:
    """Generate explanations for multiple questions."""
    ids = [int(x.strip()) for x in question_ids.split(",") if x.strip().isdigit()]

    results = []
    for qid in ids[:20]:  # Limit to 20
        result = explain_answer(question_id=qid, engine=engine)
        results.append({"id": qid, **result})

    success_count = sum(1 for r in results if r.get("success"))
    return {
        "success": True,
        "total": len(results),
        "success_count": success_count,
        "results": results
    }


@router.post("/batch-classify")
def batch_classify(
    question_ids: str = Form(...),  # Comma-separated IDs
    engine: str = Form("gemini"),
) -> dict:
    """Classify multiple questions."""
    ids = [int(x.strip()) for x in question_ids.split(",") if x.strip().isdigit()]

    results = []
    for qid in ids[:20]:  # Limit to 20
        result = classify_question(question_id=qid, engine=engine)
        results.append({"id": qid, **result})

    success_count = sum(1 for r in results if r.get("success"))
    return {
        "success": True,
        "total": len(results),
        "success_count": success_count,
        "results": results
    }
