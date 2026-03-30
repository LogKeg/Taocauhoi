"""
Matrix-based question generation endpoint.

POST /generate-matrix - Generate questions distributed across topics and difficulty levels.
Follows Vietnamese exam specification matrix (ma trận đề thi).
"""
import json
import re

from fastapi import APIRouter, Form

from app.services.ai import call_ai
from app.services.generation import (
    build_topic_prompt,
    normalize_ai_blocks,
    retrieve_similar_questions,
)
from app.api.routers.generation.helpers import (
    _is_engine_available,
    _save_text_questions_to_bank,
    _parse_explanations,
)

router = APIRouter(tags=["generation"])

# Difficulty labels for Vietnamese UI
DIFFICULTY_LABELS = {
    "easy": "Nhận biết (Dễ)",
    "medium": "Thông hiểu (Trung bình)",
    "hard": "Vận dụng (Khó)",
}


def _parse_ai_response(text: str) -> dict:
    """Parse AI response into questions, answers, explanations."""
    explanations = ""
    expl_pattern = re.compile(
        r"\n\s*-{0,3}\s*(?:LỜI GIẢI|Lời giải|EXPLANATIONS|Explanations|Giải thích)\s*-{0,3}\s*:?\s*\n",
        re.IGNORECASE,
    )
    expl_match = expl_pattern.search(text)
    if expl_match:
        explanations = text[expl_match.end():].strip()
        text = text[:expl_match.start()]

    answers = ""
    ans_pattern = re.compile(
        r"\n\s*-{0,3}\s*(?:ĐÁP ÁN|Đáp án|đáp án|ANSWERS|Answers|Answer Key)\s*-{0,3}\s*:?\s*\n",
        re.IGNORECASE,
    )
    match = ans_pattern.search(text)
    if match:
        raw_answers = text[match.end():].strip()
        text = text[:match.start()]
        answer_lines = []
        for line in raw_answers.splitlines():
            line = line.strip()
            ans_match = re.match(r'^(?:Câu\s*)?(\d+)[\.\):\s]+([A-Da-d])\b', line, re.IGNORECASE)
            if ans_match:
                answer_lines.append(f"{ans_match.group(1)}. {ans_match.group(2).upper()}")
        answers = "\n".join(answer_lines) if answer_lines else raw_answers

    questions = normalize_ai_blocks(text)
    questions = [q for q in questions if q.strip()]

    return {"questions": questions, "answers": answers, "explanations": explanations}


@router.post("/generate-matrix")
def generate_matrix(
    subject: str = Form(...),
    grade: int = Form(1),
    qtype: str = Form("mcq"),
    ai_engine: str = Form("ollama"),
    language: str = Form("vi"),
    use_rag: bool = Form(True),
    matrix_json: str = Form(...),
) -> dict:
    """
    Generate questions following an exam specification matrix.

    matrix_json format:
    [
        {"topic": "topic_key", "topic_label": "Đại số", "easy": 3, "medium": 2, "hard": 1},
        {"topic": "geometry", "topic_label": "Hình học", "easy": 2, "medium": 2, "hard": 1},
    ]
    """
    if not _is_engine_available(ai_engine):
        engine_names = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "ollama": "Ollama"}
        return {"success": False, "message": f"Chưa cấu hình {engine_names.get(ai_engine, ai_engine)}."}

    try:
        matrix = json.loads(matrix_json)
    except json.JSONDecodeError:
        return {"success": False, "message": "Dữ liệu ma trận không hợp lệ."}

    if not matrix:
        return {"success": False, "message": "Ma trận trống."}

    grade = max(1, min(12, grade))
    difficulties = ["easy", "medium", "hard"]

    # Results grouped by section
    sections = []
    all_answers = []
    all_explanations = []
    total_questions = 0
    question_number = 0
    errors = []

    for row in matrix:
        topic = row.get("topic", "")
        topic_label = row.get("topic_label", topic)

        for diff in difficulties:
            count = int(row.get(diff, 0))
            if count <= 0:
                continue

            count = min(count, 15)

            # Try RAG first
            rag_examples = []
            if use_rag:
                rag_examples = retrieve_similar_questions(
                    subject=subject,
                    topic=topic,
                    difficulty=diff,
                    question_type=qtype,
                    limit=min(3, count),
                )

            prompt = build_topic_prompt(
                subject, grade, qtype, count, topic,
                diff, rag_examples=rag_examples, language=language,
            )
            text, err = call_ai(prompt, ai_engine)

            if not text:
                errors.append(f"{topic_label}/{DIFFICULTY_LABELS.get(diff, diff)}: {err or 'Không nhận được phản hồi'}")
                continue

            result = _parse_ai_response(text)
            cell_questions = result["questions"][:count]

            if not cell_questions:
                errors.append(f"{topic_label}/{DIFFICULTY_LABELS.get(diff, diff)}: Không tạo được câu hỏi")
                continue

            # Renumber answers for this cell
            cell_answers = []
            if result["answers"]:
                for line in result["answers"].split("\n"):
                    m = re.match(r'^(\d+)\.\s*(.+)', line)
                    if m:
                        old_num = int(m.group(1))
                        if old_num <= len(cell_questions):
                            cell_answers.append(f"{question_number + old_num}. {m.group(2)}")

            sections.append({
                "topic": topic,
                "topic_label": topic_label,
                "difficulty": diff,
                "difficulty_label": DIFFICULTY_LABELS.get(diff, diff),
                "questions": cell_questions,
                "start_number": question_number + 1,
            })

            all_answers.extend(cell_answers)
            if result["explanations"]:
                all_explanations.append(
                    f"--- {topic_label} / {DIFFICULTY_LABELS.get(diff, diff)} ---\n{result['explanations']}"
                )

            question_number += len(cell_questions)
            total_questions += len(cell_questions)

    return {
        "success": True,
        "sections": sections,
        "total_questions": total_questions,
        "answers": "\n".join(all_answers),
        "explanations": "\n\n".join(all_explanations),
        "errors": errors,
        "matrix": matrix,
    }
