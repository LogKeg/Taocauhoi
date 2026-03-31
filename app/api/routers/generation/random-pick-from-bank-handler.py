"""
Random pick questions from question bank to create exams.

POST /generate-from-bank - Pick random questions by subject/topic/difficulty matrix.
Tracks usage count (times_used) for each picked question.
Prioritizes least-used questions to ensure fair distribution.
"""
import json
import random
from typing import List, Optional

from fastapi import APIRouter, Request, HTTPException
from sqlalchemy import func, or_

from app.database import SessionLocal
from app.database.models import Question

router = APIRouter(tags=["generation"])


def _pick_questions(
    db,
    subject: str,
    topic: Optional[str],
    difficulty: Optional[str],
    count: int,
    question_type: str = "mcq",
    grade: Optional[str] = None,
    exclude_ids: Optional[List[int]] = None,
) -> List[Question]:
    """
    Pick random questions from bank, prioritizing least-used ones.
    Returns list of Question objects.
    """
    query = db.query(Question).filter(Question.subject == subject)

    if topic:
        query = query.filter(Question.topic == topic)
    if difficulty:
        query = query.filter(Question.difficulty == difficulty)
    if question_type:
        query = query.filter(Question.question_type == question_type)
    if grade:
        query = query.filter(Question.grade == grade)
    if exclude_ids:
        query = query.filter(~Question.id.in_(exclude_ids))

    # Only pick questions that have answers
    query = query.filter(
        Question.answer != None,
        Question.answer != "",
    )

    # Order by times_used (ascending) to pick least-used first, then random
    available = query.order_by(Question.times_used.asc(), func.random()).limit(count * 3).all()

    if not available:
        return []

    # Shuffle among least-used group and pick requested count
    # Group by usage count, pick from lowest usage group first
    if len(available) <= count:
        return available

    random.shuffle(available)
    return available[:count]


@router.post("/generate-from-bank")
async def generate_from_bank(request: Request):
    """
    Pick random questions from question bank based on matrix specification.

    Request body JSON:
    {
        "subject": "toan_hoc",
        "question_type": "mcq",
        "matrix": [
            {"topic": "dai_so", "topic_label": "Đại số", "easy": 3, "medium": 2, "hard": 1},
            {"topic": "hinh_hoc", "topic_label": "Hình học", "easy": 2, "medium": 2, "hard": 0}
        ],
        "shuffle_questions": true,
        "shuffle_options": false
    }

    Response includes times_used for each question.
    """
    data = await request.json()
    subject = data.get("subject", "")
    question_type = data.get("question_type", "mcq")
    grade = data.get("grade", "")
    matrix = data.get("matrix", [])
    shuffle_questions = data.get("shuffle_questions", True)
    shuffle_options = data.get("shuffle_options", False)

    if not subject:
        raise HTTPException(status_code=400, detail="Chưa chọn môn học")
    if not matrix:
        raise HTTPException(status_code=400, detail="Ma trận trống")

    db = SessionLocal()
    try:
        sections = []
        all_picked_ids = []
        total_picked = 0
        total_requested = 0
        shortfalls = []
        difficulties = ["easy", "medium", "hard"]

        for row in matrix:
            topic = row.get("topic", "")
            topic_label = row.get("topic_label", topic)

            for diff in difficulties:
                count = int(row.get(diff, 0))
                if count <= 0:
                    continue

                total_requested += count

                # Pick questions, excluding already picked ones
                picked = _pick_questions(
                    db, subject, topic if topic else None,
                    diff, count, question_type,
                    grade=grade if grade else None,
                    exclude_ids=all_picked_ids,
                )

                # Track shortfall
                if len(picked) < count:
                    shortfalls.append(
                        f"{topic_label}/{diff}: cần {count}, có {len(picked)}"
                    )

                if not picked:
                    continue

                # Format questions
                diff_labels = {
                    "easy": "Nhận biết (Dễ)",
                    "medium": "Thông hiểu (Trung bình)",
                    "hard": "Vận dụng (Khó)",
                }

                section_questions = []
                for q in picked:
                    # Parse options
                    options = []
                    if q.options:
                        try:
                            options = json.loads(q.options) if isinstance(q.options, str) else q.options
                        except (json.JSONDecodeError, TypeError):
                            options = []

                    # Optionally shuffle options
                    answer = q.answer or ""
                    if shuffle_options and options and answer:
                        # Track correct answer content before shuffle
                        answer_idx = ord(answer.upper()) - 65 if answer.upper() in "ABCDE" else -1
                        correct_content = options[answer_idx] if 0 <= answer_idx < len(options) else None

                        random.shuffle(options)

                        # Update answer letter after shuffle
                        if correct_content and correct_content in options:
                            answer = chr(65 + options.index(correct_content))

                    section_questions.append({
                        "id": q.id,
                        "content": q.content,
                        "options": options,
                        "answer": answer,
                        "explanation": q.explanation or "",
                        "times_used": q.times_used or 0,
                        "source": q.source or "",
                        "image_url": q.image_url or "",
                    })

                    all_picked_ids.append(q.id)

                sections.append({
                    "topic": topic,
                    "topic_label": topic_label,
                    "difficulty": diff,
                    "difficulty_label": diff_labels.get(diff, diff),
                    "questions": section_questions,
                    "requested": count,
                    "picked": len(section_questions),
                })

                total_picked += len(section_questions)

        # Optionally shuffle all questions within sections
        if shuffle_questions:
            for sec in sections:
                random.shuffle(sec["questions"])

        # Update times_used for all picked questions
        if all_picked_ids:
            db.query(Question).filter(Question.id.in_(all_picked_ids)).update(
                {Question.times_used: Question.times_used + 1},
                synchronize_session=False,
            )
            db.commit()

        # Build flat answer key
        answer_lines = []
        q_num = 0
        for sec in sections:
            for q in sec["questions"]:
                q_num += 1
                if q["answer"]:
                    answer_lines.append(f"{q_num}. {q['answer']}")

        return {
            "success": True,
            "sections": sections,
            "total_picked": total_picked,
            "total_requested": total_requested,
            "answers": "\n".join(answer_lines),
            "shortfalls": shortfalls,
            "picked_ids": all_picked_ids,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        db.close()


@router.get("/api/bank-availability")
def check_bank_availability(
    subject: str = "",
    question_type: str = "mcq",
    grade: str = "",
):
    """
    Check how many questions are available in bank by topic and difficulty.
    Used to show availability when building matrix.
    """
    if not subject:
        return {"available": {}}

    db = SessionLocal()
    try:
        # Base filter
        base_filter = [
            Question.subject == subject,
            Question.question_type == question_type,
            Question.answer != None,
            Question.answer != "",
        ]
        if grade:
            base_filter.append(Question.grade == grade)

        # Count questions with answers, grouped by topic + difficulty
        results = db.query(
            Question.topic,
            Question.difficulty,
            func.count(Question.id).label("count"),
            func.avg(Question.times_used).label("avg_used"),
        ).filter(*base_filter).group_by(Question.topic, Question.difficulty).all()

        available = {}
        for topic, difficulty, count, avg_used in results:
            key = topic or "(chưa phân loại)"
            if key not in available:
                available[key] = {}
            available[key][difficulty or "medium"] = {
                "count": count,
                "avg_used": round(avg_used or 0, 1),
            }

        # Also get total by difficulty (no topic filter)
        totals = db.query(
            Question.difficulty,
            func.count(Question.id).label("count"),
        ).filter(*base_filter).group_by(Question.difficulty).all()

        total_by_diff = {d: c for d, c in totals}

        return {
            "available": available,
            "total_by_difficulty": total_by_diff,
        }

    finally:
        db.close()
