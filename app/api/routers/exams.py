"""
Exam management API endpoints.
"""
import json
import random
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.database import ExamCRUD, Question
from app.core import ExamCreate, ExamUpdate

router = APIRouter(prefix="/api", tags=["exams"])


@router.get("/exams")
def get_exams(
    skip: int = 0,
    limit: int = 100,
    subject: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Get all exams with optional filters"""
    exams = ExamCRUD.get_all(db, skip=skip, limit=limit, subject=subject)
    return {
        "exams": [
            {
                "id": e.id,
                "title": e.title,
                "description": e.description,
                "subject": e.subject,
                "grade": e.grade,
                "total_questions": e.total_questions,
                "duration_minutes": e.duration_minutes,
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "variants_count": len(e.variants),
            }
            for e in exams
        ]
    }


@router.get("/exams/{exam_id}")
def get_exam(exam_id: int, db: Session = Depends(get_db)):
    """Get a single exam with its questions"""
    exam = ExamCRUD.get(db, exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Không tìm thấy đề thi")

    questions = []
    for eq in sorted(exam.exam_questions, key=lambda x: x.order):
        q = eq.question
        questions.append({
            "id": q.id,
            "content": q.content,
            "options": q.options,
            "answer": q.answer,
            "order": eq.order,
            "points": eq.points,
            "difficulty": q.difficulty,
        })

    return {
        "id": exam.id,
        "title": exam.title,
        "description": exam.description,
        "subject": exam.subject,
        "grade": exam.grade,
        "total_questions": exam.total_questions,
        "duration_minutes": exam.duration_minutes,
        "created_at": exam.created_at.isoformat() if exam.created_at else None,
        "questions": questions,
        "variants": [
            {"id": v.id, "variant_code": v.variant_code, "created_at": v.created_at.isoformat()}
            for v in exam.variants
        ],
    }


@router.post("/exams")
def create_exam(data: ExamCreate, db: Session = Depends(get_db)):
    """Create a new exam"""
    exam = ExamCRUD.create(
        db,
        title=data.title,
        description=data.description,
        subject=data.subject,
        grade=data.grade,
        duration_minutes=data.duration_minutes,
    )
    return {"ok": True, "id": exam.id, "message": "Đã tạo đề thi"}


@router.put("/exams/{exam_id}")
def update_exam(exam_id: int, data: ExamUpdate, db: Session = Depends(get_db)):
    """Update an exam"""
    update_data = {k: v for k, v in data.dict().items() if v is not None}
    exam = ExamCRUD.update(db, exam_id, **update_data)
    if not exam:
        raise HTTPException(status_code=404, detail="Không tìm thấy đề thi")
    return {"ok": True, "message": "Đã cập nhật đề thi"}


@router.delete("/exams/{exam_id}")
def delete_exam(exam_id: int, db: Session = Depends(get_db)):
    """Delete an exam"""
    if ExamCRUD.delete(db, exam_id):
        return {"ok": True, "message": "Đã xóa đề thi"}
    raise HTTPException(status_code=404, detail="Không tìm thấy đề thi")


@router.post("/exams/{exam_id}/questions")
def add_questions_to_exam(exam_id: int, question_ids: List[int], db: Session = Depends(get_db)):
    """Add questions to an exam"""
    exam = ExamCRUD.add_questions(db, exam_id, question_ids)
    if not exam:
        raise HTTPException(status_code=404, detail="Không tìm thấy đề thi")
    return {"ok": True, "message": f"Đã thêm {len(question_ids)} câu hỏi"}


@router.delete("/exams/{exam_id}/questions/{question_id}")
def remove_question_from_exam(exam_id: int, question_id: int, db: Session = Depends(get_db)):
    """Remove a question from an exam"""
    if ExamCRUD.remove_question(db, exam_id, question_id):
        return {"ok": True, "message": "Đã xóa câu hỏi khỏi đề"}
    raise HTTPException(status_code=404, detail="Không tìm thấy câu hỏi trong đề")


@router.post("/exams/{exam_id}/variants")
def create_exam_variant(exam_id: int, variant_code: str, db: Session = Depends(get_db)):
    """Create a shuffled variant of an exam"""
    exam = ExamCRUD.get(db, exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Không tìm thấy đề thi")

    # Get question IDs
    question_ids = [eq.question_id for eq in exam.exam_questions]
    if not question_ids:
        raise HTTPException(status_code=400, detail="Đề thi chưa có câu hỏi")

    # Shuffle questions
    shuffled_ids = question_ids.copy()
    random.shuffle(shuffled_ids)

    # Create answer mapping for MCQ
    answer_mapping = {}
    for qid in shuffled_ids:
        q = db.query(Question).filter(Question.id == qid).first()
        if q and q.options:
            try:
                opts = json.loads(q.options)
                if isinstance(opts, list) and len(opts) >= 2:
                    shuffled_opts = opts.copy()
                    random.shuffle(shuffled_opts)
                    # Map original answer to new position
                    if q.answer in opts:
                        old_idx = opts.index(q.answer)
                        new_idx = shuffled_opts.index(q.answer)
                        answer_mapping[str(qid)] = {
                            "original": chr(65 + old_idx),
                            "shuffled": chr(65 + new_idx),
                            "shuffled_options": shuffled_opts,
                        }
            except json.JSONDecodeError:
                pass

    variant = ExamCRUD.create_variant(
        db,
        exam_id=exam_id,
        variant_code=variant_code,
        question_order=json.dumps(shuffled_ids),
        answer_mapping=json.dumps(answer_mapping) if answer_mapping else None,
    )

    return {
        "ok": True,
        "variant_id": variant.id,
        "variant_code": variant_code,
        "message": f"Đã tạo phiên bản {variant_code}",
    }
