"""
Question bank API endpoints.
"""
import importlib.util
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.database import QuestionCRUD, HistoryCRUD
from app.core import QuestionCreate, QuestionUpdate, BulkSaveRequest

# Load cache module
_services_dir = os.path.join(os.path.dirname(__file__), "..", "..", "services")
_cache_path = os.path.join(_services_dir, "database-query-cache.py")
_spec = importlib.util.spec_from_file_location("db_cache", _cache_path)
_db_cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_db_cache)

router = APIRouter(prefix="/api", tags=["questions"])


def _format_question(q) -> dict:
    """Format question for API response."""
    return {
        "id": q.id,
        "content": q.content,
        "options": q.options,
        "answer": q.answer,
        "explanation": q.explanation,
        "subject": q.subject,
        "topic": q.topic,
        "grade": q.grade,
        "question_type": q.question_type,
        "difficulty": q.difficulty,
        "difficulty_score": q.difficulty_score,
        "tags": q.tags,
        "source": q.source,
        "quality_score": q.quality_score,
        "quality_issues": q.quality_issues,
        "times_used": q.times_used,
        "created_at": q.created_at.isoformat() if q.created_at else None,
        "updated_at": q.updated_at.isoformat() if q.updated_at else None,
    }


@router.get("/questions")
def get_questions(
    skip: int = 0,
    limit: int = 100,
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
    question_type: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Get all questions with optional filters"""
    questions = QuestionCRUD.get_all(
        db,
        skip=skip,
        limit=limit,
        subject=subject,
        topic=topic,
        difficulty=difficulty,
        question_type=question_type,
        search=search,
    )
    return {
        "questions": [_format_question(q) for q in questions],
        "total": QuestionCRUD.count(db, subject=subject, topic=topic, difficulty=difficulty),
    }


@router.get("/questions/stats")
def get_questions_stats(db: Session = Depends(get_db)):
    """Get statistics about questions in the bank (cached)"""
    stats = _db_cache.get_cached_subject_stats(db)
    total = sum(s["total"] for s in stats.values())
    return {"stats": stats, "total": total}


@router.get("/questions/{question_id}")
def get_question(question_id: int, db: Session = Depends(get_db)):
    """Get a single question by ID"""
    question = QuestionCRUD.get(db, question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Không tìm thấy câu hỏi")
    return _format_question(question)


@router.post("/questions")
def create_question(data: QuestionCreate, db: Session = Depends(get_db)):
    """Create a new question"""
    question = QuestionCRUD.create(
        db,
        content=data.content,
        options=data.options,
        answer=data.answer,
        explanation=data.explanation,
        subject=data.subject,
        topic=data.topic,
        grade=data.grade,
        question_type=data.question_type,
        difficulty=data.difficulty,
        tags=data.tags,
        source=data.source,
    )
    _db_cache.invalidate_on_question_change()  # Clear cache
    return {"ok": True, "id": question.id, "message": "Đã lưu câu hỏi"}


@router.post("/questions/bulk")
def bulk_create_questions(data: BulkSaveRequest, db: Session = Depends(get_db)):
    """Create multiple questions at once"""
    questions_data = [
        {
            "content": q.content,
            "options": q.options,
            "answer": q.answer,
            "explanation": q.explanation,
            "subject": q.subject,
            "topic": q.topic,
            "grade": q.grade,
            "question_type": q.question_type,
            "difficulty": q.difficulty,
            "tags": q.tags,
            "source": q.source,
        }
        for q in data.questions
    ]
    questions = QuestionCRUD.bulk_create(db, questions_data)
    _db_cache.invalidate_on_question_change()  # Clear cache
    return {"ok": True, "count": len(questions), "message": f"Đã lưu {len(questions)} câu hỏi"}


@router.put("/questions/{question_id}")
def update_question(question_id: int, data: QuestionUpdate, db: Session = Depends(get_db)):
    """Update an existing question"""
    update_data = {k: v for k, v in data.dict().items() if v is not None}
    question = QuestionCRUD.update(db, question_id, **update_data)
    if not question:
        raise HTTPException(status_code=404, detail="Không tìm thấy câu hỏi")
    _db_cache.invalidate_on_question_change()  # Clear cache
    return {"ok": True, "message": "Đã cập nhật câu hỏi"}


@router.delete("/questions/{question_id}")
def delete_question(question_id: int, db: Session = Depends(get_db)):
    """Delete a question"""
    if QuestionCRUD.delete(db, question_id):
        _db_cache.invalidate_on_question_change()  # Clear cache
        return {"ok": True, "message": "Đã xóa câu hỏi"}
    raise HTTPException(status_code=404, detail="Không tìm thấy câu hỏi")


@router.post("/questions/bulk-delete")
def bulk_delete_questions(
    filters: dict = {},
    db: Session = Depends(get_db),
):
    """Delete questions by filter or all if no filter provided."""
    from app.database import Question

    query = db.query(Question)
    subject = filters.get("subject", "")
    difficulty = filters.get("difficulty", "")
    search = filters.get("search", "")

    if subject:
        query = query.filter(Question.subject == subject)
    if difficulty:
        query = query.filter(Question.difficulty == difficulty)
    if search:
        query = query.filter(Question.content.contains(search))

    count = query.count()
    query.delete(synchronize_session=False)
    db.commit()
    _db_cache.invalidate_on_question_change()  # Clear cache

    return {"ok": True, "deleted": count, "message": f"Đã xóa {count} câu hỏi"}


# ============================================================================
# USAGE HISTORY APIs
# ============================================================================

@router.get("/history")
def get_history(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """Get usage history"""
    history_items = HistoryCRUD.get_all(db, skip=skip, limit=limit)
    return {
        "history": [
            {
                "timestamp": h.timestamp,
                "type": h.type,
                "input_count": h.input_count,
                "output_count": h.output_count,
                "settings": h.settings,
            }
            for h in history_items
        ]
    }


@router.post("/history")
def add_history(
    type: str,
    input_count: int = 0,
    output_count: int = 0,
    settings: dict = None,
    db: Session = Depends(get_db),
):
    """Add a new history entry"""
    history = HistoryCRUD.create(
        db,
        type=type,
        input_count=input_count,
        output_count=output_count,
        settings=settings or {},
    )
    return {"ok": True, "timestamp": history.timestamp}


@router.delete("/history/{timestamp}")
def delete_history_item(timestamp: str, db: Session = Depends(get_db)):
    """Delete a history entry"""
    if HistoryCRUD.delete(db, timestamp):
        return {"ok": True, "message": "Đã xóa"}
    raise HTTPException(status_code=404, detail="Không tìm thấy")


@router.delete("/history")
def clear_history(db: Session = Depends(get_db)):
    """Clear all history"""
    HistoryCRUD.clear_all(db)
    return {"ok": True, "message": "Đã xóa tất cả lịch sử"}
