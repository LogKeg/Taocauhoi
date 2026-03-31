"""
Question bank API endpoints.
"""
import importlib.util
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.database import QuestionCRUD, HistoryCRUD
from app.database.models import Question
from app.core import QuestionCreate, QuestionUpdate, BulkSaveRequest
from app.services.image import save_question_image, download_image_sync, delete_question_images

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
        "image_url": q.image_url,
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
    grade: Optional[str] = None,
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
    question_type: Optional[str] = None,
    search: Optional[str] = None,
    id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Get all questions with optional filters. Use id param to search by exact ID."""
    # If searching by ID, return just that question
    if id is not None:
        question = QuestionCRUD.get(db, id)
        if question:
            return {"questions": [_format_question(question)], "total": 1}
        return {"questions": [], "total": 0}

    questions = QuestionCRUD.get_all(
        db,
        skip=skip,
        limit=limit,
        subject=subject,
        grade=grade,
        topic=topic,
        difficulty=difficulty,
        question_type=question_type,
        search=search,
    )
    return {
        "questions": [_format_question(q) for q in questions],
        "total": QuestionCRUD.count(db, subject=subject, grade=grade, topic=topic, difficulty=difficulty),
    }


@router.get("/questions/stats")
def get_questions_stats(db: Session = Depends(get_db)):
    """Get statistics about questions in the bank (cached)"""
    stats = _db_cache.get_cached_subject_stats(db)
    total = sum(s["total"] for s in stats.values())
    return {"stats": stats, "total": total}


@router.post("/questions/clear-cache")
def clear_questions_cache():
    """Clear all question-related cache."""
    _db_cache.invalidate_on_question_change()
    return {"ok": True, "message": "Cache cleared"}


@router.get("/questions/dashboard")
def get_dashboard_stats(db: Session = Depends(get_db)):
    """Comprehensive dashboard statistics for question bank."""
    from sqlalchemy import func, case, or_

    Q = Question

    # Total counts
    total = db.query(func.count(Q.id)).scalar()
    with_answer = db.query(func.count(Q.id)).filter(Q.answer != None, Q.answer != "").scalar()
    with_explanation = db.query(func.count(Q.id)).filter(Q.explanation != None, Q.explanation != "").scalar()
    with_image = db.query(func.count(Q.id)).filter(Q.image_url != None, Q.image_url != "").scalar()
    total_used = db.query(func.count(Q.id)).filter(Q.times_used > 0).scalar()

    # By subject
    by_subject = db.query(
        Q.subject,
        func.count(Q.id).label("total"),
        func.sum(case((Q.answer != None, 1), else_=0)).label("has_answer"),
        func.sum(Q.times_used).label("total_used"),
    ).group_by(Q.subject).order_by(func.count(Q.id).desc()).all()

    subject_data = [
        {"subject": s, "total": t, "has_answer": int(a or 0), "total_used": int(u or 0)}
        for s, t, a, u in by_subject
    ]

    # By difficulty
    by_difficulty = db.query(
        Q.difficulty, func.count(Q.id)
    ).group_by(Q.difficulty).all()
    difficulty_data = {d: c for d, c in by_difficulty}

    # By grade
    by_grade = db.query(
        Q.grade, func.count(Q.id)
    ).filter(Q.grade != None, Q.grade != "").group_by(Q.grade).order_by(func.count(Q.id).desc()).all()
    grade_data = [{"grade": g, "count": c} for g, c in by_grade]

    # By source (top 10)
    by_source = db.query(
        Q.source, func.count(Q.id)
    ).filter(Q.source != None, Q.source != "").group_by(Q.source).order_by(func.count(Q.id).desc()).limit(10).all()
    source_data = [{"source": s, "count": c} for s, c in by_source]

    # By question type
    by_type = db.query(
        Q.question_type, func.count(Q.id)
    ).group_by(Q.question_type).all()
    type_data = {t: c for t, c in by_type}

    # Recent activity (questions added per day, last 14 days)
    from datetime import datetime, timedelta
    fourteen_days_ago = datetime.utcnow() - timedelta(days=14)
    daily = db.query(
        func.date(Q.created_at).label("date"),
        func.count(Q.id).label("count"),
    ).filter(Q.created_at >= fourteen_days_ago).group_by(func.date(Q.created_at)).order_by("date").all()
    daily_data = [{"date": str(d), "count": c} for d, c in daily]

    return {
        "overview": {
            "total": total,
            "with_answer": with_answer,
            "without_answer": total - with_answer,
            "with_explanation": with_explanation,
            "with_image": with_image,
            "total_used": total_used,
            "answer_rate": round(with_answer / total * 100, 1) if total else 0,
        },
        "by_subject": subject_data,
        "by_difficulty": difficulty_data,
        "by_grade": grade_data,
        "by_source": source_data,
        "by_type": type_data,
        "daily_activity": daily_data,
    }


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
    """Create multiple questions at once, with optional image download"""
    questions_data = []
    image_downloads = []  # Store (index, image_source_url) for later download

    for i, q in enumerate(data.questions):
        q_data = {
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
            "image_url": q.image_url,  # May already have local path
        }
        questions_data.append(q_data)

        # Track external image URLs for download after questions are created
        if q.image_source_url and not q.image_url:
            image_downloads.append((i, q.image_source_url))

    # Create questions first to get their IDs
    questions = QuestionCRUD.bulk_create(db, questions_data)

    # Download and save images for questions that have external URLs
    images_downloaded = 0
    for idx, image_source_url in image_downloads:
        if idx < len(questions):
            question = questions[idx]
            try:
                image_bytes = download_image_sync(image_source_url)
                if image_bytes:
                    # Extract filename from URL
                    filename = image_source_url.split('/')[-1].split('?')[0] or "image.png"
                    image_url = save_question_image(question.id, image_bytes, filename)
                    if image_url:
                        # Update question with image_url
                        QuestionCRUD.update(db, question.id, image_url=image_url)
                        images_downloaded += 1
            except Exception as e:
                print(f"Failed to download image for question {question.id}: {e}")

    _db_cache.invalidate_on_question_change()  # Clear cache

    msg = f"Đã lưu {len(questions)} câu hỏi"
    if images_downloaded > 0:
        msg += f" ({images_downloaded} hình ảnh)"
    return {"ok": True, "count": len(questions), "images": images_downloaded, "message": msg}


@router.post("/questions/{question_id}/image")
async def upload_question_image(
    question_id: int,
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload an image for a question"""
    # Verify question exists
    question = QuestionCRUD.get(db, question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Không tìm thấy câu hỏi")

    # Validate file type
    allowed_types = {"image/png", "image/jpeg", "image/gif", "image/webp"}
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Loại file không hỗ trợ. Chấp nhận: PNG, JPEG, GIF, WebP"
        )

    # Read and save image
    image_bytes = await image.read()
    image_url = save_question_image(question_id, image_bytes, image.filename or "image.png")

    if not image_url:
        raise HTTPException(status_code=500, detail="Không thể lưu hình ảnh")

    # Update question with image_url
    QuestionCRUD.update(db, question_id, image_url=image_url)
    _db_cache.invalidate_on_question_change()

    return {"ok": True, "image_url": image_url, "message": "Đã lưu hình ảnh"}


@router.delete("/questions/{question_id}/image")
def delete_question_image(question_id: int, db: Session = Depends(get_db)):
    """Delete the image for a question"""
    question = QuestionCRUD.get(db, question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Không tìm thấy câu hỏi")

    delete_question_images(question_id)
    QuestionCRUD.update(db, question_id, image_url=None)
    _db_cache.invalidate_on_question_change()

    return {"ok": True, "message": "Đã xóa hình ảnh"}


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
    """Delete a question and its associated images"""
    if QuestionCRUD.delete(db, question_id):
        # Also delete associated images
        delete_question_images(question_id)
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
    grade = filters.get("grade", "")
    difficulty = filters.get("difficulty", "")
    search = filters.get("search", "")

    if subject:
        query = query.filter(Question.subject == subject)
    if grade:
        query = query.filter(Question.grade == grade)
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
