"""
API endpoints for Curriculum management.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional, List

from app.database import get_db, CurriculumCRUD, Curriculum
from app.services.curriculum import get_sample_curriculum, get_curriculum_urls

router = APIRouter(prefix="/api/curriculum", tags=["curriculum"])


@router.get("/")
def list_curriculum(
    subject: Optional[str] = Query(None, description="Filter by subject (toan, ly, hoa, etc.)"),
    grade: Optional[int] = Query(None, description="Filter by grade (10, 11, 12)"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Get list of curriculum items with optional filters."""
    items = CurriculumCRUD.get_all(db, skip=skip, limit=limit, subject=subject, grade=grade)
    return {
        "ok": True,
        "items": [
            {
                "id": item.id,
                "subject": item.subject,
                "grade": item.grade,
                "chapter": item.chapter,
                "topic": item.topic,
                "lesson": item.lesson,
                "knowledge": item.knowledge,
                "skills": item.skills,
                "periods": item.periods,
            }
            for item in items
        ],
        "total": CurriculumCRUD.count(db, subject=subject, grade=grade),
    }


@router.get("/stats")
def curriculum_stats(db: Session = Depends(get_db)):
    """Get curriculum statistics grouped by subject and grade."""
    stats = CurriculumCRUD.get_stats(db)
    return {"ok": True, "stats": stats}


@router.get("/subjects")
def list_subjects():
    """Get list of available subjects with their curriculum URLs."""
    urls = get_curriculum_urls()
    return {
        "ok": True,
        "subjects": [
            {
                "key": key,
                "name": data["name"],
                "description": data.get("description", ""),
                "has_program": bool(data.get("program_2018")),
            }
            for key, data in urls.items()
        ],
    }


@router.get("/chapters/{subject}/{grade}")
def get_chapters(
    subject: str,
    grade: int,
    db: Session = Depends(get_db),
):
    """Get chapters for a specific subject and grade."""
    chapters = CurriculumCRUD.get_chapters(db, subject, grade)
    return {"ok": True, "chapters": chapters}


@router.get("/topics/{subject}/{grade}")
def get_topics(
    subject: str,
    grade: int,
    db: Session = Depends(get_db),
):
    """Get topics for a specific subject and grade."""
    topics = CurriculumCRUD.get_topics(db, subject, grade)
    return {"ok": True, "topics": topics}


@router.get("/search")
def search_curriculum(
    keyword: str = Query(..., min_length=2, description="Search keyword"),
    subject: Optional[str] = Query(None),
    grade: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """Search curriculum by keyword."""
    results = CurriculumCRUD.search(db, keyword, subject, grade)
    return {
        "ok": True,
        "results": [
            {
                "id": item.id,
                "subject": item.subject,
                "grade": item.grade,
                "chapter": item.chapter,
                "topic": item.topic,
                "lesson": item.lesson,
                "knowledge": item.knowledge,
                "skills": item.skills,
            }
            for item in results
        ],
    }


@router.post("/import-sample")
def import_sample_curriculum(
    subject: Optional[str] = Query(None, description="Import specific subject only"),
    grade: Optional[int] = Query(None, description="Import specific grade only"),
    db: Session = Depends(get_db),
):
    """Import sample curriculum data into database."""
    # Get sample data
    items = get_sample_curriculum(subject=subject, grade=grade)

    if not items:
        return {"ok": False, "error": "No sample data available for the specified filters"}

    # Clear existing data for the subject/grade if specified
    if subject:
        CurriculumCRUD.delete_by_subject(db, subject)

    # Import new data
    created = CurriculumCRUD.bulk_create(db, items)

    return {
        "ok": True,
        "message": f"Imported {len(created)} curriculum items",
        "count": len(created),
    }


@router.delete("/clear")
def clear_curriculum(
    subject: Optional[str] = Query(None, description="Clear specific subject only"),
    db: Session = Depends(get_db),
):
    """Clear curriculum data from database."""
    if subject:
        count = CurriculumCRUD.delete_by_subject(db, subject)
        return {"ok": True, "message": f"Cleared {count} items for subject: {subject}"}
    else:
        count = CurriculumCRUD.delete_all(db)
        return {"ok": True, "message": f"Cleared all {count} curriculum items"}


@router.get("/for-ai/{subject}/{grade}")
def get_curriculum_for_ai(
    subject: str,
    grade: int,
    db: Session = Depends(get_db),
):
    """
    Get curriculum content formatted for AI context.
    This is used to provide curriculum reference to AI chat.
    """
    items = CurriculumCRUD.get_by_subject_grade(db, subject, grade)

    if not items:
        return {"ok": True, "content": "", "count": 0}

    # Format for AI context
    content_parts = []
    current_chapter = ""

    for item in items:
        if item.chapter != current_chapter:
            current_chapter = item.chapter
            content_parts.append(f"\n## {current_chapter}")

        if item.topic:
            content_parts.append(f"- **Chủ đề:** {item.topic}")
        if item.lesson:
            content_parts.append(f"  - Bài: {item.lesson}")
        if item.knowledge:
            content_parts.append(f"  - Kiến thức: {item.knowledge}")
        if item.skills:
            content_parts.append(f"  - Kỹ năng: {item.skills}")

    content = "\n".join(content_parts)

    return {
        "ok": True,
        "content": content,
        "count": len(items),
        "subject": subject,
        "grade": grade,
    }
