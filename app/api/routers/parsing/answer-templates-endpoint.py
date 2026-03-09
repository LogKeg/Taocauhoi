"""
Answer templates endpoint.

GET /api/answer-templates - Get available answer templates for grading.
"""
from fastapi import APIRouter

from app.core import ANSWER_TEMPLATES

router = APIRouter(tags=["parsing"])


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
