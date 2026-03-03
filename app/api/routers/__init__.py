"""
API routers for the application.
"""
from .settings import router as settings_router
from .questions import router as questions_router
from .exams import router as exams_router
from .import_export import router as import_export_router
from .ai_features import router as ai_features_router
from .generation import router as generation_router
from .grading import router as grading_router
from .parsing import router as parsing_router
from .crawler import router as crawler_router
from .storage import router as storage_router

__all__ = [
    "settings_router",
    "questions_router",
    "exams_router",
    "import_export_router",
    "ai_features_router",
    "generation_router",
    "grading_router",
    "parsing_router",
    "crawler_router",
    "storage_router",
]
