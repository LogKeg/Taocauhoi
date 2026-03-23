"""
API routers for the application.
"""
import importlib.util
import os

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
from .curriculum import router as curriculum_router

# Load kebab-case router files using importlib
_current_dir = os.path.dirname(os.path.abspath(__file__))


def _load_router(filename: str):
    filepath = os.path.join(_current_dir, filename)
    module_name = filename.replace('-', '_').replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.router


system_router = _load_router("system-status-and-cache-endpoints.py")
ai_tools_router = _load_router("ai-tools-generate-explain-classify-translate-improve.py")

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
    "curriculum_router",
    "system_router",
    "ai_tools_router",
]
