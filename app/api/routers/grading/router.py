"""
Main grading router that includes all sub-routers.

Aggregates endpoints from:
- omr-sheet-grading-endpoints.py
- handwritten-grading-endpoints.py
"""
import importlib.util
import os

from fastapi import APIRouter

router = APIRouter()

# Load sub-routers using importlib (for kebab-case filenames)
_current_dir = os.path.dirname(__file__)


def _load_router(filename: str) -> APIRouter:
    """Load router from a kebab-case Python file."""
    filepath = os.path.join(_current_dir, filename)
    module_name = filename.replace('-', '_').replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.router


# Include all sub-routers
router.include_router(_load_router("omr-sheet-grading-endpoints.py"))
router.include_router(_load_router("handwritten-grading-endpoints.py"))
