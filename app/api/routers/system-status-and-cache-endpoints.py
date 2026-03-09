"""
System status, cache management, and background task monitoring endpoints.

GET /api/system/status - System health and stats
GET /api/system/cache/stats - Cache statistics
POST /api/system/cache/clear - Clear cache
GET /api/system/tasks - List background tasks
GET /api/system/tasks/{task_id} - Get task status
"""

import importlib.util
import os

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/system", tags=["system"])

# Load services using importlib (kebab-case files)
_services_dir = os.path.join(os.path.dirname(__file__), "..", "..", "services")


def _load_module(filename: str):
    filepath = os.path.join(_services_dir, filename)
    module_name = filename.replace('-', '_').replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cache_module = _load_module("in-memory-cache-with-ttl.py")
_task_module = _load_module("background-task-manager.py")


@router.get("/status")
async def get_system_status():
    """Get system health and statistics."""
    from app.database import SessionLocal, Question

    db = SessionLocal()
    try:
        question_count = db.query(Question).count()
    finally:
        db.close()

    return {
        "status": "healthy",
        "database": {
            "questions": question_count,
        },
        "cache": _cache_module.get_cache_stats(),
        "tasks": {
            "total": len(_task_module.get_all_tasks()),
        },
    }


@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    return _cache_module.get_cache_stats()


@router.post("/cache/clear")
async def clear_cache(prefix: str = None):
    """Clear cache entries. Optionally filter by prefix."""
    count = _cache_module.clear_cache(prefix)
    return {"cleared": count, "prefix": prefix}


@router.get("/tasks")
async def list_tasks():
    """List all background tasks."""
    return {"tasks": _task_module.get_all_tasks()}


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a specific task."""
    task = _task_module.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.to_dict()
