"""
Background task manager for async AI operations.
Allows non-blocking AI generation with status tracking.
"""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskInfo:
    """Information about a background task."""
    def __init__(self, task_id: str, task_type: str):
        self.task_id = task_id
        self.task_type = task_type
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.result: Any = None
        self.error: Optional[str] = None
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# Task storage (in-memory, could be Redis in production)
_tasks: Dict[str, TaskInfo] = {}
MAX_STORED_TASKS = 100


def _cleanup_old_tasks():
    """Remove old completed tasks to prevent memory leak."""
    if len(_tasks) > MAX_STORED_TASKS:
        # Sort by created_at, keep newest
        sorted_tasks = sorted(_tasks.items(), key=lambda x: x[1].created_at, reverse=True)
        for task_id, _ in sorted_tasks[MAX_STORED_TASKS:]:
            del _tasks[task_id]


def create_task(task_type: str) -> TaskInfo:
    """Create a new task and return its info."""
    task_id = str(uuid.uuid4())[:8]
    task = TaskInfo(task_id, task_type)
    _tasks[task_id] = task
    _cleanup_old_tasks()
    return task


def get_task(task_id: str) -> Optional[TaskInfo]:
    """Get task info by ID."""
    return _tasks.get(task_id)


def get_all_tasks() -> list:
    """Get all tasks as list of dicts."""
    return [t.to_dict() for t in _tasks.values()]


async def run_task_async(
    task: TaskInfo,
    func: Callable,
    *args,
    **kwargs
) -> None:
    """Run a function as a background task."""
    task.status = TaskStatus.RUNNING
    task.started_at = datetime.utcnow()

    try:
        # If func is async, await it; otherwise run in executor
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))

        task.result = result
        task.status = TaskStatus.COMPLETED
        task.progress = 100
    except Exception as e:
        task.error = str(e)
        task.status = TaskStatus.FAILED
    finally:
        task.completed_at = datetime.utcnow()


def update_task_progress(task_id: str, progress: int) -> None:
    """Update task progress (0-100)."""
    task = _tasks.get(task_id)
    if task:
        task.progress = min(100, max(0, progress))
