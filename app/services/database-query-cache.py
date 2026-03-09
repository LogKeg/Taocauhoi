"""
Database query caching layer.
Caches frequently accessed questions and statistics.
"""

import importlib.util
import os
from typing import List, Optional

# Load cache module
_current_dir = os.path.dirname(os.path.abspath(__file__))
_cache_path = os.path.join(_current_dir, "in-memory-cache-with-ttl.py")
_spec = importlib.util.spec_from_file_location("cache", _cache_path)
_cache_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cache_module)

get_cache = _cache_module.get_cache
set_cache = _cache_module.set_cache
delete_cache = _cache_module.delete_cache
clear_cache = _cache_module.clear_cache


# Cache TTLs
STATS_TTL = 600  # 10 minutes for statistics
QUESTIONS_TTL = 300  # 5 minutes for question lists
SINGLE_QUESTION_TTL = 60  # 1 minute for single question


def get_cached_subject_stats(db_session) -> dict:
    """Get question statistics grouped by subject (cached)."""
    cache_key = "db:subject_stats"
    cached = get_cache(cache_key)
    if cached is not None:
        return cached

    from app.database import QuestionCRUD
    stats = QuestionCRUD.get_by_subject_stats(db_session)
    set_cache(cache_key, stats, STATS_TTL)
    return stats


def get_cached_questions(
    db_session,
    skip: int = 0,
    limit: int = 100,
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> List:
    """Get questions with caching."""
    cache_key = f"db:questions:{subject}:{topic}:{difficulty}:{skip}:{limit}"
    cached = get_cache(cache_key)
    if cached is not None:
        return cached

    from app.database import QuestionCRUD
    questions = QuestionCRUD.get_all(
        db_session,
        skip=skip,
        limit=limit,
        subject=subject,
        topic=topic,
        difficulty=difficulty,
    )
    set_cache(cache_key, questions, QUESTIONS_TTL)
    return questions


def get_cached_question_count(
    db_session,
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> int:
    """Get question count with caching."""
    cache_key = f"db:count:{subject}:{topic}:{difficulty}"
    cached = get_cache(cache_key)
    if cached is not None:
        return cached

    from app.database import QuestionCRUD
    count = QuestionCRUD.count(db_session, subject=subject, topic=topic, difficulty=difficulty)
    set_cache(cache_key, count, STATS_TTL)
    return count


def invalidate_question_cache():
    """Invalidate all question-related cache when data changes."""
    clear_cache("db:")


def invalidate_on_question_change():
    """Call this after create/update/delete operations."""
    invalidate_question_cache()
