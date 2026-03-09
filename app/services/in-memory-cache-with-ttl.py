"""
Simple in-memory cache for database queries and parsed content.
Uses TTL (time-to-live) to auto-expire cached data.
"""

import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

# Simple in-memory cache with TTL
_cache: Dict[str, Tuple[Any, float]] = {}
DEFAULT_TTL = 300  # 5 minutes


def get_cache(key: str) -> Optional[Any]:
    """Get cached value if not expired."""
    if key in _cache:
        value, expires_at = _cache[key]
        if time.time() < expires_at:
            return value
        # Expired, remove it
        del _cache[key]
    return None


def set_cache(key: str, value: Any, ttl: int = DEFAULT_TTL) -> None:
    """Set cached value with TTL in seconds."""
    _cache[key] = (value, time.time() + ttl)


def delete_cache(key: str) -> bool:
    """Delete cached value. Returns True if existed."""
    if key in _cache:
        del _cache[key]
        return True
    return False


def clear_cache(prefix: str = None) -> int:
    """Clear cache entries. If prefix given, only clear matching keys."""
    global _cache
    if prefix is None:
        count = len(_cache)
        _cache = {}
        return count

    keys_to_delete = [k for k in _cache if k.startswith(prefix)]
    for k in keys_to_delete:
        del _cache[k]
    return len(keys_to_delete)


def cached(ttl: int = DEFAULT_TTL, key_prefix: str = ""):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Build cache key from function name and arguments
            cache_key = f"{key_prefix}{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"

            # Check cache
            cached_value = get_cache(cache_key)
            if cached_value is not None:
                return cached_value

            # Call function and cache result
            result = func(*args, **kwargs)
            set_cache(cache_key, result, ttl)
            return result
        return wrapper
    return decorator


def get_cache_stats() -> dict:
    """Get cache statistics."""
    now = time.time()
    total = len(_cache)
    expired = sum(1 for _, (_, exp) in _cache.items() if exp <= now)
    return {
        "total_entries": total,
        "active_entries": total - expired,
        "expired_entries": expired,
    }
