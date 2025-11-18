"""Caching utilities for PeakFit.

This module provides caching mechanisms to avoid redundant computations
in fitting operations.
"""

import functools
import hashlib
from typing import Any

import numpy as np


def hash_array(arr: np.ndarray) -> str:
    """Create a hash for a numpy array.

    Args:
        arr: Numpy array to hash

    Returns:
        Hexadecimal hash string
    """
    return hashlib.md5(arr.tobytes(), usedforsecurity=False).hexdigest()


def make_cache_key(*args: Any) -> str:
    """Create a cache key from arguments.

    Args:
        *args: Arguments to hash

    Returns:
        Cache key string
    """
    key_parts = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            key_parts.append(f"arr:{hash_array(arg)}")
        elif isinstance(arg, (list, tuple)):
            key_parts.append(f"seq:{len(arg)}:{sum(hash(x) for x in arg)}")
        else:
            key_parts.append(str(arg))
    return ":".join(key_parts)


class LRUCache:
    """Simple LRU cache with size limit.

    This cache evicts least recently used items when the maximum size is reached.
    """

    def __init__(self, maxsize: int = 128) -> None:
        """Initialize cache.

        Args:
            maxsize: Maximum number of items to cache
        """
        self.maxsize = maxsize
        self._cache: dict[str, Any] = {}
        self._order: list[str] = []
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if key in self._cache:
            # Move to end (most recently used)
            self._order.remove(key)
            self._order.append(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self._cache:
            # Update existing
            self._order.remove(key)
            self._order.append(key)
            self._cache[key] = value
        else:
            # Add new
            if len(self._cache) >= self.maxsize:
                # Evict oldest
                oldest = self._order.pop(0)
                del self._cache[oldest]
            self._cache[key] = value
            self._order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._order.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        total = self._hits + self._misses
        return (self._hits / total * 100) if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics."""
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


# Global shape cache
_shape_cache = LRUCache(maxsize=256)


def cached_shape_evaluation(
    peak_name: str,
    positions: tuple[np.ndarray, ...],
    param_values: tuple[float, ...],
    evaluate_func: Any,
) -> np.ndarray:
    """Cache shape evaluations.

    Args:
        peak_name: Peak identifier
        positions: Grid positions
        param_values: Parameter values
        evaluate_func: Function to compute shape

    Returns:
        Evaluated shape array
    """
    # Create cache key
    pos_hash = ":".join(hash_array(p) for p in positions)
    param_str = ":".join(f"{v:.8f}" for v in param_values)
    key = f"{peak_name}:{pos_hash}:{param_str}"

    # Check cache
    cached = _shape_cache.get(key)
    if cached is not None:
        return cached

    # Compute and cache
    result = evaluate_func()
    _shape_cache.put(key, result)
    return result


def clear_shape_cache() -> None:
    """Clear the global shape cache."""
    _shape_cache.clear()


def get_cache_stats() -> dict[str, Any]:
    """Get global cache statistics.

    Returns:
        Dictionary with cache statistics
    """
    return _shape_cache.stats


def memoize_array_function(maxsize: int = 128):
    """Decorator to memoize functions with numpy array arguments.

    This decorator handles numpy arrays by hashing their contents.

    Args:
        maxsize: Maximum cache size

    Returns:
        Decorator function
    """

    def decorator(func):
        cache = LRUCache(maxsize=maxsize)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create key from args and kwargs
            key_parts = [func.__name__]

            for arg in args:
                if isinstance(arg, np.ndarray):
                    key_parts.append(hash_array(arg))
                else:
                    key_parts.append(str(arg))

            for k, v in sorted(kwargs.items()):
                if isinstance(v, np.ndarray):
                    key_parts.append(f"{k}={hash_array(v)}")
                else:
                    key_parts.append(f"{k}={v}")

            key = ":".join(key_parts)

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result

        # Attach cache for inspection
        wrapper.cache = cache
        return wrapper

    return decorator


@memoize_array_function(maxsize=64)
def cached_lstsq(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Cached least squares solution.

    Args:
        a: Left-hand side matrix
        b: Right-hand side vector/matrix

    Returns:
        Least squares solution
    """
    return np.linalg.lstsq(a, b, rcond=None)[0]
