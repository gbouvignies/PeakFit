"""Test caching utilities."""

import numpy as np
import pytest


class TestHashArray:
    """Tests for array hashing."""

    def test_hash_same_array(self):
        """Same array should produce same hash."""
        from peakfit.core.caching import hash_array

        arr = np.array([1.0, 2.0, 3.0])
        assert hash_array(arr) == hash_array(arr)

    def test_hash_identical_arrays(self):
        """Identical arrays should have same hash."""
        from peakfit.core.caching import hash_array

        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])
        assert hash_array(arr1) == hash_array(arr2)

    def test_hash_different_arrays(self):
        """Different arrays should have different hashes."""
        from peakfit.core.caching import hash_array

        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 4.0])
        assert hash_array(arr1) != hash_array(arr2)

    def test_hash_multidimensional(self):
        """Should handle multidimensional arrays."""
        from peakfit.core.caching import hash_array

        arr = np.array([[1, 2], [3, 4]])
        h = hash_array(arr)
        assert isinstance(h, str)
        assert len(h) > 0


class TestMakeCacheKey:
    """Tests for cache key generation."""

    def test_key_from_scalars(self):
        """Should create key from scalar values."""
        from peakfit.core.caching import make_cache_key

        key = make_cache_key(1.0, 2.0, "test")
        assert "1.0" in key
        assert "2.0" in key
        assert "test" in key

    def test_key_from_array(self):
        """Should create key from array."""
        from peakfit.core.caching import make_cache_key

        arr = np.array([1.0, 2.0])
        key = make_cache_key(arr)
        assert "arr:" in key

    def test_key_consistency(self):
        """Same arguments should produce same key."""
        from peakfit.core.caching import make_cache_key

        arr = np.array([1.0, 2.0])
        key1 = make_cache_key(arr, 3.0)
        key2 = make_cache_key(arr, 3.0)
        assert key1 == key2


class TestLRUCache:
    """Tests for LRU cache."""

    def test_basic_get_put(self):
        """Should store and retrieve values."""
        from peakfit.core.caching import LRUCache

        cache = LRUCache(maxsize=10)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_miss_returns_none(self):
        """Should return None for missing keys."""
        from peakfit.core.caching import LRUCache

        cache = LRUCache(maxsize=10)
        assert cache.get("nonexistent") is None

    def test_eviction_on_full(self):
        """Should evict oldest item when full."""
        from peakfit.core.caching import LRUCache

        cache = LRUCache(maxsize=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)  # Should evict "a"

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_access_refreshes_position(self):
        """Accessing item should move it to end."""
        from peakfit.core.caching import LRUCache

        cache = LRUCache(maxsize=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        # Access "a" to refresh it
        cache.get("a")

        # Adding new item should evict "b" (now oldest)
        cache.put("d", 4)

        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_update_existing_key(self):
        """Should update value for existing key."""
        from peakfit.core.caching import LRUCache

        cache = LRUCache(maxsize=10)
        cache.put("key", "old")
        cache.put("key", "new")
        assert cache.get("key") == "new"

    def test_clear_cache(self):
        """Should clear all items."""
        from peakfit.core.caching import LRUCache

        cache = LRUCache(maxsize=10)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()

        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.stats["size"] == 0

    def test_hit_rate_tracking(self):
        """Should track hit rate."""
        from peakfit.core.caching import LRUCache

        cache = LRUCache(maxsize=10)
        cache.put("a", 1)

        cache.get("a")  # Hit
        cache.get("a")  # Hit
        cache.get("b")  # Miss

        stats = cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert pytest.approx(stats["hit_rate"], rel=0.1) == 66.67

    def test_stats(self):
        """Should provide cache statistics."""
        from peakfit.core.caching import LRUCache

        cache = LRUCache(maxsize=100)
        cache.put("a", 1)
        cache.put("b", 2)

        stats = cache.stats
        assert stats["size"] == 2
        assert stats["maxsize"] == 100
        assert "hits" in stats
        assert "misses" in stats


class TestMemoizeArrayFunction:
    """Tests for memoization decorator."""

    def test_caches_results(self):
        """Should cache function results."""
        from peakfit.core.caching import memoize_array_function

        call_count = 0

        @memoize_array_function(maxsize=10)
        def expensive_func(arr):
            nonlocal call_count
            call_count += 1
            return np.sum(arr)

        arr = np.array([1, 2, 3])

        result1 = expensive_func(arr)
        result2 = expensive_func(arr)

        assert result1 == result2
        assert call_count == 1  # Only computed once

    def test_different_args_not_cached(self):
        """Different args should compute separately."""
        from peakfit.core.caching import memoize_array_function

        call_count = 0

        @memoize_array_function(maxsize=10)
        def expensive_func(arr):
            nonlocal call_count
            call_count += 1
            return np.sum(arr)

        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])

        expensive_func(arr1)
        expensive_func(arr2)

        assert call_count == 2

    def test_handles_kwargs(self):
        """Should handle keyword arguments."""
        from peakfit.core.caching import memoize_array_function

        call_count = 0

        @memoize_array_function(maxsize=10)
        def func_with_kwargs(arr, factor=1.0):
            nonlocal call_count
            call_count += 1
            return arr * factor

        arr = np.array([1, 2, 3])

        func_with_kwargs(arr, factor=2.0)
        func_with_kwargs(arr, factor=2.0)
        func_with_kwargs(arr, factor=3.0)

        assert call_count == 2

    def test_cache_accessible(self):
        """Should expose cache for inspection."""
        from peakfit.core.caching import memoize_array_function

        @memoize_array_function(maxsize=10)
        def func(x):
            return x * 2

        arr = np.array([1, 2])
        func(arr)

        assert hasattr(func, "cache")
        assert func.cache.stats["size"] == 1


class TestCachedLstsq:
    """Tests for cached least squares."""

    def test_basic_lstsq(self):
        """Should compute least squares correctly."""
        from peakfit.core.caching import cached_lstsq

        a = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        b = np.array([1, 2, 3], dtype=float)

        result = cached_lstsq(a, b)
        expected = np.linalg.lstsq(a, b, rcond=None)[0]

        np.testing.assert_allclose(result, expected)

    def test_caching_works(self):
        """Should cache results."""
        from peakfit.core.caching import cached_lstsq

        a = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([1, 2], dtype=float)

        # Clear cache first
        cached_lstsq.cache.clear()

        result1 = cached_lstsq(a, b)
        result2 = cached_lstsq(a, b)

        np.testing.assert_array_equal(result1, result2)
        # Should have 1 miss (first call) and 1 hit (second call)
        assert cached_lstsq.cache.stats["hits"] >= 1


class TestGlobalShapeCache:
    """Tests for global shape cache."""

    def test_clear_shape_cache(self):
        """Should clear global cache."""
        from peakfit.core.caching import clear_shape_cache, get_cache_stats

        # Add something to cache
        clear_shape_cache()
        stats = get_cache_stats()
        assert stats["size"] == 0

    def test_get_cache_stats(self):
        """Should return cache statistics."""
        from peakfit.core.caching import get_cache_stats

        stats = get_cache_stats()
        assert "size" in stats
        assert "maxsize" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
