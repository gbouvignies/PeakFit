"""Tests for backend selection and management."""

import pytest

from peakfit.core.backend import (
    auto_select_backend,
    get_available_backends,
    get_backend,
    get_best_backend,
    set_backend,
)


class TestGetAvailableBackends:
    """Tests for get_available_backends function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        backends = get_available_backends()
        assert isinstance(backends, list)

    def test_numpy_always_available(self):
        """Test that numpy backend is always available."""
        backends = get_available_backends()
        assert "numpy" in backends

    def test_at_least_one_backend(self):
        """Test that at least numpy is available."""
        backends = get_available_backends()
        assert len(backends) >= 1


class TestGetBestBackend:
    """Tests for get_best_backend function."""

    def test_returns_string(self):
        """Test that function returns a string."""
        backend = get_best_backend()
        assert isinstance(backend, str)

    def test_returns_available_backend(self):
        """Test that returned backend is available."""
        backend = get_best_backend()
        available = get_available_backends()
        assert backend in available

    def test_prefers_numba_if_available(self):
        """Test that numba is preferred if available."""
        available = get_available_backends()
        best = get_best_backend()

        if "numba" in available:
            assert best == "numba"
        else:
            assert best == "numpy"


class TestSetBackend:
    """Tests for set_backend function."""

    def test_set_numpy_backend(self):
        """Test setting numpy backend."""
        set_backend("numpy")
        assert get_backend() == "numpy"

    def test_set_invalid_backend_raises_error(self):
        """Test that setting invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="not available"):
            set_backend("invalid_backend")

    def test_backend_persists(self):
        """Test that backend setting persists."""
        set_backend("numpy")
        backend1 = get_backend()
        backend2 = get_backend()
        assert backend1 == backend2 == "numpy"


class TestAutoSelectBackend:
    """Tests for auto_select_backend function."""

    def test_auto_select_returns_backend_name(self):
        """Test that auto_select returns a backend name."""
        backend = auto_select_backend()
        assert isinstance(backend, str)
        assert backend in get_available_backends()

    def test_auto_select_sets_backend(self):
        """Test that auto_select actually sets the backend."""
        selected = auto_select_backend()
        current = get_backend()
        assert selected == current


class TestBackendFunctions:
    """Tests for backend function retrieval."""

    def test_get_gaussian_func_returns_callable(self):
        """Test that gaussian function getter returns callable."""
        from peakfit.core.backend import get_gaussian_func

        set_backend("numpy")
        func = get_gaussian_func()
        assert callable(func)

    def test_get_lorentzian_func_returns_callable(self):
        """Test that lorentzian function getter returns callable."""
        from peakfit.core.backend import get_lorentzian_func

        set_backend("numpy")
        func = get_lorentzian_func()
        assert callable(func)

    def test_get_pvoigt_func_returns_callable(self):
        """Test that pvoigt function getter returns callable."""
        from peakfit.core.backend import get_pvoigt_func

        set_backend("numpy")
        func = get_pvoigt_func()
        assert callable(func)

    def test_backend_functions_work_after_switch(self):
        """Test that functions work after switching backends."""
        import numpy as np

        from peakfit.core.backend import get_gaussian_func

        # Test with numpy backend
        set_backend("numpy")
        func_numpy = get_gaussian_func()
        result_numpy = func_numpy(np.array([0.0, 1.0]), 10.0)

        assert isinstance(result_numpy, np.ndarray)
        assert len(result_numpy) == 2
        assert result_numpy[0] == pytest.approx(1.0)  # At center

        # Test with numba if available
        if "numba" in get_available_backends():
            set_backend("numba")
            func_numba = get_gaussian_func()
            result_numba = func_numba(np.array([0.0, 1.0]), 10.0)

            assert isinstance(result_numba, np.ndarray)
            assert len(result_numba) == 2
            # Results should be approximately the same
            assert np.allclose(result_numpy, result_numba)
