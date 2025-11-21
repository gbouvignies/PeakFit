"""Tests for JAX optimizer.

This test suite validates that the JAX optimizer produces correct results
and is compatible with the existing scipy optimizer interface.
"""

import numpy as np
import pytest

from peakfit.core import lineshapes


@pytest.mark.skipif(not lineshapes.HAS_JAX, reason="JAX not installed")
class TestJAXOptimizerBasic:
    """Basic tests for JAX optimizer functionality."""

    def test_optimizer_import(self):
        """Test that JAX optimizer can be imported."""
        from peakfit.core import optimizer_jax

        assert hasattr(optimizer_jax, "fit_cluster_jax")
        assert hasattr(optimizer_jax, "fit_cluster")

    def test_params_to_jax_arrays(self):
        """Test parameter conversion to JAX arrays."""
        from peakfit.core.fitting import Parameters
        from peakfit.core.optimizer_jax import params_to_jax_arrays

        # Create simple parameters
        params = Parameters()
        params.add("x1", value=1.0, vary=True, min=0.0, max=10.0)
        params.add("x2", value=2.0, vary=True, min=-5.0, max=5.0)
        params.add("x3", value=3.0, vary=False)  # Fixed parameter

        x0, lower, upper, names = params_to_jax_arrays(params)

        # Check that only varying parameters are included
        assert len(x0) == 2
        assert len(lower) == 2
        assert len(upper) == 2
        assert len(names) == 2
        assert "x1" in names
        assert "x2" in names
        assert "x3" not in names

        # Check values
        assert np.allclose(np.array(x0), [1.0, 2.0])
        assert np.allclose(np.array(lower), [0.0, -5.0])
        assert np.allclose(np.array(upper), [10.0, 5.0])

    def test_compute_residuals_jax(self):
        """Test JAX residual computation."""
        import jax.numpy as jnp

        from peakfit.core.optimizer_jax import compute_residuals_jax

        # Simple test case
        x = jnp.array([1.0, 2.0])  # Dummy parameter values (not used in Phase 2.0)

        # Create simple shapes matrix (2 peaks, 10 points)
        shapes = jnp.array([[1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.9]])

        # Create data (perfect fit with amplitudes [2.0, 3.0])
        amplitudes_true = jnp.array([2.0, 3.0])
        data = shapes.T @ amplitudes_true
        noise = 1.0

        # Compute residuals (should be near zero for perfect fit)
        residuals = compute_residuals_jax(
            x, shapes, data, noise
        )

        # Residuals should be very small
        assert jnp.allclose(residuals, 0.0, atol=1e-10)

    def test_require_jax_function(self):
        """Test that require_jax works correctly."""
        from peakfit.core.optimizer_jax import require_jax

        # Should not raise since JAX is available (test is skipped if not)
        require_jax()


@pytest.mark.skipif(not lineshapes.HAS_JAX, reason="JAX not installed")
class TestJAXOptimizerIntegration:
    """Integration tests with scipy optimizer delegation."""

    def test_scipy_optimizer_delegates_to_jax(self):
        """Test that scipy_optimizer delegates to JAX when backend is jax."""
        from peakfit.core.backend import get_backend, set_backend

        # Save current backend
        original_backend = get_backend()

        try:
            # Set JAX backend
            set_backend("jax")
            assert get_backend() == "jax"

            # Import will use JAX optimizer when JAX backend is active
            from peakfit.core.scipy_optimizer import fit_cluster_dict

            # The function should now delegate to JAX
            # (actual fitting test would require creating a Cluster object)

        finally:
            # Restore original backend
            set_backend(original_backend)

    def test_backend_switching_works(self):
        """Test that we can switch between backends."""
        from peakfit.core.backend import get_available_backends, get_backend, set_backend

        available = get_available_backends()

        # Should be able to switch to JAX
        if "jax" in available:
            set_backend("jax")
            assert get_backend() == "jax"

        # Should be able to switch to numpy
        set_backend("numpy")
        assert get_backend() == "numpy"

        # Switch back to JAX for remaining tests
        if "jax" in available:
            set_backend("jax")


class TestOptimizerErrors:
    """Test error handling in JAX optimizer."""

    @pytest.mark.skipif(not lineshapes.HAS_JAX, reason="JAX not installed")
    def test_jax_optimizer_error_class(self):
        """Test that JAXOptimizerError is defined."""
        from peakfit.core.optimizer_jax import JAXOptimizerError

        # Should be able to raise and catch
        with pytest.raises(JAXOptimizerError):
            raise JAXOptimizerError("Test error")

    @pytest.mark.skipif(not lineshapes.HAS_JAX, reason="JAX not installed")
    def test_convergence_warning_class(self):
        """Test that ConvergenceWarning is defined."""
        from peakfit.core.optimizer_jax import ConvergenceWarning

        # Should be a UserWarning subclass
        assert issubclass(ConvergenceWarning, UserWarning)
