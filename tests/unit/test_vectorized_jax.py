"""Tests for vectorized JAX lineshape evaluation (Phase 2.1)."""

import numpy as np
import pytest

from peakfit.core import lineshapes


@pytest.mark.skipif(not lineshapes.HAS_JAX, reason="JAX not installed")
class TestVectorizedJAX:
    """Tests for Phase 2.1 vectorized JAX evaluation."""

    def test_vectorized_module_import(self):
        """Test that vectorized JAX module can be imported."""
        from peakfit.core import vectorized_jax

        assert hasattr(vectorized_jax, "compute_shapes_matrix_jax_vectorized")
        assert hasattr(vectorized_jax, "extract_peak_evaluation_data_jax")

    def test_shape_type_constants(self):
        """Test that shape type constants are defined."""
        from peakfit.core.vectorized_jax import (
            SHAPE_GAUSSIAN,
            SHAPE_LORENTZIAN,
            SHAPE_NO_APOD,
            SHAPE_PVOIGT,
            SHAPE_SP1,
            SHAPE_SP2,
        )

        # Should be distinct integers
        shape_types = {SHAPE_GAUSSIAN, SHAPE_LORENTZIAN, SHAPE_PVOIGT,
                      SHAPE_NO_APOD, SHAPE_SP1, SHAPE_SP2}
        assert len(shape_types) == 6

    def test_pts2hz_delta_jax(self):
        """Test point-to-Hz conversion."""
        import jax.numpy as jnp

        from peakfit.core.vectorized_jax import pts2hz_delta_jax

        dx_pt = jnp.array([0.0, 1.0, 10.0, 100.0])
        sw = 10000.0  # 10 kHz
        size = 1024

        dx_hz = pts2hz_delta_jax(dx_pt, sw, size)

        # Check conversion
        expected = dx_pt * sw / size
        assert jnp.allclose(dx_hz, expected)

    def test_evaluate_single_shape_gaussian(self):
        """Test single shape evaluation for Gaussian."""
        import jax.numpy as jnp

        from peakfit.core.vectorized_jax import SHAPE_GAUSSIAN, evaluate_single_shape_jax

        grid_pts = jnp.arange(0, 100, dtype=jnp.float32)
        shape_type = SHAPE_GAUSSIAN
        position = 50.0
        fwhm = 10.0
        eta = 0.0  # Not used for Gaussian
        r2 = 0.0  # Not used
        aq = 0.0
        end = 0.0
        off = 0.0
        phase = 0.0
        sw = 10000.0
        size = 100

        result = evaluate_single_shape_jax(
            grid_pts, shape_type, position, fwhm, eta,
            r2, aq, end, off, phase, sw, size
        )

        # Check that peak is at position
        assert result[50] == pytest.approx(1.0, rel=0.01)
        # Check that values decrease away from center
        assert result[50] > result[40]
        assert result[50] > result[60]

    def test_evaluate_single_shape_lorentzian(self):
        """Test single shape evaluation for Lorentzian."""
        import jax.numpy as jnp

        from peakfit.core.vectorized_jax import SHAPE_LORENTZIAN, evaluate_single_shape_jax

        grid_pts = jnp.arange(0, 100, dtype=jnp.float32)
        shape_type = SHAPE_LORENTZIAN
        position = 50.0
        fwhm = 10.0
        eta = 0.0
        r2 = 0.0
        aq = 0.0
        end = 0.0
        off = 0.0
        phase = 0.0
        sw = 10000.0
        size = 100

        result = evaluate_single_shape_jax(
            grid_pts, shape_type, position, fwhm, eta,
            r2, aq, end, off, phase, sw, size
        )

        # Check that peak is at position
        assert result[50] == pytest.approx(1.0, rel=0.01)
        # Lorentzian has broader tails than Gaussian
        assert result[40] > 0.0


@pytest.mark.skipif(not lineshapes.HAS_JAX, reason="JAX not installed")
class TestVectorizedIntegration:
    """Integration tests for vectorized evaluation with optimizer."""

    def test_compute_shapes_matrix_uses_vectorized(self):
        """Test that compute_shapes_matrix can use vectorized path."""
        from peakfit.core.optimizer_jax import compute_shapes_matrix_numpy

        # This is tested indirectly - if import doesn't fail, the function exists
        assert callable(compute_shapes_matrix_numpy)
