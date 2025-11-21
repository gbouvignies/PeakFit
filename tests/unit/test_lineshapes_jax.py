"""Tests for JAX lineshape functions.

This test suite validates that JAX implementations produce identical results
to NumPy implementations, and tests JAX-specific features like autodiff.
"""

import numpy as np
import pytest

# Import the lineshapes module
from peakfit.core import lineshapes


class TestJaxAvailability:
    """Tests for JAX backend availability."""

    def test_check_jax_available_returns_bool(self):
        """Test that check_jax_available returns a boolean."""
        result = lineshapes.check_jax_available()
        assert isinstance(result, bool)

    def test_get_backend_info_returns_dict(self):
        """Test that get_backend_info returns proper dict."""
        info = lineshapes.get_backend_info()
        assert isinstance(info, dict)
        assert "jax_available" in info
        assert "backend" in info
        assert "jit_enabled" in info
        assert "precision" in info

    def test_backend_is_jax_or_numpy(self):
        """Test that backend is either jax or numpy."""
        info = lineshapes.get_backend_info()
        assert info["backend"] in ["jax", "numpy"]


@pytest.mark.skipif(not lineshapes.HAS_JAX, reason="JAX not installed")
class TestJaxLineshapeEquivalence:
    """Test that JAX lineshapes match NumPy reference implementations."""

    def test_gaussian_matches_numpy(self):
        """Test JAX gaussian matches NumPy implementation."""
        import numpy as np

        dx = np.array([0.0, 1.0, 2.0, 5.0, 10.0, -5.0, -10.0])
        fwhm = 10.0

        # NumPy reference
        c = 4.0 * np.log(2.0) / (fwhm * fwhm)
        expected = np.exp(-dx * dx * c)

        # JAX implementation
        result = lineshapes.gaussian(dx, fwhm)
        result_np = np.asarray(result)

        assert np.allclose(result_np, expected, rtol=1e-14, atol=1e-14)

    def test_lorentzian_matches_numpy(self):
        """Test JAX lorentzian matches NumPy implementation."""
        import numpy as np

        dx = np.array([0.0, 1.0, 2.0, 5.0, 10.0, -5.0, -10.0])
        fwhm = 10.0

        # NumPy reference
        half_width_sq = (0.5 * fwhm) ** 2
        expected = half_width_sq / (dx * dx + half_width_sq)

        # JAX implementation
        result = lineshapes.lorentzian(dx, fwhm)
        result_np = np.asarray(result)

        assert np.allclose(result_np, expected, rtol=1e-14, atol=1e-14)

    def test_pvoigt_matches_numpy(self):
        """Test JAX pvoigt matches NumPy implementation."""
        import numpy as np

        dx = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
        fwhm = 10.0
        eta = 0.5

        # NumPy reference
        c = 4.0 * np.log(2.0) / (fwhm * fwhm)
        gauss = np.exp(-dx * dx * c)
        half_width_sq = (0.5 * fwhm) ** 2
        lorentz = half_width_sq / (dx * dx + half_width_sq)
        expected = (1.0 - eta) * gauss + eta * lorentz

        # JAX implementation
        result = lineshapes.pvoigt(dx, fwhm, eta)
        result_np = np.asarray(result)

        assert np.allclose(result_np, expected, rtol=1e-14, atol=1e-14)

    def test_no_apod_matches_numpy(self):
        """Test JAX no_apod matches NumPy implementation."""
        import numpy as np

        dx = np.linspace(-50, 50, 100)
        r2 = 10.0
        aq = 0.1
        phase = 0.0

        # NumPy reference
        z1 = aq * (1j * dx + r2)
        spec = aq * (1.0 - np.exp(-z1)) / z1
        expected = (spec * np.exp(1j * np.deg2rad(phase))).real

        # JAX implementation
        result = lineshapes.no_apod(dx, r2, aq, phase)
        result_np = np.asarray(result)

        assert np.allclose(result_np, expected, rtol=1e-12, atol=1e-12)

    def test_no_apod_with_phase_matches_numpy(self):
        """Test JAX no_apod with phase matches NumPy implementation."""
        import numpy as np

        dx = np.linspace(-50, 50, 100)
        r2 = 10.0
        aq = 0.1
        phase = 45.0

        # NumPy reference
        z1 = aq * (1j * dx + r2)
        spec = aq * (1.0 - np.exp(-z1)) / z1
        expected = (spec * np.exp(1j * np.deg2rad(phase))).real

        # JAX implementation
        result = lineshapes.no_apod(dx, r2, aq, phase)
        result_np = np.asarray(result)

        assert np.allclose(result_np, expected, rtol=1e-12, atol=1e-12)

    def test_sp1_matches_numpy(self):
        """Test JAX sp1 matches NumPy implementation."""
        import numpy as np

        dx = np.linspace(-50, 50, 100)
        r2 = 10.0
        aq = 0.1
        end = 1.0
        off = 0.35
        phase = 0.0

        # NumPy reference
        z1 = aq * (1j * dx + r2)
        f1, f2 = 1j * off * np.pi, 1j * (end - off) * np.pi
        a1 = (np.exp(+f2) - np.exp(+z1)) * np.exp(-z1 + f1) / (2 * (z1 - f2))
        a2 = (np.exp(+z1) - np.exp(-f2)) * np.exp(-z1 - f1) / (2 * (z1 + f2))
        spec = 1j * aq * (a1 + a2)
        expected = (spec * np.exp(1j * np.deg2rad(phase))).real

        # JAX implementation
        result = lineshapes.sp1(dx, r2, aq, end, off, phase)
        result_np = np.asarray(result)

        assert np.allclose(result_np, expected, rtol=1e-12, atol=1e-12)

    def test_sp2_matches_numpy(self):
        """Test JAX sp2 matches NumPy implementation."""
        import numpy as np

        dx = np.linspace(-50, 50, 100)
        r2 = 10.0
        aq = 0.1
        end = 1.0
        off = 0.35
        phase = 0.0

        # NumPy reference
        z1 = aq * (1j * dx + r2)
        f1, f2 = 1j * off * np.pi, 1j * (end - off) * np.pi
        a1 = (np.exp(+2 * f2) - np.exp(z1)) * np.exp(-z1 + 2 * f1) / (4 * (z1 - 2 * f2))
        a2 = (np.exp(-2 * f2) - np.exp(z1)) * np.exp(-z1 - 2 * f1) / (4 * (z1 + 2 * f2))
        a3 = (1.0 - np.exp(-z1)) / (2 * z1)
        spec = aq * (a1 + a2 + a3)
        expected = (spec * np.exp(1j * np.deg2rad(phase))).real

        # JAX implementation
        result = lineshapes.sp2(dx, r2, aq, end, off, phase)
        result_np = np.asarray(result)

        assert np.allclose(result_np, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not lineshapes.HAS_JAX, reason="JAX not installed")
class TestJaxLineshapeProperties:
    """Test mathematical properties of JAX lineshapes."""

    def test_gaussian_peak_is_one(self):
        """Test Gaussian peak value is 1.0."""
        result = lineshapes.gaussian(np.array([0.0]), 10.0)
        assert np.asarray(result)[0] == pytest.approx(1.0, rel=1e-14)

    def test_gaussian_symmetry(self):
        """Test Gaussian is symmetric."""
        dx = np.array([-5.0, 5.0])
        result = lineshapes.gaussian(dx, 10.0)
        result_np = np.asarray(result)
        assert result_np[0] == pytest.approx(result_np[1], rel=1e-14)

    def test_lorentzian_peak_is_one(self):
        """Test Lorentzian peak value is 1.0."""
        result = lineshapes.lorentzian(np.array([0.0]), 10.0)
        assert np.asarray(result)[0] == pytest.approx(1.0, rel=1e-14)

    def test_lorentzian_symmetry(self):
        """Test Lorentzian is symmetric."""
        dx = np.array([-5.0, 5.0])
        result = lineshapes.lorentzian(dx, 10.0)
        result_np = np.asarray(result)
        assert result_np[0] == pytest.approx(result_np[1], rel=1e-14)

    def test_pvoigt_peak_is_one(self):
        """Test Pseudo-Voigt peak value is 1.0."""
        result = lineshapes.pvoigt(np.array([0.0]), 10.0, 0.5)
        assert np.asarray(result)[0] == pytest.approx(1.0, rel=1e-14)

    def test_pvoigt_eta_zero_is_gaussian(self):
        """Test pvoigt with eta=0 is pure Gaussian."""
        dx = np.linspace(-20, 20, 50)
        pv_result = lineshapes.pvoigt(dx, 10.0, 0.0)
        gauss_result = lineshapes.gaussian(dx, 10.0)
        assert np.allclose(np.asarray(pv_result), np.asarray(gauss_result), rtol=1e-14)

    def test_pvoigt_eta_one_is_lorentzian(self):
        """Test pvoigt with eta=1 is pure Lorentzian."""
        dx = np.linspace(-20, 20, 50)
        pv_result = lineshapes.pvoigt(dx, 10.0, 1.0)
        lor_result = lineshapes.lorentzian(dx, 10.0)
        assert np.allclose(np.asarray(pv_result), np.asarray(lor_result), rtol=1e-14)


@pytest.mark.skipif(not lineshapes.HAS_JAX, reason="JAX not installed")
class TestCalculateLstsqAmplitude:
    """Test JAX linear least squares amplitude calculation."""

    def test_lstsq_amplitude_simple_case(self):
        """Test lstsq amplitude calculation for simple case."""
        import numpy as np

        # Simple test: single peak, perfect fit
        shapes = np.array([[1.0, 0.5, 0.2, 0.1]])
        data = np.array([2.0, 1.0, 0.4, 0.2])

        result = lineshapes.calculate_lstsq_amplitude(shapes, data)
        result_np = np.asarray(result)

        # Expected amplitude is 2.0 (scales the shape to match data)
        assert result_np.shape == (1,)
        assert result_np[0] == pytest.approx(2.0, rel=1e-10)

    def test_lstsq_amplitude_multiple_peaks(self):
        """Test lstsq with multiple peaks."""
        import numpy as np

        # Two peaks with different amplitudes
        shapes = np.array([[1.0, 0.5, 0.2], [0.2, 0.5, 1.0]])
        # Data is sum: 3*shape1 + 2*shape2
        data = 3.0 * shapes[0] + 2.0 * shapes[1]

        result = lineshapes.calculate_lstsq_amplitude(shapes, data)
        result_np = np.asarray(result)

        assert result_np.shape == (2,)
        assert result_np[0] == pytest.approx(3.0, rel=1e-10)
        assert result_np[1] == pytest.approx(2.0, rel=1e-10)

    def test_lstsq_matches_numpy(self):
        """Test JAX lstsq matches NumPy result."""
        import numpy as np

        # Random test case
        rng = np.random.default_rng(42)
        shapes = rng.random((3, 50))
        data = rng.random(50)

        # NumPy reference
        ata = np.dot(shapes, shapes.T)
        atb = np.dot(shapes, data)
        expected = np.linalg.solve(ata, atb)

        # JAX implementation
        result = lineshapes.calculate_lstsq_amplitude(shapes, data)
        result_np = np.asarray(result)

        assert np.allclose(result_np, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not lineshapes.HAS_JAX, reason="JAX not installed")
class TestJaxAutodiff:
    """Test JAX automatic differentiation capabilities."""

    def test_gaussian_gradient(self):
        """Test that we can compute gradient of gaussian wrt fwhm."""
        import jax
        import numpy as np

        dx = np.array([0.0, 1.0, 2.0, 5.0])

        def loss(fwhm):
            return jax.numpy.sum(lineshapes.gaussian(dx, fwhm) ** 2)

        # Compute gradient
        grad_fn = jax.grad(loss)
        gradient = grad_fn(10.0)

        # Gradient should be finite and reasonable
        assert np.isfinite(gradient)
        assert gradient != 0.0  # Should have non-zero gradient

    def test_lorentzian_gradient(self):
        """Test that we can compute gradient of lorentzian wrt fwhm."""
        import jax
        import numpy as np

        dx = np.array([0.0, 1.0, 2.0, 5.0])

        def loss(fwhm):
            return jax.numpy.sum(lineshapes.lorentzian(dx, fwhm) ** 2)

        # Compute gradient
        grad_fn = jax.grad(loss)
        gradient = grad_fn(10.0)

        # Gradient should be finite and reasonable
        assert np.isfinite(gradient)
        assert gradient != 0.0

    def test_pvoigt_gradient_wrt_eta(self):
        """Test that we can compute gradient of pvoigt wrt eta."""
        import jax
        import numpy as np

        dx = np.array([0.0, 1.0, 2.0, 5.0])
        fwhm = 10.0

        def loss(eta):
            return jax.numpy.sum(lineshapes.pvoigt(dx, fwhm, eta) ** 2)

        # Compute gradient
        grad_fn = jax.grad(loss)
        gradient = grad_fn(0.5)

        # Gradient should be finite
        assert np.isfinite(gradient)


class TestRequireJax:
    """Test require_jax function."""

    @pytest.mark.skipif(lineshapes.HAS_JAX, reason="JAX is installed")
    def test_require_jax_raises_when_not_available(self):
        """Test that require_jax raises RuntimeError when JAX not available."""
        with pytest.raises(RuntimeError, match="JAX is required"):
            lineshapes.require_jax()

    @pytest.mark.skipif(not lineshapes.HAS_JAX, reason="JAX not installed")
    def test_require_jax_succeeds_when_available(self):
        """Test that require_jax doesn't raise when JAX is available."""
        lineshapes.require_jax()  # Should not raise


class TestNumpyFallback:
    """Test that NumPy fallback works when JAX unavailable."""

    @pytest.mark.skipif(lineshapes.HAS_JAX, reason="JAX is installed")
    def test_gaussian_works_without_jax(self):
        """Test gaussian works with NumPy fallback."""
        import numpy as np

        dx = np.array([0.0, 1.0, 2.0])
        result = lineshapes.gaussian(dx, 10.0)

        assert isinstance(result, np.ndarray)
        assert result[0] == pytest.approx(1.0)

    @pytest.mark.skipif(lineshapes.HAS_JAX, reason="JAX is installed")
    def test_lorentzian_works_without_jax(self):
        """Test lorentzian works with NumPy fallback."""
        import numpy as np

        dx = np.array([0.0, 1.0, 2.0])
        result = lineshapes.lorentzian(dx, 10.0)

        assert isinstance(result, np.ndarray)
        assert result[0] == pytest.approx(1.0)
