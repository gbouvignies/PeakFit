"""Numerical correctness tests for Numba implementations.

These tests ensure that Numba optimizations do not affect numerical accuracy.
All tests use strict tolerances (rtol=1e-14 or better).

Run with: uv run pytest tests/test_lineshapes_correctness.py -v
"""

import numpy as np
import pytest


def gaussian_reference(dx, fwhm):
    """Reference NumPy implementation of Gaussian."""
    c = 4.0 * np.log(2.0) / (fwhm * fwhm)
    return np.exp(-dx * dx * c)


def lorentzian_reference(dx, fwhm):
    """Reference NumPy implementation of Lorentzian."""
    half_width_sq = (0.5 * fwhm) ** 2
    return half_width_sq / (dx * dx + half_width_sq)


@pytest.mark.parametrize("fwhm", [1.0, 5.0, 10.0, 50.0, 100.0])
@pytest.mark.parametrize("n_points", [100, 1000, 10000])
def test_gaussian_vs_reference(fwhm, n_points):
    """Test Numba Gaussian matches NumPy reference with machine precision."""
    from peakfit.lineshapes import gaussian

    dx = np.linspace(-500, 500, n_points)

    result_numba = gaussian(dx, fwhm)
    result_ref = gaussian_reference(dx, fwhm)

    # Strict tolerance: machine precision
    np.testing.assert_allclose(result_numba, result_ref, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize("fwhm", [1.0, 5.0, 10.0, 50.0, 100.0])
@pytest.mark.parametrize("n_points", [100, 1000, 10000])
def test_lorentzian_vs_reference(fwhm, n_points):
    """Test Numba Lorentzian matches NumPy reference with machine precision."""
    from peakfit.lineshapes import lorentzian

    dx = np.linspace(-500, 500, n_points)

    result_numba = lorentzian(dx, fwhm)
    result_ref = lorentzian_reference(dx, fwhm)

    np.testing.assert_allclose(result_numba, result_ref, rtol=1e-15, atol=1e-15)


def test_reproducibility():
    """Test that Numba functions give identical results across runs."""
    from peakfit.lineshapes import gaussian

    dx = np.linspace(-100, 100, 1000)
    fwhm = 10.0

    # Run 100 times and check all results are identical
    results = [gaussian(dx, fwhm) for _ in range(100)]

    for i in range(1, 100):
        np.testing.assert_array_equal(results[0], results[i])


@pytest.mark.parametrize("eta", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_pvoigt_correctness(eta):
    """Test Pseudo-Voigt correctness at various eta values."""
    from peakfit.lineshapes import gaussian, lorentzian, pvoigt

    dx = np.linspace(-100, 100, 1000)
    fwhm = 10.0

    result_pvoigt = pvoigt(dx, fwhm, eta)
    result_manual = (1.0 - eta) * gaussian(dx, fwhm) + eta * lorentzian(dx, fwhm)

    np.testing.assert_allclose(result_pvoigt, result_manual, rtol=1e-15, atol=1e-15)


def test_multi_peak_consistency():
    """Test multi-peak functions are consistent with single-peak versions."""
    from peakfit.lineshapes import (
        compute_all_gaussian_shapes,
        compute_all_lorentzian_shapes,
        compute_all_pvoigt_shapes,
        gaussian,
        lorentzian,
        pvoigt,
    )

    positions = np.linspace(-200, 200, 1000)
    centers = np.array([0.0])
    fwhms = np.array([15.0])
    etas = np.array([0.3])

    # Test Gaussian
    multi_gauss = compute_all_gaussian_shapes(positions, centers, fwhms)
    single_gauss = gaussian(positions - centers[0], fwhms[0])
    np.testing.assert_allclose(multi_gauss[0], single_gauss, rtol=1e-15, atol=1e-15)

    # Test Lorentzian
    multi_lorentz = compute_all_lorentzian_shapes(positions, centers, fwhms)
    single_lorentz = lorentzian(positions - centers[0], fwhms[0])
    np.testing.assert_allclose(multi_lorentz[0], single_lorentz, rtol=1e-15, atol=1e-15)

    # Test Pseudo-Voigt
    multi_pvoigt = compute_all_pvoigt_shapes(positions, centers, fwhms, etas)
    single_pvoigt = pvoigt(positions - centers[0], fwhms[0], etas[0])
    np.testing.assert_allclose(multi_pvoigt[0], single_pvoigt, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize("n_peaks", [5, 10, 50])
def test_multi_peak_gaussian_independence(n_peaks):
    """Test that multi-peak Gaussian computes each peak independently."""
    from peakfit.lineshapes import compute_all_gaussian_shapes, gaussian

    positions = np.linspace(0, 1000, 2000)
    centers = np.linspace(100, 900, n_peaks)
    fwhms = np.linspace(5.0, 20.0, n_peaks)

    shapes = compute_all_gaussian_shapes(positions, centers, fwhms)

    # Check each peak independently
    for i in range(n_peaks):
        dx = positions - centers[i]
        expected = gaussian(dx, fwhms[i])
        np.testing.assert_allclose(shapes[i], expected, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize("phase", [0.0, 30.0, 60.0, 90.0, 120.0, 180.0])
def test_no_apod_phase_consistency(phase):
    """Test no_apod phase correction is numerically accurate."""
    from peakfit.lineshapes import no_apod

    dx = np.linspace(-100, 100, 500)
    r2 = 5.0
    aq = 0.05

    # Compute with Numba
    result_numba = no_apod(dx, r2, aq, phase)

    # Compute with pure NumPy (reference)
    z1 = aq * (1j * dx + r2)
    spec = aq * (1.0 - np.exp(-z1)) / z1
    result_ref = (spec * np.exp(1j * np.deg2rad(phase))).real

    # Should match to high precision
    np.testing.assert_allclose(result_numba, result_ref, rtol=1e-12, atol=1e-15)


def test_gaussian_small_fwhm():
    """Test Gaussian with very small FWHM."""
    from peakfit.lineshapes import gaussian

    dx = np.linspace(-10, 10, 1001)  # Odd number so center is exact
    fwhm = 0.01  # Very narrow peak

    result = gaussian(dx, fwhm)

    # Should be normalized at center
    center_idx = len(dx) // 2
    assert result[center_idx] == pytest.approx(1.0, rel=1e-10)

    # Should decay very quickly
    assert result[0] < 1e-100
    assert result[-1] < 1e-100


def test_gaussian_large_fwhm():
    """Test Gaussian with very large FWHM."""
    from peakfit.lineshapes import gaussian

    dx = np.linspace(-1000, 1000, 2001)  # Odd number so center is exact
    fwhm = 500.0  # Very wide peak

    result = gaussian(dx, fwhm)

    # Should be normalized at center
    center_idx = len(dx) // 2
    assert result[center_idx] == pytest.approx(1.0, rel=1e-10)

    # Should decay slowly (but not too slowly for FWHM=500 over range -1000 to 1000)
    assert result[0] > 1e-6  # Still measurable at edges


def test_lorentzian_small_fwhm():
    """Test Lorentzian with very small FWHM."""
    from peakfit.lineshapes import lorentzian

    dx = np.linspace(-10, 10, 1001)  # Odd number so center is exact
    fwhm = 0.01  # Very narrow peak

    result = lorentzian(dx, fwhm)

    # Should be normalized at center
    center_idx = len(dx) // 2
    assert result[center_idx] == pytest.approx(1.0, rel=1e-10)


def test_lorentzian_large_fwhm():
    """Test Lorentzian with very large FWHM."""
    from peakfit.lineshapes import lorentzian

    dx = np.linspace(-1000, 1000, 2001)  # Odd number so center is exact
    fwhm = 500.0  # Very wide peak

    result = lorentzian(dx, fwhm)

    # Should be normalized at center
    center_idx = len(dx) // 2
    assert result[center_idx] == pytest.approx(1.0, rel=1e-10)


@pytest.mark.parametrize("seed", range(10))
def test_multi_peak_random_parameters(seed):
    """Test multi-peak functions with random parameters."""
    from peakfit.lineshapes import compute_all_gaussian_shapes, gaussian

    rng = np.random.default_rng(seed)

    n_peaks = rng.integers(5, 20)
    n_points = rng.integers(500, 2000)

    positions = np.linspace(0, 1000, n_points)
    centers = rng.uniform(100, 900, n_peaks)
    fwhms = rng.uniform(5.0, 30.0, n_peaks)

    shapes = compute_all_gaussian_shapes(positions, centers, fwhms)

    # Verify each peak independently
    for i in range(n_peaks):
        dx = positions - centers[i]
        expected = gaussian(dx, fwhms[i])
        np.testing.assert_allclose(shapes[i], expected, rtol=1e-15, atol=1e-15)


def test_deterministic_random_state():
    """Test that results are deterministic with same random seed."""
    from peakfit.lineshapes import compute_all_gaussian_shapes

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    positions = np.linspace(0, 1000, 1000)

    # Generate same random parameters
    centers1 = rng1.uniform(100, 900, 50)
    fwhms1 = rng1.uniform(5.0, 30.0, 50)

    centers2 = rng2.uniform(100, 900, 50)
    fwhms2 = rng2.uniform(5.0, 30.0, 50)

    shapes1 = compute_all_gaussian_shapes(positions, centers1, fwhms1)
    shapes2 = compute_all_gaussian_shapes(positions, centers2, fwhms2)

    np.testing.assert_array_equal(shapes1, shapes2)
