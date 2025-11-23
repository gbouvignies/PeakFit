"""Functional tests for Numba lineshape functions.

Run with: uv run pytest tests/test_lineshapes.py -v
"""

import numpy as np
import pytest

from peakfit.lineshapes import (
    compute_all_gaussian_shapes,
    compute_all_lorentzian_shapes,
    compute_all_pvoigt_shapes,
    gaussian,
    lorentzian,
    no_apod,
    pvoigt,
    sp1,
    sp2,
)


def test_gaussian_basic():
    """Test Gaussian lineshape properties."""
    dx = np.linspace(-100, 100, 1001)  # Use 1001 points so dx[500] is exactly 0
    fwhm = 10.0

    result = gaussian(dx, fwhm)

    # Check normalization at center
    assert np.isclose(result[500], 1.0, rtol=1e-10)

    # Check FWHM
    half_max_idx = np.where(result >= 0.5)[0]
    fwhm_measured = dx[half_max_idx[-1]] - dx[half_max_idx[0]]
    assert np.isclose(fwhm_measured, fwhm, rtol=0.01)

    # Check symmetry (exclude center point)
    assert np.allclose(result[:500], result[501:][::-1], rtol=1e-12)


def test_lorentzian_basic():
    """Test Lorentzian lineshape properties."""
    dx = np.linspace(-100, 100, 1001)  # Use 1001 points so dx[500] is exactly 0
    fwhm = 10.0

    result = lorentzian(dx, fwhm)

    # Check normalization at center
    assert np.isclose(result[500], 1.0, rtol=1e-10)

    # Check FWHM
    half_max_idx = np.where(result >= 0.5)[0]
    fwhm_measured = dx[half_max_idx[-1]] - dx[half_max_idx[0]]
    assert np.isclose(fwhm_measured, fwhm, rtol=0.01)


def test_pvoigt_pure_limits():
    """Test Pseudo-Voigt at pure Gaussian and Lorentzian limits."""
    dx = np.linspace(-50, 50, 500)
    fwhm = 10.0

    # Pure Gaussian (eta=0)
    pv_gauss = pvoigt(dx, fwhm, eta=0.0)
    pure_gauss = gaussian(dx, fwhm)
    np.testing.assert_allclose(pv_gauss, pure_gauss, rtol=1e-14)

    # Pure Lorentzian (eta=1)
    pv_lorentz = pvoigt(dx, fwhm, eta=1.0)
    pure_lorentz = lorentzian(dx, fwhm)
    np.testing.assert_allclose(pv_lorentz, pure_lorentz, rtol=1e-14)


@pytest.mark.parametrize("phase", [0.0, 45.0, 90.0, 180.0, 270.0])
def test_no_apod_phases(phase):
    """Test no_apod with various phase values."""
    dx = np.linspace(-100, 100, 500)
    r2 = 5.0
    aq = 0.05

    result = no_apod(dx, r2, aq, phase)

    # Result should be real (dtype check)
    assert result.dtype == np.float64

    # Should be finite
    assert np.all(np.isfinite(result))

    # Should have reasonable magnitude
    assert np.max(np.abs(result)) < 100.0


def test_sp1_basic():
    """Test SP1 apodization."""
    dx = np.linspace(-100, 100, 500)
    r2 = 5.0
    aq = 0.05
    end = 2.0
    off = 0.5

    result = sp1(dx, r2, aq, end, off, phase=0.0)

    assert result.dtype == np.float64
    assert np.all(np.isfinite(result))


def test_sp2_basic():
    """Test SP2 apodization."""
    dx = np.linspace(-100, 100, 500)
    r2 = 5.0
    aq = 0.05
    end = 2.0
    off = 0.5

    result = sp2(dx, r2, aq, end, off, phase=0.0)

    assert result.dtype == np.float64
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("n_peaks", [1, 5, 10, 50, 100])
def test_multi_peak_gaussian(n_peaks):
    """Test parallel multi-peak Gaussian computation."""
    n_points = 500
    positions = np.linspace(0, 1000, n_points)
    centers = np.linspace(100, 900, n_peaks)
    fwhms = np.full(n_peaks, 10.0)

    shapes = compute_all_gaussian_shapes(positions, centers, fwhms)

    # Check output shape
    assert shapes.shape == (n_peaks, n_points)

    # Check dtype
    assert shapes.dtype == np.float64

    # Each peak should be maximum near its center
    for i, center in enumerate(centers):
        center_idx = np.argmin(np.abs(positions - center))
        # Check peak is close to 1.0 at center (within 5%)
        assert shapes[i, center_idx] > 0.95


def test_multi_peak_lorentzian():
    """Test parallel multi-peak Lorentzian computation."""
    positions = np.linspace(0, 1000, 500)
    centers = np.array([100, 300, 500, 700, 900])
    fwhms = np.full(5, 10.0)

    shapes = compute_all_lorentzian_shapes(positions, centers, fwhms)

    assert shapes.shape == (5, 500)
    assert shapes.dtype == np.float64


def test_multi_peak_pvoigt():
    """Test parallel multi-peak Pseudo-Voigt computation."""
    positions = np.linspace(0, 1000, 500)
    centers = np.array([100, 300, 500, 700, 900])
    fwhms = np.full(5, 10.0)
    etas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    shapes = compute_all_pvoigt_shapes(positions, centers, fwhms, etas)

    assert shapes.shape == (5, 500)
    assert shapes.dtype == np.float64


def test_gaussian_array_input():
    """Test that Gaussian handles array inputs correctly."""
    dx = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    fwhm = 10.0

    result = gaussian(dx, fwhm)

    assert len(result) == len(dx)
    assert result.dtype == np.float64
    # Check center is maximum
    assert result[2] == pytest.approx(1.0)
    # Check symmetry
    assert result[0] == pytest.approx(result[4])
    assert result[1] == pytest.approx(result[3])


def test_lorentzian_array_input():
    """Test that Lorentzian handles array inputs correctly."""
    dx = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    fwhm = 10.0

    result = lorentzian(dx, fwhm)

    assert len(result) == len(dx)
    assert result.dtype == np.float64
    # Check center is maximum
    assert result[2] == pytest.approx(1.0)
    # Check symmetry
    assert result[0] == pytest.approx(result[4])
    assert result[1] == pytest.approx(result[3])


def test_pvoigt_mixed():
    """Test Pseudo-Voigt with mixed eta value."""
    dx = np.linspace(-50, 50, 100)
    fwhm = 10.0
    eta = 0.5

    result = pvoigt(dx, fwhm, eta)

    # Should be between pure Gaussian and pure Lorentzian
    gauss = gaussian(dx, fwhm)
    lorentz = lorentzian(dx, fwhm)

    # At all points, pvoigt should be a weighted average
    expected = 0.5 * gauss + 0.5 * lorentz
    np.testing.assert_allclose(result, expected, rtol=1e-14)


def test_no_apod_zero_phase():
    """Test no_apod with zero phase."""
    dx = np.linspace(-50, 50, 200)
    r2 = 10.0
    aq = 0.1
    phase = 0.0

    result = no_apod(dx, r2, aq, phase)

    # Should be symmetric around center for zero phase
    center_idx = len(dx) // 2
    left_half = result[:center_idx]
    right_half = result[center_idx:][::-1]

    # Relaxed tolerance for FID-based lineshapes
    np.testing.assert_allclose(left_half, right_half, rtol=0.01, atol=1e-6)


def test_multi_peak_gaussian_single_peak():
    """Test multi-peak Gaussian with single peak matches single-peak function."""
    positions = np.linspace(-100, 100, 500)
    centers = np.array([0.0])
    fwhms = np.array([10.0])

    multi_result = compute_all_gaussian_shapes(positions, centers, fwhms)
    single_result = gaussian(positions - centers[0], fwhms[0])

    # Relaxed tolerance due to minor floating point differences in parallel computation
    np.testing.assert_allclose(multi_result[0], single_result, rtol=1e-13)


def test_multi_peak_lorentzian_single_peak():
    """Test multi-peak Lorentzian with single peak matches single-peak function."""
    positions = np.linspace(-100, 100, 500)
    centers = np.array([0.0])
    fwhms = np.array([10.0])

    multi_result = compute_all_lorentzian_shapes(positions, centers, fwhms)
    single_result = lorentzian(positions - centers[0], fwhms[0])

    np.testing.assert_allclose(multi_result[0], single_result, rtol=1e-14)


def test_multi_peak_pvoigt_single_peak():
    """Test multi-peak Pseudo-Voigt with single peak matches single-peak function."""
    positions = np.linspace(-100, 100, 500)
    centers = np.array([0.0])
    fwhms = np.array([10.0])
    etas = np.array([0.5])

    multi_result = compute_all_pvoigt_shapes(positions, centers, fwhms, etas)
    single_result = pvoigt(positions - centers[0], fwhms[0], etas[0])

    np.testing.assert_allclose(multi_result[0], single_result, rtol=1e-14)


def test_gaussian_different_fwhm():
    """Test Gaussian with different FWHM values."""
    dx = np.linspace(-100, 100, 1001)  # Use 1001 points so center is exact

    result_narrow = gaussian(dx, 5.0)
    result_wide = gaussian(dx, 20.0)

    # Narrow peak should be taller and narrower
    assert result_narrow.max() == pytest.approx(1.0)
    assert result_wide.max() == pytest.approx(1.0)

    # Check that narrow peak decays faster
    idx_50 = 250  # offset from center
    assert result_narrow[idx_50] < result_wide[idx_50]


def test_sp1_sp2_different():
    """Test that SP1 and SP2 produce different results."""
    dx = np.linspace(-100, 100, 500)
    r2 = 5.0
    aq = 0.05
    end = 2.0
    off = 0.5
    phase = 0.0

    result_sp1 = sp1(dx, r2, aq, end, off, phase)
    result_sp2 = sp2(dx, r2, aq, end, off, phase)

    # Results should be different
    assert not np.allclose(result_sp1, result_sp2)

    # Both should be finite and reasonable
    assert np.all(np.isfinite(result_sp1))
    assert np.all(np.isfinite(result_sp2))
