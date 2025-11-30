"""Tests for lineshape models."""

import pytest

import numpy as np

from peakfit.core.lineshapes.functions import (
    gaussian,
    lorentzian,
    no_apod,
    pvoigt,
    make_sp1_evaluator,
    make_sp2_evaluator,
)

# Note: clean() function was removed during refactoring


class TestGaussianFunction:
    """Tests for gaussian lineshape function."""

    def test_gaussian_at_center(self):
        """Test Gaussian is 1.0 at center."""
        dx = np.array([0.0])
        fwhm = 10.0
        result = gaussian(dx, fwhm)
        assert result[0] == pytest.approx(1.0)

    def test_gaussian_at_fwhm(self):
        """Test Gaussian is 0.5 at FWHM."""
        fwhm = 10.0
        dx = np.array([fwhm / 2.0])
        result = gaussian(dx, fwhm)
        assert result[0] == pytest.approx(0.5, abs=1e-10)

    def test_gaussian_symmetry(self):
        """Test Gaussian is symmetric."""
        dx = np.array([-5.0, 5.0])
        fwhm = 10.0
        result = gaussian(dx, fwhm)
        assert result[0] == pytest.approx(result[1])

    def test_gaussian_array_input(self):
        """Test Gaussian works with array input."""
        dx = np.linspace(-20, 20, 100)
        fwhm = 10.0
        result = gaussian(dx, fwhm)
        assert result.shape == dx.shape
        assert np.all(result > 0)
        assert np.all(result <= 1.0)


class TestLorentzianFunction:
    """Tests for lorentzian lineshape function."""

    def test_lorentzian_at_center(self):
        """Test Lorentzian is 1.0 at center."""
        dx = np.array([0.0])
        fwhm = 10.0
        result = lorentzian(dx, fwhm)
        assert result[0] == pytest.approx(1.0)

    def test_lorentzian_at_fwhm(self):
        """Test Lorentzian is 0.5 at FWHM."""
        fwhm = 10.0
        dx = np.array([fwhm / 2.0])
        result = lorentzian(dx, fwhm)
        assert result[0] == pytest.approx(0.5, abs=1e-10)

    def test_lorentzian_symmetry(self):
        """Test Lorentzian is symmetric."""
        dx = np.array([-5.0, 5.0])
        fwhm = 10.0
        result = lorentzian(dx, fwhm)
        assert result[0] == pytest.approx(result[1])

    def test_lorentzian_array_input(self):
        """Test Lorentzian works with array input."""
        dx = np.linspace(-20, 20, 100)
        fwhm = 10.0
        result = lorentzian(dx, fwhm)
        assert result.shape == dx.shape
        assert np.all(result > 0)
        assert np.all(result <= 1.0)

    def test_lorentzian_broader_tails_than_gaussian(self):
        """Test Lorentzian has broader tails than Gaussian."""
        dx = np.array([20.0])
        fwhm = 10.0
        lor = lorentzian(dx, fwhm)
        gauss = gaussian(dx, fwhm)
        assert lor[0] > gauss[0]


class TestPVoigtFunction:
    """Tests for pseudo-Voigt lineshape function."""

    def test_pvoigt_pure_gaussian(self):
        """Test pseudo-Voigt with eta=0 is pure Gaussian."""
        dx = np.linspace(-20, 20, 50)
        fwhm = 10.0
        pv_result = pvoigt(dx, fwhm, eta=0.0)
        gauss_result = gaussian(dx, fwhm)
        assert np.allclose(pv_result, gauss_result)

    def test_pvoigt_pure_lorentzian(self):
        """Test pseudo-Voigt with eta=1 is pure Lorentzian."""
        dx = np.linspace(-20, 20, 50)
        fwhm = 10.0
        pv_result = pvoigt(dx, fwhm, eta=1.0)
        lor_result = lorentzian(dx, fwhm)
        assert np.allclose(pv_result, lor_result)

    def test_pvoigt_mixed(self):
        """Test pseudo-Voigt with eta=0.5 is between Gaussian and Lorentzian."""
        dx = np.array([15.0])
        fwhm = 10.0
        gauss_val = gaussian(dx, fwhm)[0]
        lor_val = lorentzian(dx, fwhm)[0]
        pv_val = pvoigt(dx, fwhm, eta=0.5)[0]
        assert gauss_val < pv_val < lor_val

    def test_pvoigt_at_center(self):
        """Test pseudo-Voigt is 1.0 at center."""
        dx = np.array([0.0])
        fwhm = 10.0
        result = pvoigt(dx, fwhm, eta=0.5)
        assert result[0] == pytest.approx(1.0)


class TestNoApodFunction:
    """Tests for non-apodized lineshape function."""

    def test_no_apod_finite_output(self):
        """Test no_apod produces finite output."""
        dx = np.linspace(-50, 50, 100)
        r2 = 10.0
        aq = 0.1
        result = no_apod(dx, r2, aq)
        assert np.all(np.isfinite(result))

    def test_no_apod_with_phase(self):
        """Test no_apod with phase correction."""
        dx = np.linspace(-50, 50, 100)
        r2 = 10.0
        aq = 0.1
        result_no_phase = no_apod(dx, r2, aq, phase=0.0)
        result_with_phase = no_apod(dx, r2, aq, phase=90.0)
        # Results should be different with phase
        assert not np.allclose(result_no_phase, result_with_phase)

    def test_no_apod_array_shape(self):
        """Test no_apod preserves array shape."""
        dx = np.linspace(-50, 50, 100)
        r2 = 10.0
        aq = 0.1
        result = no_apod(dx, r2, aq)
        assert result.shape == dx.shape


class TestSp1Function:
    """Tests for SP1 apodization lineshape function."""

    def test_sp1_finite_output(self):
        """Test SP1 produces finite output."""
        dx = np.linspace(-50, 50, 100)
        r2 = 10.0
        aq = 0.1
        end = 1.0
        off = 0.35
        sp1_eval = make_sp1_evaluator(aq, end, off)
        result = sp1_eval(dx, r2)
        assert np.all(np.isfinite(result))

    def test_sp1_with_phase(self):
        """Test SP1 with phase correction."""
        dx = np.linspace(-50, 50, 100)
        r2 = 10.0
        aq = 0.1
        end = 1.0
        off = 0.35
        sp1_eval = make_sp1_evaluator(aq, end, off)
        result_no_phase = sp1_eval(dx, r2, phase=0.0)
        result_with_phase = sp1_eval(dx, r2, phase=90.0)
        # Results should be different with phase
        assert not np.allclose(result_no_phase, result_with_phase)

    def test_sp1_array_shape(self):
        """Test SP1 preserves array shape."""
        dx = np.linspace(-50, 50, 100)
        r2 = 10.0
        aq = 0.1
        end = 1.0
        off = 0.35
        sp1_eval = make_sp1_evaluator(aq, end, off)
        result = sp1_eval(dx, r2)
        assert result.shape == dx.shape


class TestSp2Function:
    """Tests for SP2 apodization lineshape function."""

    def test_sp2_finite_output(self):
        """Test SP2 produces finite output."""
        dx = np.linspace(-50, 50, 100)
        r2 = 10.0
        aq = 0.1
        end = 1.0
        off = 0.35
        sp2_eval = make_sp2_evaluator(aq, end, off)
        result = sp2_eval(dx, r2)
        assert np.all(np.isfinite(result))

    def test_sp2_with_phase(self):
        """Test SP2 with phase correction."""
        dx = np.linspace(-50, 50, 100)
        r2 = 10.0
        aq = 0.1
        end = 1.0
        off = 0.35
        sp2_eval = make_sp2_evaluator(aq, end, off)
        result_no_phase = sp2_eval(dx, r2, phase=0.0)
        result_with_phase = sp2_eval(dx, r2, phase=90.0)
        # Results should be different with phase
        assert not np.allclose(result_no_phase, result_with_phase)

    def test_sp2_array_shape(self):
        """Test SP2 preserves array shape."""
        dx = np.linspace(-50, 50, 100)
        r2 = 10.0
        aq = 0.1
        end = 1.0
        off = 0.35
        sp2_eval = make_sp2_evaluator(aq, end, off)
        result = sp2_eval(dx, r2)
        assert result.shape == dx.shape


class TestRegisteredShapes:
    """Tests for registered shapes."""

    def test_shapes_dict_exists(self):
        """Test that SHAPES dictionary exists."""
        from peakfit.core.lineshapes import SHAPES

        assert isinstance(SHAPES, dict)
        assert len(SHAPES) > 0

    def test_standard_shapes_registered(self):
        """Test that standard shapes are registered."""
        from peakfit.core.lineshapes import SHAPES

        # These shapes should be registered by default
        assert "gaussian" in SHAPES
        assert "lorentzian" in SHAPES
        assert "pvoigt" in SHAPES
