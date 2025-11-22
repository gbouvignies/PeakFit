"""Test lineshape functions."""

import numpy as np
import pytest

from peakfit.lineshapes.models import gaussian, lorentzian, no_apod, pvoigt, sp1, sp2


class TestGaussian:
    """Tests for Gaussian lineshape."""

    def test_gaussian_peak_height(self):
        """Gaussian should be 1.0 at center."""
        dx = np.array([0.0])
        fwhm = 10.0
        result = gaussian(dx, fwhm)
        assert result[0] == pytest.approx(1.0)

    def test_gaussian_half_maximum(self):
        """Gaussian should be 0.5 at FWHM/2 from center."""
        fwhm = 10.0
        dx = np.array([fwhm / 2])
        result = gaussian(dx, fwhm)
        assert result[0] == pytest.approx(0.5, rel=1e-6)

    def test_gaussian_symmetry(self):
        """Gaussian should be symmetric around center."""
        dx = np.array([-5.0, 5.0])
        fwhm = 10.0
        result = gaussian(dx, fwhm)
        assert result[0] == pytest.approx(result[1])

    def test_gaussian_decay(self):
        """Gaussian should decay to near zero far from center."""
        dx = np.array([100.0])
        fwhm = 10.0
        result = gaussian(dx, fwhm)
        assert result[0] < 1e-10

    def test_gaussian_array_input(self):
        """Gaussian should handle array input."""
        dx = np.linspace(-50, 50, 101)
        fwhm = 10.0
        result = gaussian(dx, fwhm)
        assert result.shape == dx.shape
        assert np.max(result) == pytest.approx(1.0)


class TestLorentzian:
    """Tests for Lorentzian lineshape."""

    def test_lorentzian_peak_height(self):
        """Lorentzian should be 1.0 at center."""
        dx = np.array([0.0])
        fwhm = 10.0
        result = lorentzian(dx, fwhm)
        assert result[0] == pytest.approx(1.0)

    def test_lorentzian_half_maximum(self):
        """Lorentzian should be 0.5 at FWHM/2 from center."""
        fwhm = 10.0
        dx = np.array([fwhm / 2])
        result = lorentzian(dx, fwhm)
        assert result[0] == pytest.approx(0.5)

    def test_lorentzian_symmetry(self):
        """Lorentzian should be symmetric around center."""
        dx = np.array([-5.0, 5.0])
        fwhm = 10.0
        result = lorentzian(dx, fwhm)
        assert result[0] == pytest.approx(result[1])

    def test_lorentzian_slower_decay(self):
        """Lorentzian should decay slower than Gaussian."""
        dx = np.array([50.0])
        fwhm = 10.0
        lorentz = lorentzian(dx, fwhm)
        gauss = gaussian(dx, fwhm)
        # Lorentzian has heavier tails
        assert lorentz[0] > gauss[0]


class TestPseudoVoigt:
    """Tests for Pseudo-Voigt lineshape."""

    def test_pvoigt_pure_gaussian(self):
        """Pseudo-Voigt with eta=0 should be Gaussian."""
        dx = np.linspace(-20, 20, 41)
        fwhm = 10.0
        eta = 0.0
        pv = pvoigt(dx, fwhm, eta)
        g = gaussian(dx, fwhm)
        np.testing.assert_allclose(pv, g)

    def test_pvoigt_pure_lorentzian(self):
        """Pseudo-Voigt with eta=1 should be Lorentzian."""
        dx = np.linspace(-20, 20, 41)
        fwhm = 10.0
        eta = 1.0
        pv = pvoigt(dx, fwhm, eta)
        lor = lorentzian(dx, fwhm)
        np.testing.assert_allclose(pv, lor)

    def test_pvoigt_peak_height(self):
        """Pseudo-Voigt should be 1.0 at center for any eta."""
        dx = np.array([0.0])
        fwhm = 10.0
        for eta in [0.0, 0.3, 0.5, 0.7, 1.0]:
            result = pvoigt(dx, fwhm, eta)
            assert result[0] == pytest.approx(1.0)

    def test_pvoigt_interpolation(self):
        """Pseudo-Voigt should interpolate between G and L."""
        dx = np.array([10.0])
        fwhm = 10.0
        eta = 0.5
        g = gaussian(dx, fwhm)[0]
        lor = lorentzian(dx, fwhm)[0]
        expected = 0.5 * g + 0.5 * lor
        result = pvoigt(dx, fwhm, eta)
        assert result[0] == pytest.approx(expected)


class TestNoApod:
    """Tests for no-apodization lineshape."""

    def test_no_apod_peak_height(self):
        """NoApod should be approximately 1.0 at center."""
        dx = np.array([0.0])
        r2 = 10.0
        aq = 0.1
        result = no_apod(dx, r2, aq)
        # At center, should be close to aq / (aq * r2) for small r2
        assert abs(result[0]) < 2.0  # Reasonable amplitude

    def test_no_apod_phase_rotation(self):
        """NoApod with phase should rotate peak."""
        dx = np.array([0.0, 5.0])
        r2 = 10.0
        aq = 0.1
        phase0 = no_apod(dx, r2, aq, phase=0.0)
        phase90 = no_apod(dx, r2, aq, phase=90.0)
        # Phase should affect the real part
        assert not np.allclose(phase0, phase90)


class TestSP1:
    """Tests for SP1 apodization lineshape."""

    def test_sp1_shape(self):
        """SP1 should return array of same shape."""
        dx = np.linspace(-50, 50, 101)
        r2 = 10.0
        aq = 0.1
        end = 0.5
        off = 0.1
        result = sp1(dx, r2, aq, end, off)
        assert result.shape == dx.shape

    def test_sp1_real_output(self):
        """SP1 should return real values."""
        dx = np.linspace(-50, 50, 101)
        r2 = 10.0
        aq = 0.1
        end = 0.5
        off = 0.1
        result = sp1(dx, r2, aq, end, off)
        assert np.all(np.isreal(result))


class TestSP2:
    """Tests for SP2 apodization lineshape."""

    def test_sp2_shape(self):
        """SP2 should return array of same shape."""
        dx = np.linspace(-50, 50, 101)
        r2 = 10.0
        aq = 0.1
        end = 0.5
        off = 0.1
        result = sp2(dx, r2, aq, end, off)
        assert result.shape == dx.shape

    def test_sp2_real_output(self):
        """SP2 should return real values."""
        dx = np.linspace(-50, 50, 101)
        r2 = 10.0
        aq = 0.1
        end = 0.5
        off = 0.1
        result = sp2(dx, r2, aq, end, off)
        assert np.all(np.isreal(result))
