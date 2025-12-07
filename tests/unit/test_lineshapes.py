"""Test lineshape functions."""

import numpy as np
import pytest

from peakfit.core.lineshapes import (
    GaussianEvaluator,
    LorentzianEvaluator,
    NoApodEvaluator,
    PseudoVoigtEvaluator,
    SP1Evaluator,
    SP2Evaluator,
)


class TestGaussian:
    """Tests for Gaussian lineshape."""

    def setup_method(self):
        self.evaluator = GaussianEvaluator()

    def test_gaussian_peak_height(self):
        """Gaussian should be 1.0 at center."""
        dx = np.array([0.0])
        fwhm = 10.0
        result = self.evaluator.evaluate(dx, fwhm)
        assert result[0] == pytest.approx(1.0)
        assert result[0] == pytest.approx(1.0)

    def test_gaussian_half_maximum(self):
        """Gaussian should be 0.5 at FWHM/2 from center."""
        fwhm = 10.0
        dx = np.array([fwhm / 2])
        result = self.evaluator.evaluate(dx, fwhm)
        assert result[0] == pytest.approx(0.5, rel=1e-6)

    def test_gaussian_symmetry(self):
        """Gaussian should be symmetric around center."""
        dx = np.array([-5.0, 5.0])
        fwhm = 10.0
        result = self.evaluator.evaluate(dx, fwhm)
        assert result[0] == pytest.approx(result[1])


class TestLorentzian:
    """Tests for Lorentzian lineshape."""

    def setup_method(self):
        self.evaluator = LorentzianEvaluator()

    def test_lorentzian_peak_height(self):
        """Lorentzian should be 1.0 at center."""
        dx = np.array([0.0])
        fwhm = 10.0
        result = self.evaluator.evaluate(dx, fwhm)
        assert result[0] == pytest.approx(1.0)
        assert result[0] == pytest.approx(1.0)

    def test_lorentzian_half_maximum(self):
        """Lorentzian should be 0.5 at FWHM/2 from center."""
        fwhm = 10.0
        dx = np.array([fwhm / 2])
        result = self.evaluator.evaluate(dx, fwhm)
        assert result[0] == pytest.approx(0.5)

    def test_lorentzian_symmetry(self):
        """Lorentzian should be symmetric around center."""
        dx = np.array([-5.0, 5.0])
        fwhm = 10.0
        result = self.evaluator.evaluate(dx, fwhm)
        assert result[0] == pytest.approx(result[1])


class TestPseudoVoigt:
    """Tests for Pseudo-Voigt lineshape."""

    def setup_method(self):
        self.evaluator = PseudoVoigtEvaluator()

    def test_pvoigt_pure_gaussian(self):
        """Pseudo-Voigt with eta=0 should be Gaussian."""
        dx = np.linspace(-20, 20, 41)
        fwhm = 10.0
        eta = 0.0
        pv = self.evaluator.evaluate(dx, fwhm, eta)
        g = GaussianEvaluator().evaluate(dx, fwhm)
        np.testing.assert_allclose(pv, g)

    def test_pvoigt_pure_lorentzian(self):
        """Pseudo-Voigt with eta=1 should be Lorentzian."""
        dx = np.linspace(-20, 20, 41)
        fwhm = 10.0
        eta = 1.0
        pv = self.evaluator.evaluate(dx, fwhm, eta)
        lor = LorentzianEvaluator().evaluate(dx, fwhm)
        np.testing.assert_allclose(pv, lor)

    def test_pvoigt_peak_height(self):
        """Pseudo-Voigt should be 1.0 at center for any eta."""
        dx = np.array([0.0])
        fwhm = 10.0
        for eta in [0.0, 0.3, 0.5, 0.7, 1.0]:
            result = self.evaluator.evaluate(dx, fwhm, eta)
            assert result[0] == pytest.approx(1.0)
            assert result[0] == pytest.approx(1.0)


class TestNoApod:
    """Tests for no-apodization lineshape."""

    def test_no_apod_peak_height(self):
        """NoApod should be approximately 1.0 at center."""
        dx = np.array([0.0])
        r2 = 10.0
        aq = 0.1
        evaluator = NoApodEvaluator(aq)
        result = evaluator.evaluate(dx, r2)
        # At center, should be close to aq / (aq * r2) for small r2
        # val = (1 - exp(-aq*r2)) / r2
        val = (1.0 - np.exp(-aq * r2)) / r2
        assert result[0] == pytest.approx(val)

    def test_no_apod_phase_rotation(self):
        """NoApod with phase should rotate peak."""
        dx = np.array([0.0, 5.0])
        r2 = 10.0
        aq = 0.1
        evaluator = NoApodEvaluator(aq)
        phase0 = evaluator.evaluate(dx, r2, phase=0.0)
        phase90 = evaluator.evaluate(dx, r2, phase=90.0)
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
        evaluator = SP1Evaluator(aq, end, off)
        result = evaluator.evaluate(dx, r2)
        assert result.shape == dx.shape

    def test_sp1_real_output(self):
        """SP1 should return real values."""
        dx = np.linspace(-50, 50, 101)
        r2 = 10.0
        aq = 0.1
        end = 0.5
        off = 0.1
        evaluator = SP1Evaluator(aq, end, off)
        result = evaluator.evaluate(dx, r2)
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
        evaluator = SP2Evaluator(aq, end, off)
        result = evaluator.evaluate(dx, r2)
        assert result.shape == dx.shape

    def test_sp2_real_output(self):
        """SP2 should return real values."""
        dx = np.linspace(-50, 50, 101)
        r2 = 10.0
        aq = 0.1
        end = 0.5
        off = 0.1
        evaluator = SP2Evaluator(aq, end, off)
        result = evaluator.evaluate(dx, r2)
        assert np.all(np.isreal(result))
