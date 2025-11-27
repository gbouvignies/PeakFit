"""Tests for amplitude uncertainty computation."""

from __future__ import annotations

import numpy as np
import pytest

from peakfit.core.fitting.computation import (
    calculate_amplitude_covariance,
    calculate_amplitudes,
    calculate_amplitudes_with_uncertainty,
)


class TestAmplitudeCovariance:
    """Tests for calculate_amplitude_covariance function."""

    def test_single_peak_covariance(self):
        """Test covariance for a single peak."""
        # Single peak with unit shape
        shapes = np.array([[1.0, 1.0, 1.0, 1.0]])  # (1, 4) - flat peak
        noise = 1.0

        cov = calculate_amplitude_covariance(shapes, noise)

        # Cov(a) = (S^T S)^{-1} * sigma^2
        # S^T S = 4 (sum of 1^2 over 4 points)
        # So covariance should be 1/4 * 1^2 = 0.25
        assert cov.shape == (1, 1)
        assert cov[0, 0] == pytest.approx(0.25)

    def test_two_peaks_covariance(self):
        """Test covariance for two orthogonal peaks."""
        # Two orthogonal peaks (no overlap)
        shapes = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        noise = 1.0

        cov = calculate_amplitude_covariance(shapes, noise)

        # Orthogonal shapes: covariance matrix should be diagonal
        assert cov.shape == (2, 2)
        # Each peak has only 1 non-zero point
        assert cov[0, 0] == pytest.approx(1.0)
        assert cov[1, 1] == pytest.approx(1.0)
        # Off-diagonal should be zero
        assert cov[0, 1] == pytest.approx(0.0)
        assert cov[1, 0] == pytest.approx(0.0)

    def test_noise_scaling(self):
        """Test that covariance scales with noise^2."""
        shapes = np.array([[1.0, 1.0, 1.0, 1.0]])

        cov_noise1 = calculate_amplitude_covariance(shapes, noise=1.0)
        cov_noise2 = calculate_amplitude_covariance(shapes, noise=2.0)

        # Covariance should scale with noise^2
        assert cov_noise2[0, 0] == pytest.approx(cov_noise1[0, 0] * 4)

    def test_overlapping_peaks_covariance(self):
        """Test that overlapping peaks have correlated uncertainties."""
        # Two overlapping peaks
        shapes = np.array(
            [
                [1.0, 0.5, 0.0, 0.0],
                [0.0, 0.5, 1.0, 0.0],
            ]
        )
        noise = 1.0

        cov = calculate_amplitude_covariance(shapes, noise)

        # Off-diagonal elements should be non-zero for overlapping peaks
        assert cov.shape == (2, 2)
        # Diagonal should be positive
        assert cov[0, 0] > 0
        assert cov[1, 1] > 0
        # Off-diagonal represents correlation due to overlap
        assert cov[0, 1] != 0  # Non-zero correlation


class TestAmplitudesWithUncertainty:
    """Tests for calculate_amplitudes_with_uncertainty function."""

    def test_returns_correct_amplitudes(self):
        """Test that amplitudes match basic calculation."""
        shapes = np.array(
            [
                [1.0, 0.5, 0.0],
                [0.0, 0.5, 1.0],
            ]
        )
        data = np.array([2.0, 1.5, 3.0])
        noise = 0.5

        amps, _errors, _cov = calculate_amplitudes_with_uncertainty(shapes, data, noise)
        amps_basic = calculate_amplitudes(shapes, data)

        np.testing.assert_array_almost_equal(amps, amps_basic)

    def test_returns_correct_errors(self):
        """Test that errors are sqrt of covariance diagonal."""
        shapes = np.array([[1.0, 1.0, 1.0, 1.0]])
        data = np.array([1.0, 1.0, 1.0, 1.0])
        noise = 2.0

        _amps, errors, cov = calculate_amplitudes_with_uncertainty(shapes, data, noise)

        # Error should be sqrt of covariance diagonal
        expected_errors = np.sqrt(np.diag(cov))
        np.testing.assert_array_almost_equal(errors, expected_errors)

    def test_2d_data(self):
        """Test with 2D data (multiple planes)."""
        shapes = np.array([[1.0, 0.5, 0.0]])  # (1, 3) - 1 peak, 3 points
        # Data should be (n_planes, n_points) = (2, 3)
        # But lstsq(shapes.T, data) expects shapes.T=(3, 1) and data=(3,) or (3, n_planes)
        # So data.T should be (3, 2) if we want (n_planes, n_points)
        data = np.array(
            [
                [2.0, 1.0, 0.0],  # plane 0
                [4.0, 2.0, 0.0],  # plane 1
            ]
        ).T  # Transpose to (3, 2) - (n_points, n_planes)
        noise = 1.0

        amps, errors, _cov = calculate_amplitudes_with_uncertainty(shapes, data, noise)

        # Amplitudes should be (1, 2) - one peak, two planes
        assert amps.shape == (1, 2)
        # Error shape is (1,) - same for all planes (depends only on shapes)
        assert errors.shape == (1,)


class TestAmplitudeUncertaintyIntegration:
    """Integration tests for amplitude uncertainty in fitting context."""

    def test_gaussian_peak_uncertainty(self):
        """Test uncertainty for a Gaussian-like peak."""
        # Create a Gaussian-like shape
        x = np.linspace(-2, 2, 50)
        shape = np.exp(-(x**2))  # Gaussian shape
        shapes = shape.reshape(1, -1)

        # Simulate data with known amplitude
        true_amplitude = 10.0
        data = true_amplitude * shape

        noise = 0.1

        amps, errors, _cov = calculate_amplitudes_with_uncertainty(shapes, data, noise)

        # Should recover true amplitude
        assert amps[0] == pytest.approx(true_amplitude, rel=1e-10)
        # Error should be reasonable given noise level
        assert errors[0] > 0
        assert errors[0] < noise  # Error should be reduced by having many data points

    def test_uncertainty_increases_with_noise(self):
        """Test that uncertainty increases with noise level."""
        x = np.linspace(-2, 2, 50)
        shapes = np.exp(-(x**2)).reshape(1, -1)
        data = 5.0 * shapes.flatten()

        _, errors_low, _ = calculate_amplitudes_with_uncertainty(shapes, data, noise=0.1)
        _, errors_high, _ = calculate_amplitudes_with_uncertainty(shapes, data, noise=1.0)

        assert errors_high[0] > errors_low[0]
        # Should scale linearly with noise
        assert errors_high[0] == pytest.approx(errors_low[0] * 10, rel=1e-10)
