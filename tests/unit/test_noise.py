"""Tests for noise estimation module."""

import numpy as np

from peakfit.data.noise import estimate_noise


class TestEstimateNoise:
    """Tests for noise estimation function."""

    def test_estimate_noise_with_pure_noise(self):
        """Test noise estimation with pure Gaussian noise."""
        rng = np.random.default_rng(42)
        noise_level = 5.0
        data = rng.normal(0, noise_level, 10000)

        estimated = estimate_noise(data)
        # Should be within 20% of true noise level for large sample
        assert 4.0 < estimated < 6.0

    def test_estimate_noise_with_signal_and_noise(self):
        """Test noise estimation with signal + noise."""
        rng = np.random.default_rng(42)
        noise_level = 2.0

        # Create signal + noise
        x = np.linspace(-50, 50, 10000)
        signal = 100 * np.exp(-(x**2) / 100)  # Large peak in center
        noise = rng.normal(0, noise_level, 10000)
        data = signal + noise

        # Truncation should remove signal, giving good noise estimate
        estimated = estimate_noise(data)
        assert 1.5 < estimated < 3.0

    def test_estimate_noise_reproducible(self):
        """Test that noise estimation is reproducible."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 3.0, 5000)

        est1 = estimate_noise(data)
        est2 = estimate_noise(data)

        assert est1 == est2

    def test_estimate_noise_positive_result(self):
        """Test that noise estimation always returns positive value."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1.0, 1000)

        estimated = estimate_noise(data)
        assert estimated > 0

    def test_estimate_noise_different_distributions(self):
        """Test noise estimation with different noise levels."""
        rng = np.random.default_rng(42)

        for noise_level in [1.0, 5.0, 10.0]:
            data = rng.normal(0, noise_level, 10000)
            estimated = estimate_noise(data)
            # Within 30% of true value
            assert noise_level * 0.7 < estimated < noise_level * 1.3
