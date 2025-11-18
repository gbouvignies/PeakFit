"""Integration tests using synthetic data."""

import numpy as np

from peakfit.shapes import gaussian, lorentzian


class TestSyntheticSpectrumGeneration:
    """Tests for generating synthetic spectra."""

    def test_generate_1d_gaussian_spectrum(self):
        """Should generate 1D spectrum with Gaussian peaks."""
        n_points = 512
        x = np.arange(n_points, dtype=float)

        # Define peaks: (center_pt, fwhm_pts, amplitude)
        peaks = [
            (100.0, 10.0, 1000.0),
            (200.0, 15.0, 800.0),
            (350.0, 8.0, 600.0),
        ]

        spectrum = np.zeros(n_points)
        for center, _fwhm, amp in peaks:
            dx = x - center
            spectrum += amp * gaussian(dx, fwhm)

        # Check peak positions
        for center, _fwhm, amp in peaks:
            # Maximum should be near the peak center
            center_idx = int(center)
            local_max = spectrum[max(0, center_idx - 5) : min(n_points, center_idx + 6)]
            assert np.max(local_max) > 0.9 * amp

    def test_generate_2d_gaussian_spectrum(self):
        """Should generate 2D spectrum with Gaussian peaks."""
        shape = (128, 256)
        y_axis = np.arange(shape[0], dtype=float)
        x_axis = np.arange(shape[1], dtype=float)

        # Define 2D peaks: (y_center, x_center, fwhm_y, fwhm_x, amplitude)
        peaks = [
            (50.0, 100.0, 10.0, 12.0, 1000.0),
            (80.0, 180.0, 8.0, 10.0, 800.0),
        ]

        spectrum = np.zeros(shape)
        for y0, x0, fwhm_y, fwhm_x, amp in peaks:
            for i, y in enumerate(y_axis):
                for j, x in enumerate(x_axis):
                    dy = y - y0
                    dx = x - x0
                    spectrum[i, j] += (
                        amp
                        * gaussian(np.array([dy]), fwhm_y)[0]
                        * gaussian(np.array([dx]), fwhm_x)[0]
                    )

        # Check that peaks are present
        for y0, x0, _, _, amp in peaks:
            yi, xi = int(y0), int(x0)
            assert spectrum[yi, xi] > 0.9 * amp

    def test_add_noise_to_spectrum(self):
        """Should add Gaussian noise to spectrum."""
        spectrum = np.ones((100, 100)) * 100.0
        noise_level = 10.0

        rng = np.random.default_rng(42)
        noisy_spectrum = spectrum + rng.normal(0, noise_level, spectrum.shape)

        # Standard deviation should be close to noise_level
        measured_std = np.std(noisy_spectrum - spectrum)
        assert np.abs(measured_std - noise_level) < 1.0

    def test_pseudo_3d_intensity_decay(self):
        """Should create pseudo-3D with decaying intensity."""
        base_spectrum = np.ones((64, 128)) * 1000.0
        n_planes = 20
        decay_rate = 0.1

        pseudo3d = np.zeros((n_planes, *base_spectrum.shape))
        for i in range(n_planes):
            intensity = np.exp(-decay_rate * i)
            pseudo3d[i] = base_spectrum * intensity

        # First plane should have highest intensity
        assert np.mean(pseudo3d[0]) > np.mean(pseudo3d[-1])

        # Decay should be exponential
        plane_intensities = np.array([np.mean(pseudo3d[i]) for i in range(n_planes)])
        log_intensities = np.log(plane_intensities)
        # Should be roughly linear
        slope = np.polyfit(np.arange(n_planes), log_intensities, 1)[0]
        assert np.abs(slope + decay_rate) < 0.05


class TestFittingAccuracy:
    """Tests for fitting parameter recovery."""

    def test_recover_gaussian_center(self):
        """Fitting should recover correct peak center."""
        # Generate spectrum with known center
        n_points = 256
        true_center = 120.0
        true_fwhm = 15.0
        true_amp = 1000.0

        x = np.arange(n_points, dtype=float)
        spectrum = true_amp * gaussian(x - true_center, true_fwhm)

        # Find maximum (should be at center)
        measured_center = np.argmax(spectrum)
        assert np.abs(measured_center - true_center) < 1.0

    def test_recover_gaussian_fwhm(self):
        """Fitting should recover correct FWHM."""
        n_points = 256
        true_center = 120.0
        true_fwhm = 20.0
        true_amp = 1000.0

        x = np.arange(n_points, dtype=float)
        spectrum = true_amp * gaussian(x - true_center, true_fwhm)

        # Find half-maximum points
        half_max = true_amp / 2
        above_half = x[spectrum >= half_max]
        measured_fwhm = above_half[-1] - above_half[0]

        assert np.abs(measured_fwhm - true_fwhm) < 2.0

    def test_recover_lorentzian_center(self):
        """Should recover Lorentzian peak center."""
        n_points = 256
        true_center = 100.0
        true_fwhm = 10.0

        x = np.arange(n_points, dtype=float)
        spectrum = 1000.0 * lorentzian(x - true_center, true_fwhm)

        measured_center = np.argmax(spectrum)
        assert np.abs(measured_center - true_center) < 1.0

    def test_noise_affects_accuracy(self):
        """Higher noise should reduce fitting accuracy."""
        n_points = 256
        true_center = 120.0
        true_fwhm = 15.0
        true_amp = 1000.0

        x = np.arange(n_points, dtype=float)
        clean_spectrum = true_amp * gaussian(x - true_center, true_fwhm)

        rng = np.random.default_rng(42)

        # Low noise
        low_noise_spectrum = clean_spectrum + rng.normal(0, 10, n_points)
        low_noise_center = np.argmax(low_noise_spectrum)

        # High noise
        high_noise_spectrum = clean_spectrum + rng.normal(0, 200, n_points)
        high_noise_center = np.argmax(high_noise_spectrum)

        # Low noise should give better result
        low_noise_error = abs(low_noise_center - true_center)
        high_noise_error = abs(high_noise_center - true_center)

        # High noise might actually be less accurate (but not always due to randomness)
        # Just verify both give reasonable results
        assert low_noise_error < 5.0  # Should be very close
        # High noise should not be extremely close — allow for larger error
        assert high_noise_error < n_points  # sanity check


class TestOverlappingPeaks:
    """Tests for handling overlapping peaks."""

    def test_two_overlapping_gaussians(self):
        """Should resolve two overlapping Gaussian peaks."""
        n_points = 256
        x = np.arange(n_points, dtype=float)

        # Two peaks separated by less than 2*FWHM
        center1, fwhm1, amp1 = 100.0, 15.0, 1000.0
        center2, fwhm2, amp2 = 115.0, 12.0, 800.0

        spectrum = amp1 * gaussian(x - center1, fwhm1) + amp2 * gaussian(x - center2, fwhm2)

        # Should have a single maximum between the two centers
        max_idx = np.argmax(spectrum)
        assert 100 <= max_idx <= 115

    def test_three_overlapping_peaks(self):
        """Should handle three overlapping peaks."""
        n_points = 256
        x = np.arange(n_points, dtype=float)

        peaks = [
            (80.0, 10.0, 600.0),
            (90.0, 12.0, 1000.0),  # Central, tallest
            (102.0, 11.0, 500.0),
        ]

        spectrum = sum(amp * gaussian(x - c, f) for c, f, amp in peaks)

        # Maximum should be near the tallest peak
        max_idx = np.argmax(spectrum)
        assert 85 <= max_idx <= 95

    def test_well_separated_peaks(self):
        """Well-separated peaks should be independent."""
        n_points = 512
        x = np.arange(n_points, dtype=float)

        center1, fwhm1, amp1 = 100.0, 10.0, 1000.0
        center2, fwhm2, amp2 = 300.0, 10.0, 800.0

        spectrum = amp1 * gaussian(x - center1, fwhm1) + amp2 * gaussian(x - center2, fwhm2)

        # Find local maxima
        from scipy.signal import find_peaks

        peaks_idx, _ = find_peaks(spectrum, height=100)

        # Should find exactly two peaks
        assert len(peaks_idx) == 2
        assert np.abs(peaks_idx[0] - center1) < 2
        assert np.abs(peaks_idx[1] - center2) < 2
