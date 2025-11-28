"""Pytest fixtures for PeakFit tests."""

import pytest

import numpy as np


@pytest.fixture
def sample_1d_spectrum():
    """Generate a simple 1D spectrum with known peaks."""
    x = np.arange(512)
    spectrum = np.zeros(512)

    # Add two Gaussian peaks
    center1, fwhm1, amp1 = 100, 10, 1000
    center2, fwhm2, amp2 = 150, 15, 800

    spectrum += amp1 * np.exp(-((x - center1) ** 2) * 4 * np.log(2) / fwhm1**2)
    spectrum += amp2 * np.exp(-((x - center2) ** 2) * 4 * np.log(2) / fwhm2**2)

    # Add noise
    rng = np.random.default_rng(42)
    spectrum += rng.normal(0, 10, 512)

    return spectrum, [(center1, fwhm1, amp1), (center2, fwhm2, amp2)]


@pytest.fixture
def sample_2d_spectrum():
    """Generate a 2D spectrum with known peaks."""
    shape = (128, 256)
    spectrum = np.zeros(shape)

    # Add peaks
    peaks = [
        (50, 100, 10, 12, 1000),  # y, x, fwhm_y, fwhm_x, amp
        (70, 150, 8, 10, 800),
        (52, 102, 11, 13, 600),  # Overlapping with first peak
    ]

    for y0, x0, fwhm_y, fwhm_x, amp in peaks:
        y, x = np.ogrid[: shape[0], : shape[1]]
        gaussian_2d = np.exp(
            -((y - y0) ** 2) * 4 * np.log(2) / fwhm_y**2
            - ((x - x0) ** 2) * 4 * np.log(2) / fwhm_x**2
        )
        spectrum += amp * gaussian_2d

    # Add noise
    rng = np.random.default_rng(42)
    spectrum += rng.normal(0, 10, shape)

    return spectrum, peaks


@pytest.fixture
def sample_pseudo3d_spectrum(sample_2d_spectrum):
    """Generate a pseudo-3D spectrum (multiple 2D planes)."""
    base_spectrum, peaks = sample_2d_spectrum
    n_planes = 10

    # Create pseudo-3D by stacking with varying intensities
    pseudo3d = np.zeros((n_planes, *base_spectrum.shape))
    intensities = np.linspace(1.0, 0.5, n_planes)  # Decay

    for i, intensity in enumerate(intensities):
        pseudo3d[i] = base_spectrum * intensity

    return pseudo3d, peaks, intensities


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_peaklist_file(tmp_path):
    """Create a sample Sparky peak list file."""
    peaklist_path = tmp_path / "peaks.list"
    content = """# Sparky peak list
Assignment  w1   w2
Peak1  8.50  120.5
Peak2  7.80  115.3
Peak3  8.52  120.8
"""
    peaklist_path.write_text(content)
    return peaklist_path


@pytest.fixture
def sample_config_file(tmp_path):
    """Create a sample TOML configuration file."""
    config_path = tmp_path / "peakfit.toml"
    content = """
[fitting]
lineshape = "gaussian"
refine_iterations = 2
fix_positions = false

[clustering]
contour_factor = 5.0

[output]
directory = "Results"
formats = ["txt", "csv"]
"""
    config_path.write_text(content)
    return config_path
