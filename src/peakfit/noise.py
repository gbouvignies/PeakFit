import numpy as np
from scipy.optimize import curve_fit

from peakfit.spectra import Spectra
from peakfit.typing import FittingOptions, FloatArray
from peakfit.ui import PeakFitUI as ui


def prepare_noise_level(clargs: FittingOptions, spectra: Spectra) -> float:
    """Prepare the noise level for fitting."""
    if clargs.noise is not None and clargs.noise < 0.0:
        clargs.noise = None

    if clargs.noise is None:
        clargs.noise = estimate_noise(spectra.data)
        ui.info(f"Estimated noise level: {clargs.noise:.2f}")

    return clargs.noise


def _gaussian(x: FloatArray, amplitude: float, sigma: float) -> FloatArray:
    """Gaussian function centered at 0.

    Args:
        x: Input array
        amplitude: Amplitude of the Gaussian
        sigma: Standard deviation (width)

    Returns:
        Gaussian function values
    """
    return amplitude * np.exp(-(x**2) / (2 * sigma**2))


def estimate_noise(data: FloatArray) -> float:
    """Estimate the noise level in the data.

    Uses a Gaussian fit to the histogram of truncated data to estimate
    the standard deviation (sigma) of the noise distribution.

    Args:
        data: Input data array

    Returns:
        Estimated noise level (sigma of the distribution)
    """
    std = np.std(data)
    truncated_data = data[np.abs(data) < std]
    y, x_edges = np.histogram(truncated_data.flatten(), bins=100)
    x = (x_edges[1:] + x_edges[:-1]) / 2

    # Initial guess: amplitude from max y, sigma from std of truncated data
    amplitude_guess = float(np.max(y))
    sigma_guess = float(np.std(truncated_data))

    # Fit Gaussian (center fixed at 0)
    popt, _ = curve_fit(
        _gaussian,
        x,
        y.astype(float),
        p0=[amplitude_guess, sigma_guess],
        bounds=([0, 0], [np.inf, np.inf]),
    )

    # Return sigma (second parameter)
    return float(popt[1])
