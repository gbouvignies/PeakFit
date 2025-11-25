import numpy as np
from scipy.optimize import curve_fit

from peakfit.core.domain.spectrum import Spectra
from peakfit.core.shared.typing import FittingOptions, FloatArray


def prepare_noise_level(clargs: FittingOptions, spectra: Spectra) -> float:
    """Prepare the noise level for fitting."""
    if clargs.noise is not None and clargs.noise <= 0.0:
        clargs.noise = None

    if clargs.noise is None:
        clargs.noise = estimate_noise(spectra.data)

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


def _mad_sigma(values: FloatArray) -> float:
    """Robust noise estimate using the MAD heuristic."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        return 0.0
    mad_to_sigma = 1 / 0.6744897501960817  # Approximation for normal distribution
    return mad * mad_to_sigma


def estimate_noise(data: FloatArray) -> float:
    """Estimate the noise level in the data.

    Primary estimate fits a zero-centered Gaussian to a histogram of truncated
    data. If the fit fails or yields a non-positive sigma, fall back to a robust
    MAD-based estimate to avoid reporting zero noise levels.
    """
    flattened = np.asarray(data, dtype=float).ravel()
    if flattened.size == 0:
        return 0.0

    std = float(np.std(flattened))
    if std == 0.0:
        return 0.0

    truncated = flattened[np.abs(flattened) < std]
    if truncated.size < 10:
        truncated = flattened

    hist, x_edges = np.histogram(truncated, bins=100)
    if not np.any(hist):
        return _mad_sigma(flattened)

    x = (x_edges[1:] + x_edges[:-1]) / 2
    amplitude_guess = float(np.max(hist))
    sigma_guess = float(np.std(truncated)) or std

    try:
        popt, _ = curve_fit(
            _gaussian,
            x,
            hist.astype(float),
            p0=[amplitude_guess, sigma_guess],
            bounds=([0, 0], [np.inf, np.inf]),
        )
        sigma = float(popt[1])
        if sigma > 0:
            return sigma
    except Exception:  # pragma: no cover - scipy may raise RuntimeError/ValueError
        pass

    fallback = _mad_sigma(flattened)
    return fallback if fallback > 0 else std
