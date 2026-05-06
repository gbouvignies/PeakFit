"""Noise estimation algorithms."""

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.shared.typing import FloatArray


def prepare_noise_level(noise_level: float | None, spectra: Spectra) -> float:
    """Prepare the noise level for fitting.

    If noise_level is provided and positive, it is returned.
    Otherwise, noise is estimated from the spectra data.
    """
    if noise_level is not None and noise_level <= 0.0:
        noise_level = None

    if noise_level is None:
        noise_level = estimate_noise(spectra.data)

    return noise_level


def _gaussian(x: FloatArray, amplitude: float, sigma: float) -> np.ndarray:
    """Gaussian function centered at 0."""
    return cast("np.ndarray", amplitude * np.exp(-(x**2) / (2 * sigma**2)))


def _mad_sigma(values: FloatArray) -> float:
    """Robust noise estimate using the MAD heuristic."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        return 0.0
    # Sigma approx = 1.4826 * MAD
    return mad * 1.4825796886589388


def estimate_noise(data: np.ndarray) -> float:
    """Estimate the noise level in the data.

    Primary estimate fits a zero-centered Gaussian to a histogram of truncated
    data. If the fit fails or yields a non-positive sigma, fall back to a robust
    MAD-based estimate.
    """
    flattened = np.asarray(data, dtype=np.float64).ravel()
    if flattened.size == 0:
        return 0.0

    std = float(np.std(flattened))
    if std == 0.0:
        return 0.0

    min_truncated_samples = 10

    # Truncate high intensity signals to isolate noise
    truncated = flattened[np.abs(flattened) < std]
    if truncated.size < min_truncated_samples:
        # If nearly everything is signal, just use everything (heuristic fallback)
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
    except (RuntimeError, ValueError) as exc:
        logging.debug("Gaussian fit for noise estimation failed: %s", exc)

    fallback = _mad_sigma(flattened)
    return fallback if fallback > 0 else std
