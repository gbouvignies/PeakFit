"""Pure NumPy lineshape functions for NMR peak fitting.

This module provides vectorized NumPy implementations of common NMR lineshapes.
All functions are optimized using NumPy broadcasting and vectorization for performance.
"""

from typing import cast

import numpy as np

from peakfit.core.shared.typing import FloatArray


def gaussian(dx: FloatArray, fwhm: float) -> FloatArray:
    """Gaussian lineshape (normalized to 1 at center).

    Args:
        dx: Frequency offset from peak center (Hz)
        fwhm: Full width at half maximum (Hz)

    Returns:
        Gaussian profile values

    Notes:
        Uses vectorized NumPy operations for performance.
        Optimized constant calculation avoids repeated computation.
    """
    c = 4.0 * np.log(2.0) / (fwhm * fwhm)
    return cast(FloatArray, np.asarray(np.exp(-dx * dx * c), dtype=float))


def lorentzian(dx: FloatArray, fwhm: float) -> FloatArray:
    """Lorentzian lineshape (normalized to 1 at center).

    Args:
        dx: Frequency offset from peak center (Hz)
        fwhm: Full width at half maximum (Hz)

    Returns:
        Lorentzian profile values
    """
    half_width_sq = (0.5 * fwhm) ** 2
    return cast(FloatArray, np.asarray(half_width_sq / (dx * dx + half_width_sq), dtype=float))


def pvoigt(dx: FloatArray, fwhm: float, eta: float) -> FloatArray:
    """Pseudo-Voigt lineshape (mixture of Gaussian and Lorentzian).

    Args:
        dx: Frequency offset from peak center (Hz)
        fwhm: Full width at half maximum (Hz)
        eta: Lorentzian fraction (0=pure Gaussian, 1=pure Lorentzian)

    Returns:
        Pseudo-Voigt profile values
    """
    c_gauss = 4.0 * np.log(2.0) / (fwhm * fwhm)
    gauss = np.exp(-dx * dx * c_gauss)
    half_width_sq = (0.5 * fwhm) ** 2
    lorentz = half_width_sq / (dx * dx + half_width_sq)
    return cast(FloatArray, np.asarray((1.0 - eta) * gauss + eta * lorentz, dtype=float))


def no_apod(dx: FloatArray, r2: float, aq: float, phase: float = 0.0) -> FloatArray:
    """Non-apodized FID-based lineshape.

    Args:
        dx: Frequency offset from peak center (Hz)
        r2: Transverse relaxation rate (Hz)
        aq: Acquisition time (s)
        phase: Phase correction (degrees)

    Returns:
        Real part of lineshape after phase correction

    Notes:
        This represents the Fourier transform of a non-apodized
        exponentially decaying FID.
    """
    z1 = aq * (1j * dx + r2)
    spec = aq * (1.0 - np.exp(-z1)) / z1
    return cast(FloatArray, np.asarray((spec * np.exp(1j * np.deg2rad(phase))).real, dtype=float))


def sp1(
    dx: FloatArray, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> FloatArray:
    """SP1 (sine bell) apodization lineshape.

    Args:
        dx: Frequency offset from peak center (Hz)
        r2: Transverse relaxation rate (Hz)
        aq: Acquisition time (s)
        end: End parameter for sine bell
        off: Offset parameter for sine bell
        phase: Phase correction (degrees)

    Returns:
        Real part of lineshape after phase correction
    """
    z1 = aq * (1j * dx + r2)
    f1, f2 = 1j * off * np.pi, 1j * (end - off) * np.pi
    a1 = (np.exp(+f2) - np.exp(+z1)) * np.exp(-z1 + f1) / (2 * (z1 - f2))
    a2 = (np.exp(+z1) - np.exp(-f2)) * np.exp(-z1 - f1) / (2 * (z1 + f2))
    spec = 1j * aq * (a1 + a2)
    return cast(FloatArray, np.asarray((spec * np.exp(1j * np.deg2rad(phase))).real, dtype=float))


def sp2(
    dx: FloatArray, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> FloatArray:
    """SP2 (sine squared bell) apodization lineshape.

    Args:
        dx: Frequency offset from peak center (Hz)
        r2: Transverse relaxation rate (Hz)
        aq: Acquisition time (s)
        end: End parameter for sine bell
        off: Offset parameter for sine bell
        phase: Phase correction (degrees)

    Returns:
        Real part of lineshape after phase correction
    """
    z1 = aq * (1j * dx + r2)
    f1, f2 = 1j * off * np.pi, 1j * (end - off) * np.pi
    a1 = (np.exp(+2 * f2) - np.exp(z1)) * np.exp(-z1 + 2 * f1) / (4 * (z1 - 2 * f2))
    a2 = (np.exp(-2 * f2) - np.exp(z1)) * np.exp(-z1 - 2 * f1) / (4 * (z1 + 2 * f2))
    a3 = (1.0 - np.exp(-z1)) / (2 * z1)
    spec = aq * (a1 + a2 + a3)
    return cast(FloatArray, np.asarray((spec * np.exp(1j * np.deg2rad(phase))).real, dtype=float))
