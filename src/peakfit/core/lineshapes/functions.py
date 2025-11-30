"""Pure NumPy lineshape functions for NMR peak fitting.

This module provides vectorized NumPy implementations of common NMR lineshapes.
All functions are optimized using NumPy broadcasting and vectorization for performance.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from peakfit.core.shared.typing import FloatArray

# =============================================================================
# Simple lineshapes
# =============================================================================


def gaussian(dx: FloatArray, fwhm: float) -> FloatArray:
    """Gaussian lineshape (normalized to 1 at center).

    Args:
        dx: Frequency offset from peak center (Hz)
        fwhm: Full width at half maximum (Hz)

    Returns
    -------
        Gaussian profile values

    Notes
    -----
        Uses vectorized NumPy operations for performance.
    """
    c = 4.0 * np.log(2.0) / (fwhm * fwhm)
    return np.exp(-dx * dx * c)


def lorentzian(dx: FloatArray, fwhm: float) -> FloatArray:
    """Lorentzian lineshape (normalized to 1 at center).

    Args:
        dx: Frequency offset from peak center (Hz)
        fwhm: Full width at half maximum (Hz)

    Returns
    -------
        Lorentzian profile values
    """
    half_width_sq = (0.5 * fwhm) ** 2
    return half_width_sq / (dx * dx + half_width_sq)


def pvoigt(dx: FloatArray, fwhm: float, eta: float) -> FloatArray:
    """Pseudo-Voigt lineshape (mixture of Gaussian and Lorentzian).

    Args:
        dx: Frequency offset from peak center (Hz)
        fwhm: Full width at half maximum (Hz)
        eta: Lorentzian fraction (0=pure Gaussian, 1=pure Lorentzian)

    Returns
    -------
        Pseudo-Voigt profile values
    """
    dx_sq = dx * dx
    c_gauss = 4.0 * np.log(2.0) / (fwhm * fwhm)
    gauss = np.exp(-dx_sq * c_gauss)
    half_width_sq = (0.5 * fwhm) ** 2
    lorentz = half_width_sq / (dx_sq + half_width_sq)
    return (1.0 - eta) * gauss + eta * lorentz


# =============================================================================
# Apodization lineshape evaluator factories
#
# These factory functions return closures with pre-computed exponentials.
# Parameters `aq`, `end`, `off` are spectrum/dimension-specific and don't
# change during fitting, so we compute their exponentials once upfront.
# =============================================================================

# Type alias for evaluator functions
Evaluator = Callable[[FloatArray, float, float], FloatArray]


def no_apod(dx: FloatArray, r2: float, aq: float, phase: float = 0.0) -> FloatArray:
    """Non-apodized FID-based lineshape.

    Args:
        dx: Frequency offset from peak center (rad/s)
        r2: Transverse relaxation rate (Hz)
        aq: Acquisition time (s)
        phase: Phase correction (degrees)

    Returns
    -------
        Real part of lineshape after phase correction

    Notes
    -----
        This represents the Fourier transform of a non-apodized
        exponentially decaying FID.
    """
    z1 = aq * (1j * dx + r2)
    exp_nz1 = np.exp(-z1)
    spec = aq * (1.0 - exp_nz1) / z1
    if phase != 0.0:
        spec = spec * np.exp(1j * np.deg2rad(phase))
    return spec.real


def make_sp1_evaluator(aq: float, end: float, off: float) -> Evaluator:
    """Create an SP1 (sine bell) apodization lineshape evaluator.

    Pre-computes exponentials that depend on aq, end, off (static during fitting).

    Args:
        aq: Acquisition time (s)
        end: End parameter for sine bell
        off: Offset parameter for sine bell

    Returns
    -------
        Evaluator function: (dx, r2, phase) -> lineshape
    """
    f1 = 1j * off * np.pi
    f2 = 1j * (end - off) * np.pi
    exp_pf1 = np.exp(f1)
    exp_nf1 = 1.0 / exp_pf1
    exp_pf2 = np.exp(f2)
    exp_nf2 = 1.0 / exp_pf2

    def evaluate(dx: FloatArray, r2: float, phase: float = 0.0) -> FloatArray:
        """Evaluate SP1 lineshape.

        Args:
            dx: Frequency offset from peak center (rad/s)
            r2: Transverse relaxation rate (Hz)
            phase: Phase correction (degrees)

        Returns
        -------
            Real part of lineshape after phase correction
        """
        z1 = aq * (1j * dx + r2)
        exp_pz1 = np.exp(z1)
        exp_nz1 = 1.0 / exp_pz1
        a1 = (exp_pf2 - exp_pz1) * exp_nz1 * exp_pf1 / (2 * (z1 - f2))
        a2 = (exp_pz1 - exp_nf2) * exp_nz1 * exp_nf1 / (2 * (z1 + f2))
        spec = 1j * aq * (a1 + a2)
        if phase != 0.0:
            spec = spec * np.exp(1j * np.deg2rad(phase))
        return spec.real

    return evaluate


def make_sp2_evaluator(aq: float, end: float, off: float) -> Evaluator:
    """Create an SP2 (sine squared bell) apodization lineshape evaluator.

    Pre-computes exponentials that depend on aq, end, off (static during fitting).

    Args:
        aq: Acquisition time (s)
        end: End parameter for sine bell
        off: Offset parameter for sine bell

    Returns
    -------
        Evaluator function: (dx, r2, phase) -> lineshape
    """
    two_f1 = 2j * off * np.pi
    two_f2 = 2j * (end - off) * np.pi
    exp_p2f1 = np.exp(two_f1)
    exp_n2f1 = 1.0 / exp_p2f1
    exp_p2f2 = np.exp(two_f2)
    exp_n2f2 = 1.0 / exp_p2f2

    def evaluate(dx: FloatArray, r2: float, phase: float = 0.0) -> FloatArray:
        """Evaluate SP2 lineshape.

        Args:
            dx: Frequency offset from peak center (rad/s)
            r2: Transverse relaxation rate (Hz)
            phase: Phase correction (degrees)

        Returns
        -------
            Real part of lineshape after phase correction
        """
        z1 = aq * (1j * dx + r2)
        exp_pz1 = np.exp(z1)
        exp_nz1 = 1.0 / exp_pz1
        a1 = (exp_p2f2 - exp_pz1) * exp_nz1 * exp_p2f1 / (4 * (z1 - two_f2))
        a2 = (exp_n2f2 - exp_pz1) * exp_nz1 * exp_n2f1 / (4 * (z1 + two_f2))
        a3 = (1.0 - exp_nz1) / (2 * z1)
        spec = aq * (a1 + a2 + a3)
        if phase != 0.0:
            spec = spec * np.exp(1j * np.deg2rad(phase))
        return spec.real

    return evaluate
