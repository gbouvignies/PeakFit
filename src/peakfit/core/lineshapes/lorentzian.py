"""Lorentzian lineshape model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.core.lineshapes.base import BaseEvaluator, PeakShape
from peakfit.core.lineshapes.registry import register_shape

if TYPE_CHECKING:
    from peakfit.core.shared.typing import FloatArray


class LorentzianEvaluator(BaseEvaluator):
    """Lorentzian lineshape evaluator.

    L(dx) = γ² / (γ² + dx²)  where γ = fwhm/2

    Height-normalized to 1.0 at center.
    """

    def _compute_single(
        self, dx: FloatArray, fwhm: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        gamma = 0.5 * fwhm
        gamma2 = gamma * gamma
        dx2 = dx * dx
        denom = gamma2 + dx2
        lorentz = gamma2 / denom

        if not calc_derivs:
            return lorentz, None, None

        denom_inv2 = 1.0 / (denom * denom)
        d_dx = -2.0 * gamma2 * dx * denom_inv2
        d_fwhm = gamma * dx2 * denom_inv2
        return lorentz, d_dx, d_fwhm

    def integral(self, fwhm: float, j_hz: float = 0.0) -> float:
        """Analytical integral: π * fwhm / 2."""
        integral = np.pi * fwhm / 2.0
        return integral * 2.0 if j_hz != 0.0 else integral


@register_shape("lorentzian")
class Lorentzian(PeakShape):
    """Lorentzian lineshape."""

    def _create_evaluator(self) -> Any:
        return LorentzianEvaluator()


# Module-level evaluator instance
_lorentzian_evaluator = LorentzianEvaluator()


def lorentzian(dx: FloatArray, fwhm: float, j_hz: float = 0.0) -> FloatArray:
    """Evaluate Lorentzian lineshape.

    Convenience function for simple evaluation without managing evaluator instances.
    For repeated evaluations or when derivatives are needed, use LorentzianEvaluator
    directly to benefit from caching.

    Args:
        dx: Frequency offset array (Hz)
        fwhm: Full width at half maximum (Hz)
        j_hz: J-coupling constant (Hz), default 0.0

    Returns
    -------
        Lineshape values at given offsets
    """
    return _lorentzian_evaluator.evaluate(np.asarray(dx), fwhm, j_hz)
