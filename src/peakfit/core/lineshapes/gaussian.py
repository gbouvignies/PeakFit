"""Gaussian lineshape model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.core.lineshapes.base import BaseEvaluator, PeakShape
from peakfit.core.lineshapes.registry import register_shape
from peakfit.core.lineshapes.utils import _LN2, _SQRT_PI_4LN2

if TYPE_CHECKING:
    from peakfit.core.shared.typing import FloatArray


class GaussianEvaluator(BaseEvaluator):
    """Gaussian lineshape evaluator.

    G(dx) = exp(-c * dx²)  where c = 4*ln(2) / fwhm²

    Height-normalized to 1.0 at center.
    """

    def _compute_single(
        self, dx: FloatArray, fwhm: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        c = 4.0 * _LN2 / (fwhm * fwhm)
        dx2 = dx * dx
        gauss = np.exp(-c * dx2)

        if not calc_derivs:
            return gauss, None, None

        c2 = 2.0 * c
        d_dx = -c2 * dx * gauss
        d_fwhm = (c2 / fwhm) * dx2 * gauss
        return gauss, d_dx, d_fwhm

    def integral(self, fwhm: float, j_hz: float = 0.0) -> float:
        """Analytical integral: fwhm * sqrt(π / (4*ln(2)))."""
        integral = fwhm * _SQRT_PI_4LN2
        return integral * 2.0 if j_hz != 0.0 else integral


@register_shape("gaussian")
class Gaussian(PeakShape):
    """Gaussian lineshape."""

    def _create_evaluator(self) -> Any:
        return GaussianEvaluator()


# Module-level evaluator instance
_gaussian_evaluator = GaussianEvaluator()


def gaussian(dx: FloatArray, fwhm: float, j_hz: float = 0.0) -> FloatArray:
    """Evaluate Gaussian lineshape.

    Convenience function for simple evaluation without managing evaluator instances.
    For repeated evaluations or when derivatives are needed, use GaussianEvaluator
    directly to benefit from caching.

    Args:
        dx: Frequency offset array (Hz)
        fwhm: Full width at half maximum (Hz)
        j_hz: J-coupling constant (Hz), default 0.0

    Returns
    -------
        Lineshape values at given offsets
    """
    return _gaussian_evaluator.evaluate(np.asarray(dx), fwhm, j_hz)
