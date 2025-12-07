"""Gaussian lineshape model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.core.lineshapes.base import BaseEvaluator, PeakShape
from peakfit.core.lineshapes.registry import register_shape
from peakfit.core.lineshapes.utils import _LN2

if TYPE_CHECKING:
    from peakfit.core.shared.typing import FloatArray


class GaussianEvaluator(BaseEvaluator):
    """Gaussian lineshape: G(dx) = exp(-c * dx²) where c = 4*ln(2) / fwhm²."""

    def _compute_single(
        self, dx: FloatArray, fwhm: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        c = 4.0 * _LN2 / (fwhm * fwhm)
        dx_arr = np.asarray(dx)
        dx2 = dx_arr * dx_arr
        gauss = np.exp(-c * dx2)

        # Handle non-finite inputs explicitly: NaN stays NaN, ±inf -> 0
        mask_nan = np.isnan(dx_arr)
        mask_inf = np.isinf(dx_arr)
        if mask_nan.any() or mask_inf.any():
            gauss = np.where(mask_inf, 0.0, gauss)
            gauss = np.where(mask_nan, np.nan, gauss)

        if not calc_derivs:
            return gauss, None, None

        c2 = 2.0 * c
        d_dx = np.zeros_like(gauss)
        d_fwhm = np.zeros_like(gauss)
        mask_finite = ~(mask_nan | mask_inf)
        if mask_finite.any():
            d_dx_finite = -c2 * dx_arr[mask_finite] * gauss[mask_finite]
            d_fwhm_finite = (c2 / fwhm) * dx2[mask_finite] * gauss[mask_finite]
            d_dx[mask_finite] = d_dx_finite
            d_fwhm[mask_finite] = d_fwhm_finite
        if mask_nan.any():
            d_dx = np.where(mask_nan, np.nan, d_dx)
            d_fwhm = np.where(mask_nan, np.nan, d_fwhm)
        return gauss, d_dx, d_fwhm


@register_shape("gaussian")
class Gaussian(PeakShape):
    """Gaussian lineshape."""

    def _create_evaluator(self) -> Any:
        return GaussianEvaluator()


_gaussian_evaluator = GaussianEvaluator()


def gaussian(dx: FloatArray, fwhm: float, j_hz: float = 0.0) -> FloatArray:
    """Evaluate Gaussian lineshape."""
    return _gaussian_evaluator.evaluate(np.asarray(dx), fwhm, j_hz)
