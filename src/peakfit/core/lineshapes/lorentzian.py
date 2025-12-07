"""Lorentzian lineshape model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.core.lineshapes.base import BaseEvaluator, PeakShape
from peakfit.core.lineshapes.registry import register_shape

if TYPE_CHECKING:
    from peakfit.core.shared.typing import FloatArray


class LorentzianEvaluator(BaseEvaluator):
    """Lorentzian lineshape: L(dx) = γ² / (γ² + dx²) where γ = fwhm/2."""

    def _compute_single(
        self, dx: FloatArray, fwhm: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        gamma = 0.5 * fwhm
        gamma2 = gamma * gamma
        dx_arr = np.asarray(dx)
        dx2 = dx_arr * dx_arr
        denom = gamma2 + dx2
        lorentz = gamma2 / denom

        mask_nan = np.isnan(dx_arr)
        mask_inf = np.isinf(dx_arr)
        if mask_inf.any():
            lorentz = np.where(mask_inf, 0.0, lorentz)
        if mask_nan.any():
            lorentz = np.where(mask_nan, np.nan, lorentz)

        if not calc_derivs:
            return lorentz, None, None

        denom_inv2 = 1.0 / (denom * denom)
        d_dx = np.zeros_like(lorentz)
        d_fwhm = np.zeros_like(lorentz)
        mask_finite = ~(mask_nan | mask_inf)
        if mask_finite.any():
            d_dx_finite = -2.0 * gamma2 * dx_arr[mask_finite] * denom_inv2[mask_finite]
            d_fwhm_finite = gamma * dx2[mask_finite] * denom_inv2[mask_finite]
            d_dx[mask_finite] = d_dx_finite
            d_fwhm[mask_finite] = d_fwhm_finite
        if mask_nan.any():
            d_dx = np.where(mask_nan, np.nan, d_dx)
            d_fwhm = np.where(mask_nan, np.nan, d_fwhm)
        return lorentz, d_dx, d_fwhm


@register_shape("lorentzian")
class Lorentzian(PeakShape):
    """Lorentzian lineshape."""

    def _create_evaluator(self) -> Any:
        return LorentzianEvaluator()


_lorentzian_evaluator = LorentzianEvaluator()


def lorentzian(dx: FloatArray, fwhm: float, j_hz: float = 0.0) -> FloatArray:
    """Evaluate Lorentzian lineshape."""
    return _lorentzian_evaluator.evaluate(np.asarray(dx), fwhm, j_hz)
