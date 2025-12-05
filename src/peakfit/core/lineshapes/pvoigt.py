"""Pseudo-Voigt lineshape model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.core.lineshapes.base import PeakShape
from peakfit.core.lineshapes.registry import register_shape
from peakfit.core.lineshapes.utils import _LN2, _SQRT_PI_4LN2, CachedResult

if TYPE_CHECKING:
    from peakfit.core.fitting.parameters import Parameters
    from peakfit.core.shared.typing import FloatArray, IntArray


class PseudoVoigtEvaluator:
    """Pseudo-Voigt lineshape evaluator.

    V(dx) = η * L(dx) + (1-η) * G(dx)

    Linear combination of Lorentzian and Gaussian with mixing parameter η ∈ [0,1].
    Height-normalized to 1.0 at center.

    Caching Strategy:
        Similar to BaseEvaluator, caches results from evaluate() calls
        for efficient subsequent get_cached_derivatives() calls.
    """

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache = CachedResult()

    def _compute_single(
        self, dx: FloatArray, fwhm: float, eta: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None, FloatArray | None]:
        """Compute PseudoVoigt for single component."""
        gamma = 0.5 * fwhm
        gamma2 = gamma * gamma
        c = 4.0 * _LN2 / (fwhm * fwhm)
        dx2 = dx * dx

        # Component values
        denom = gamma2 + dx2
        lorentz = gamma2 / denom
        gauss = np.exp(-c * dx2)
        pvoigt = eta * lorentz + (1.0 - eta) * gauss

        if not calc_derivs:
            return pvoigt, None, None, None

        # Lorentzian derivatives
        denom_inv2 = 1.0 / (denom * denom)
        d_lor_dx = -2.0 * gamma2 * dx * denom_inv2
        d_lor_fwhm = gamma * dx2 * denom_inv2

        # Gaussian derivatives
        c2 = 2.0 * c
        d_gau_dx = -c2 * dx * gauss
        d_gau_fwhm = (c2 / fwhm) * dx2 * gauss

        # Combined derivatives
        one_minus_eta = 1.0 - eta
        d_dx = eta * d_lor_dx + one_minus_eta * d_gau_dx
        d_fwhm = eta * d_lor_fwhm + one_minus_eta * d_gau_fwhm
        d_eta = lorentz - gauss

        return pvoigt, d_dx, d_fwhm, d_eta

    def _core(
        self, dx: FloatArray, fwhm: float, eta: float, j_hz: float, calc_derivs: bool
    ) -> tuple[
        FloatArray, FloatArray | None, FloatArray | None, FloatArray | None, FloatArray | None
    ]:
        """Core computation with doublet handling."""
        if j_hz == 0.0:
            val, d_dx, d_fwhm, d_eta = self._compute_single(dx, fwhm, eta, calc_derivs)
            if not calc_derivs:
                return val, None, None, None, None
            return val, d_dx, d_fwhm, d_eta, np.zeros_like(val)

        # Doublet case
        j_shift = 0.5 * j_hz
        val_p, d_dx_p, d_fwhm_p, d_eta_p = self._compute_single(
            dx + j_shift, fwhm, eta, calc_derivs
        )
        val_m, d_dx_m, d_fwhm_m, d_eta_m = self._compute_single(
            dx - j_shift, fwhm, eta, calc_derivs
        )

        val = val_p + val_m
        if not calc_derivs:
            return val, None, None, None, None

        assert d_dx_p is not None and d_dx_m is not None
        assert d_fwhm_p is not None and d_fwhm_m is not None
        assert d_eta_p is not None and d_eta_m is not None
        return (
            val,
            d_dx_p + d_dx_m,
            d_fwhm_p + d_fwhm_m,
            d_eta_p + d_eta_m,
            0.5 * (d_dx_p - d_dx_m),
        )

    def evaluate(self, dx: FloatArray, fwhm: float, eta: float, j_hz: float = 0.0) -> FloatArray:
        """Evaluate Pseudo-Voigt lineshape."""
        dx_arr = np.asarray(dx)

        # Check cache first
        if self._cache.matches(dx_arr, fwhm, eta, j_hz) and self._cache.value is not None:
            return self._cache.value

        # Compute with derivatives and cache
        val, d_dx, d_fwhm, d_eta, d_j = self._core(dx_arr, fwhm, eta, j_hz, calc_derivs=True)

        # Update cache
        self._cache.update_key(dx_arr, fwhm, eta, j_hz)
        self._cache.value = val
        self._cache.d_dx = d_dx
        self._cache.d_fwhm = d_fwhm
        self._cache.d_eta = d_eta
        self._cache.d_j = d_j

        return val

    def evaluate_derivatives(
        self, dx: FloatArray, fwhm: float, eta: float, j_hz: float = 0.0
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Evaluate Pseudo-Voigt and all derivatives."""
        dx_arr = np.asarray(dx)

        # Check cache first
        if self._cache.matches(dx_arr, fwhm, eta, j_hz) and self._cache.value is not None:
            assert self._cache.d_dx is not None
            assert self._cache.d_fwhm is not None
            assert self._cache.d_eta is not None
            assert self._cache.d_j is not None
            return (
                self._cache.value,
                self._cache.d_dx,
                self._cache.d_fwhm,
                self._cache.d_eta,
                self._cache.d_j,
            )

        # Compute fresh and update cache
        val, d_dx, d_fwhm, d_eta, d_j = self._core(dx_arr, fwhm, eta, j_hz, calc_derivs=True)

        self._cache.update_key(dx_arr, fwhm, eta, j_hz)
        self._cache.value = val
        self._cache.d_dx = d_dx
        self._cache.d_fwhm = d_fwhm
        self._cache.d_eta = d_eta
        self._cache.d_j = d_j

        return val, d_dx, d_fwhm, d_eta, d_j

    def get_cached_derivatives(
        self, dx: FloatArray, fwhm: float, eta: float, j_hz: float = 0.0
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None:
        """Get cached derivatives if they match the given parameters."""
        dx_arr = np.asarray(dx)
        if self._cache.matches(dx_arr, fwhm, eta, j_hz) and self._cache.d_dx is not None:
            assert self._cache.d_fwhm is not None
            assert self._cache.d_eta is not None
            assert self._cache.d_j is not None
            return self._cache.d_dx, self._cache.d_fwhm, self._cache.d_eta, self._cache.d_j
        return None

    def clear_cache(self) -> None:
        """Clear the cached results."""
        self._cache = CachedResult()

    def height(self, j_hz: float = 0.0) -> float:
        """Height at peak center (always 1.0)."""
        del j_hz
        return 1.0

    def integral(self, fwhm: float, eta: float, j_hz: float = 0.0) -> float:
        """Analytical integral as weighted sum of components."""
        integral = eta * (np.pi * fwhm / 2.0) + (1.0 - eta) * (fwhm * _SQRT_PI_4LN2)
        return integral * 2.0 if j_hz != 0.0 else integral


@register_shape("pvoigt")
class PseudoVoigt(PeakShape):
    """Pseudo-Voigt lineshape (mixture of Gaussian and Lorentzian)."""

    def _create_evaluator(self) -> Any:
        return PseudoVoigtEvaluator()

    def create_params(self) -> Parameters:
        """Create parameters including eta mixing factor."""
        params = super().create_params()

        # Eta (fraction) parameter using ParameterId
        eta_id = self._fraction_id()
        params.add(
            eta_id,
            value=0.5,
            min=-1.0,
            max=1.0,
            unit="",
        )
        self._param_ids.append(eta_id)
        self.param_names = list(params.keys())
        return params

    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        """Evaluate pseudo-Voigt at given points."""
        pos_name = self._position_id().name
        lw_name = self._linewidth_id().name
        eta_name = self._fraction_id().name
        x0 = params[pos_name].value
        fwhm = params[lw_name].value
        eta = params[eta_name].value
        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_hz = self.spec_params.pts2hz_delta(dx_pt)
        # Evaluator returns tuple, first element is value
        res_tuple = self.evaluator.evaluate(dx_hz, fwhm, eta)
        res: FloatArray = np.asarray(sign * res_tuple[0], dtype=float)
        return res

    def evaluate_derivatives(
        self, x_pt: IntArray, params: Parameters
    ) -> tuple[FloatArray, dict[str, FloatArray]]:
        """Evaluate pseudo-Voigt shape and derivatives."""
        pos_name = self._position_id().name
        lw_name = self._linewidth_id().name
        eta_name = self._fraction_id().name
        x0 = params[pos_name].value
        fwhm = params[lw_name].value
        eta = params[eta_name].value

        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_hz = self.spec_params.pts2hz_delta(dx_pt)

        # Call evaluator
        # (dx, fwhm, eta, j_hz=0.0) -> (val, d_dx, d_fwhm, d_eta, d_jhz)
        val, d_dx, d_fwhm, d_eta, _ = self.evaluator.evaluate(dx_hz, fwhm, eta)

        # Apply sign
        val = sign * val
        d_dx = sign * d_dx
        d_fwhm = sign * d_fwhm
        d_eta = sign * d_eta

        derivs = {}
        hz_per_ppm = self.spec_params.ppm2hz(1.0)
        derivs[pos_name] = d_dx * (-hz_per_ppm)
        derivs[lw_name] = d_fwhm
        derivs[eta_name] = d_eta

        return val, derivs


# Module-level evaluator instance
_pvoigt_evaluator = PseudoVoigtEvaluator()


def pvoigt(dx: FloatArray, fwhm: float, eta: float, j_hz: float = 0.0) -> FloatArray:
    """Evaluate Pseudo-Voigt lineshape.

    Convenience function for simple evaluation without managing evaluator instances.
    For repeated evaluations or when derivatives are needed, use PseudoVoigtEvaluator
    directly to benefit from caching.

    Args:
        dx: Frequency offset array (Hz)
        fwhm: Full width at half maximum (Hz)
        eta: Mixing parameter (0=Gaussian, 1=Lorentzian)
        j_hz: J-coupling constant (Hz), default 0.0

    Returns
    -------
        Lineshape values at given offsets
    """
    return _pvoigt_evaluator.evaluate(np.asarray(dx), fwhm, eta, j_hz)
