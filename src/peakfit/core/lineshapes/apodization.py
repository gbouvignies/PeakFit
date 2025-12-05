"""Apodization-based lineshape models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.core.lineshapes.base import BaseShape
from peakfit.core.lineshapes.utils import CachedResult

if TYPE_CHECKING:
    from peakfit.core.domain.spectrum import Spectra
    from peakfit.core.fitting.parameters import ParameterId, Parameters
    from peakfit.core.shared.typing import FittingOptions, FloatArray, IntArray


class ApodizationEvaluator:
    """Base for apodization-based lineshapes (NoApod, SP1, SP2).

    Complex-valued internally, returning real part after phase correction.
    Parameterized by R2 relaxation rate and phase.
    """

    def __init__(self, aq: float) -> None:
        self.aq = aq
        self._cache = CachedResult()

    def _compute_complex(
        self, dx: FloatArray, r2: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        raise NotImplementedError

    def _apply_phase(
        self,
        val: FloatArray,
        phase: float,
        d_dx: FloatArray | None = None,
        d_r2: FloatArray | None = None,
        d_jhz: FloatArray | None = None,
    ) -> tuple[
        FloatArray, FloatArray | None, FloatArray | None, FloatArray | None, FloatArray | None
    ]:
        if d_dx is None:
            if phase != 0.0:
                val = val * np.exp(1j * np.deg2rad(phase))
            return val.real, None, None, None, None

        assert d_r2 is not None
        d_phase = 1j * val

        if phase != 0.0:
            pf = np.exp(1j * np.deg2rad(phase))
            val = val * pf
            d_dx = d_dx * pf
            d_r2 = d_r2 * pf
            d_phase = d_phase * pf
            if d_jhz is not None:
                d_jhz = d_jhz * pf

        d_jhz_real = d_jhz.real if d_jhz is not None else np.zeros_like(val.real)
        d_dx_real: FloatArray = d_dx.real  # type: ignore[union-attr]
        d_r2_real: FloatArray = d_r2.real  # type: ignore[union-attr]
        return val.real, d_dx_real, d_r2_real, d_phase.real, d_jhz_real

    def _core(
        self, dx: FloatArray, r2: float, phase: float, j_hz: float, calc_derivs: bool
    ) -> tuple[
        FloatArray, FloatArray | None, FloatArray | None, FloatArray | None, FloatArray | None
    ]:
        if j_hz == 0.0:
            val, d_dx, d_r2 = self._compute_complex(dx, r2, calc_derivs)
            if not calc_derivs:
                return self._apply_phase(val, phase)
            return self._apply_phase(val, phase, d_dx, d_r2, np.zeros_like(val))

        # Doublet case
        j_rad = np.pi * j_hz
        val_p, d_dx_p, d_r2_p = self._compute_complex(dx + j_rad, r2, calc_derivs)
        val_m, d_dx_m, d_r2_m = self._compute_complex(dx - j_rad, r2, calc_derivs)

        val = val_p + val_m
        if not calc_derivs:
            return self._apply_phase(val, phase)

        assert d_dx_p is not None and d_dx_m is not None
        assert d_r2_p is not None and d_r2_m is not None
        d_dx = d_dx_p + d_dx_m
        d_r2 = d_r2_p + d_r2_m
        d_jhz = np.pi * (d_dx_p - d_dx_m)

        return self._apply_phase(val, phase, d_dx, d_r2, d_jhz)

    def evaluate(
        self, dx: FloatArray, r2: float, phase: float = 0.0, j_hz: float = 0.0
    ) -> FloatArray:
        dx_arr = np.asarray(dx)
        if self._cache.matches(dx_arr, r2, phase, j_hz) and self._cache.value is not None:
            return self._cache.value

        val, _, _, _, _ = self._core(dx_arr, r2, phase, j_hz, calc_derivs=False)
        self._cache.update_key(dx_arr, r2, phase, j_hz)
        self._cache.value = val  # type: ignore[assignment]
        self._cache.d_dx = None
        self._cache.d_r2 = None
        self._cache.d_phase = None
        self._cache.d_j = None
        return val  # type: ignore[return-value]

    def evaluate_derivatives(
        self, dx: FloatArray, r2: float, phase: float = 0.0, j_hz: float = 0.0
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        dx_arr = np.asarray(dx)
        if (
            self._cache.matches(dx_arr, r2, phase, j_hz)
            and self._cache.value is not None
            and self._cache.d_dx is not None
        ):
            return (
                self._cache.value,
                self._cache.d_dx,
                self._cache.d_r2,  # type: ignore[return-value]
                self._cache.d_phase,  # type: ignore[return-value]
                self._cache.d_j,  # type: ignore[return-value]
            )

        val, d_dx, d_r2, d_phase, d_j = self._core(dx_arr, r2, phase, j_hz, calc_derivs=True)
        self._cache.update_key(dx_arr, r2, phase, j_hz)
        self._cache.value = val
        self._cache.d_dx = d_dx
        self._cache.d_r2 = d_r2
        self._cache.d_phase = d_phase
        self._cache.d_j = d_j

        assert d_dx is not None and d_r2 is not None and d_phase is not None and d_j is not None
        return val, d_dx, d_r2, d_phase, d_j

    def get_cached_derivatives(
        self, dx: FloatArray, r2: float, phase: float = 0.0, j_hz: float = 0.0
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None:
        dx_arr = np.asarray(dx)
        if self._cache.matches(dx_arr, r2, phase, j_hz) and self._cache.d_dx is not None:
            assert self._cache.d_r2 is not None and self._cache.d_phase is not None
            assert self._cache.d_j is not None
            return self._cache.d_dx, self._cache.d_r2, self._cache.d_phase, self._cache.d_j
        return None

    def clear_cache(self) -> None:
        self._cache = CachedResult()

    def __call__(
        self, dx: FloatArray, r2: float, phase: float = 0.0, j_hz: float = 0.0
    ) -> FloatArray:
        return self.evaluate(dx, r2, phase, j_hz)


class ApodShape(BaseShape):
    """Base class for apodization-based lineshapes."""

    R2_START = 20.0
    FWHM_START = 25.0
    shape_name: str = "no_apod"

    def __init__(
        self,
        name: str,
        center: float,
        spectra: Spectra,
        dim: int,
        args: FittingOptions,
        evaluator: ApodizationEvaluator | None = None,
    ) -> None:
        super().__init__(name, center, spectra, dim, args)
        self._evaluator = evaluator if evaluator is not None else self._create_evaluator()

    def _create_evaluator(self) -> Any:
        raise NotImplementedError("Must provide evaluator or override _create_evaluator")

    def create_params(self) -> Parameters:
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        param_ids: list[ParameterId] = []

        pos_id = self._position_id()
        params.add(
            pos_id,
            value=self.center,
            min=self.center - self.spec_params.hz2ppm(self.FWHM_START),
            max=self.center + self.spec_params.hz2ppm(self.FWHM_START),
            unit="ppm",
        )
        param_ids.append(pos_id)

        lw_id = self._linewidth_id()
        params.add(lw_id, value=self.R2_START, min=0.1, max=200.0, unit="Hz")
        param_ids.append(lw_id)

        if self.args.jx and self.spec_params.direct:
            j_id = self._jcoupling_id()
            params.add(j_id, value=5.0, min=1.0, max=10.0, unit="Hz")
            param_ids.append(j_id)

        if (self.args.phx and self.spec_params.direct) or self.args.phy:
            phase_id = self._phase_id()
            params.add(phase_id, value=0.0, min=-5.0, max=5.0, unit="deg")
            param_ids.append(phase_id)

        self._param_ids = param_ids
        self.param_names = list(params.keys())
        return params

    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        pos_name = self._position_id().name
        lw_name = self._linewidth_id().name
        phase_name = self._phase_id().name
        j_name = self._jcoupling_id().name

        x0 = params[pos_name].value
        r2 = params[lw_name].value
        p0 = params[phase_name].value if phase_name in params else 0.0
        j_hz = params[j_name].value if j_name in params else 0.0

        # Optimization: Use x_pt
        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_rads = self.spec_params.pts2hz_delta(dx_pt) * 2 * np.pi

        # Use memoized evaluator
        # Evaluator handles J-coupling internally
        # Normalization: evaluate at 0 offset (dx=0) to get peak height
        norm = self._evaluator.evaluate(0.0, r2, p0, j_hz)
        shape = self._evaluator.evaluate(dx_rads, r2, p0, j_hz)

        res: FloatArray = np.asarray(sign * shape / norm, dtype=float)
        return res

    def evaluate_derivatives(
        self, x_pt: IntArray, params: Parameters
    ) -> tuple[FloatArray, dict[str, FloatArray]]:
        """Evaluate apodization shape and derivatives."""
        pos_name = self._position_id().name
        lw_name = self._linewidth_id().name
        phase_name = self._phase_id().name
        j_name = self._jcoupling_id().name

        x0 = params[pos_name].value
        r2 = params[lw_name].value
        p0 = params[phase_name].value if phase_name in params else 0.0
        j_hz = params[j_name].value if j_name in params else 0.0

        # Optimization: Use x_pt
        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_rads = self.spec_params.pts2hz_delta(dx_pt) * 2 * np.pi

        # Evaluate at peak center for normalization
        norm_val, _, norm_dr2, norm_dphase, norm_dj = self._evaluator.evaluate_derivatives(
            0.0, r2, p0, j_hz
        )

        # Evaluate at requested points only
        val, d_dx, d_r2, d_phase, d_j = self._evaluator.evaluate_derivatives(dx_rads, r2, p0, j_hz)

        s = sign

        # Apply normalization and sign
        res = s * val / norm_val

        # Derivatives
        inv_norm = 1.0 / norm_val

        # d_res_dr2 = s * (d_r2 * norm - val * d_norm_dr2) / norm^2
        d_res_dr2 = s * inv_norm * (d_r2 - (val / norm_val) * norm_dr2)

        d_res_dphase = s * inv_norm * (d_phase - (val / norm_val) * norm_dphase)

        d_res_dj = s * inv_norm * (d_j - (val / norm_val) * norm_dj)

        d_res_dx = s * d_dx * inv_norm

        derivs = {}

        # Position (ppm)
        hz_per_ppm = self.spec_params.ppm2hz(1.0)
        derivs[pos_name] = d_res_dx * (-hz_per_ppm * 2 * np.pi)

        # Linewidth (R2)
        derivs[lw_name] = d_res_dr2

        # Phase (if exists)
        if phase_name in params:
            derivs[phase_name] = d_res_dphase

        # J-coupling (if exists)
        if j_name in params:
            derivs[j_name] = d_res_dj

        return res, derivs
