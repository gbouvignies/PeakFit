"""Base classes for lineshape models and evaluators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from peakfit.core.fitting.parameters import ParameterId
from peakfit.core.lineshapes.utils import CachedResult, get_axis_label

if TYPE_CHECKING:
    from peakfit.core.domain.spectrum import Spectra
    from peakfit.core.fitting.parameters import Parameters
    from peakfit.core.shared.typing import FittingOptions, FloatArray, IntArray


@runtime_checkable
class Shape(Protocol):
    """Protocol for lineshape models."""

    name: str
    axis: str
    center: float
    spec_params: Any
    size: int
    param_names: list[str]
    cluster_id: int
    args: FittingOptions

    def create_params(self) -> Parameters: ...
    def fix_params(self, params: Parameters) -> None: ...
    def release_params(self, params: Parameters) -> None: ...
    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray: ...
    def print(self, params: Parameters) -> str: ...
    @property
    def center_i(self) -> int: ...
    def evaluate_derivatives(
        self, x_pt: IntArray, params: Parameters
    ) -> tuple[FloatArray, dict[str, FloatArray]]: ...


class BaseShape:
    """Base class for all lineshape models."""

    def __init__(
        self, name: str, center: float, spectra: Spectra, dim: int, args: FittingOptions
    ) -> None:
        self.name = name
        self.axis = get_axis_label(dim)
        self.center = center
        self.spec_params = spectra.params[dim]
        self.size = self.spec_params.size
        self.param_names: list[str] = []
        self._param_ids: list[ParameterId] = []
        self.cluster_id = 0
        self.args = args
        self.full_grid = np.arange(self.size)

    def _position_id(self) -> ParameterId:
        return ParameterId.position(self.name, self.axis)

    def _linewidth_id(self) -> ParameterId:
        return ParameterId.linewidth(self.name, self.axis)

    def _fraction_id(self) -> ParameterId:
        return ParameterId.fraction(self.name, self.axis)

    def _jcoupling_id(self) -> ParameterId:
        return ParameterId.jcoupling(self.name, self.axis)

    def _phase_id(self) -> ParameterId:
        return ParameterId.phase(self.cluster_id, self.axis)

    def create_params(self) -> Parameters:
        raise NotImplementedError

    def fix_params(self, params: Parameters) -> None:
        for name in self.param_names:
            params[name].vary = False

    def release_params(self, params: Parameters) -> None:
        for name in self.param_names:
            params[name].vary = True

    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        raise NotImplementedError

    def print(self, params: Parameters) -> str:
        lines = []
        for name in self.param_names:
            param = params[name]
            if param.param_id is not None:
                shortname = param.param_id.user_name
            else:
                shortname = name.split(".")[-1] if "." in name else name
            line = f"# {shortname:<10s}: {param.value:10.5f} ± {param.stderr:10.5f}"
            lines.append(line)
        return "\n".join(lines)

    @property
    def center_i(self) -> int:
        return self.spec_params.ppm2pt_i(self.center)

    def _compute_dx_and_sign(self, x_pt: IntArray, x0: float) -> tuple[FloatArray, FloatArray]:
        """Compute frequency offset and aliasing sign."""
        x0_pt = self.spec_params.ppm2pts(x0)
        dx_pt = x_pt - x0_pt
        if not self.spec_params.direct:
            aliasing = (dx_pt + 0.5 * self.size) // self.size
        else:
            aliasing = np.zeros_like(dx_pt)
        dx_pt_corrected = dx_pt - self.size * aliasing
        sign = np.power(-1.0, aliasing) if self.spec_params.p180 else np.ones_like(aliasing)
        return dx_pt_corrected, sign

    def evaluate_derivatives(
        self, x_pt: IntArray, params: Parameters
    ) -> tuple[FloatArray, dict[str, FloatArray]]:
        raise NotImplementedError


class BaseEvaluator:
    """Base evaluator with caching and doublet logic for FWHM-based lineshapes."""

    def __init__(self) -> None:
        self._cache = CachedResult()

    def _compute_single(
        self, dx: FloatArray, fwhm: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        raise NotImplementedError

    def _core(
        self, dx: FloatArray, fwhm: float, j_hz: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None, FloatArray | None]:
        if j_hz == 0.0:
            val, d_dx, d_fwhm = self._compute_single(dx, fwhm, calc_derivs)
            if not calc_derivs:
                return val, None, None, None
            return val, d_dx, d_fwhm, np.zeros_like(val)

        j_shift = 0.5 * j_hz
        val_p, d_dx_p, d_fwhm_p = self._compute_single(dx + j_shift, fwhm, calc_derivs)
        val_m, d_dx_m, d_fwhm_m = self._compute_single(dx - j_shift, fwhm, calc_derivs)

        val = val_p + val_m
        if not calc_derivs:
            return val, None, None, None

        assert d_dx_p is not None and d_dx_m is not None
        assert d_fwhm_p is not None and d_fwhm_m is not None
        return val, d_dx_p + d_dx_m, d_fwhm_p + d_fwhm_m, 0.5 * (d_dx_p - d_dx_m)

    def _get_or_compute(
        self, dx: FloatArray, fwhm: float, j_hz: float = 0.0, *, calc_derivs: bool = True
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None, FloatArray | None]:
        dx_arr = np.asarray(dx)

        if self._cache.matches(dx_arr, fwhm, j_hz) and self._cache.value is not None:
            if not calc_derivs:
                return self._cache.value, None, None, None
            if self._cache.d_dx is not None:
                return self._cache.value, self._cache.d_dx, self._cache.d_fwhm, self._cache.d_j

        val, d_dx, d_fwhm, d_j = self._core(dx_arr, fwhm, j_hz, calc_derivs=calc_derivs)

        self._cache.update_key(dx_arr, fwhm, j_hz)
        self._cache.value = val
        if calc_derivs:
            self._cache.d_dx = d_dx
            self._cache.d_fwhm = d_fwhm
            self._cache.d_j = d_j

        return val, d_dx, d_fwhm, d_j

    def evaluate(self, dx: FloatArray, fwhm: float, j_hz: float = 0.0) -> FloatArray:
        val, _, _, _ = self._get_or_compute(dx, fwhm, j_hz, calc_derivs=True)
        return val

    def evaluate_derivatives(
        self, dx: FloatArray, fwhm: float, j_hz: float = 0.0
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        val, d_dx, d_fwhm, d_j = self._get_or_compute(dx, fwhm, j_hz, calc_derivs=True)
        assert d_dx is not None and d_fwhm is not None and d_j is not None
        return val, d_dx, d_fwhm, d_j

    def get_cached_derivatives(
        self, dx: FloatArray, fwhm: float, j_hz: float = 0.0
    ) -> tuple[FloatArray, FloatArray, FloatArray] | None:
        dx_arr = np.asarray(dx)
        if self._cache.matches(dx_arr, fwhm, j_hz) and self._cache.d_dx is not None:
            assert self._cache.d_fwhm is not None and self._cache.d_j is not None
            return self._cache.d_dx, self._cache.d_fwhm, self._cache.d_j
        return None

    def clear_cache(self) -> None:
        self._cache = CachedResult()


class PeakShape(BaseShape):
    """Base class for simple peak shapes (Gaussian, Lorentzian, Pseudo-Voigt)."""

    FWHM_START = 25.0
    evaluator: Any

    def __init__(
        self, name: str, center: float, spectra: Spectra, dim: int, args: FittingOptions
    ) -> None:
        super().__init__(name, center, spectra, dim, args)
        self.evaluator = self._create_evaluator()

    def _create_evaluator(self) -> Any:
        raise NotImplementedError

    def _prepare_eval_args(
        self, x_pt: IntArray, params: Parameters
    ) -> tuple[FloatArray, float, FloatArray, str, str]:
        pos_name = self._position_id().name
        lw_name = self._linewidth_id().name
        x0 = params[pos_name].value
        fwhm = params[lw_name].value
        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_hz = self.spec_params.pts2hz_delta(dx_pt)
        return dx_hz, fwhm, sign, pos_name, lw_name

    def create_params(self) -> Parameters:
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        pos_id = self._position_id()
        params.add(
            pos_id,
            value=self.center,
            min=self.center - self.spec_params.hz2ppm(self.FWHM_START),
            max=self.center + self.spec_params.hz2ppm(self.FWHM_START),
            unit="ppm",
        )
        lw_id = self._linewidth_id()
        params.add(lw_id, value=self.FWHM_START, min=0.1, max=200.0, unit="Hz")
        self._param_ids = [pos_id, lw_id]
        self.param_names = list(params.keys())
        return params

    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        dx_hz, fwhm, sign, _, _ = self._prepare_eval_args(x_pt, params)
        res: FloatArray = np.asarray(sign * self.evaluator.evaluate(dx_hz, fwhm), dtype=float)
        return res

    def evaluate_derivatives(
        self, x_pt: IntArray, params: Parameters
    ) -> tuple[FloatArray, dict[str, FloatArray]]:
        dx_hz, fwhm, sign, pos_name, lw_name = self._prepare_eval_args(x_pt, params)
        val, d_dx, d_fwhm, _ = self.evaluator.evaluate_derivatives(dx_hz, fwhm)

        val = sign * val
        d_dx = sign * d_dx
        d_fwhm = sign * d_fwhm

        hz_per_ppm = self.spec_params.ppm2hz(1.0)
        derivs = {pos_name: d_dx * (-hz_per_ppm), lw_name: d_fwhm}
        return val, derivs
