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
    """Base class for all lineshape models.

    Uses the ParameterId system for consistent parameter naming with dot-notation.
    """

    def __init__(
        self, name: str, center: float, spectra: Spectra, dim: int, args: FittingOptions
    ) -> None:
        """Initialize shape.

        Args:
            name: Peak name
            center: Center position in ppm
            spectra: Spectra object with spectral parameters
            dim: Dimension index (1-based, 1=first spectral dim after pseudo)
            args: Command-line arguments
        """
        self.name = name
        # Use NMRPipe Fn convention for axis labels
        self.axis = get_axis_label(dim)
        self.center = center
        self.spec_params = spectra.params[dim]
        self.size = self.spec_params.size
        self.param_names: list[str] = []
        self._param_ids: list[ParameterId] = []  # Store ParameterIds for this shape
        self.cluster_id = 0
        self.args = args
        self.full_grid = np.arange(self.size)

    def _position_id(self) -> ParameterId:
        """Create ParameterId for position (chemical shift) parameter."""
        return ParameterId.position(self.name, self.axis)

    def _linewidth_id(self) -> ParameterId:
        """Create ParameterId for linewidth parameter."""
        return ParameterId.linewidth(self.name, self.axis)

    def _fraction_id(self) -> ParameterId:
        """Create ParameterId for fraction (eta) parameter."""
        return ParameterId.fraction(self.name, self.axis)

    def _jcoupling_id(self) -> ParameterId:
        """Create ParameterId for J-coupling parameter."""
        return ParameterId.jcoupling(self.name, self.axis)

    def _phase_id(self) -> ParameterId:
        """Create ParameterId for phase parameter (cluster-level)."""
        return ParameterId.phase(self.cluster_id, self.axis)

    def create_params(self) -> Parameters:
        """Create parameters for this shape."""
        raise NotImplementedError

    def fix_params(self, params: Parameters) -> None:
        """Fix (freeze) all parameters for this shape."""
        for name in self.param_names:
            params[name].vary = False

    def release_params(self, params: Parameters) -> None:
        """Release (unfreeze) all parameters for this shape."""
        for name in self.param_names:
            params[name].vary = True

    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        """Evaluate shape at given points."""
        raise NotImplementedError

    def print(self, params: Parameters) -> str:
        """Format parameters as string for output."""
        lines = []
        for name in self.param_names:
            param = params[name]
            # Use ParameterId's user_name if available
            if param.param_id is not None:
                shortname = param.param_id.user_name
            else:
                shortname = name.split(".")[-1] if "." in name else name
            value = param.value
            stderr_val = param.stderr
            line = f"# {shortname:<10s}: {value:10.5f} ± {stderr_val:10.5f}"
            lines.append(line)
        return "\n".join(lines)

    @property
    def center_i(self) -> int:
        """Center position in points (integer)."""
        return self.spec_params.ppm2pt_i(self.center)

    def _compute_dx_and_sign(self, x_pt: IntArray, x0: float) -> tuple[FloatArray, FloatArray]:
        """Compute frequency offset and aliasing sign.

        Args:
            x_pt: Points to evaluate
            x0: Center position in ppm

        Returns
        -------
            Tuple of (corrected dx in points, sign from aliasing)
        """
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
        """Evaluate shape and derivatives."""
        raise NotImplementedError


class BaseEvaluator:
    """Base evaluator with shared caching and doublet logic.

    These lineshapes are:
    - Real-valued
    - Height-normalized to 1.0 at center for singlets
    - Parameterized by FWHM (full width at half maximum)

    Caching Strategy:
        When evaluate() is called, it computes both values AND derivatives via _core()
        and caches them. Subsequent calls to get_cached_derivatives() with the same
        parameters return the cached derivatives without recomputation.
        This is optimized for scipy.optimize.least_squares which calls fun() then jac().
    """

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache = CachedResult()

    def _compute_single(
        self, dx: FloatArray, fwhm: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        """Compute lineshape for a single component (no J-coupling)."""
        raise NotImplementedError

    def _core(
        self, dx: FloatArray, fwhm: float, j_hz: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None, FloatArray | None]:
        """Core computation with doublet handling."""
        if j_hz == 0.0:
            val, d_dx, d_fwhm = self._compute_single(dx, fwhm, calc_derivs)
            if not calc_derivs:
                return val, None, None, None
            return val, d_dx, d_fwhm, np.zeros_like(val)

        # Doublet case
        j_shift = 0.5 * j_hz
        val_p, d_dx_p, d_fwhm_p = self._compute_single(dx + j_shift, fwhm, calc_derivs)
        val_m, d_dx_m, d_fwhm_m = self._compute_single(dx - j_shift, fwhm, calc_derivs)

        val = val_p + val_m
        if not calc_derivs:
            return val, None, None, None

        # d_dx_p and d_dx_m are guaranteed non-None when calc_derivs=True
        assert d_dx_p is not None and d_dx_m is not None
        assert d_fwhm_p is not None and d_fwhm_m is not None
        return (
            val,
            d_dx_p + d_dx_m,
            d_fwhm_p + d_fwhm_m,
            0.5 * (d_dx_p - d_dx_m),  # d/dj = 0.5 * (d/dx_p - d/dx_m)
        )

    def evaluate(self, dx: FloatArray, fwhm: float, j_hz: float = 0.0) -> FloatArray:
        """Evaluate lineshape values.

        Also computes and caches derivatives for efficient subsequent jac() calls.

        Args:
            dx: Frequency offset array (Hz)
            fwhm: Full width at half maximum (Hz)
            j_hz: J-coupling constant (Hz), default 0.0

        Returns
        -------
            Lineshape values at given offsets
        """
        dx_arr = np.asarray(dx)

        # Check cache first
        if self._cache.matches(dx_arr, fwhm, j_hz) and self._cache.value is not None:
            return self._cache.value

        # Compute with derivatives and cache
        val, d_dx, d_fwhm, d_j = self._core(dx_arr, fwhm, j_hz, calc_derivs=True)

        # Update cache
        self._cache.update_key(dx_arr, fwhm, j_hz)
        self._cache.value = val
        self._cache.d_dx = d_dx
        self._cache.d_fwhm = d_fwhm
        self._cache.d_j = d_j

        return val

    def evaluate_derivatives(
        self, dx: FloatArray, fwhm: float, j_hz: float = 0.0
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Evaluate lineshape and all derivatives.

        Returns cached derivatives if available from prior evaluate() call.
        Otherwise computes fresh.

        Args:
            dx: Frequency offset array (Hz)
            fwhm: Full width at half maximum (Hz)
            j_hz: J-coupling constant (Hz), default 0.0

        Returns
        -------
            Tuple of (value, d_value_dx, d_value_dfwhm, d_value_dj)
        """
        dx_arr = np.asarray(dx)

        # Check cache first
        if self._cache.matches(dx_arr, fwhm, j_hz) and self._cache.value is not None:
            assert self._cache.d_dx is not None
            assert self._cache.d_fwhm is not None
            assert self._cache.d_j is not None
            return self._cache.value, self._cache.d_dx, self._cache.d_fwhm, self._cache.d_j

        # Compute fresh and update cache
        val, d_dx, d_fwhm, d_j = self._core(dx_arr, fwhm, j_hz, calc_derivs=True)

        self._cache.update_key(dx_arr, fwhm, j_hz)
        self._cache.value = val
        self._cache.d_dx = d_dx
        self._cache.d_fwhm = d_fwhm
        self._cache.d_j = d_j

        assert d_dx is not None and d_fwhm is not None and d_j is not None
        return val, d_dx, d_fwhm, d_j

    def get_cached_derivatives(
        self, dx: FloatArray, fwhm: float, j_hz: float = 0.0
    ) -> tuple[FloatArray, FloatArray, FloatArray] | None:
        """Get cached derivatives if they match the given parameters."""
        dx_arr = np.asarray(dx)
        if self._cache.matches(dx_arr, fwhm, j_hz) and self._cache.d_dx is not None:
            assert self._cache.d_fwhm is not None and self._cache.d_j is not None
            return self._cache.d_dx, self._cache.d_fwhm, self._cache.d_j
        return None

    def clear_cache(self) -> None:
        """Clear the cached results."""
        self._cache = CachedResult()

    def height(self, j_hz: float = 0.0) -> float:
        """Height at peak center (always 1.0 for normalized lineshapes)."""
        return 1.0

    def integral(self, fwhm: float, j_hz: float = 0.0) -> float:
        """Compute analytical integral."""
        raise NotImplementedError


class PeakShape(BaseShape):
    """Base class for simple peak shapes (Gaussian, Lorentzian, Pseudo-Voigt)."""

    FWHM_START = 25.0
    evaluator: Any  # Should be a Protocol for Evaluator

    def __init__(
        self, name: str, center: float, spectra: Spectra, dim: int, args: FittingOptions
    ) -> None:
        super().__init__(name, center, spectra, dim, args)
        self.evaluator = self._create_evaluator()

    def _create_evaluator(self) -> Any:
        """Create the evaluator for this shape."""
        raise NotImplementedError

    def create_params(self) -> Parameters:
        """Create parameters for peak shape (position and FWHM)."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()

        # Position parameter using ParameterId
        pos_id = self._position_id()
        params.add(
            pos_id,
            value=self.center,
            min=self.center - self.spec_params.hz2ppm(self.FWHM_START),
            max=self.center + self.spec_params.hz2ppm(self.FWHM_START),
            unit="ppm",
        )

        # Linewidth parameter using ParameterId
        lw_id = self._linewidth_id()
        params.add(
            lw_id,
            value=self.FWHM_START,
            min=0.1,
            max=200.0,
            unit="Hz",
        )

        self._param_ids = [pos_id, lw_id]
        self.param_names = list(params.keys())
        return params

    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        """Evaluate peak shape at given points."""
        pos_name = self._position_id().name
        lw_name = self._linewidth_id().name
        x0 = params[pos_name].value
        fwhm = params[lw_name].value
        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_hz = self.spec_params.pts2hz_delta(dx_pt)
        # Evaluator returns tuple, first element is value
        res_tuple = self.evaluator.evaluate(dx_hz, fwhm)
        res: FloatArray = np.asarray(sign * res_tuple[0], dtype=float)
        return res

    def evaluate_derivatives(
        self, x_pt: IntArray, params: Parameters
    ) -> tuple[FloatArray, dict[str, FloatArray]]:
        """Evaluate shape and derivatives at given points.

        Returns
        -------
            Tuple of (values, dict of derivatives w.r.t parameter names)
        """
        pos_name = self._position_id().name
        lw_name = self._linewidth_id().name
        x0 = params[pos_name].value
        fwhm = params[lw_name].value

        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_hz = self.spec_params.pts2hz_delta(dx_pt)

        # Call evaluator
        # Expects (dx, fwhm, j_hz=0.0) -> (val, d_dx, d_fwhm, d_jhz)
        val, d_dx, d_fwhm, _ = self.evaluator.evaluate(dx_hz, fwhm)

        # Apply sign (aliasing)
        val = sign * val
        d_dx = sign * d_dx
        d_fwhm = sign * d_fwhm

        # Derivatives w.r.t parameters
        derivs = {}

        # Position (ppm)
        # d_model/d_x0_ppm = d_model/d_dx_hz * d_dx_hz/d_x0_ppm
        # d_dx_hz/d_x0_ppm = -Hz/ppm
        hz_per_ppm = self.spec_params.ppm2hz(1.0)
        derivs[pos_name] = d_dx * (-hz_per_ppm)

        # Linewidth (Hz)
        derivs[lw_name] = d_fwhm

        return val, derivs
