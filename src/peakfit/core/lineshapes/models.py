"""Lineshape models for NMR peak fitting.

This module provides various lineshape model classes (Gaussian, Lorentzian, Pseudo-Voigt,
and apodization-based shapes) for fitting NMR peaks.

All lineshape models use the unified ParameterId system for parameter naming,
producing names like "2N-H.F1.cs" instead of legacy "_2N_H_F10".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from peakfit.core.domain.spectrum import Spectra
    from peakfit.core.fitting.parameters import Parameters
    from peakfit.core.shared.typing import FittingOptions, FloatArray, IntArray
from peakfit.core.fitting.parameters import ParameterId
from peakfit.core.lineshapes import functions
from peakfit.core.lineshapes.registry import register_shape


def get_axis_label(dim_index: int) -> str:
    """Get the axis label for a dimension using Bruker Topspin convention.

    For pseudo-3D experiments:
    - F1 = pseudo-dimension (intensities, CEST offsets, etc.)
    - F2 = first spectral dimension (indirect, e.g., 15N)
    - F3 = second spectral dimension (direct/acquisition, e.g., 1H)

    Args:
        dim_index: 1-based dimension index (1 = first spectral dim after pseudo)

    Returns
    -------
        Axis label like "F2", "F3", "F4"
    """
    # Offset by 1: F1 is reserved for pseudo-dimension
    return f"F{dim_index + 1}"


class BaseShape(ABC):
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

    @abstractmethod
    def create_params(self) -> Parameters:
        """Create parameters for this shape."""
        ...

    def fix_params(self, params: Parameters) -> None:
        """Fix (freeze) all parameters for this shape."""
        for name in self.param_names:
            params[name].vary = False

    def release_params(self, params: Parameters) -> None:
        """Release (unfreeze) all parameters for this shape."""
        for name in self.param_names:
            params[name].vary = True

    @abstractmethod
    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        """Evaluate shape at given points."""
        ...

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


class PeakShape(BaseShape):
    """Base class for simple peak shapes (Gaussian, Lorentzian, Pseudo-Voigt)."""

    FWHM_START = 25.0
    shape_func: Callable[..., FloatArray]

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
        res: FloatArray = np.asarray(sign * self.shape_func(dx_hz, fwhm), dtype=float)
        return res


@register_shape("lorentzian")
class Lorentzian(PeakShape):
    """Lorentzian lineshape."""

    shape_func = staticmethod(functions.lorentzian)


@register_shape("gaussian")
class Gaussian(PeakShape):
    """Gaussian lineshape."""

    shape_func = staticmethod(functions.gaussian)


@register_shape("pvoigt")
class PseudoVoigt(PeakShape):
    """Pseudo-Voigt lineshape (mixture of Gaussian and Lorentzian)."""

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
        res: FloatArray = np.asarray(sign * functions.pvoigt(dx_hz, fwhm, eta), dtype=float)
        return res


class ApodShape(BaseShape):
    """Base class for apodization-based lineshapes.

    Subclasses must override `_create_evaluator()` to return an evaluator
    function created via the factory functions in the functions module.
    The evaluator pre-computes static exponentials that depend on aq, end,
    and off parameters (which don't change during fitting).
    """

    R2_START = 20.0
    FWHM_START = 25.0
    shape_name: str = "no_apod"  # Override in subclasses

    def __init__(
        self, name: str, center: float, spectra: Spectra, dim: int, args: FittingOptions
    ) -> None:
        """Initialize apodization shape with pre-computed evaluator."""
        super().__init__(name, center, spectra, dim, args)
        # Create memoized evaluator with static parameters
        self._evaluator = self._create_evaluator()

    def __getstate__(self) -> dict:
        """Prepare state for pickling - exclude the closure."""
        state = self.__dict__.copy()
        # Remove the unpicklable closure
        state.pop("_evaluator", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state from pickle - recreate the closure."""
        self.__dict__.update(state)
        # Recreate the evaluator
        self._evaluator = self._create_evaluator()

    @abstractmethod
    def _create_evaluator(self) -> Callable[..., FloatArray]:
        """Create the appropriate evaluator for this shape type."""
        ...

    def create_params(self) -> Parameters:
        """Create parameters for apodization shape."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        param_ids: list[ParameterId] = []

        # Position parameter
        pos_id = self._position_id()
        params.add(
            pos_id,
            value=self.center,
            min=self.center - self.spec_params.hz2ppm(self.FWHM_START),
            max=self.center + self.spec_params.hz2ppm(self.FWHM_START),
            unit="ppm",
        )
        param_ids.append(pos_id)

        # Linewidth (R2) parameter
        lw_id = self._linewidth_id()
        params.add(
            lw_id,
            value=self.R2_START,
            min=0.1,
            max=200.0,
            unit="Hz",
        )
        param_ids.append(lw_id)

        # J-coupling parameter (optional)
        if self.args.jx and self.spec_params.direct:
            j_id = self._jcoupling_id()
            params.add(
                j_id,
                value=5.0,
                min=1.0,
                max=10.0,
                unit="Hz",
            )
            param_ids.append(j_id)

        # Phase parameter (optional, cluster-level)
        if (self.args.phx and self.spec_params.direct) or self.args.phy:
            phase_id = self._phase_id()
            params.add(
                phase_id,
                value=0.0,
                min=-5.0,
                max=5.0,
                unit="deg",
            )
            param_ids.append(phase_id)

        self._param_ids = param_ids
        self.param_names = list(params.keys())
        return params

    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        """Evaluate apodization shape at given points using memoized evaluator."""
        pos_name = self._position_id().name
        lw_name = self._linewidth_id().name
        phase_name = self._phase_id().name
        j_name = self._jcoupling_id().name

        x0 = params[pos_name].value
        r2 = params[lw_name].value
        p0 = params[phase_name].value if phase_name in params else 0.0
        j_hz = params[j_name].value if j_name in params else 0.0

        dx_pt, sign = self._compute_dx_and_sign(self.full_grid, x0)
        dx_rads = self.spec_params.pts2hz_delta(dx_pt) * 2 * np.pi
        j_rads = np.array([[0.0]]).T if j_hz == 0.0 else j_hz * np.pi * np.array([[1.0, -1.0]]).T
        dx_rads = dx_rads + j_rads

        # Use memoized evaluator instead of plain function
        norm = np.sum(self._evaluator(j_rads, r2, p0), axis=0)
        shape = np.sum(self._evaluator(dx_rads, r2, p0), axis=0)

        res: FloatArray = np.asarray(sign[x_pt] * shape[x_pt] / norm, dtype=float)
        return res


@register_shape("no_apod")
class NoApod(ApodShape):
    """Non-apodized lineshape."""

    shape_name = "no_apod"

    def _create_evaluator(self) -> Callable[..., FloatArray]:
        """Create NoApod evaluator with aq bound via partial."""
        from functools import partial

        return partial(functions.no_apod, aq=self.spec_params.aq_time)


@register_shape("sp1")
class SP1(ApodShape):
    """SP1 apodization lineshape."""

    shape_name = "sp1"

    def _create_evaluator(self) -> Callable[..., FloatArray]:
        """Create SP1 evaluator with pre-computed exponentials."""
        return functions.make_sp1_evaluator(
            aq=self.spec_params.aq_time,
            end=self.spec_params.apodq2,
            off=self.spec_params.apodq1,
        )


@register_shape("sp2")
class SP2(ApodShape):
    """SP2 apodization lineshape."""

    shape_name = "sp2"

    def _create_evaluator(self) -> Callable[..., FloatArray]:
        """Create SP2 evaluator with pre-computed exponentials."""
        return functions.make_sp2_evaluator(
            aq=self.spec_params.aq_time,
            end=self.spec_params.apodq2,
            off=self.spec_params.apodq1,
        )
