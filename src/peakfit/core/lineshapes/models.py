"""Lineshape models for NMR peak fitting.

This module provides various lineshape model classes (Gaussian, Lorentzian, Pseudo-Voigt,
and apodization-based shapes) for fitting NMR peaks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from peakfit.core.domain.spectrum import Spectra
    from peakfit.core.fitting.parameters import Parameters
from peakfit.core.lineshapes import functions
from peakfit.core.lineshapes.registry import register_shape
from peakfit.core.shared.typing import FittingOptions, FloatArray, IntArray

AXIS_NAMES = ("x", "y", "z", "a")


class BaseShape(ABC):
    """Base class for all lineshape models."""

    def __init__(
        self, name: str, center: float, spectra: Spectra, dim: int, args: FittingOptions
    ) -> None:
        """Initialize shape.

        Args:
            name: Peak name
            center: Center position in ppm
            spectra: Spectra object with spectral parameters
            dim: Dimension index (1-based)
            args: Command-line arguments
        """
        self.name = name
        self.axis = AXIS_NAMES[spectra.data[0].ndim - dim]
        self.center = center
        self.spec_params = spectra.params[dim]
        self.size = self.spec_params.size
        self.param_names: list[str] = []
        self.cluster_id = 0
        self.args = args
        self.full_grid = np.arange(self.size)

    @property
    def prefix(self) -> str:
        """Parameter name prefix for this shape."""
        import re

        return re.sub(r"\W+|^(?=\d)", "_", f"{self.name}_{self.axis}")

    @property
    def prefix_phase(self) -> str:
        """Parameter name prefix for phase parameters."""
        import re

        return re.sub(r"\W+|^(?=\d)", "_", f"{self.cluster_id}_{self.axis}")

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
            fullname = name
            shortname = name.replace(self.prefix[:-1], "").replace(self.prefix_phase[:-1], "")
            value = params[fullname].value
            stderr_val = params[fullname].stderr
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

        Returns:
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
    shape_func: Callable

    def create_params(self) -> Parameters:
        """Create parameters for peak shape (position and FWHM)."""
        # Import ParameterType at runtime to avoid circular imports during module import
        from peakfit.core.fitting.parameters import Parameters, ParameterType

        params = Parameters()
        params.add(
            f"{self.prefix}0",
            value=self.center,
            min=self.center - self.spec_params.hz2ppm(self.FWHM_START),
            max=self.center + self.spec_params.hz2ppm(self.FWHM_START),
            param_type=ParameterType.POSITION,
            unit="ppm",
        )
        params.add(
            f"{self.prefix}_fwhm",
            value=self.FWHM_START,
            min=0.1,
            max=200.0,
            param_type=ParameterType.FWHM,
            unit="Hz",
        )
        self.param_names = list(params.keys())
        return params

    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        """Evaluate peak shape at given points."""
        x0 = params[f"{self.prefix}0"].value
        fwhm = params[f"{self.prefix}_fwhm"].value
        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_hz = self.spec_params.pts2hz_delta(dx_pt)
        return sign * self.shape_func(dx_hz, fwhm)


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
        # Import ParameterType at runtime to avoid circular imports
        from peakfit.core.fitting.parameters import ParameterType

        params.add(
            f"{self.prefix}_eta",
            value=0.5,
            min=-1.0,
            max=1.0,
            param_type=ParameterType.FRACTION,
            unit="",
        )
        self.param_names = list(params.keys())
        return params

    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        """Evaluate pseudo-Voigt at given points."""
        x0 = params[f"{self.prefix}0"].value
        fwhm = params[f"{self.prefix}_fwhm"].value
        eta = params[f"{self.prefix}_eta"].value
        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_hz = self.spec_params.pts2hz_delta(dx_pt)
        return sign * functions.pvoigt(dx_hz, fwhm, eta)


class ApodShape(BaseShape):
    """Base class for apodization-based lineshapes."""

    R2_START = 20.0
    FWHM_START = 25.0
    shape_func: Callable
    shape_name: str = "no_apod"  # Override in subclasses

    def create_params(self) -> Parameters:
        """Create parameters for apodization shape."""
        # Import ParameterType and Parameters at runtime to avoid circular imports
        from peakfit.core.fitting.parameters import Parameters, ParameterType

        params = Parameters()
        params.add(
            f"{self.prefix}0",
            value=self.center,
            min=self.center - self.spec_params.hz2ppm(self.FWHM_START),
            max=self.center + self.spec_params.hz2ppm(self.FWHM_START),
            param_type=ParameterType.POSITION,
            unit="ppm",
        )
        params.add(
            f"{self.prefix}_r2",
            value=self.R2_START,
            min=0.1,
            max=200.0,
            param_type=ParameterType.FWHM,  # R2 is related to linewidth
            unit="Hz",
        )
        if self.args.jx and self.spec_params.direct:
            params.add(
                f"{self.prefix}_j",
                value=5.0,
                min=1.0,
                max=10.0,
                param_type=ParameterType.JCOUPLING,
                unit="Hz",
            )
        if (self.args.phx and self.spec_params.direct) or self.args.phy:
            params.add(
                f"{self.prefix_phase}p",
                value=0.0,
                min=-5.0,
                max=5.0,
                param_type=ParameterType.PHASE,
                unit="deg",
            )
        self.param_names = list(params.keys())
        return params

    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        """Evaluate apodization shape at given points."""
        parvalues = params.valuesdict()
        x0 = parvalues[f"{self.prefix}0"]
        r2 = parvalues[f"{self.prefix}_r2"]
        p0 = parvalues.get(f"{self.prefix_phase}p", 0.0)
        j_hz = parvalues.get(f"{self.prefix}_j", 0.0)

        dx_pt, sign = self._compute_dx_and_sign(self.full_grid, x0)
        dx_rads = self.spec_params.pts2hz_delta(dx_pt) * 2 * np.pi
        j_rads = np.array([[0.0]]).T if j_hz == 0.0 else j_hz * np.pi * np.array([[1.0, -1.0]]).T
        dx_rads = dx_rads + j_rads

        # Select shape function
        if self.shape_name == "sp1":
            func = functions.sp1
        elif self.shape_name == "sp2":
            func = functions.sp2
        else:
            func = functions.no_apod

        shape_args = (r2, self.spec_params.aq_time)
        if self.shape_name in ("sp1", "sp2"):
            shape_args += (self.spec_params.apodq2, self.spec_params.apodq1)
        shape_args += (p0,)

        norm = np.sum(func(j_rads, *shape_args), axis=0)
        shape = np.sum(func(dx_rads, *shape_args), axis=0)

        return sign[x_pt] * shape[x_pt] / norm


@register_shape("no_apod")
class NoApod(ApodShape):
    """Non-apodized lineshape."""

    shape_func = staticmethod(functions.no_apod)
    shape_name = "no_apod"


@register_shape("sp1")
class SP1(ApodShape):
    """SP1 apodization lineshape."""

    shape_func = staticmethod(functions.sp1)
    shape_name = "sp1"


@register_shape("sp2")
class SP2(ApodShape):
    """SP2 apodization lineshape."""

    shape_func = staticmethod(functions.sp2)
    shape_name = "sp2"
