"""Lineshape models for NMR peak fitting.

This module provides various lineshape models (Gaussian, Lorentzian, Pseudo-Voigt,
and apodization-based shapes) for fitting NMR peaks.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Protocol, TypeVar

import numpy as np

# Import backend registry for dynamic backend selection
from peakfit.core.backend import (
    get_gaussian_func,
    get_lorentzian_func,
    get_no_apod_func,
    get_pvoigt_func,
    get_sp1_func,
    get_sp2_func,
)
from peakfit.core.fitting import Parameters, ParameterType

# Import optimized lineshape functions (JIT-compiled if Numba available)
from peakfit.core.optimized import (
    gaussian_jit,
    lorentzian_jit,
    no_apod_jit,
    sp1_jit,
    sp2_jit,
)
from peakfit.nmrpipe import SpectralParameters
from peakfit.spectra import Spectra
from peakfit.typing import FittingOptions, FloatArray, IntArray

T = TypeVar("T")

AXIS_NAMES = ("x", "y", "z", "a")


def clean(name: str) -> str:
    """Clean a string to be a valid parameter name."""
    return re.sub(r"\W+|^(?=\d)", "_", name)


class Shape(Protocol):
    """Protocol for lineshape models."""

    axis: str
    name: str
    cluster_id: int
    center: float
    spec_params: SpectralParameters
    size: int

    def create_params(self) -> Parameters: ...
    def fix_params(self, params: Parameters) -> None: ...
    def release_params(self, params: Parameters) -> None: ...
    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray: ...
    def print(self, params: Parameters) -> str: ...
    @property
    def center_i(self) -> int: ...
    @property
    def prefix(self) -> str: ...


SHAPES: dict[str, Callable[..., Shape]] = {}


def register_shape(
    shape_names: str | Iterable[str],
) -> Callable[[type[Shape]], type[Shape]]:
    """Decorator to register a shape class."""
    if isinstance(shape_names, str):
        shape_names = [shape_names]

    def decorator(shape_class: type[Shape]) -> type[Shape]:
        for name in shape_names:
            SHAPES[name] = shape_class
        return shape_class

    return decorator


def gaussian(dx: FloatArray, fwhm: float) -> FloatArray:
    """Gaussian lineshape (normalized to 1 at center)."""
    return np.exp(-(dx**2) * 4 * np.log(2) / (fwhm**2))


def lorentzian(dx: FloatArray, fwhm: float) -> FloatArray:
    """Lorentzian lineshape (normalized to 1 at center)."""
    return (0.5 * fwhm) ** 2 / (dx**2 + (0.5 * fwhm) ** 2)


def pvoigt(dx: FloatArray, fwhm: float, eta: float) -> FloatArray:
    """Pseudo-Voigt lineshape (linear combination of Gaussian and Lorentzian)."""
    return (1.0 - eta) * gaussian(dx, fwhm) + eta * lorentzian(dx, fwhm)


def no_apod(dx: FloatArray, r2: float, aq: float, phase: float = 0.0) -> FloatArray:
    """Non-apodized lineshape with optional phase."""
    z1 = aq * (1j * dx + r2)
    spec = aq * (1.0 - np.exp(-z1)) / z1
    return (spec * np.exp(1j * np.deg2rad(phase))).real


def sp1(
    dx: FloatArray, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> FloatArray:
    """SP1 apodization lineshape."""
    z1 = aq * (1j * dx + r2)
    f1, f2 = 1j * off * np.pi, 1j * (end - off) * np.pi
    a1 = (np.exp(+f2) - np.exp(+z1)) * np.exp(-z1 + f1) / (2 * (z1 - f2))
    a2 = (np.exp(+z1) - np.exp(-f2)) * np.exp(-z1 - f1) / (2 * (z1 + f2))
    spec = 1j * aq * (a1 + a2)
    return (spec * np.exp(1j * np.deg2rad(phase))).real


def sp2(
    dx: FloatArray, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> FloatArray:
    """SP2 apodization lineshape."""
    z1 = aq * (1j * dx + r2)
    f1, f2 = 1j * off * np.pi, 1j * (end - off) * np.pi
    a1 = (np.exp(+2 * f2) - np.exp(z1)) * np.exp(-z1 + 2 * f1) / (4 * (z1 - 2 * f2))
    a2 = (np.exp(-2 * f2) - np.exp(z1)) * np.exp(-z1 - 2 * f1) / (4 * (z1 + 2 * f2))
    a3 = (1.0 - np.exp(-z1)) / (2 * z1)
    spec = aq * (a1 + a2 + a3)
    return (spec * np.exp(1j * np.deg2rad(phase))).real


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
        return clean(f"{self.name}_{self.axis}")

    @property
    def prefix_phase(self) -> str:
        """Parameter name prefix for phase parameters."""
        return clean(f"{self.cluster_id}_{self.axis}")

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

    shape_func = staticmethod(lorentzian_jit)

    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        """Evaluate Lorentzian shape using current backend."""
        x0 = params[f"{self.prefix}0"].value
        fwhm = params[f"{self.prefix}_fwhm"].value
        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_hz = self.spec_params.pts2hz_delta(dx_pt)
        # Use backend registry for dynamic backend selection
        func = get_lorentzian_func()
        return sign * func(dx_hz, fwhm)


@register_shape("gaussian")
class Gaussian(PeakShape):
    """Gaussian lineshape."""

    shape_func = staticmethod(gaussian_jit)

    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        """Evaluate Gaussian shape using current backend."""
        x0 = params[f"{self.prefix}0"].value
        fwhm = params[f"{self.prefix}_fwhm"].value
        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_hz = self.spec_params.pts2hz_delta(dx_pt)
        # Use backend registry for dynamic backend selection
        func = get_gaussian_func()
        return sign * func(dx_hz, fwhm)


@register_shape("pvoigt")
class PseudoVoigt(PeakShape):
    """Pseudo-Voigt lineshape (mixture of Gaussian and Lorentzian)."""

    def create_params(self) -> Parameters:
        """Create parameters including eta mixing factor."""
        params = super().create_params()
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
        # Use backend registry for dynamic backend selection
        func = get_pvoigt_func()
        return sign * func(dx_hz, fwhm, eta)


class ApodShape(BaseShape):
    """Base class for apodization-based lineshapes."""

    R2_START = 20.0
    FWHM_START = 25.0
    shape_func: Callable
    shape_name: str = "no_apod"  # Override in subclasses

    def create_params(self) -> Parameters:
        """Create parameters for apodization shape."""
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

    def _get_shape_func(self) -> Callable:
        """Get the shape function from backend registry."""
        if self.shape_name == "sp1":
            return get_sp1_func()
        if self.shape_name == "sp2":
            return get_sp2_func()
        return get_no_apod_func()

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

        # Get the shape function from backend registry
        func = self._get_shape_func()

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

    shape_func = staticmethod(no_apod_jit)
    shape_name = "no_apod"


@register_shape("sp1")
class SP1(ApodShape):
    """SP1 apodization lineshape."""

    shape_func = staticmethod(sp1_jit)
    shape_name = "sp1"


@register_shape("sp2")
class SP2(ApodShape):
    """SP2 apodization lineshape."""

    shape_func = staticmethod(sp2_jit)
    shape_name = "sp2"
