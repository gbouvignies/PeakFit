"""Shape models - all shapes treated symmetrically with uniform J-coupling support."""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

import numpy as np

from peakfit.lineshapes import functions
from peakfit.lineshapes.functions import KERNEL_REGISTRY
from peakfit.typing import FittingOptions, FloatArray, IntArray

try:
    from peakfit.fitting.parameters import Parameters, ParameterType
except ImportError:
    Parameters = object  # type: ignore[misc, assignment]
    ParameterType = object  # type: ignore[misc, assignment]


# Shape registry
SHAPES: dict[str, type] = {}


def register_shape(names: str | Iterable[str]) -> Callable:
    """Register shape class with one or more names."""
    name_list = [names] if isinstance(names, str) else list(names)

    def decorator(cls):
        for name in name_list:
            SHAPES[name] = cls
        return cls

    return decorator


def get_shape(name: str) -> type:
    """Get shape class by name."""
    return SHAPES[name]


def list_shapes() -> list[str]:
    """List registered shape names."""
    return list(SHAPES.keys())


# Base class
class BaseShape(ABC):
    """Base for all shapes - uniform interface with J-coupling support."""

    shape_func: Callable
    shape_name: str

    def __init__(self, name: str, center: float, spectra, dim: int, args: FittingOptions):
        self.name = name
        self.axis = ("x", "y", "z", "a")[spectra.data[0].ndim - dim]
        self.center = center
        self.spec_params = spectra.params[dim]
        self.size = self.spec_params.size
        self.cluster_id = 0
        self.args = args
        self.full_grid = np.arange(self.size)
        self.param_names: list[str] = []

        # Cached prefixes
        self._prefix = re.sub(r"\W+|^(?=\d)", "_", f"{name}_{self.axis}")
        self._prefix_phase = re.sub(r"\W+|^(?=\d)", "_", f"{self.cluster_id}_{self.axis}")

    @abstractmethod
    def create_params(self) -> Parameters:
        """Create parameters."""
        ...

    def fix_params(self, params: Parameters) -> None:
        """Fix all parameters."""
        for name in self.param_names:
            params[name].vary = False

    def release_params(self, params: Parameters) -> None:
        """Release all parameters."""
        for name in self.param_names:
            params[name].vary = True

    @abstractmethod
    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        """Evaluate shape."""
        ...

    @classmethod
    def batch_evaluate(cls, shapes: list, x_pt: IntArray, params: Parameters) -> FloatArray:
        """Batch evaluation using catalog."""
        from peakfit.lineshapes.catalog import batch_evaluate_with_catalog

        return batch_evaluate_with_catalog(shapes, x_pt, params)

    def print(self, params: Parameters) -> str:
        """Format parameters."""
        lines = []
        for name in self.param_names:
            short = name.replace(self._prefix[:-1], "").replace(self._prefix_phase[:-1], "")
            val = params[name].value
            err = params[name].stderr
            lines.append(f"# {short:<10s}: {val:10.5f} ± {err:10.5f}")
        return "\n".join(lines)

    @property
    def center_i(self) -> int:
        """Center in points."""
        return self.spec_params.ppm2pt_i(self.center)

    def _compute_dx_and_sign(self, x_pt: IntArray, x0: float) -> tuple[FloatArray, FloatArray]:
        """Compute offset and aliasing sign."""
        x0_pt = self.spec_params.ppm2pts(x0)
        dx_pt = x_pt - x0_pt
        if not self.spec_params.direct:
            aliasing = (dx_pt + 0.5 * self.size) // self.size
        else:
            aliasing = np.zeros_like(dx_pt)
        dx_pt_corrected = dx_pt - self.size * aliasing
        sign = np.power(-1.0, aliasing) if self.spec_params.p180 else np.ones_like(aliasing)
        return dx_pt_corrected, sign


# All shapes use same pattern - only difference is which parameters they need
class FreqShape(BaseShape):
    """Frequency-domain shapes (gaussian, lorentzian, pvoigt).

    All frequency-domain shapes support uniform J-coupling.
    """

    needs_eta = False

    def create_params(self) -> Parameters:
        params = Parameters()
        params.add(
            f"{self._prefix}0",
            value=self.center,
            min=self.center - self.spec_params.hz2ppm(25.0),
            max=self.center + self.spec_params.hz2ppm(25.0),
            param_type=ParameterType.POSITION,
            unit="ppm",
        )
        params.add(
            f"{self._prefix}_fwhm",
            value=25.0,
            min=0.1,
            max=200.0,
            param_type=ParameterType.FWHM,
            unit="Hz",
        )
        if self.needs_eta:
            params.add(
                f"{self._prefix}_eta",
                value=0.5,
                min=-1.0,
                max=1.0,
                param_type=ParameterType.FRACTION,
                unit="",
            )
        # J-coupling (uniform across all shapes)
        if self.args.jx and self.spec_params.direct:
            params.add(
                f"{self._prefix}_j",
                value=5.0,
                min=1.0,
                max=10.0,
                param_type=ParameterType.JCOUPLING,
                unit="Hz",
            )
        self.param_names = list(params.keys())
        return params

    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        """Evaluate using KERNEL_REGISTRY (zero manual math)."""
        parvals = params.valuesdict()
        x0 = parvals[f"{self._prefix}0"]
        fwhm = parvals[f"{self._prefix}_fwhm"]
        j_hz = parvals.get(f"{self._prefix}_j", 0.0)

        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_hz = self.spec_params.pts2hz_delta(dx_pt)

        # Retrieve kernel from registry
        kernel_func = KERNEL_REGISTRY[self.shape_name]

        # Prepare arguments
        if self.needs_eta:
            eta = parvals[f"{self._prefix}_eta"]
            args = (fwhm, eta)
        else:
            args = (fwhm,)

        # Execute kernel (handles J-coupling and normalization internally)
        return sign * kernel_func(dx_hz, j_hz, *args)


class TimeShape(BaseShape):
    """FID-parameterized frequency-domain shapes (no_apod, sp1, sp2).

    These shapes are parameterized by FID properties (R2, acquisition time)
    but are mathematically Fourier-transformed into the frequency domain.
    All support uniform J-coupling.
    """

    needs_apod_params = False

    def create_params(self) -> Parameters:
        params = Parameters()
        params.add(
            f"{self._prefix}0",
            value=self.center,
            min=self.center - self.spec_params.hz2ppm(25.0),
            max=self.center + self.spec_params.hz2ppm(25.0),
            param_type=ParameterType.POSITION,
            unit="ppm",
        )
        params.add(
            f"{self._prefix}_r2",
            value=20.0,
            min=0.1,
            max=200.0,
            param_type=ParameterType.FWHM,
            unit="Hz",
        )
        # J-coupling (uniform across all shapes)
        if self.args.jx and self.spec_params.direct:
            params.add(
                f"{self._prefix}_j",
                value=5.0,
                min=1.0,
                max=10.0,
                param_type=ParameterType.JCOUPLING,
                unit="Hz",
            )
        # Phase correction
        if (self.args.phx and self.spec_params.direct) or self.args.phy:
            params.add(
                f"{self._prefix_phase}p",
                value=0.0,
                min=-5.0,
                max=5.0,
                param_type=ParameterType.PHASE,
                unit="deg",
            )
        self.param_names = list(params.keys())
        return params

    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        """Evaluate using KERNEL_REGISTRY (zero manual math)."""
        parvals = params.valuesdict()
        x0 = parvals[f"{self._prefix}0"]
        r2 = parvals[f"{self._prefix}_r2"]
        phase = parvals.get(f"{self._prefix_phase}p", 0.0)
        j_hz = parvals.get(f"{self._prefix}_j", 0.0)

        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        # Convert to radians/s (FID kernels expect this unit)
        dx_rads = self.spec_params.pts2hz_delta(dx_pt) * 2 * np.pi

        # Retrieve kernel from registry
        kernel_func = KERNEL_REGISTRY[self.shape_name]

        # Prepare arguments
        if self.needs_apod_params:
            args = (
                r2,
                self.spec_params.aq_time,
                self.spec_params.apodq2,
                self.spec_params.apodq1,
                phase,
            )
        else:
            args = (r2, self.spec_params.aq_time, phase)

        # Execute kernel (handles J-coupling and normalization internally)
        return sign * kernel_func(dx_rads, j_hz, *args)


# Register all shapes (all treated equally)
@register_shape("gaussian")
class Gaussian(FreqShape):
    """Gaussian lineshape."""

    shape_func = staticmethod(functions.gaussian)
    shape_name = "gaussian"


@register_shape("lorentzian")
class Lorentzian(FreqShape):
    """Lorentzian lineshape."""

    shape_func = staticmethod(functions.lorentzian)
    shape_name = "lorentzian"


@register_shape("pvoigt")
class PseudoVoigt(FreqShape):
    """Pseudo-Voigt lineshape."""

    shape_func = staticmethod(functions.pvoigt)
    shape_name = "pvoigt"
    needs_eta = True


@register_shape("no_apod")
class NoApod(TimeShape):
    """Non-apodized FID-based frequency-domain lineshape."""

    shape_func = staticmethod(functions.no_apod)
    shape_name = "no_apod"


@register_shape("sp1")
class SP1(TimeShape):
    """SP1 apodization FID-based frequency-domain lineshape."""

    shape_func = staticmethod(functions.sp1)
    shape_name = "sp1"
    needs_apod_params = True


@register_shape("sp2")
class SP2(TimeShape):
    """SP2 apodization FID-based frequency-domain lineshape."""

    shape_func = staticmethod(functions.sp2)
    shape_name = "sp2"
    needs_apod_params = True


# Legacy exports
Shape = BaseShape
PeakShape = FreqShape
ApodShape = TimeShape
