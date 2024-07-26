import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Protocol, TypeVar

import lmfit as lf
import numpy as np

from peakfit.cli import Arguments
from peakfit.nmrpipe import SpectralParameters
from peakfit.spectra import Spectra
from peakfit.typing import FloatArray, IntArray

T = TypeVar("T")

AXIS_NAMES = ("x", "y", "z", "a")


def clean(name: str) -> str:
    return re.sub(r"\W+|^(?=\d)", "_", name)


class Shape(Protocol):
    axis: str
    name: str
    cluster_id: int
    center: float
    spec_params: SpectralParameters
    size: int

    def create_params(self) -> lf.Parameters: ...
    def fix_params(self, params: lf.Parameters) -> None: ...
    def release_params(self, params: lf.Parameters) -> None: ...
    def evaluate(self, x_pt: IntArray, params: lf.Parameters) -> FloatArray: ...
    def print(self, params: lf.Parameters) -> str: ...
    @property
    def center_i(self) -> int: ...
    @property
    def prefix(self) -> str: ...


SHAPES: dict[str, Callable[..., Shape]] = {}


def register_shape(
    shape_names: str | Iterable[str],
) -> Callable[[type[Shape]], type[Shape]]:
    if isinstance(shape_names, str):
        shape_names = [shape_names]

    def decorator(shape_class: type[Shape]) -> type[Shape]:
        for name in shape_names:
            SHAPES[name] = shape_class
        return shape_class

    return decorator


def gaussian(dx: FloatArray, fwhm: float) -> FloatArray:
    return np.exp(-(dx**2) * 4 * np.log(2) / (fwhm**2))


def lorentzian(dx: FloatArray, fwhm: float) -> FloatArray:
    return (0.5 * fwhm) ** 2 / (dx**2 + (0.5 * fwhm) ** 2)


def pvoigt(dx: FloatArray, fwhm: float, eta: float) -> FloatArray:
    return (1.0 - eta) * gaussian(dx, fwhm) + eta * lorentzian(dx, fwhm)


def no_apod(dx: FloatArray, r2: float, aq: float, phase: float = 0.0) -> FloatArray:
    z1 = aq * (1j * dx + r2)
    spec = aq * (1.0 - np.exp(-z1)) / z1
    return (spec * np.exp(1j * np.deg2rad(phase))).real


def sp1(
    dx: FloatArray, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> FloatArray:
    z1 = aq * (1j * dx + r2)
    f1, f2 = 1j * off * np.pi, 1j * (end - off) * np.pi
    a1 = (np.exp(+f2) - np.exp(+z1)) * np.exp(-z1 + f1) / (2 * (z1 - f2))
    a2 = (np.exp(+z1) - np.exp(-f2)) * np.exp(-z1 - f1) / (2 * (z1 + f2))
    spec = 1j * aq * (a1 + a2)
    return (spec * np.exp(1j * np.deg2rad(phase))).real


def sp2(
    dx: FloatArray, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> FloatArray:
    z1 = aq * (1j * dx + r2)
    f1, f2 = 1j * off * np.pi, 1j * (end - off) * np.pi
    a1 = (np.exp(+2 * f2) - np.exp(z1)) * np.exp(-z1 + 2 * f1) / (4 * (z1 - 2 * f2))
    a2 = (np.exp(-2 * f2) - np.exp(z1)) * np.exp(-z1 - 2 * f1) / (4 * (z1 + 2 * f2))
    a3 = (1.0 - np.exp(-z1)) / (2 * z1)
    spec = aq * (a1 + a2 + a3)
    return (spec * np.exp(1j * np.deg2rad(phase))).real


class BaseShape(ABC):
    def __init__(
        self, name: str, center: float, spectra: Spectra, dim: int, args: Arguments
    ) -> None:
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
        return clean(f"{self.name}_{self.axis}")

    @property
    def prefix_phase(self) -> str:
        return clean(f"{self.cluster_id}_{self.axis}")

    @abstractmethod
    def create_params(self) -> lf.Parameters: ...

    def fix_params(self, params: lf.Parameters) -> None:
        for name in self.param_names:
            params[name].vary = False

    def release_params(self, params: lf.Parameters) -> None:
        for name in self.param_names:
            params[name].vary = True

    @abstractmethod
    def evaluate(self, x_pt: IntArray, params: lf.Parameters) -> FloatArray: ...

    def print(self, params: lf.Parameters) -> str:
        lines = []
        for name in self.param_names:
            fullname = name
            shortname = name.replace(self.prefix[:-1], "").replace(
                self.prefix_phase[:-1], ""
            )
            value = params[fullname].value
            stderr = params[fullname].stderr
            stderr_str = stderr if stderr is not None else 0.0
            line = f"# {shortname:<10s}: {value:10.5f} ± {stderr_str:10.5f}"
            lines.append(line)
        return "\n".join(lines)

    @property
    def center_i(self) -> int:
        return self.spec_params.ppm2pt_i(self.center)

    def _compute_dx_and_sign(
        self, x_pt: IntArray, x0: float
    ) -> tuple[FloatArray, FloatArray]:
        x0_pt = self.spec_params.ppm2pts(x0)
        dx_pt = x_pt - x0_pt
        if not self.spec_params.direct:
            aliasing = (dx_pt + 0.5 * self.size) // self.size
        else:
            aliasing = np.zeros_like(dx_pt)
        dx_pt_corrected = dx_pt - self.size * aliasing
        sign = (
            np.power(-1.0, aliasing)
            if self.spec_params.p180
            else np.ones_like(aliasing)
        )
        return dx_pt_corrected, sign


class PeakShape(BaseShape):
    FWHM_START = 25.0
    shape_func: Callable

    def create_params(self) -> lf.Parameters:
        params = lf.Parameters()
        params.add(
            f"{self.prefix}0",
            value=self.center,
            min=self.center - self.spec_params.hz2ppm(self.FWHM_START),
            max=self.center + self.spec_params.hz2ppm(self.FWHM_START),
        )
        params.add(f"{self.prefix}_fwhm", value=self.FWHM_START, min=0.1, max=200.0)
        self.param_names = list(params.keys())
        return params

    def evaluate(self, x_pt: IntArray, params: lf.Parameters) -> FloatArray:
        x0 = params[f"{self.prefix}0"].value
        fwhm = params[f"{self.prefix}_fwhm"].value
        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_hz = self.spec_params.pts2hz_delta(dx_pt)
        return sign * self.shape_func(dx_hz, fwhm)


@register_shape("lorentzian")
class Lorentzian(PeakShape):
    shape_func = staticmethod(lorentzian)


@register_shape("gaussian")
class Gaussian(PeakShape):
    shape_func = staticmethod(gaussian)


@register_shape("pvoigt")
class PseudoVoigt(PeakShape):
    def create_params(self) -> lf.Parameters:
        params = super().create_params()
        params.add(f"{self.prefix}_eta", value=0.5, min=-1.0, max=1.0)
        self.param_names = list(params.keys())
        return params

    def evaluate(self, x_pt: IntArray, params: lf.Parameters) -> FloatArray:
        x0 = params[f"{self.prefix}0"].value
        fwhm = params[f"{self.prefix}_fwhm"].value
        eta = params[f"{self.prefix}_eta"].value
        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_hz = self.spec_params.pts2hz_delta(dx_pt)
        return sign * pvoigt(dx_hz, fwhm, eta)


class ApodShape(BaseShape):
    R2_START = 20.0
    FWHM_START = 25.0
    shape_func: Callable

    def create_params(self) -> lf.Parameters:
        params = lf.Parameters()
        params.add(
            f"{self.prefix}0",
            value=self.center,
            min=self.center - self.spec_params.hz2ppm(self.FWHM_START),
            max=self.center + self.spec_params.hz2ppm(self.FWHM_START),
        )
        params.add(f"{self.prefix}_r2", value=self.R2_START, min=0.1, max=200.0)
        if self.args.jx and self.spec_params.direct:
            params.add(f"{self.prefix}_j", value=5.0, min=1.0, max=10.0)
        if (self.args.phx and self.spec_params.direct) or self.args.phy:
            params.add(f"{self.prefix_phase}p", value=0.0, min=-5.0, max=5.0)
        self.param_names = list(params.keys())
        return params

    def evaluate(self, x_pt: IntArray, params: lf.Parameters) -> FloatArray:
        parvalues = params.valuesdict()
        x0 = parvalues[f"{self.prefix}0"]
        r2 = parvalues[f"{self.prefix}_r2"]
        p0 = parvalues.get(f"{self.prefix_phase}p", 0.0)
        j_hz = parvalues.get(f"{self.prefix}_j", 0.0)

        dx_pt, sign = self._compute_dx_and_sign(self.full_grid, x0)
        dx_rads = self.spec_params.pts2hz_delta(dx_pt) * 2 * np.pi
        j_rads = (
            np.array([[0.0]]).T
            if j_hz == 0.0
            else j_hz * np.pi * np.array([[1.0, -1.0]]).T
        )
        dx_rads = dx_rads + j_rads

        shape_args = (r2, self.spec_params.aq_time)
        if self.shape_func in (sp1, sp2):
            shape_args += (self.spec_params.apodq2, self.spec_params.apodq1)
        shape_args += (p0,)

        norm = np.sum(self.shape_func(j_rads, *shape_args), axis=0)
        shape = np.sum(self.shape_func(dx_rads, *shape_args), axis=0)

        return sign[x_pt] * shape[x_pt] / norm


@register_shape("no_apod")
class NoApod(ApodShape):
    shape_func = staticmethod(no_apod)


@register_shape("sp1")
class SP1(ApodShape):
    shape_func = staticmethod(sp1)


@register_shape("sp2")
class SP2(ApodShape):
    shape_func = staticmethod(sp2)
