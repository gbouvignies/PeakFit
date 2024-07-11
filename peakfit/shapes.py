import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Protocol

import lmfit as lf
import numpy as np
import numpy.typing as npt

from peakfit.cli import Arguments
from peakfit.nmrpipe import SpectralParameters
from peakfit.spectra import Spectra

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int32]


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
    f1 = 1j * off * np.pi
    f2 = 1j * (end - off) * np.pi
    a1 = (np.exp(+f2) - np.exp(+z1)) * np.exp(-z1 + f1) / (2 * (z1 - f2))
    a2 = (np.exp(+z1) - np.exp(-f2)) * np.exp(-z1 - f1) / (2 * (z1 + f2))
    spec = 1j * aq * (a1 + a2)
    return (spec * np.exp(1j * np.deg2rad(phase))).real


def sp2(
    dx: FloatArray, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> FloatArray:
    z1 = aq * (1j * dx + r2)
    f1 = 1j * off * np.pi
    f2 = 1j * (end - off) * np.pi
    a1 = (np.exp(+2 * f2) - np.exp(z1)) * np.exp(-z1 + 2 * f1) / (4 * (z1 - 2 * f2))
    a2 = (np.exp(-2 * f2) - np.exp(z1)) * np.exp(-z1 - 2 * f1) / (4 * (z1 + 2 * f2))
    a3 = (1.0 - np.exp(-z1)) / (2 * z1)
    spec = aq * (a1 + a2 + a3)
    return (spec * np.exp(1j * np.deg2rad(phase))).real


class BaseShape(ABC):
    def __init__(
        self, name: str, center: float, spectra: Spectra, dim: int, args: Arguments
    ) -> None:
        """Initializes the Lorentzian object.

        Args:
            name (str): Peak name.
            center (float): Center value of the Pseudo Voigt profile.
            spectra (Spectra): Spectra object.
            dim (int): Dimension of the shape.
        """
        self.name = name
        self.axis = AXIS_NAMES[spectra.data[0].ndim - dim]
        self.center = center
        self.spec_params = spectra.params[dim]
        self.size = self.spec_params.size
        self.param_names: list[str] = []
        self.cluster_id = 0
        self.args = args

    @property
    def prefix(self) -> str:
        return clean(f"{self.name}_{self.axis}")

    @property
    def prefix_phase(self) -> str:
        return clean(f"{self.cluster_id}_{self.axis}")

    @abstractmethod
    def create_params(self) -> lf.Parameters:
        """Creates and initializes fitting parameters for the Pseudo Voigt profile.

        Returns:
            lf.Parameters: Initialized parameters.
        """

    def fix_params(self, params: lf.Parameters) -> None:
        """Fixes all parameters of the shape."""
        for name in self.param_names:
            params[name].vary = False

    def release_params(self, params: lf.Parameters) -> None:
        """Varies all parameters of the shape."""
        for name in self.param_names:
            params[name].vary = True

    @abstractmethod
    def evaluate(self, x_pt: IntArray, params: lf.Parameters) -> FloatArray:
        """Evaluates the Pseudo Voigt profile at given points.

        Args:
            x_pt (FloatArray): Array of x-values.
            params (lf.Parameters): Parameters of the Pseudo Voigt profile.

        Returns:
            FloatArray: Evaluated profile.
        """

    def print(self, params: lf.Parameters) -> str:
        """Prints the fitting parameters.

        Args:
            params (lf.Parameters): Parameters to print.

        Returns:
            str: Formatted parameter string.
        """
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


@register_shape("lorentzian")
class Lorentzian(BaseShape):
    FWHM_START = 25.0

    def create_params(self) -> lf.Parameters:
        params = lf.Parameters()
        params.add(
            f"{self.prefix}0",
            value=self.center,
            min=self.center - self.spec_params.hz2ppm(self.FWHM_START),
            max=self.center + self.spec_params.hz2ppm(self.FWHM_START),
        )
        params.add(f"{self.prefix}_fwhm", value=self.FWHM_START, min=0.1, max=200.0)
        self.param_names = [*params.keys()]
        return params

    def evaluate(self, x_pt: IntArray, params: lf.Parameters) -> FloatArray:
        x0 = params[f"{self.prefix}0"].value
        fwhm = params[f"{self.prefix}_fwhm"].value

        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_hz = self.spec_params.pts2hz_delta(dx_pt)

        return sign * lorentzian(dx_hz, fwhm)


@register_shape("gaussian")
class Gaussian(BaseShape):
    FWHM_START = 25.0

    def create_params(self) -> lf.Parameters:
        params = lf.Parameters()
        params.add(
            f"{self.prefix}0",
            value=self.center,
            min=self.center - self.spec_params.hz2ppm(self.FWHM_START),
            max=self.center + self.spec_params.hz2ppm(self.FWHM_START),
        )
        params.add(f"{self.prefix}_fwhm", value=self.FWHM_START, min=0.1, max=200.0)
        self.param_names = [*params.keys()]
        return params

    def evaluate(self, x_pt: IntArray, params: lf.Parameters) -> FloatArray:
        x0 = params[f"{self.prefix}0"].value
        fwhm = params[f"{self.prefix}_fwhm"].value

        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_hz = self.spec_params.pts2hz_delta(dx_pt)

        return sign * gaussian(dx_hz, fwhm)


@register_shape("pvoigt")
class PseudoVoigt(BaseShape):
    FWHM_START = 25.0

    def create_params(self) -> lf.Parameters:
        params = lf.Parameters()
        params.add(
            f"{self.prefix}0",
            value=self.center,
            min=self.center - self.spec_params.hz2ppm(self.FWHM_START),
            max=self.center + self.spec_params.hz2ppm(self.FWHM_START),
        )
        params.add(f"{self.prefix}_fwhm", value=self.FWHM_START, min=0.1, max=200.0)
        params.add(f"{self.prefix}_eta", value=0.5, min=-1.0, max=1.0)
        self.param_names = [*params.keys()]
        return params

    def evaluate(self, x_pt: IntArray, params: lf.Parameters) -> FloatArray:
        x0 = params[f"{self.prefix}0"].value
        fwhm = params[f"{self.prefix}_fwhm"].value
        eta = params[f"{self.prefix}_eta"].value

        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_hz = self.spec_params.pts2hz_delta(dx_pt)

        return sign * pvoigt(dx_hz, fwhm, eta)


@register_shape("no_apod")
class NoApod(BaseShape):
    R2_START = 20.0
    FWHM_START = 25.0

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
        x0 = parvalues.get(f"{self.prefix}0", 0.0)
        r2 = parvalues.get(f"{self.prefix}_r2", 0.0)
        p0 = parvalues.get(f"{self.prefix_phase}p", 0.0)
        j_hz = parvalues.get(f"{self.prefix}_j", 0.0)

        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_rads = self.spec_params.pts2hz_delta(dx_pt) * 2 * np.pi

        if j_hz == 0.0:
            j_rads = np.array([[0.0]]).T
        else:
            j_rads = j_hz * np.pi * np.array([[1.0, -1.0]]).T

        dx_rads = dx_rads + j_rads

        norm = np.sum(no_apod(j_rads, r2, self.spec_params.aq_time, p0), axis=0)
        shape = np.sum(no_apod(dx_rads, r2, self.spec_params.aq_time, p0), axis=0)

        return sign * shape / norm


@register_shape("sp1")
class SP1(BaseShape):
    R2_START = 20.0
    FWHM_START = 25.0

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
        x0 = parvalues.get(f"{self.prefix}0", 0.0)
        r2 = parvalues.get(f"{self.prefix}_r2", 0.0)
        p0 = parvalues.get(f"{self.prefix_phase}p", 0.0)
        j_hz = parvalues.get(f"{self.prefix}_j", 0.0)

        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_rads = self.spec_params.pts2hz_delta(dx_pt) * 2 * np.pi

        if j_hz == 0.0:
            j_rads = np.array([[0.0]]).T
        else:
            j_rads = j_hz * np.pi * np.array([[1.0, -1.0]]).T

        dx_rads = dx_rads + j_rads

        norm = np.sum(
            sp1(
                j_rads,
                r2,
                self.spec_params.aq_time,
                self.spec_params.apodq2,
                self.spec_params.apodq1,
                p0,
            ),
            axis=0,
        )
        shape = np.sum(
            sp1(
                dx_rads,
                r2,
                self.spec_params.aq_time,
                self.spec_params.apodq2,
                self.spec_params.apodq1,
                p0,
            ),
            axis=0,
        )

        return sign * shape / norm


@register_shape("sp2")
class SP2(BaseShape):
    R2_START = 20.0
    FWHM_START = 25.0

    def create_params(self) -> lf.Parameters:
        params = lf.Parameters()
        params.add(
            f"{self.prefix}0",
            value=self.center,
            min=self.center - self.spec_params.hz2ppm(self.FWHM_START),
            max=self.center + self.spec_params.hz2ppm(self.FWHM_START),
        )
        params.add(f"{self.prefix}_r2", value=self.R2_START, min=1.0, max=200.0)
        if self.args.jx and self.spec_params.direct:
            params.add(f"{self.prefix}_j", value=5.0, min=1.0, max=10.0)
        if (self.args.phx and self.spec_params.direct) or self.args.phy:
            params.add(f"{self.prefix_phase}p", value=0.0, min=-5.0, max=5.0)
        self.param_names = list(params.keys())
        return params

    def evaluate(self, x_pt: IntArray, params: lf.Parameters) -> FloatArray:
        parvalues = params.valuesdict()
        x0 = parvalues.get(f"{self.prefix}0", 0.0)
        r2 = parvalues.get(f"{self.prefix}_r2", 0.0)
        p0 = parvalues.get(f"{self.prefix_phase}p", 0.0)
        j_hz = parvalues.get(f"{self.prefix}_j", 0.0)

        dx_pt, sign = self._compute_dx_and_sign(x_pt, x0)
        dx_rads = self.spec_params.pts2hz_delta(dx_pt) * 2 * np.pi

        if j_hz == 0.0:
            j_rads = np.array([[0.0]]).T
        else:
            j_rads = j_hz * np.pi * np.array([[1.0, -1.0]]).T

        dx_rads = dx_rads + j_rads

        norm = np.sum(
            sp2(
                j_rads,
                r2,
                self.spec_params.aq_time,
                self.spec_params.apodq2,
                self.spec_params.apodq1,
                p0,
            ),
            axis=0,
        )
        shape = np.sum(
            sp2(
                dx_rads,
                r2,
                self.spec_params.aq_time,
                self.spec_params.apodq2,
                self.spec_params.apodq1,
                p0,
            ),
            axis=0,
        )

        return sign * shape / norm
