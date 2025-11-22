"""NMR spectrum representation and NMRPipe parameter handling."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from nmrglue.fileio.pipe import guess_udic, read
from numpy.typing import NDArray

from peakfit.typing import FittingOptions, FloatArray

ArrayInt = NDArray[np.int_]
T = TypeVar("T", float, FloatArray)

P1_MIN = 175.0
P1_MAX = 185.0


@dataclass
class SpectralParameters:
    size: int
    sw: float
    obs: float
    car: float
    aq_time: float
    apocode: float
    apodq1: float
    apodq2: float
    apodq3: float
    p180: bool
    direct: bool
    ft: bool
    delta: float = field(init=False)
    first: float = field(init=False)

    def __post_init__(self) -> None:
        # derived units (these are in ppm)
        self.delta = -self.sw / (self.size * self.obs) if self.size * self.obs != 0.0 else 0.0
        self.first = self.car / self.obs - self.delta * self.size / 2.0 if self.obs != 0.0 else 0.0

    def hz2pts_delta(self, hz: T) -> T:
        return hz / (self.obs * self.delta)

    def pts2hz_delta(self, pts: T) -> T:
        return pts * self.obs * self.delta

    def hz2pts(self, hz: T) -> T:
        return ((hz / self.obs) - self.first) / self.delta

    def hz2pt_i(self, hz: float) -> int:
        return round(self.hz2pts(hz)) % self.size

    def pts2hz(self, pts: T) -> T:
        return (pts * self.delta + self.first) * self.obs

    def ppm2pts(self, ppm: T) -> T:
        return (ppm - self.first) / self.delta

    def ppm2pt_i(self, ppm: float) -> int:
        return round(self.ppm2pts(ppm)) % self.size

    def pts2ppm(self, pts: T) -> T:
        return (pts * self.delta) + self.first

    def hz2ppm(self, hz: T) -> T:
        return hz / self.obs


def read_spectral_parameters(dic: dict[str, Any], data: FloatArray) -> list[SpectralParameters]:
    spec_params: list[SpectralParameters] = []

    for i in range(data.ndim):
        size = data.shape[i]
        fdf = f"FDF{int(dic['FDDIMORDER'][data.ndim - 1 - i])}"
        is_direct = i == data.ndim - 1
        ft = dic.get(f"{fdf}FTFLAG", 0.0) == 1.0

        if ft:
            sw = dic.get(f"{fdf}SW", 1.0)
            orig = dic.get(f"{fdf}ORIG", 0.0)
            obs = dic.get(f"{fdf}OBS", 1.0)
            car = orig + sw / 2.0 - sw / size
            aq_time = dic.get(f"{fdf}APOD", 0.0) / max(sw, 1e-6)
            p180 = P1_MIN <= abs(dic.get(f"{fdf}P1", 0.0)) <= P1_MAX
        else:
            sw = obs = car = aq_time = 1.0
            p180 = False

        spec_params.append(
            SpectralParameters(
                size=size,
                sw=sw,
                obs=obs,
                car=car,
                aq_time=aq_time,
                apocode=dic.get(f"{fdf}APODCODE", 0.0),
                apodq1=dic.get(f"{fdf}APODQ1", 0.0),
                apodq2=dic.get(f"{fdf}APODQ2", 0.0),
                apodq3=dic.get(f"{fdf}APODQ3", 0.0),
                p180=p180,
                direct=is_direct,
                ft=ft,
            )
        )

    return spec_params


@dataclass
class Spectra:
    dic: dict
    data: FloatArray
    z_values: np.ndarray
    pseudo_dim_added: bool = False

    def __post_init__(self) -> None:
        udic = guess_udic(self.dic, self.data)
        no_pseudo_dim = udic[0]["freq"]
        if no_pseudo_dim:
            self.data = np.expand_dims(self.data, axis=0)
            self.pseudo_dim_added = True
        if self.z_values.size == 0:
            self.z_values = np.arange(self.data.shape[0])

    @cached_property
    def params(self) -> list[SpectralParameters]:
        return read_spectral_parameters(self.dic, self.data)

    def exclude_planes(self, exclude_list: Sequence[int] | None) -> None:
        if exclude_list is None:
            return
        mask = ~np.isin(range(self.data.shape[0]), exclude_list)
        self.data, self.z_values = self.data[mask], self.z_values[mask]


def read_spectra(
    path_spectra: Path,
    path_z_values: Path | None = None,
    exclude_list: Sequence[int] | None = None,
) -> Spectra:
    """Read NMRPipe spectra and z-values, returning a Spectra object."""
    dic, data = read(path_spectra)
    data = data.astype(np.float32)

    if path_z_values is not None:
        z_values = np.genfromtxt(path_z_values, dtype=None, encoding="utf-8")
    else:
        z_values = np.array([])

    spectra = Spectra(dic, data, z_values)
    spectra.exclude_planes(exclude_list)

    return spectra


def get_shape_names(clargs: FittingOptions, spectra: Spectra) -> list[str]:
    """Determine shape names for fitting based on CLI args or spectral parameters."""
    match (clargs.pvoigt, clargs.lorentzian, clargs.gaussian):
        case (True, _, _):
            shape = "pvoigt"
        case (_, True, _):
            shape = "lorentzian"
        case (_, _, True):
            shape = "gaussian"
        case _:
            return [determine_shape_name(param) for param in spectra.params[1:]]

    return [shape] * (spectra.data.ndim - 1)


def determine_shape_name(dim_params: SpectralParameters) -> str:
    """Determine the shape name based on spectral parameters."""
    if dim_params.apocode == 1.0:
        if dim_params.apodq3 == 1.0:
            return "sp1"
        if dim_params.apodq3 == 2.0:
            return "sp2"
    if dim_params.apocode in {0.0, 2.0}:
        return "no_apod"
    return "pvoigt"
