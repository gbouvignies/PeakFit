from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.number]
ArrayInt = NDArray[np.int_]


@dataclass()
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
    delta: float = field(init=False)
    first: float = field(init=False)

    def __post_init__(self) -> None:
        # derived units (these are in ppm)
        self.delta = -self.sw / (self.size * self.obs)
        self.first = self.car / self.obs - self.delta * self.size / 2.0

    def hz2pts_delta[T: (float, Array)](self, hz: T) -> T:
        return hz / (self.obs * self.delta)

    def pts2hz_delta[T: (float, Array)](self, pts: T) -> T:
        return pts * self.obs * self.delta

    def hz2pts[T: (float, Array)](self, hz: T) -> T:
        return ((hz / self.obs) - self.first) / self.delta

    def hz2pt_i(self, hz: float) -> int:
        return int(round(self.hz2pts(hz))) % self.size

    def pts2hz[T: (float, Array)](self, pts: T) -> T:
        return (pts * self.delta + self.first) * self.obs

    def ppm2pts[T: (float, Array)](self, ppm: T) -> T:
        return (ppm - self.first) / self.delta

    def ppm2pt_i(self, ppm: float) -> int:
        return int(round(self.ppm2pts(ppm))) % self.size

    def pts2ppm[T: (float, Array)](self, pts: T) -> T:
        return (pts * self.delta) + self.first

    def hz2ppm[T: (float, Array)](self, hz: T) -> T:
        return hz / self.obs


def read_spectral_parameters(
    dic: dict[str, Any], data: Array
) -> list[SpectralParameters]:
    fdf_in_order = [f"FDF{int(dic[f'FDDIMORDER{i}'])}" for i in range(1, data.ndim + 1)]
    direct = fdf_in_order[0]
    labels = ["FDSIZE", "FDSPECNUM", "FDF3SIZE", "FDF4SIZE"]
    ndsizes = {
        fdf: int(dic.get(label, 1))
        for fdf, label in zip(fdf_in_order, labels, strict=False)
    }

    spec_params = []
    for fdf in reversed(fdf_in_order):
        size = ndsizes[fdf]
        aq_time = calculate_acquisition_time(dic, fdf, size, direct)
        sw, obs, orig = (
            dic.get(f"{fdf}SW", 1.0),
            dic.get(f"{fdf}OBS", 1.0),
            dic.get(f"{fdf}ORIG", 0.0),
        )
        car = orig + sw / 2.0 - sw / size

        spec_params.append(
            SpectralParameters(
                size,
                sw,
                obs,
                car,
                aq_time,
                dic.get(f"{fdf}APODCODE", 0.0),
                dic.get(f"{fdf}APODQ1", 0.0),
                dic.get(f"{fdf}APODQ2", 0.0),
                dic.get(f"{fdf}APODQ3", 0.0),
            )
        )

    return spec_params


def calculate_acquisition_time(
    dic: dict[str, Any], fdf: str, size: int, direct: str
) -> float:
    aq_time = 0.0
    if dic.get(f"{fdf}FTSIZE", 0.0) != 0.0:
        aq_time = dic[f"{fdf}TDSIZE"] / dic[f"{fdf}SW"] * size / dic[f"{fdf}FTSIZE"]
        if fdf == direct:
            aq_time *= 1.0 - dic.get("FDDMXVAL", 0.0) / dic[f"{fdf}TDSIZE"]
        correction_factor = 1.0 if dic.get(f"{fdf}APODCODE", 0.0) == 1.0 else 0.5
        aq_time *= 1.0 - correction_factor / dic[f"{fdf}TDSIZE"]
    return aq_time
