from collections.abc import Sequence

import lmfit as lf
import numpy as np
from numpy.typing import NDArray

from peakfit.shapes import SHAPES, Shape
from peakfit.spectra import Spectra

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


class Peak:
    def __init__(self, name: str, positions: FloatArray, shapes: list[Shape]) -> None:
        self.name = name
        self.positions = positions
        self.positions_start = positions
        self.shapes = shapes

    def create_params(self) -> lf.Parameters:
        params = lf.Parameters()
        for shape in self.shapes:
            params.update(shape.create_params())
        return params

    def evaluate(self, grid: Sequence[IntArray], params: lf.Parameters) -> FloatArray:
        evaluations = [
            shape.evaluate(pts, params)
            for pts, shape in zip(grid, self.shapes, strict=False)
        ]
        return np.prod(evaluations, axis=0)

    def print(self, params: lf.Parameters) -> str:
        return "\n".join(shape.print(params) for shape in self.shapes)

    @property
    def positions_i(self) -> IntArray:
        return np.array([shape.center_i for shape in self.shapes], dtype=np.int_)

    def update_positions(self, params: lf.Parameters) -> None:
        self.positions = np.array(
            [params[f"{shape.prefix}0"].value for shape in self.shapes]
        )
        for shape, position in zip(self.shapes, self.positions, strict=False):
            shape.center = position


def create_peak(
    name: str,
    positions: Sequence[float],
    shape_names: list[str],
    spectra: Spectra,
) -> Peak:
    shapes = [
        SHAPES[shape_name](name, center, spectra, dim)
        for dim, (center, shape_name) in enumerate(
            zip(positions, shape_names, strict=False), start=1
        )
    ]
    return Peak(name, np.array(positions), shapes)


def create_params(peaks: list[Peak], *, fixed: bool = False) -> lf.Parameters:
    params = lf.Parameters()
    for peak in peaks:
        params.update(peak.create_params())

    if fixed:
        for name in params:
            if name.endswith("0"):
                params[name].vary = False

    return params
