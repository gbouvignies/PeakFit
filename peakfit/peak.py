from collections.abc import Sequence

import lmfit as lf
import numpy as np

from peakfit.cli import Arguments
from peakfit.shapes import SHAPES, Shape
from peakfit.spectra import Spectra
from peakfit.typing import FloatArray, IntArray


class Peak:
    def __init__(self, name: str, positions: FloatArray, shapes: list[Shape]) -> None:
        self.name = name
        self.positions = positions
        self.positions_start = positions
        self.shapes = shapes

    def set_cluster_id(self, cluster_id: int) -> None:
        for shape in self.shapes:
            shape.cluster_id = cluster_id

    def create_params(self) -> lf.Parameters:
        params = lf.Parameters()
        for shape in self.shapes:
            params.update(shape.create_params())
        return params

    def fix_params(self, params: lf.Parameters) -> None:
        for shape in self.shapes:
            shape.fix_params(params)

    def release_params(self, params: lf.Parameters) -> None:
        for shape in self.shapes:
            shape.release_params(params)

    def evaluate(self, grid: Sequence[IntArray], params: lf.Parameters) -> FloatArray:
        evaluations = [
            shape.evaluate(pts, params)
            for pts, shape in zip(grid, self.shapes, strict=False)
        ]
        return np.prod(evaluations, axis=0)

    def print(self, params: lf.Parameters) -> str:
        result = f"# Name: {self.name}\n"
        result += "\n".join(shape.print(params) for shape in self.shapes)
        return result

    @property
    def positions_i(self) -> IntArray:
        return np.array([shape.center_i for shape in self.shapes], dtype=np.int_)

    @property
    def positions_hz(self) -> FloatArray:
        return np.array(
            [shape.spec_params.pts2hz(shape.center_i) for shape in self.shapes],
            dtype=np.float64,
        )

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
    args: Arguments,
) -> Peak:
    shapes = [
        SHAPES[shape_name](name, center, spectra, dim, args)
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
