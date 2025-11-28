"""Domain representation of peaks and related helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from peakfit.core.fitting.parameters import Parameters
from peakfit.core.lineshapes import LineshapeFactory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.core.lineshapes import Shape
    from peakfit.core.shared.typing import FittingOptions, FloatArray, IntArray

if TYPE_CHECKING:
    from peakfit.core.domain.spectrum import Spectra


@dataclass
class Peak:
    """Represents a single NMR peak with parameterized shapes."""

    name: str
    positions: FloatArray
    shapes: list[Shape]
    positions_start: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        """Store a copy of initial positions for later reference."""
        self.positions_start = self.positions.copy()

    def set_cluster_id(self, cluster_id: int) -> None:
        """Assign cluster_id to all shapes belonging to this peak."""
        for shape in self.shapes:
            shape.cluster_id = cluster_id

    def create_params(self) -> Parameters:
        """Create `Parameters` for each shape in this peak and return combined result."""
        params = Parameters()
        for shape in self.shapes:
            params.update(shape.create_params())
        return params

    def fix_params(self, params: Parameters) -> None:
        """Fix (set `vary` to False) all parameters for this peak's shapes."""
        for shape in self.shapes:
            shape.fix_params(params)

    def release_params(self, params: Parameters) -> None:
        """Release (set `vary` to True) all parameters for this peak's shapes."""
        for shape in self.shapes:
            shape.release_params(params)

    def evaluate(self, grid: Sequence[IntArray], params: Parameters) -> FloatArray:
        """Evaluate the peak's combined lineshape product across provided grid points."""
        raw_evals: list[FloatArray] = [
            np.asarray(shape.evaluate(pts, params), dtype=float)
            for pts, shape in zip(grid, self.shapes, strict=False)
        ]
        raw_evals_arr = np.stack(raw_evals, axis=0)
        evaluations: FloatArray = np.asarray(raw_evals_arr, dtype=float)
        prod_res = np.prod(evaluations, axis=0)
        result: FloatArray = np.asarray(prod_res, dtype=float)
        return result

    def print(self, params: Parameters) -> str:
        """Return textual representation of the peak parameters for output."""
        result = f"# Name: {self.name}\n"
        result += "\n".join(shape.print(params) for shape in self.shapes)
        return result

    @property
    def positions_i(self) -> IntArray:
        """Integer position indices for each shape in the peak."""
        return np.array([shape.center_i for shape in self.shapes], dtype=np.int_)

    @property
    def positions_hz(self) -> FloatArray:
        """Position centers in Hz for the peak shapes, converted from point indices."""
        return np.array(
            [shape.spec_params.pts2hz(shape.center_i) for shape in self.shapes],
            dtype=np.float64,
        )

    def update_positions(self, params: Parameters) -> None:
        """Update the peak's positions array based on parameter values from `params`."""
        self.positions = np.array([params[f"{shape.prefix}0"].value for shape in self.shapes])
        for shape, position in zip(self.shapes, self.positions, strict=False):
            shape.center = position


def create_peak(
    name: str,
    positions: Sequence[float],
    shape_names: list[str],
    spectra: Spectra,
    args: FittingOptions,
) -> Peak:
    """Create a `Peak` object with shapes constructed from `shape_names`.

    Args:
        name: Peak name
        positions: Positions per dimension (ppm)
        shape_names: Shape names per dimension
        spectra: Spectra metadata object
        args: CLI fitting options
    """
    factory = LineshapeFactory(spectra, args)
    shapes = factory.create_shapes(name, positions, shape_names)
    return Peak(name, np.array(positions), shapes)


def create_params(peaks: list[Peak], *, fixed: bool = False) -> Parameters:
    """Combine parameters from a list of `Peak` objects into a single `Parameters`.

    Args:
        peaks: List of peaks
        fixed: If True, set position parameters to not vary
    """
    params = Parameters()
    for peak in peaks:
        params.update(peak.create_params())

    if fixed:
        for name in params:
            if name.endswith("0"):
                params[name].vary = False

    return params


__all__ = ["Peak", "create_params", "create_peak"]
