"""Peak representation and parameter management.

This module provides the Peak class for representing NMR peaks and
functions for creating fitting parameters.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from peakfit.core.fitting import Parameters
from peakfit.shapes import SHAPES, Shape
from peakfit.spectra import Spectra
from peakfit.typing import FittingOptions, FloatArray, IntArray


@dataclass
class Peak:
    """Represents a single NMR peak with its shapes in each dimension.

    Attributes:
        name: Peak identifier
        positions: Peak positions in ppm for each dimension
        shapes: List of Shape objects for each dimension
        positions_start: Initial positions (set automatically)
    """

    name: str
    positions: FloatArray
    shapes: list[Shape]
    positions_start: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        """Initialize derived fields after dataclass construction."""
        self.positions_start = self.positions.copy()

    def set_cluster_id(self, cluster_id: int) -> None:
        """Set cluster ID for all shapes.

        Args:
            cluster_id: Cluster identifier
        """
        for shape in self.shapes:
            shape.cluster_id = cluster_id

    def create_params(self) -> Parameters:
        """Create parameters for all shapes in this peak.

        Returns:
            Parameters object with all shape parameters
        """
        params = Parameters()
        for shape in self.shapes:
            params.update(shape.create_params())
        return params

    def fix_params(self, params: Parameters) -> None:
        """Fix all parameters for this peak.

        Args:
            params: Parameters to modify
        """
        for shape in self.shapes:
            shape.fix_params(params)

    def release_params(self, params: Parameters) -> None:
        """Release (unfreeze) all parameters for this peak.

        Args:
            params: Parameters to modify
        """
        for shape in self.shapes:
            shape.release_params(params)

    def evaluate(self, grid: Sequence[IntArray], params: Parameters) -> FloatArray:
        """Evaluate peak shape at grid points.

        The total peak shape is the product of shapes in each dimension.

        Args:
            grid: List of point arrays for each dimension
            params: Current parameter values

        Returns:
            Evaluated peak shape (product over dimensions)
        """
        evaluations = [
            shape.evaluate(pts, params)
            for pts, shape in zip(grid, self.shapes, strict=False)
        ]
        return np.prod(evaluations, axis=0)

    def print(self, params: Parameters) -> str:
        """Format peak parameters as string.

        Args:
            params: Current parameter values

        Returns:
            Formatted string with peak information
        """
        result = f"# Name: {self.name}\n"
        result += "\n".join(shape.print(params) for shape in self.shapes)
        return result

    @property
    def positions_i(self) -> IntArray:
        """Peak positions as integer points."""
        return np.array([shape.center_i for shape in self.shapes], dtype=np.int_)

    @property
    def positions_hz(self) -> FloatArray:
        """Peak positions in Hz."""
        return np.array(
            [shape.spec_params.pts2hz(shape.center_i) for shape in self.shapes],
            dtype=np.float64,
        )

    def update_positions(self, params: Parameters) -> None:
        """Update peak positions from fitted parameters.

        Args:
            params: Fitted parameter values
        """
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
    args: FittingOptions,
) -> Peak:
    """Create a Peak object from positions and shape names.

    Args:
        name: Peak identifier
        positions: Peak positions in ppm for each dimension
        shape_names: Name of lineshape to use for each dimension
        spectra: Spectra object with spectral parameters
        args: Command-line arguments

    Returns:
        Peak object ready for fitting
    """
    shapes = [
        SHAPES[shape_name](name, center, spectra, dim, args)
        for dim, (center, shape_name) in enumerate(
            zip(positions, shape_names, strict=False), start=1
        )
    ]
    return Peak(name, np.array(positions), shapes)


def create_params(peaks: list[Peak], *, fixed: bool = False) -> Parameters:
    """Create combined parameters for all peaks.

    Args:
        peaks: List of peaks to create parameters for
        fixed: If True, fix position parameters (don't vary during fitting)

    Returns:
        Parameters object with all peak parameters
    """
    params = Parameters()
    for peak in peaks:
        params.update(peak.create_params())

    if fixed:
        for name in params:
            if name.endswith("0"):
                params[name].vary = False

    return params
