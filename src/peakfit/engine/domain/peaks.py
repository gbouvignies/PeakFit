"""Domain representation of peaks and related helpers."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.types import Shape
from peakfit.shared.typing import FloatArray, IntArray

if TYPE_CHECKING:
    from peakfit.engine.types import Shape
    from peakfit.shared.typing import FloatArray, IntArray


@dataclass(slots=True)
class Peak:
    """Represents a single NMR peak with parameterized shapes.

    A peak is defined by its position in N-dimensions and its lineshape profile
    along each dimension.
    """

    name: str
    positions: FloatArray
    shapes: list[Shape]
    positions_start: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        """Initialize calculated fields."""
        self.positions = np.asarray(self.positions, dtype=np.float64)

        if len(self.positions) != len(self.shapes):
            msg = (
                f"Peak '{self.name}': dimensionality mismatch - "
                f"{len(self.positions)} positions but {len(self.shapes)} shapes"
            )
            raise ValueError(msg)

        # Store initial positions
        self.positions_start = self.positions.copy()

    def set_cluster_id(self, cluster_id: int) -> None:
        """Assign cluster_id to all shapes belonging to this peak."""
        for shape in self.shapes:
            shape.cluster_id = cluster_id

    def create_params(self) -> Parameters:
        """Create Parameters for each shape in this peak."""
        params = Parameters()
        for shape in self.shapes:
            params.update(shape.create_params())
        return params

    def fix_params(self, params: Parameters) -> None:
        """Fix (set vary=False) all parameters for this peak's shapes."""
        for shape in self.shapes:
            shape.fix_params(params)

    def release_params(self, params: Parameters) -> None:
        """Release (set vary=True) all parameters for this peak's shapes."""
        for shape in self.shapes:
            shape.release_params(params)

    def print(self, params: Parameters) -> str:
        """Return textual representation of the peak parameters for output."""
        result = f"# Name: {self.name}\n"
        result += "\n".join(shape.print(params) for shape in self.shapes)
        return result

    @property
    def positions_i(self) -> IntArray:
        """Integer position indices for each shape in the peak."""
        return np.array([shape.center_i for shape in self.shapes], dtype=np.int_)


__all__ = ["Peak"]
