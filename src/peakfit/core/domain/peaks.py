"""Domain representation of peaks and related helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.core.fitting.parameters import Parameters
    from peakfit.core.shared.typing import FloatArray, IntArray


class Peak(BaseModel):
    """Represents a single NMR peak with parameterized shapes."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    positions: Any = Field(description="Peak positions (FloatArray)")
    shapes: list[Any] = Field(description="List of Shape objects")
    positions_start: Any = Field(
        default=None, init=False, description="Initial positions (FloatArray)"
    )

    @model_validator(mode="after")
    def initialize_positions_start(self) -> Peak:
        """Store a copy of initial positions for later reference."""
        if self.positions_start is None:
            if not isinstance(self.positions, np.ndarray):
                self.positions = np.array(self.positions, dtype=float)
            self.positions_start = self.positions.copy()
        return self

    def set_cluster_id(self, cluster_id: int) -> None:
        """Assign cluster_id to all shapes belonging to this peak."""
        for shape in self.shapes:
            shape.cluster_id = cluster_id

    def create_params(self) -> Parameters:
        """Create Parameters for each shape in this peak."""
        from peakfit.core.fitting.parameters import Parameters

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

    def evaluate(self, grid: Sequence[IntArray], params: Parameters) -> FloatArray:
        """Evaluate the peak's combined lineshape product across grid points."""
        raw_evals: list[FloatArray] = [
            np.asarray(shape.evaluate(pts, params), dtype=float)
            for pts, shape in zip(grid, self.shapes, strict=False)
        ]
        raw_evals_arr = np.stack(raw_evals, axis=0)
        evaluations: FloatArray = np.asarray(raw_evals_arr, dtype=float)
        prod_res = np.prod(evaluations, axis=0)
        result: FloatArray = np.asarray(prod_res, dtype=float)
        return result

    def evaluate_derivatives(
        self, grid: Sequence[IntArray], params: Parameters
    ) -> tuple[FloatArray, dict[str, FloatArray]]:
        """Evaluate peak and its derivatives w.r.t parameters."""
        evaluations = []
        shape_derivs = []

        for pts, shape in zip(grid, self.shapes, strict=False):
            val, derivs = shape.evaluate_derivatives(pts, params)
            evaluations.append(np.asarray(val, dtype=float))
            shape_derivs.append(derivs)

        if not evaluations:
            return np.array([]), {}

        raw_evals_arr = np.stack(evaluations, axis=0)
        peak_val = np.prod(raw_evals_arr, axis=0)

        # d(S1*S2*...)/dtheta = (dS_i/dtheta) * product(S_j for j!=i)
        total_derivs = {}

        for i, shape in enumerate(self.shapes):
            others = evaluations[:i] + evaluations[i + 1 :]
            if others:
                others_arr = np.stack(others, axis=0)
                others_prod = np.prod(others_arr, axis=0)
            else:
                others_prod = 1.0

            for param_name, d_shape in shape_derivs[i].items():
                total_derivs[param_name] = d_shape * others_prod

        return peak_val, total_derivs

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


def create_params(peaks: list[Peak], *, fixed: bool = False) -> Parameters:
    """Combine parameters from a list of Peak objects into a single Parameters."""
    from peakfit.core.fitting.parameters import Parameters

    params = Parameters()
    for peak in peaks:
        params.update(peak.create_params())

    if fixed:
        for name in params:
            if name.endswith("0"):
                params[name].vary = False

    return params


__all__ = ["Peak", "create_params"]
