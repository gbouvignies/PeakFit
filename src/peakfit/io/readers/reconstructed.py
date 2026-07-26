"""Minimal reconstructed shapes for result-reader fallback state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.types import ClusterParameters, LineshapeResult, ParamSpec

if TYPE_CHECKING:
    from peakfit.shared.typing import FloatArray


@dataclass
class ReconstructedShape:
    """Minimal shape used when rebuilding state from summary JSON alone."""

    center: float
    axis: str
    name: str = ""
    shape_name: str = "reconstructed"
    param_names: list[str] = field(default_factory=list)
    cluster_id: int = 0

    @property
    def dim_ctx(self) -> Any:
        """Dimension context is unavailable for reconstructed shapes."""
        return None

    @property
    def center_i(self) -> int:
        """Integer index of the center position."""
        return int(self.center)

    def print(self, _params: Parameters) -> str:
        """Return string representation."""
        return f"# ReconstructedShape: {self.name}"

    def evaluate(self, x_pt: Any, _params: Parameters) -> Any:
        """Evaluate lineshape, returning zeros for reconstructed shapes."""
        return np.zeros_like(x_pt)

    def evaluate_derivatives(
        self, x_pt: FloatArray, _params: Parameters
    ) -> tuple[FloatArray, dict[str, FloatArray]]:
        """Evaluate lineshape with derivatives."""
        return np.zeros_like(x_pt), {}

    def create_params(self) -> Parameters:
        """Create empty parameters."""
        return Parameters()

    def get_parameter_spec(self) -> list[ParamSpec]:
        """Get parameter specifications."""
        return []

    def fix_params(self, params: Parameters) -> None:
        """Fix parameters."""

    def release_params(self, params: Parameters) -> None:
        """Release parameters."""

    def evaluate_cluster(
        self,
        x_grid: Any,
        cluster_params: ClusterParameters,
        _compute_derivs: bool = False,
    ) -> LineshapeResult:
        """Evaluate for cluster, returning zeros."""
        n_points = len(x_grid)
        n_peaks = cluster_params.n_peaks if cluster_params.n_peaks > 0 else 1
        return LineshapeResult(values=np.zeros((n_points, n_peaks)))

    def get_cluster_parameters(
        self, _peaks: Any, _params: Parameters, _param_map: dict[str, int] | None = None
    ) -> ClusterParameters:
        """Get cluster parameters."""
        return ClusterParameters()


__all__ = ["ReconstructedShape"]
