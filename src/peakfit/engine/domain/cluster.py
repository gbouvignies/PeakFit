"""Domain model representing a cluster of peaks."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.domain.peaks import Peak
from peakfit.shared.typing import FloatArray, IntArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.engine.domain.param_map import ParameterMap
    from peakfit.engine.domain.params_scalar import Parameters


@dataclass(slots=True)
class Cluster:
    """Grouped peaks sharing a contiguous spectral segment."""

    cluster_id: int
    peaks: list[Peak]
    grid_indices: list[IntArray]
    data: FloatArray
    corrections: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        """Initialize calculated fields."""
        self.data = np.asarray(self.data, dtype=np.float64)

        self.corrections = np.zeros_like(self.data, dtype=np.float64)

    @classmethod
    def from_clusters(cls, clusters: Sequence[Cluster]) -> Cluster:
        """Create a single cluster by merging a list of clusters."""
        if not clusters:
            msg = "clusters list cannot be empty"
            raise ValueError(msg)
        result = clusters[0]
        for other in clusters[1:]:
            result = result + other
        return result

    @property
    def corrected_data(self) -> FloatArray:
        """Return data with corrections subtracted (ready for processing)."""
        return self.data - self.corrections

    @property
    def n_series(self) -> int:
        """Return the number of spectra in the pseudo dimension."""
        return self.corrected_data.shape[0] if self.corrected_data.ndim > 1 else 1

    @property
    def n_amplitude_params(self) -> int:
        """Return the number of amplitude parameters (DOF)."""
        return len(self.peaks) * self.n_series

    def evaluate(
        self,
        params: Parameters,
        grid_indices: Sequence[IntArray] | None = None,
        d_matrix: np.ndarray | None = None,
        param_map: ParameterMap | None = None,
    ) -> np.ndarray:
        """Evaluate shapes for all peaks on the provided grid positions."""
        grid = list(self.grid_indices) if grid_indices is None else list(grid_indices)
        return self.evaluate_peaks(
            self.peaks,
            params,
            grid,
            d_matrix=d_matrix,
            param_map=param_map,
        )

    @staticmethod
    def evaluate_peaks(
        peaks: Sequence[Peak],
        params: Parameters,
        grid_indices: Sequence[IntArray],
        *,
        d_matrix: np.ndarray | None = None,
        param_map: ParameterMap | None = None,
    ) -> np.ndarray:
        """Vectorized evaluation for peaks on provided grid positions.

        Returns a consistent shapes matrix of shape (n_peaks, n_points) for any
        n_peaks >= 0.
        """
        if not grid_indices:
            msg = "grid_indices cannot be empty"
            raise ValueError(msg)

        grid_dims: list[np.ndarray] | np.ndarray = list(grid_indices)
        if isinstance(grid_dims, np.ndarray):
            grid_dims = [grid_dims]

        n_peaks = len(peaks)
        n_points = grid_dims[0].size
        n_dims = len(grid_dims)
        compute_derivatives = d_matrix is not None and param_map is not None

        # accum_val: (n_points, n_peaks) starting as ones
        accum_val = np.ones((n_points, n_peaks), dtype=np.float64)

        if n_peaks == 0:
            return accum_val.T

        dim_vals: list[np.ndarray] = []
        dim_deriv_maps: list[dict[str, FloatArray]] = []
        param_indices_map: list[dict[str, IntArray]] = []

        for d in range(n_dims):
            val, current_derivs, idx_map = _evaluate_single_dimension(
                d, peaks, params, grid_dims[d], param_map, compute_derivatives
            )

            dim_vals.append(val)
            accum_val *= val

            if compute_derivatives:
                dim_deriv_maps.append(current_derivs)
                param_indices_map.append(idx_map)

        if compute_derivatives and d_matrix is not None:
            for d in range(n_dims):
                others_prod = np.ones((n_points, n_peaks))
                for other_d in range(n_dims):
                    if other_d != d:
                        others_prod *= dim_vals[other_d]

                p_map = param_indices_map[d]
                deriv_map = dim_deriv_maps[d]

                for param_type, indices in p_map.items():
                    if param_type in deriv_map:
                        deriv_total = deriv_map[param_type] * others_prod
                        _accumulate_derivatives(d_matrix, indices, deriv_total)

        return accum_val.T

    def __add__(self, other: object) -> Cluster:
        """Concatenate two clusters, preserving peaks and data arrays."""
        if not isinstance(other, Cluster):
            return NotImplemented

        # Concatenate positions
        new_positions = [
            np.concatenate((positions_self, positions_other))
            for positions_self, positions_other in zip(
                self.grid_indices,
                other.grid_indices,
                strict=False,
            )
        ]

        # Concatenate data
        new_data = np.concatenate((self.data, other.data), axis=0)

        return type(self)(
            cluster_id=self.cluster_id,
            peaks=self.peaks + other.peaks,
            grid_indices=new_positions,
            data=new_data,
        )


def _evaluate_single_dimension(
    d: int,
    peaks: Sequence[Peak],
    params: Parameters,
    grid_dim: np.ndarray,
    param_map: ParameterMap | None,
    compute_derivatives: bool,
) -> tuple[FloatArray, dict[str, FloatArray], dict[str, IntArray]]:
    """Evaluate shapes for a single dimension.

    Uses the unified evaluation pipeline where the Shape class IS the evaluator.
    All parameters flow through ClusterParameters uniformly.
    """
    sample_shape = peaks[0].shapes[d]

    cluster_params = sample_shape.get_cluster_parameters(
        peaks, params, param_map=param_map if compute_derivatives else None
    )

    x_grid = grid_dim.ravel().astype(np.float64)
    result = sample_shape.evaluate_cluster(
        x_grid, cluster_params, compute_derivs=compute_derivatives
    )

    return result.values, result.derivatives, cluster_params.index_map


def _accumulate_derivatives(
    d_matrix: np.ndarray,
    indices: IntArray,
    deriv_total: FloatArray,
) -> None:
    """Accumulate derivatives for mapped parameters, supporting shared indices."""
    valid_mask = indices >= 0
    if not np.any(valid_mask):
        return

    np.add.at(d_matrix, indices[valid_mask], deriv_total[:, valid_mask].T)
