"""On-the-fly spectra reconstruction from fitting results."""

from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.fitting.computation import calculate_shape_heights
from peakfit.io.readers import ResultsLoader
from peakfit.io.state import default_state_path, load_state

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.domain.state import FittingState
    from peakfit.shared.typing import FloatArray


_MIN_SHAPES_FOR_HZ_PER_POINT = 2


class SpectraReconstructor:
    """Reconstructs simulated spectra from persisted fitting state on demand."""

    def __init__(self, results_dir: Path) -> None:
        """Initialize reconstructor with path to results directory.

        Args:
            results_dir: Path to results directory containing summary/fit_summary.json
        """
        self.results_dir = results_dir
        self._loader = ResultsLoader(results_dir)
        self._state: FittingState | None = None
        self._cluster_all: Cluster | None = None
        self._amplitudes: FloatArray | None = None

    @property
    def state(self) -> FittingState:
        """Lazy-loaded fitting state."""
        if self._state is None:
            state_path = default_state_path(self.results_dir)
            if state_path.exists():
                self._state = load_state(state_path)
            else:
                self._state = self._loader.load_fitting_state()
        return self._state

    def _prepare_model(self) -> None:
        """Prepare unified cluster model and amplitudes if not already done."""
        if self._cluster_all is not None:
            return

        # Calculate all shape heights (intensities)
        amplitudes_list: list[FloatArray] = []
        for cluster in self.state.clusters:
            # Note: calculate_shape_heights needs the shape information.
            # In FittingState, clusters have peaks, and peaks have shapes.
            # We must ensure the parameters in self.state.scalar_params allow evaluation.
            _shapes, amplitudes = calculate_shape_heights(self.state.scalar_params, cluster)
            amplitudes_list.append(amplitudes)

        if not amplitudes_list:
            self._amplitudes = np.array([])
            return

        self._amplitudes = np.concatenate(amplitudes_list)

        # Merge all clusters into one for vectorized evaluation
        self._cluster_all = Cluster.from_clusters(self.state.clusters)

    def reconstruct_plane(self, plane_index: int, grid_shape_2d: tuple[int, int]) -> FloatArray:
        """Reconstruct a single 2D plane of the spectrum.

        Args:
            plane_index: Index of the plane (Z-dimension)
            grid_shape_2d: Shape of the 2D plane (Y, X)

        Returns:
        -------
            2D numpy array containing the simulated spectrum for this plane.
        """
        # Ensure model is ready
        self._prepare_model()

        if self._cluster_all is None or self._amplitudes is None or len(self._amplitudes) == 0:
            return np.zeros(grid_shape_2d, dtype=np.float64)

        simulated = np.zeros(grid_shape_2d, dtype=np.float64)

        # Pre-calculate Hertz per point for optimization logic
        hz_per_pt = self._calculate_hz_per_point(self._cluster_all)

        for i, peak in enumerate(self._cluster_all.peaks):
            amp = self._amplitudes[i]

            # Amplitude for this specific plane
            if np.ndim(amp) == 0:
                current_amp = float(amp)
            else:
                # Vector amplitude (n_series,)
                current_amp = float(amp[plane_index]) if plane_index < len(amp) else 0.0

            if current_amp == 0:
                continue

            # Evaluate peak
            self._evaluate_peak_on_grid(
                peak,
                current_amp,
                simulated,
                self.state.scalar_params,
                grid_shape_2d,
                hz_per_pt,
            )

        return simulated

    def _evaluate_peak_on_grid(
        self,
        peak: Any,
        current_amp: float,
        simulated: FloatArray,
        params: Any,
        grid_shape_2d: tuple[int, int],
        hz_per_pt: tuple[float, float],
    ) -> None:
        """Evaluate a single peak on the grid, optimizing for bounding box if possible."""
        y_hz_per_pt, x_hz_per_pt = hz_per_pt
        cutoff_factor = 4.0

        # Get peak parameters
        y_shape = peak.shapes[0]
        x_shape = peak.shapes[1]

        # Get parameter specs to identify position and linewidth names
        y_specs = y_shape.get_parameter_spec()
        x_specs = x_shape.get_parameter_spec()

        # Position is generally the first parameter
        y_pos_name = y_specs[0].name
        x_pos_name = x_specs[0].name

        # Attempt to find linewidth parameter names
        y_lw_name = next((s.name for s in y_specs if s.name in ("lw", "fwhm", "sigma")), None)
        x_lw_name = next((s.name for s in x_specs if s.name in ("lw", "fwhm", "sigma")), None)

        # Get values using the unified interface
        y_cluster_params = y_shape.get_cluster_parameters([peak], params)
        x_cluster_params = x_shape.get_cluster_parameters([peak], params)

        y_pos_val = float(y_cluster_params.get(y_pos_name).item())
        x_pos_val = float(x_cluster_params.get(x_pos_name).item())

        # If we can't determine linewidth, we can't optimize the bounding box safely
        if y_lw_name and x_lw_name:
            y_fwhm_val = float(y_cluster_params.get(y_lw_name).item())
            x_fwhm_val = float(x_cluster_params.get(x_lw_name).item())

            # Convert to points
            y_cen_pt = y_shape.dim_ctx.ppm2pts(y_pos_val)
            y_fwhm_pt = y_fwhm_val / y_hz_per_pt

            x_cen_pt = x_shape.dim_ctx.ppm2pts(x_pos_val)
            x_fwhm_pt = x_fwhm_val / x_hz_per_pt
        else:
            # Force fallback to full grid
            y_cen_pt = 0.0
            y_fwhm_pt = float("inf")
            x_cen_pt = 0.0
            x_fwhm_pt = float("inf")

        # Define Box
        y_margin = cutoff_factor * y_fwhm_pt
        x_margin = cutoff_factor * x_fwhm_pt

        y_min = int(np.floor(y_cen_pt - y_margin))
        y_max = int(np.ceil(y_cen_pt + y_margin))

        x_min = int(np.floor(x_cen_pt - x_margin))
        x_max = int(np.ceil(x_cen_pt + x_margin))

        # Check bounds (strict containment to avoid aliasing complexity)
        if y_min >= 0 and y_max < grid_shape_2d[0] and x_min >= 0 and x_max < grid_shape_2d[1]:
            # Fast Path: Evaluate Subgrid
            y_slice = slice(y_min, y_max)
            x_slice = slice(x_min, x_max)

            # Construct coordinate mesh for the slice
            yy, xx = np.mgrid[y_slice, x_slice]
            grid_indices = [yy.ravel(), xx.ravel()]

            # Evaluate (single source of truth)
            vals = Cluster.evaluate_peaks([peak], params, grid_indices)[0]

            # Add to simulation
            simulated[y_slice, x_slice] += (current_amp * vals).reshape(
                y_max - y_min, x_max - x_min
            )

        else:
            # Slow Path: Full Grid (handles aliasing/wrapping/edges)
            full_grid = np.indices(grid_shape_2d)
            grid_indices = [indices.ravel() for indices in full_grid]
            vals = Cluster.evaluate_peaks([peak], params, grid_indices)[0]
            simulated += (current_amp * vals).reshape(grid_shape_2d)

    def _calculate_hz_per_point(self, cluster: Cluster) -> tuple[float, float]:
        """Calculate Hz per point for Y and X dimensions."""
        y_hz_per_pt = 1.0
        x_hz_per_pt = 1.0

        if len(cluster.peaks) > 0:
            first_peak = cluster.peaks[0]
            if len(first_peak.shapes) >= _MIN_SHAPES_FOR_HZ_PER_POINT:
                # Calculate Hz per point
                # pts2hz_delta(1.0) returns the Hz delta for 1 point
                y_hz_per_pt = abs(first_peak.shapes[0].dim_ctx.pts2hz_delta(np.array([1.0]))[0])
                x_hz_per_pt = abs(first_peak.shapes[1].dim_ctx.pts2hz_delta(np.array([1.0]))[0])

        return y_hz_per_pt, x_hz_per_pt
