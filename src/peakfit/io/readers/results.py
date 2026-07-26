"""Read versioned completed-fit JSON and its limited continuation fallback."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.params_scalar import Parameter, Parameters
from peakfit.engine.domain.params_vector import FitParameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.state import FittingState
from peakfit.io.readers.reconstructed import ReconstructedShape
from peakfit.io.schemas import ClusterResultSchema, FitSummarySchema
from peakfit.shared.paths import format_path

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.types import Shape

__all__ = ["ResultsLoader"]
_PARAM_NAME_PARTS = 3
type MCMCChainRecord = tuple[Any, list[str], int, int, int]


class ResultsLoader:
    """Loader for PeakFit results from structured output files.

    The summary is the canonical completed-result record.  The optional state
    reconstructed here is deliberately minimal; pickle state remains the
    continuation path for numerical workflows.

    Example:
        >>> loader = ResultsLoader(Path("Fits/20260129_120000"))
        >>> summary = loader.load_summary()
        >>> state = loader.load_fitting_state()
        >>> chains = loader.load_mcmc_chains()
    """

    def __init__(self, directory: Path) -> None:
        """Initialize the loader.

        Args:
            directory: Path to results directory

        Raises:
            FileNotFoundError: If summary/fit.json is not found
        """
        self.directory = directory
        self.summary_path = directory / "summary" / "fit.json"

        if not self.summary_path.exists():
            raise FileNotFoundError(f"Results file not found: {format_path(self.summary_path)}")

    def load_summary(self) -> FitSummarySchema:
        """Load the raw summary schema.

        Returns:
            FitSummarySchema with parsed JSON data
        """
        with self.summary_path.open(encoding="utf-8") as f:
            data = json.load(f)
        return FitSummarySchema(**data)

    def load_fitting_state(self) -> FittingState:
        """Reconstruct the FittingState from the summary JSON.

        This method parses the JSON summary and reconstructs a FittingState
        object that can be used for analysis or visualization.

        Returns:
            FittingState object populated with results
        """
        summary = self.load_summary()

        all_peaks, clusters = self._reconstruct_clusters_and_peaks(summary.clusters)
        params = self._reconstruct_parameters(summary.clusters)
        noise = self._get_noise_level(summary)

        fit_params = FitParameters.from_parameters(params, all_peaks)

        return FittingState(
            clusters=clusters,
            params=fit_params,
            scalar_params=params,
            noise=noise,
        )

    def _reconstruct_clusters_and_peaks(
        self, cluster_schemas: list[ClusterResultSchema]
    ) -> tuple[list[Peak], list[Cluster]]:
        """Reconstruct clusters and peaks from schema data."""
        all_peaks: list[Peak] = []
        clusters: list[Cluster] = []

        for c_data in cluster_schemas:
            cluster_peaks = []
            for name in c_data.peak_names:
                # Find canonical position params (peak.F*.cs).
                axis_values: dict[str, float] = {}
                for param in c_data.final_nonlinear_parameters:
                    if not param.name.startswith(f"{name}."):
                        continue
                    if not param.name.endswith(".cs"):
                        continue
                    parts = param.name.split(".")
                    if len(parts) < _PARAM_NAME_PARTS:
                        continue
                    axis_values[parts[-2]] = param.value

                x_val = axis_values.get("F2", 0.0)
                y_val = axis_values.get("F3", 0.0)

                if x_val == 0.0 and y_val == 0.0 and axis_values:
                    ordered_axes = sorted(
                        axis_values,
                        key=lambda axis: int(axis[1:]) if axis.startswith("F") else 999,
                    )
                    x_val = axis_values[ordered_axes[0]]
                    y_val = axis_values[ordered_axes[1]] if len(ordered_axes) > 1 else 0.0

                shapes_list: list[Shape] = [
                    cast("Shape", ReconstructedShape(x_val, "F2")),
                    cast("Shape", ReconstructedShape(y_val, "F1")),
                ]

                peak = Peak(
                    name=name,
                    positions=np.array([x_val, y_val]),
                    shapes=shapes_list,
                )
                cluster_peaks.append(peak)
                all_peaks.append(peak)

            # Minimal 1-point cluster reconstruction for post-fit workflows.
            dummy_grid_indices = [np.array([0])]
            dummy_data = np.array([[0.0]])

            cluster = Cluster(
                cluster_id=c_data.cluster_id,
                peaks=cluster_peaks,
                grid_indices=dummy_grid_indices,
                data=dummy_data,
            )
            clusters.append(cluster)

        return all_peaks, clusters

    def _reconstruct_parameters(self, cluster_schemas: list[ClusterResultSchema]) -> Parameters:
        """Reconstruct Parameters from cluster schema data."""
        params = Parameters()
        for c_data in cluster_schemas:
            for p_data in c_data.final_nonlinear_parameters:
                if p_data.name in params:
                    continue

                param = Parameter(
                    name=p_data.name,
                    value=p_data.value,
                    min=p_data.min_bound if p_data.min_bound is not None else -float("inf"),
                    max=p_data.max_bound if p_data.max_bound is not None else float("inf"),
                    vary=p_data.vary,
                    stderr=(p_data.standard_error if p_data.standard_error is not None else 0.0),
                )
                params.add(
                    param.name,
                    value=param.value,
                    min_value=param.min,
                    max_value=param.max,
                    vary=param.vary,
                    stderr=param.stderr,
                )
        return params

    def _get_noise_level(self, summary: FitSummarySchema) -> float:
        """Extract noise level from summary."""
        return summary.noise

    def load_mcmc_chains(self) -> list[MCMCChainRecord]:
        """Load MCMC chains from HDF5 files for all clusters.

        Returns:
            List of tuples containing:
            - chains: numpy array of shape (n_chains, n_samples, n_params)
            - param_names: list of parameter names
            - cluster_id: cluster identifier
            - burn_in_idx: burn-in index
            - thin: thinning factor
        """
        chains_dir = self.directory / "chains"
        if not chains_dir.exists():
            return []

        return [
            record
            for h5_path in sorted(chains_dir.glob("cluster_*_chains.h5"))
            if (record := _load_mcmc_chain_file(h5_path))
        ]


def _load_mcmc_chain_file(path: Path) -> MCMCChainRecord | None:
    """Load one MCMC chain file, returning None when it is incomplete."""
    try:
        cluster_id = int(path.name.split("_")[1])
        with h5py.File(str(path), "r") as handle:
            group = handle[f"cluster_{cluster_id}"]
            if "chains" not in group:
                return None

            chain = group["chains"][()]
            nonlinear_names = [name.decode() for name in group["nonlinear_names"][:]]
            burn_in, thin = _load_burn_in_metadata(group)
            return chain, nonlinear_names, cluster_id, burn_in, thin
    except (KeyError, OSError, ValueError):
        return None


def _load_burn_in_metadata(group: Any) -> tuple[int, int]:
    """Read burn-in and thinning metadata from an MCMC HDF5 group."""
    if "burn_in" not in group:
        return 0, 1

    burn_in_group = group["burn_in"]
    thin = int(burn_in_group.attrs.get("thin", 1))
    if "burn_in_idx" in burn_in_group.attrs:
        return int(burn_in_group.attrs["burn_in_idx"]), thin
    if "burn_in" in burn_in_group.attrs:
        burn_in = int(np.ceil(int(burn_in_group.attrs["burn_in"]) / thin))
        return burn_in, thin
    return 0, thin
