"""Algorithms bridging domain objects with numerical routines."""

from peakfit.core.algorithms.clustering import (
    assign_peaks_to_segments,
    create_clusters,
    group_connected_pairs,
    merge_connected_segments,
    segment_data,
)
from peakfit.core.algorithms.global_optimization import (
    GlobalFitResult,
    fit_basin_hopping,
    fit_differential_evolution,
)
from peakfit.core.algorithms.mcmc import UncertaintyResult, estimate_uncertainties_mcmc
from peakfit.core.algorithms.noise import estimate_noise, prepare_noise_level
from peakfit.core.algorithms.varpro import (
    ScipyOptimizerError,
    VarProOptimizer,
    fit_cluster,
    fit_clusters,
)

__all__ = [
    "GlobalFitResult",
    "ScipyOptimizerError",
    "UncertaintyResult",
    "VarProOptimizer",
    "assign_peaks_to_segments",
    "create_clusters",
    "estimate_noise",
    "estimate_uncertainties_mcmc",
    "fit_basin_hopping",
    "fit_cluster",
    "fit_clusters",
    "fit_differential_evolution",
    "group_connected_pairs",
    "merge_connected_segments",
    "prepare_noise_level",
    "segment_data",
]
