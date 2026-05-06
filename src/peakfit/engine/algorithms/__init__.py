"""Algorithms bridging domain objects with numerical routines."""

from peakfit.engine.algorithms.clustering import (
    assign_peaks_to_segments,
    create_clusters,
    group_connected_pairs,
    merge_connected_segments,
    segment_data,
)
from peakfit.engine.algorithms.global_optimization import (
    GlobalFitResult,
    fit_basin_hopping,
    fit_differential_evolution,
)
from peakfit.engine.algorithms.mcmc import UncertaintyResult, estimate_uncertainties_mcmc
from peakfit.engine.algorithms.noise import estimate_noise, prepare_noise_level
from peakfit.engine.algorithms.varpro import (
    ScipyOptimizerError,
    VarProOptimizer,
    fit_cluster,
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
    "fit_differential_evolution",
    "group_connected_pairs",
    "merge_connected_segments",
    "prepare_noise_level",
    "segment_data",
]
