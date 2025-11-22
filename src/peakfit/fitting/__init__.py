"""Peak fitting optimization and computation.

This module provides the core fitting functionality for NMR peak analysis,
including parameter management, optimization algorithms, parallel processing,
and simulation capabilities.
"""

# Parameter management
from peakfit.fitting.parameters import Parameter, Parameters, ParameterType

# Fit results
from peakfit.fitting.results import FitResult

# Basic optimization
from peakfit.fitting.optimizer import (
    fit_cluster,
    fit_cluster_dict,
    fit_clusters,
    fit_clusters_sequential,
)

# Advanced optimization
from peakfit.fitting.advanced import (
    GlobalFitResult,
    UncertaintyResult,
    compute_profile_likelihood,
    estimate_uncertainties_mcmc,
    fit_basin_hopping,
    fit_differential_evolution,
)

# Parallel fitting
from peakfit.fitting.parallel import (
    fit_clusters_parallel,
    fit_clusters_parallel_refined,
)

# Core computation functions
from peakfit.fitting.computation import (
    calculate_amplitudes,
    calculate_shape_heights,
    calculate_shapes,
    residuals,
    update_cluster_corrections,
)

# Simulation
from peakfit.fitting.simulation import simulate_data

__all__ = [
    # Parameters
    "Parameter",
    "Parameters",
    "ParameterType",
    # Results
    "FitResult",
    "GlobalFitResult",
    "UncertaintyResult",
    # Basic optimization
    "fit_cluster",
    "fit_cluster_dict",
    "fit_clusters",
    "fit_clusters_sequential",
    # Advanced optimization
    "fit_basin_hopping",
    "fit_differential_evolution",
    "compute_profile_likelihood",
    "estimate_uncertainties_mcmc",
    # Parallel fitting
    "fit_clusters_parallel",
    "fit_clusters_parallel_refined",
    # Computation
    "calculate_shapes",
    "calculate_amplitudes",
    "calculate_shape_heights",
    "residuals",
    "update_cluster_corrections",
    # Simulation
    "simulate_data",
]
