"""Peak fitting optimization and computation.

This module provides the core fitting functionality for NMR peak analysis,
including parameter management, optimization algorithms, and simulation capabilities.
"""

# Parameter management
# Advanced optimization
from peakfit.core.fitting.advanced import (
    GlobalFitResult,
    UncertaintyResult,
    compute_profile_likelihood,
    estimate_uncertainties_mcmc,
    fit_basin_hopping,
    fit_differential_evolution,
)

# Core computation functions
from peakfit.core.fitting.computation import (
    calculate_amplitudes,
    calculate_shape_heights,
    calculate_shapes,
    residuals,
    update_cluster_corrections,
)

# Basic optimization
from peakfit.core.fitting.optimizer import (
    fit_cluster,
    fit_cluster_dict,
    fit_clusters,
    fit_clusters_sequential,
)
from peakfit.core.fitting.parameters import Parameter, Parameters, ParameterType

# Fit results
from peakfit.core.fitting.results import FitResult

# Simulation
from peakfit.core.fitting.simulation import simulate_data
from peakfit.core.fitting.strategies import (
    BasinHoppingStrategy,
    DifferentialEvolutionStrategy,
    LeastSquaresStrategy,
    OptimizationResult,
    OptimizationStrategy,
    get_strategy,
    register_strategy,
)

__all__ = [
    "BasinHoppingStrategy",
    "DifferentialEvolutionStrategy",
    "FitResult",
    "GlobalFitResult",
    "LeastSquaresStrategy",
    "OptimizationResult",
    "OptimizationStrategy",
    "Parameter",
    "ParameterType",
    "Parameters",
    "UncertaintyResult",
    "calculate_amplitudes",
    "calculate_shape_heights",
    "calculate_shapes",
    "compute_profile_likelihood",
    "estimate_uncertainties_mcmc",
    "fit_basin_hopping",
    "fit_cluster",
    "fit_cluster_dict",
    "fit_clusters",
    "fit_clusters_sequential",
    "fit_differential_evolution",
    "get_strategy",
    "register_strategy",
    "residuals",
    "simulate_data",
    "update_cluster_corrections",
]
