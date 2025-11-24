"""Peak fitting optimization and computation.

This module provides the core fitting functionality for NMR peak analysis,
including parameter management, optimization algorithms, parallel processing,
and simulation capabilities.
"""

# Parameter management
# Advanced optimization
from peakfit.fitting.advanced import (
    GlobalFitResult,
    UncertaintyResult,
    compute_profile_likelihood,
    estimate_uncertainties_mcmc,
    fit_basin_hopping,
    fit_differential_evolution,
)

# Core computation functions
from peakfit.fitting.computation import (
    calculate_amplitudes,
    calculate_shape_heights,
    calculate_shapes,
    residuals,
    update_cluster_corrections,
)

# Basic optimization
from peakfit.fitting.optimizer import (
    fit_cluster,
    fit_cluster_dict,
    fit_clusters,
    fit_clusters_sequential,
)

# Output file generation
from peakfit.fitting.output import write_profiles, write_shifts
from peakfit.fitting.parameters import Parameter, Parameters, ParameterType

# Fit results
from peakfit.fitting.results import FitResult

# Simulation
from peakfit.fitting.simulation import simulate_data

__all__ = [
    # Results
    "FitResult",
    "GlobalFitResult",
    # Parameters
    "Parameter",
    "ParameterType",
    "Parameters",
    "UncertaintyResult",
    "calculate_amplitudes",
    "calculate_shape_heights",
    # Computation
    "calculate_shapes",
    "compute_profile_likelihood",
    "estimate_uncertainties_mcmc",
    # Advanced optimization
    "fit_basin_hopping",
    # Basic optimization
    "fit_cluster",
    "fit_cluster_dict",
    "fit_clusters",
    "fit_clusters_sequential",
    "fit_differential_evolution",
    "residuals",
    # Simulation
    "simulate_data",
    "update_cluster_corrections",
    # Output
    "write_profiles",
    "write_shifts",
]
