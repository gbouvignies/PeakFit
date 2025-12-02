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
    calculate_amplitude_covariance,
    calculate_amplitudes,
    calculate_amplitudes_with_uncertainty,
    calculate_shape_heights,
    calculate_shapes,
    inject_amplitude_parameters,
    residuals,
    update_cluster_corrections,
)

# Constraints system
from peakfit.core.fitting.constraints import (
    ConstraintResolver,
    ParameterConfig,
    ParameterConstraint,
    ParameterDefaults,
    PeakConstraints,
    PositionWindowConfig,
    apply_constraints,
    constraints_from_cli,
)

# Basic optimization
from peakfit.core.fitting.optimizer import (
    fit_cluster,
    fit_cluster_dict,
    fit_clusters,
)
from peakfit.core.fitting.parameters import Parameter, Parameters, ParameterType

# Multi-step protocol
from peakfit.core.fitting.protocol import (
    FitProtocol,
    FitStep,
    ProtocolExecutor,
    ProtocolResult,
    StepResult,
    apply_step_constraints,
    create_protocol_from_config,
)

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
    "ConstraintResolver",
    "DifferentialEvolutionStrategy",
    "FitProtocol",
    "FitResult",
    "FitStep",
    "GlobalFitResult",
    "LeastSquaresStrategy",
    "OptimizationResult",
    "OptimizationStrategy",
    "Parameter",
    "ParameterConfig",
    "ParameterConstraint",
    "ParameterDefaults",
    "ParameterType",
    "Parameters",
    "PeakConstraints",
    "PositionWindowConfig",
    "ProtocolExecutor",
    "ProtocolResult",
    "StepResult",
    "UncertaintyResult",
    "apply_constraints",
    "apply_step_constraints",
    "calculate_amplitude_covariance",
    "calculate_amplitudes",
    "calculate_amplitudes_with_uncertainty",
    "calculate_shape_heights",
    "calculate_shapes",
    "compute_profile_likelihood",
    "constraints_from_cli",
    "create_protocol_from_config",
    "estimate_uncertainties_mcmc",
    "fit_basin_hopping",
    "fit_cluster",
    "fit_cluster_dict",
    "fit_clusters",
    "fit_differential_evolution",
    "get_strategy",
    "inject_amplitude_parameters",
    "register_strategy",
    "residuals",
    "simulate_data",
    "update_cluster_corrections",
]
