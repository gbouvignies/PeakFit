"""Peak fitting optimization and computation.

This module provides the core fitting functionality for NMR peak analysis,
including parameter management, optimization algorithms, and simulation capabilities.
"""

from peakfit.engine.algorithms.common import (
    calculate_shape_heights,
    inject_amplitude_parameters,
    residuals,
    update_cluster_corrections,
)
from peakfit.engine.algorithms.global_optimization import (
    fit_basin_hopping,
    fit_differential_evolution,
)
from peakfit.engine.algorithms.linear_algebra import (
    calculate_amplitude_covariance,
    calculate_amplitudes,
    calculate_amplitudes_with_uncertainty,
)
from peakfit.engine.algorithms.mcmc import UncertaintyResult, estimate_uncertainties_mcmc
from peakfit.engine.algorithms.varpro import (
    VarProOptimizer,
    fit_cluster,
)
from peakfit.engine.domain.constraints import (
    ConstraintResolver,
    ParameterConfig,
    ParameterConstraint,
    ParameterDefaults,
    PeakConstraints,
    PositionWindowConfig,
    apply_constraints,
    constraints_from_cli,
)
from peakfit.engine.domain.params_scalar import Parameter, Parameters
from peakfit.engine.domain.protocol import (
    FitProtocol,
    FitStep,
    ProtocolResult,
    StepResult,
    apply_step_constraints,
    create_protocol_from_config,
)
from peakfit.engine.fitting.simulation import simulate_data
from peakfit.engine.fitting.strategies import (
    BasinHoppingStrategy,
    DifferentialEvolutionStrategy,
    FitStrategy,
    get_strategy,
    register_strategy,
)
from peakfit.engine.results import FitResult

__all__ = [
    "BasinHoppingStrategy",
    "ConstraintResolver",
    "DifferentialEvolutionStrategy",
    "FitProtocol",
    "FitResult",
    "FitStep",
    "FitStrategy",
    "Parameter",
    "ParameterConfig",
    "ParameterConstraint",
    "ParameterDefaults",
    "Parameters",
    "PeakConstraints",
    "PositionWindowConfig",
    "ProtocolResult",
    "StepResult",
    "UncertaintyResult",
    "VarProOptimizer",
    "apply_constraints",
    "apply_step_constraints",
    "calculate_amplitude_covariance",
    "calculate_amplitudes",
    "calculate_amplitudes_with_uncertainty",
    "calculate_shape_heights",
    "constraints_from_cli",
    "create_protocol_from_config",
    "estimate_uncertainties_mcmc",
    "fit_basin_hopping",
    "fit_cluster",
    "fit_differential_evolution",
    "get_strategy",
    "inject_amplitude_parameters",
    "register_strategy",
    "residuals",
    "simulate_data",
    "update_cluster_corrections",
]
