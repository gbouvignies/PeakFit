"""Multi-step fitting protocol for NMR peak fitting.

This module provides a flexible protocol system for defining multi-step
fitting workflows. Users can define sequences of fitting steps with
different parameter constraints at each step.

Example use cases:
- Fix positions first, then release them for fine-tuning
- Fit linewidths with fixed eta, then optimize eta
- Progressive refinement with increasing parameter freedom

Example TOML configuration:
    [[fitting.steps]]
    name = "fix_positions"
    fix = ["*.*.cs"]
    iterations = 1

    [[fitting.steps]]
    name = "refine_linewidths"
    fix = ["*.*.cs", "*.*.eta"]
    vary = ["*.*.lw"]
    iterations = 2

    [[fitting.steps]]
    name = "full_optimization"
    vary = ["*"]
    iterations = 1
"""

import fnmatch
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from peakfit.engine.domain.params_scalar import Parameters


class FitStep(BaseModel):
    """Single step in a multi-step fitting protocol.

    Each step can modify which parameters are fixed/varied before
    running the optimization.

    Attributes:
    ----------
        name: Human-readable name for this step (for logging)
        fix: Glob patterns for parameters to fix (vary=False)
        vary: Glob patterns for parameters to vary (vary=True)
        iterations: Number of refinement iterations for this step
        description: Optional description for documentation

    Note:
        Patterns in `vary` are applied after `fix`, so they can
        override fixed parameters. This allows patterns like:
            fix = ["*"]  # Fix everything
            vary = ["*.*.lw"]  # Except linewidths
    """

    name: str = Field(default="", description="Step name for logging")
    fix: list[str] = Field(default_factory=list, description="Patterns to fix")
    vary: list[str] = Field(default_factory=list, description="Patterns to vary")
    iterations: int = Field(default=1, ge=1, description="Refinement iterations")
    description: str = Field(default="", description="Step description")


class FitProtocol(BaseModel):
    """Multi-step fitting protocol.

    Defines a sequence of fitting steps to be executed in order.
    If no steps are defined, a default single-step protocol is used.

    Attributes:
    ----------
        steps: List of fitting steps to execute
        continue_on_failure: Whether to continue if a step fails
    """

    steps: list[FitStep] = Field(default_factory=list)
    continue_on_failure: bool = Field(
        default=True,
        description="Continue to next step even if current step fails to converge",
    )

    def is_empty(self) -> bool:
        """Check if protocol has any steps defined."""
        return len(self.steps) == 0

    @classmethod
    def create(
        cls,
        steps: list[FitStep] | None = None,
    ) -> FitProtocol:
        """Create a protocol from steps."""
        return cls(steps=steps or [])

    @classmethod
    def default(cls, refine_iterations: int = 1) -> FitProtocol:
        """Create default single-step protocol.

        Args:
            refine_iterations: Number of refinement iterations

        Returns:
        -------
            Protocol with single step that varies all parameters
        """
        return cls(
            steps=[
                FitStep(
                    name="default",
                    vary=["*"],
                    iterations=refine_iterations,
                )
            ]
        )

    @classmethod
    def positions_then_full(cls) -> FitProtocol:
        """Create common protocol: fix positions first, then full optimization.

        Returns:
        -------
            Two-step protocol
        """
        return cls(
            steps=[
                FitStep(
                    name="fix_positions",
                    fix=["*.*.cs"],
                    iterations=1,
                    description="Optimize linewidths with fixed positions",
                ),
                FitStep(
                    name="full_optimization",
                    vary=["*"],
                    iterations=1,
                    description="Full optimization of all parameters",
                ),
            ]
        )


@dataclass
class StepResult:
    """Result from a single protocol step.

    Attributes:
    ----------
        step: The step configuration
        step_index: Index of this step (0-based)
        success: Whether all clusters converged
        n_converged: Number of clusters that converged
        n_total: Total number of clusters
        total_time: Time taken for this step (seconds)
        message: Status message
    """

    step: FitStep
    step_index: int
    success: bool
    n_converged: int
    n_total: int
    total_time: float
    message: str = ""


@dataclass
class ProtocolResult:
    """Result from executing a complete protocol.

    Attributes:
    ----------
        steps: Results from each step
        success: Whether all steps completed successfully
        total_time: Total time for all steps
        final_params: Final fitted parameters
    """

    steps: list[StepResult] = field(default_factory=list)
    success: bool = True
    total_time: float = 0.0
    final_params: Parameters | None = None


def apply_step_constraints(params: Parameters, step: FitStep) -> Parameters:
    """Apply fix/vary patterns from a step to parameters.

    Args:
        params: Parameters to modify
        step: Step with fix/vary patterns

    Returns:
    -------
        Modified parameters (same instance)
    """
    # First apply fix patterns
    for pattern in step.fix:
        _apply_pattern(params, pattern, vary=False)

    # Then apply vary patterns (can override fix)
    for pattern in step.vary:
        _apply_pattern(params, pattern, vary=True)

    return params


def _apply_pattern(params: Parameters, pattern: str, *, vary: bool) -> None:
    """Apply a single pattern to parameters.

    Args:
        params: Parameters to modify
        pattern: Glob pattern to match
        vary: Value to set for matching parameters
    """
    # Convert glob to regex
    regex = fnmatch.translate(pattern)

    for name, param in params.items():
        if re.match(regex, name) and not param.computed:
            param.vary = vary


def create_protocol_from_config(
    steps: list[FitStep] | None,
    refine_iterations: int = 1,
) -> FitProtocol:
    """Create a FitProtocol from configuration.

    Args:
        steps: Explicit steps from config (if any)
        refine_iterations: Legacy refine iterations count

    Returns:
    -------
        Configured FitProtocol
    """
    # If explicit steps provided, use them
    if steps:
        return FitProtocol(steps=steps)

    # Default: vary everything
    return FitProtocol.default(refine_iterations=refine_iterations + 1)


__all__ = [
    "FitProtocol",
    "FitStep",
    "ProtocolResult",
    "StepResult",
    "apply_step_constraints",
    "create_protocol_from_config",
]
