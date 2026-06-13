"""Fitting step constraints for NMR peak fitting.

Users can define a sequence of fitting steps with different parameter
constraints at each step.

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


def build_fit_steps(
    steps: list[FitStep] | None,
    refine_iterations: int = 1,
) -> list[FitStep]:
    """Return configured fit steps.

    Args:
        steps: Explicit steps from config (if any)
        refine_iterations: Refine iteration count used by the default step

    Returns:
    -------
        Configured fitting steps
    """
    if steps:
        return steps

    return [
        FitStep(
            name="default",
            vary=["*"],
            iterations=refine_iterations + 1,
        )
    ]


__all__ = [
    "FitStep",
    "apply_step_constraints",
    "build_fit_steps",
]
