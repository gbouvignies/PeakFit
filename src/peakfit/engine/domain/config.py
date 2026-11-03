"""Domain configuration and result models for PeakFit."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from peakfit.engine.domain.constraints import ParameterConfig
from peakfit.engine.domain.fit_steps import FitStep

LineshapeName = Literal[
    "auto",
    "gaussian",
    "gaussian_doublet",
    "lorentzian",
    "lorentzian_doublet",
    "pvoigt",
    "pvoigt_doublet",
    "sp1",
    "sp1_doublet",
    "sp2",
    "sp2_doublet",
    "no_apod",
    "no_apod_doublet",
]
OutputFormat = Literal["csv", "json", "txt"]


class FitConfig(BaseModel):
    """Configuration for the fitting process.

    Supports the common one-step configuration and optional multi-step fit steps
    with parameter constraints.

    Simple usage:
        [fitting]
        lineshape = "auto"
        refine_iterations = 2

    Advanced multi-step fit:
        [[fitting.steps]]
        name = "fix_positions"
        fix = ["*.*.cs"]
        iterations = 1

        [[fitting.steps]]
        name = "full_optimization"
        vary = ["*"]
        iterations = 2

    Parameter constraints:
        [parameters]
        position_window = 0.1

        [parameters.position_windows]
        F2 = 0.5
        F3 = 0.05

        [parameters.peaks."2N-H"]
        position_window = 0.02
        "F2.cs" = { vary = false }
    """

    model_config = ConfigDict(extra="forbid")

    lineshape: LineshapeName = Field(
        default="auto",
        description="Lineshape model to use. 'auto' detects from NMRPipe apodization.",
    )
    refine_iterations: Annotated[int, Field(le=20)] = Field(
        default=1,
        description="Number of optimizer passes; corrections update between passes.",
    )
    fit_phase: list[str] = Field(
        default_factory=list,
        description="Dimensions to fit phase correction for (e.g., ['F1', 'F2']).",
    )
    max_iterations: Annotated[int, Field(gt=0)] = Field(
        default=1000,
        description="Maximum iterations for optimizer.",
    )
    tolerance: Annotated[float, Field(gt=0)] = Field(
        default=1e-8,
        description="Convergence tolerance for optimizer.",
    )

    # Multi-step fitting steps
    steps: list[FitStep] = Field(
        default_factory=list,
        description="Multi-step fitting steps. If empty, uses refine_iterations.",
    )

    @field_validator("refine_iterations")
    @classmethod
    def _require_positive_refine_iterations(cls, value: int) -> int:
        if value < 1:
            raise ValueError("refine_iterations must be at least 1")
        return value


class ClusterConfig(BaseModel):
    """Configuration for peak clustering and segmentation.

    Controls contour thresholds and other clustering parameters used during
    segmentation of peaks into clusters prior to fitting.
    """

    model_config = ConfigDict(extra="forbid")

    contour_factor: Annotated[float, Field(gt=0)] = Field(
        default=5.0,
        description="Factor multiplied by noise level for contour threshold.",
    )
    contour_level: float | None = Field(
        default=None,
        description="Explicit contour level (overrides contour_factor if set).",
    )


class OutputConfig(BaseModel):
    """Configuration for output file generation.

    Keep this model limited to fit-output controls that are actually
    implemented by the writer path.
    """

    model_config = ConfigDict(extra="forbid")

    directory: Path = Field(default=Path("Fits"), description="Output directory for results.")
    formats: list[OutputFormat] = Field(
        default=["csv", "txt"],
        description="Presentation formats: CSV and a short report by default; JSON is explicit.",
    )
    detailed_evidence: bool = Field(
        default=False,
        description="Include observations and predictions for portable review/continuation.",
    )
    save_simulated: bool = Field(
        default=False,
        description="Export simulated.npy in float64, with calibration in scientific.npz.",
    )
    include_timestamp: bool = Field(
        default=True,
        description="Include timestamp in output directory name.",
    )
    headless: bool = Field(
        default=False,
        description="Disable interactive/live display (use reporter-only output).",
    )


class PeakFitConfig(BaseModel):
    """Top-level PeakFit configuration for fitting, clustering, and output.

    Example TOML configuration:
        [fitting]
        lineshape = "auto"
        refine_iterations = 2

        # Optional: multi-step fit
        [[fitting.steps]]
        name = "fix_positions"
        fix = ["*.*.cs"]
        iterations = 1

        [[fitting.steps]]
        name = "full_optimization"
        vary = ["*"]
        iterations = 2

        [clustering]
        contour_factor = 5.0

        [output]
        directory = "Fits"
        formats = ["csv", "txt"]

        # Parameter constraints
        [parameters]
        position_window = 0.1

        [parameters.position_windows]
        F2 = 0.5   # 15N dimension
        F3 = 0.05  # 1H dimension

        [parameters.defaults]
        "*.*.lw" = { min = 5.0, max = 100.0 }

        [parameters.peaks."2N-H"]
        position_window = 0.02
        "F2.cs" = { vary = false }
    """

    model_config = ConfigDict(extra="forbid")

    fitting: FitConfig = Field(default_factory=FitConfig)
    clustering: ClusterConfig = Field(default_factory=ClusterConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    parameters: ParameterConfig = Field(
        default_factory=ParameterConfig,
        description="Parameter constraints and position windows.",
    )
    noise_level: float | None = Field(
        default=None,
        description="Manual noise level. If None, estimated automatically.",
        gt=0,
    )
    exclude_planes: list[int] = Field(
        default_factory=list,
        description="List of plane indices to exclude from fitting.",
    )

    @field_validator("exclude_planes")
    @classmethod
    def validate_exclude_planes(cls, v: list[int]) -> list[int]:
        """Validate list of plane indices to ensure no negative indices are provided."""
        if any(idx < 0 for idx in v):
            msg = "Plane indices must be non-negative"
            raise ValueError(msg)
        return sorted(set(v))


# =============================================================================
# Optimizer Configuration
# =============================================================================


@dataclass(frozen=True)
class VarProConfig:
    """Configuration for Variable Projection optimizer."""

    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    max_nfev: int = 1000
    verbose: int = 0


__all__ = [
    "ClusterConfig",
    "FitConfig",
    "FitStep",
    "LineshapeName",
    "OutputConfig",
    "OutputFormat",
    "ParameterConfig",
    "PeakFitConfig",
    "VarProConfig",
]
