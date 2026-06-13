"""Domain configuration and result models for PeakFit."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from peakfit.engine.domain.constraints import ParameterConfig
from peakfit.engine.domain.data import PeakData
from peakfit.engine.domain.fit_steps import FitStep
from peakfit.shared.constants import (
    BASIN_HOPPING_NITER,
    BASIN_HOPPING_STEPSIZE,
    BASIN_HOPPING_TEMPERATURE,
)

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
        fix_positions = false

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
    refine_iterations: Annotated[int, Field(ge=0, le=20)] = Field(
        default=1,
        description="Number of refinement iterations for cross-talk correction.",
    )
    fix_positions: bool = Field(default=False, description="Fix peak positions during fitting.")
    fit_j_coupling: bool = Field(
        default=False,
        description="Fit J-coupling constant in direct dimension.",
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
    optimizer_seed: Annotated[int, Field(ge=0)] | None = Field(
        default=None,
        description="Random seed for stochastic optimizers such as basin-hopping.",
    )

    # Multi-step fitting steps
    steps: list[FitStep] = Field(
        default_factory=list,
        description="Multi-step fitting steps. If empty, uses refine_iterations.",
    )


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


class AutoPeakConfig(BaseModel):
    """Configuration for automatic peak picking when no peak list is provided."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Enable automatic peak picking when no peak list is given.",
    )
    start_threshold_sigma: Annotated[float, Field(gt=0)] = Field(
        default=15.0,
        description="Stop when residual maximum falls below this noise multiple.",
    )
    add_threshold_sigma: Annotated[float, Field(gt=0)] = Field(
        default=3.0,
        description="Minimum residual-noise multiple required to add a peak within an ROI.",
    )
    f_test_pvalue: Annotated[float, Field(gt=0, lt=1)] = Field(
        default=1e-6,
        description="P-value cutoff used when accepting additional peaks in an ROI.",
    )
    max_clusters: Annotated[int, Field(gt=0)] = Field(
        default=2000,
        description="Maximum number of ROI iterations during automatic picking.",
    )
    max_peaks_per_roi: int | None = Field(
        default=None,
        gt=0,
        description="Optional maximum number of peaks to add in a single ROI.",
    )
    min_peak_separation_pts: Annotated[int, Field(ge=0)] = Field(
        default=5,
        description="Minimum separation between automatically inserted peaks (in points).",
    )
    position_window_ppm: Annotated[float, Field(gt=0)] = Field(
        default=0.05,
        description="Initial +/- window for peak position parameters during ROI fitting.",
    )
    max_nfev_per_fit: Annotated[int, Field(gt=0)] = Field(
        default=250,
        description="Maximum VARPRO function evaluations for each ROI trial fit.",
    )
    position_constraint_factor: Annotated[float, Field(gt=0)] = Field(
        default=1.5,
        description="CS bound factor in units of fitted linewidth during constrained release.",
    )
    max_constraint_refits: Annotated[int, Field(ge=0)] = Field(
        default=3,
        description=(
            "Maximum number of additional constrained CS-release retries when CS lands near bounds."
        ),
    )
    proton_constraint_margin_ppm: Annotated[float, Field(gt=0)] = Field(
        default=0.002,
        description="Boundary proximity threshold for 1H CS constraints (ppm).",
    )
    heteronuclear_constraint_margin_ppm: Annotated[float, Field(gt=0)] = Field(
        default=0.02,
        description="Boundary proximity threshold for heteronuclear CS constraints (ppm).",
    )
    amplitude_zero_tolerance: Annotated[float, Field(ge=0)] = Field(
        default=1e-12,
        description="Tolerance used to reject peaks with near-zero amplitudes in all spectra.",
    )


class OutputConfig(BaseModel):
    """Configuration for output file generation.

    Keep this model limited to fit-output controls that are actually
    implemented by the writer path.
    """

    model_config = ConfigDict(extra="forbid")

    directory: Path = Field(default=Path("Fits"), description="Output directory for results.")
    formats: list[OutputFormat] = Field(
        default=["json", "csv"],
        description="Output formats for results. Add 'txt' to write a Markdown report.",
    )
    save_simulated: bool = Field(default=False, description="Save simulated spectrum to file.")
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
        formats = ["json", "csv"]

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
    auto_peak: AutoPeakConfig = Field(default_factory=AutoPeakConfig)
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


class ValidationResult(BaseModel):
    """Result of input validation operations (spectrum/peaklist)."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    info: dict[str, object] = Field(default_factory=dict)


# =============================================================================
# Optimizer Configurations
# =============================================================================


@dataclass(frozen=True)
class VarProConfig:
    """Configuration for Variable Projection optimizer."""

    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    max_nfev: int = 1000
    verbose: int = 0


@dataclass(frozen=True)
class BasinHoppingConfig:
    """Configuration for the basin-hopping optimizer."""

    n_iterations: int = BASIN_HOPPING_NITER
    temperature: float = BASIN_HOPPING_TEMPERATURE
    step_size: float = BASIN_HOPPING_STEPSIZE
    seed: int | None = None


OptimizerConfig = VarProConfig | BasinHoppingConfig


__all__ = [
    "AutoPeakConfig",
    "BasinHoppingConfig",
    "ClusterConfig",
    "FitConfig",
    "FitStep",
    "LineshapeName",
    "OptimizerConfig",
    "OutputConfig",
    "OutputFormat",
    "ParameterConfig",
    "PeakData",
    "PeakFitConfig",
    "ValidationResult",
    "VarProConfig",
]
