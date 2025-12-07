"""Domain configuration and result models for PeakFit."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from peakfit.core.fitting.constraints import ParameterConfig
from peakfit.core.fitting.protocol import FitStep

LineshapeName = Literal["auto", "gaussian", "lorentzian", "pvoigt", "sp1", "sp2", "no_apod"]
OutputFormat = Literal["csv", "json", "txt"]
OutputVerbosity = Literal["minimal", "standard", "full"]
LogFormat = Literal["text", "json"]


class FitConfig(BaseModel):
    """Configuration for the fitting process.

    Supports both simple configuration (legacy) and advanced multi-step
    protocols with parameter constraints.

    Simple usage (legacy):
        [fitting]
        lineshape = "auto"
        refine_iterations = 2
        fix_positions = false

    Advanced multi-step protocol:
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
    # Phase fitting: list of dimension labels to fit phase for
    # e.g., ["F2"] for direct dimension only, ["F1", "F2"] for both
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
        description="Random seed for stochastic optimizers (e.g., basin-hopping, differential evolution).",
    )

    # Multi-step fitting protocol
    steps: list[FitStep] = Field(
        default_factory=list,
        description="Multi-step fitting protocol. If empty, uses refine_iterations.",
    )

    def get_phase_dimensions(self, n_spectral_dims: int = 2) -> list[str]:
        """Get list of dimension labels to fit phase for.

        Args:
            n_spectral_dims: Number of spectral dimensions (for Fn labeling)

        Returns
        -------
            List of dimension labels like ['F1', 'F2']
        """
        return self.fit_phase

    def has_protocol(self) -> bool:
        """Check if a multi-step protocol is defined."""
        return len(self.steps) > 0


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

    Supports both new structured output system and legacy formats.
    """

    model_config = ConfigDict(extra="forbid")

    directory: Path = Field(default=Path("Fits"), description="Output directory for results.")
    formats: list[OutputFormat] = Field(
        default=["json", "csv", "txt"],
        description="Output formats for results. Default includes all formats.",
    )
    verbosity: OutputVerbosity = Field(
        default="standard",
        description="Output verbosity: minimal (essential), standard (default), full (all).",
    )
    save_simulated: bool = Field(default=True, description="Save simulated spectrum to file.")
    save_html_report: bool = Field(default=False, description="Save HTML report of fitting.")
    save_chains: bool = Field(
        default=False,
        description="Save MCMC chains to disk (requires significant storage).",
    )
    save_figures: bool = Field(default=True, description="Generate and save diagnostic figures.")
    include_timestamp: bool = Field(
        default=False,
        description="Include timestamp in output directory name.",
    )
    headless: bool = Field(
        default=False,
        description="Disable interactive/live display (use reporter-only output).",
    )
    include_legacy: bool = Field(
        default=False,
        description="Write legacy .out outputs alongside structured outputs (opt-in).",
    )
    log_format: LogFormat = Field(
        default="text",
        description="Format for log file: text (human-readable) or json (structured).",
    )


class PeakFitConfig(BaseModel):
    """Top-level PeakFit configuration for fitting, clustering, and output.

    Example TOML configuration:
        [fitting]
        lineshape = "auto"
        refine_iterations = 2

        # Optional: multi-step protocol
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


class PeakData(BaseModel):
    """Peak data with N-dimensional position support."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Peak identifier")
    positions: list[float] = Field(
        default_factory=list,
        description="Peak positions in ppm, ordered from F1 to Fn (indirect to direct).",
    )
    cluster_id: int | None = Field(default=None, description="Assigned cluster ID")

    def get_positions(self) -> list[float]:
        """Get positions as a list."""
        return self.positions


class FitResultPeak(BaseModel):
    """Fitted peak result with N-dimensional support."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    # N-dimensional positions and linewidths as lists
    positions: list[float] = Field(
        default_factory=list,
        description="Fitted positions in ppm, ordered F1 to Fn.",
    )
    position_errors: list[float] = Field(
        default_factory=list,
        description="Position errors in ppm, ordered F1 to Fn.",
    )
    fwhms: list[float] = Field(
        default_factory=list,
        description="Fitted linewidths (FWHM) in Hz, ordered F1 to Fn.",
    )
    fwhm_errors: list[float] = Field(
        default_factory=list,
        description="Linewidth errors in Hz, ordered F1 to Fn.",
    )
    # Dimension labels for clarity in output
    dimension_labels: list[str] = Field(
        default_factory=list,
        description="Dimension labels like ['F1', 'F2'].",
    )
    amplitudes: list[float] = Field(default_factory=list)
    amplitude_errors: list[float] = Field(default_factory=list)


class FitResult(BaseModel):
    """Result record for a single fit operation (legacy model)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cluster_id: int
    peaks: list[FitResultPeak]
    residual_norm: float
    n_iterations: int
    success: bool
    message: str = ""
    n_function_evals: int = 0
    cost: float = 0.0


class ValidationResult(BaseModel):
    """Result of input validation operations (spectrum/peaklist)."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    info: dict[str, object] = Field(default_factory=dict)


__all__ = [
    "ClusterConfig",
    "FitConfig",
    "FitResult",
    "FitResultPeak",
    "FitStep",
    "LineshapeName",
    "LogFormat",
    "OutputConfig",
    "OutputFormat",
    "OutputVerbosity",
    "ParameterConfig",
    "PeakData",
    "PeakFitConfig",
    "ValidationResult",
]
