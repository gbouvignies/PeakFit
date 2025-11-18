"""Pydantic models for PeakFit configuration and data structures."""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FitConfig(BaseModel):
    """Configuration for the fitting process."""

    model_config = ConfigDict(extra="forbid")

    lineshape: Literal["auto", "gaussian", "lorentzian", "pvoigt", "sp1", "sp2", "no_apod"] = Field(
        default="auto",
        description="Lineshape model to use. 'auto' detects from NMRPipe apodization.",
    )
    refine_iterations: Annotated[int, Field(ge=0, le=20)] = Field(
        default=1,
        description="Number of refinement iterations for cross-talk correction.",
    )
    fix_positions: bool = Field(
        default=False,
        description="Fix peak positions during fitting.",
    )
    fit_j_coupling: bool = Field(
        default=False,
        description="Fit J-coupling constant in direct dimension.",
    )
    fit_phase_x: bool = Field(
        default=False,
        description="Fit phase correction in X dimension.",
    )
    fit_phase_y: bool = Field(
        default=False,
        description="Fit phase correction in Y dimension.",
    )
    max_iterations: Annotated[int, Field(gt=0)] = Field(
        default=1000,
        description="Maximum iterations for optimizer.",
    )
    tolerance: Annotated[float, Field(gt=0)] = Field(
        default=1e-8,
        description="Convergence tolerance for optimizer.",
    )


class ClusterConfig(BaseModel):
    """Configuration for peak clustering."""

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
    """Configuration for output files."""

    model_config = ConfigDict(extra="forbid")

    directory: Path = Field(
        default=Path("Fits"),
        description="Output directory for results.",
    )
    formats: list[Literal["csv", "json", "txt"]] = Field(
        default=["txt"],
        description="Output formats for results.",
    )
    save_simulated: bool = Field(
        default=True,
        description="Save simulated spectrum to file.",
    )
    save_html_report: bool = Field(
        default=True,
        description="Save HTML report of fitting.",
    )


class PeakFitConfig(BaseModel):
    """Main configuration for PeakFit."""

    model_config = ConfigDict(extra="forbid")

    fitting: FitConfig = Field(default_factory=FitConfig)
    clustering: ClusterConfig = Field(default_factory=ClusterConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
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
        """Ensure plane indices are non-negative."""
        if any(idx < 0 for idx in v):
            msg = "Plane indices must be non-negative"
            raise ValueError(msg)
        return sorted(set(v))


class PeakData(BaseModel):
    """Data structure for a single peak."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Peak identifier")
    position_x: float = Field(description="X position in ppm")
    position_y: float = Field(description="Y position in ppm")
    position_z: float | None = Field(default=None, description="Z position in ppm (if applicable)")
    cluster_id: int | None = Field(default=None, description="Assigned cluster ID")


class FitResultPeak(BaseModel):
    """Fitting results for a single peak."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    position_x: float
    position_x_error: float | None = None
    position_y: float
    position_y_error: float | None = None
    fwhm_x: float | None = None
    fwhm_x_error: float | None = None
    fwhm_y: float | None = None
    fwhm_y_error: float | None = None
    amplitudes: list[float] = Field(default_factory=list)
    amplitude_errors: list[float] = Field(default_factory=list)


class FitResult(BaseModel):
    """Complete fitting result for a cluster."""

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
    """Result of input validation."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    info: dict[str, Any] = Field(default_factory=dict)
