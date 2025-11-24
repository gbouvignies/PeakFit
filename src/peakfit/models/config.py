"""Pydantic models for PeakFit configuration and data structures.

This module contains configuration models and I/O functions for loading/saving
configuration files in TOML format.
"""

import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

import tomli_w
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


# ============================================================================
# Configuration I/O Functions
# ============================================================================


def load_config(path: Path) -> PeakFitConfig:
    """Load configuration from a TOML file.

    Args:
        path: Path to the TOML configuration file.

    Returns:
        PeakFitConfig: Validated configuration object.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("rb") as f:
        data = tomllib.load(f)

    return PeakFitConfig.model_validate(data)


def save_config(config: PeakFitConfig, path: Path) -> None:
    """Save configuration to a TOML file.

    Args:
        config: Configuration object to save.
        path: Path where to save the TOML file.
    """
    data = config.model_dump(mode="json", exclude_none=True)

    # Convert Path objects to strings
    def convert_paths(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_paths(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    data = convert_paths(data)

    with path.open("wb") as f:
        tomli_w.dump(data, f)


def generate_default_config() -> str:
    """Generate a default configuration file as a string.

    Returns:
        str: TOML-formatted default configuration.
    """
    return """# PeakFit Configuration File
# Generated automatically - edit as needed

[fitting]
lineshape = "auto"  # auto, gaussian, lorentzian, pvoigt, sp1, sp2, no_apod
refine_iterations = 1
fix_positions = false
fit_j_coupling = false
fit_phase_x = false
fit_phase_y = false
max_iterations = 1000
tolerance = 1e-8

[clustering]
contour_factor = 5.0
# contour_level = 1000.0  # Uncomment to set explicit contour level

[output]
directory = "Fits"
formats = ["txt"]
save_simulated = true
save_html_report = true

# Optional settings
# noise_level = 100.0  # Uncomment to set manual noise level
# exclude_planes = []  # List of plane indices to exclude
"""
