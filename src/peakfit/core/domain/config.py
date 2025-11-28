"""Domain configuration and result models for PeakFit."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

LineshapeName = Literal["auto", "gaussian", "lorentzian", "pvoigt", "sp1", "sp2", "no_apod"]
OutputFormat = Literal["csv", "json", "txt"]
OutputVerbosity = Literal["minimal", "standard", "full"]


class FitConfig(BaseModel):
    """Configuration for the fitting process."""

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
    # Legacy aliases for backward compatibility
    fit_phase_x: bool = Field(
        default=False,
        description="Fit phase correction in direct dimension (deprecated, use fit_phase).",
    )
    fit_phase_y: bool = Field(
        default=False,
        description="Fit phase correction in indirect dimension (deprecated, use fit_phase).",
    )
    max_iterations: Annotated[int, Field(gt=0)] = Field(
        default=1000,
        description="Maximum iterations for optimizer.",
    )
    tolerance: Annotated[float, Field(gt=0)] = Field(
        default=1e-8,
        description="Convergence tolerance for optimizer.",
    )

    def get_phase_dimensions(self, n_spectral_dims: int = 2) -> list[str]:
        """Get list of dimension labels to fit phase for.

        Handles both new fit_phase list and legacy fit_phase_x/y flags.

        Args:
            n_spectral_dims: Number of spectral dimensions (for Fn labeling)

        Returns
        -------
            List of dimension labels like ['F1', 'F2']
        """
        if self.fit_phase:
            return self.fit_phase

        # Convert legacy flags to dimension labels
        # For 2D: x = F2 (direct), y = F1 (indirect)
        # For 3D: x = F3 (direct), y = F2 (first indirect)
        dims = []
        if self.fit_phase_y:
            # Indirect dimension (F1 for 2D, F2 for 3D)
            dims.append(f"F{n_spectral_dims - 1}")
        if self.fit_phase_x:
            # Direct dimension (F2 for 2D, F3 for 3D)
            dims.append(f"F{n_spectral_dims}")
        return dims


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
    include_legacy: bool = Field(
        default=False,
        description="Generate legacy format output files in legacy/ subdirectory.",
    )
    save_chains: bool = Field(
        default=False,
        description="Save MCMC chains to disk (requires significant storage).",
    )
    save_figures: bool = Field(default=True, description="Generate and save diagnostic figures.")
    include_timestamp: bool = Field(
        default=False,
        description="Include timestamp in output directory name.",
    )


class PeakFitConfig(BaseModel):
    """Top-level PeakFit configuration for fitting, clustering, and output."""

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
    # Legacy fields for backward compatibility (2D spectra)
    position_x: float | None = Field(
        default=None, description="X/direct dimension position in ppm (deprecated)"
    )
    position_y: float | None = Field(
        default=None, description="Y/indirect dimension position in ppm (deprecated)"
    )
    position_z: float | None = Field(
        default=None, description="Z position in ppm for 3D (deprecated)"
    )
    cluster_id: int | None = Field(default=None, description="Assigned cluster ID")

    def get_positions(self) -> list[float]:
        """Get positions as a list, handling both new and legacy formats."""
        if self.positions:
            return self.positions
        # Convert legacy x/y/z to list
        pos = []
        if self.position_y is not None:
            pos.append(self.position_y)  # F1 (indirect)
        if self.position_z is not None:
            pos.append(self.position_z)  # F2 for 3D
        if self.position_x is not None:
            pos.append(self.position_x)  # Fn (direct)
        return pos


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
    # Legacy fields for backward compatibility
    position_x: float | None = None
    position_x_error: float | None = None
    position_y: float | None = None
    position_y_error: float | None = None
    fwhm_x: float | None = None
    fwhm_x_error: float | None = None
    fwhm_y: float | None = None
    fwhm_y_error: float | None = None
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
    "LineshapeName",
    "OutputConfig",
    "OutputFormat",
    "OutputVerbosity",
    "PeakData",
    "PeakFitConfig",
    "ValidationResult",
]
