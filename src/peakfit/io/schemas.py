"""JSON schema definitions for PeakFit output files.

This module provides Pydantic models that define the structure of
JSON output files. These serve as both documentation and validation
for the output format.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OutputFormat(str, Enum):
    """Supported output formats."""

    JSON = "json"
    CSV = "csv"
    TXT = "txt"  # Legacy
    TOML = "toml"
    MARKDOWN = "markdown"


# =============================================================================
# Run Metadata Schema
# =============================================================================


class InputFileInfo(BaseModel):
    """Information about an input file."""

    path: str = Field(description="Relative path to the file")
    checksum_sha256: str = Field(description="SHA-256 checksum for verification")


class FittingConfiguration(BaseModel):
    """Configuration used for fitting."""

    lineshape: str = Field(
        default="auto",
        description="Lineshape model (auto, gaussian, lorentzian, pvoigt, etc.)",
    )
    refine_iterations: int = Field(default=1, ge=0)
    fix_positions: bool = Field(default=False)
    fit_j_coupling: bool = Field(default=False)
    fit_phase_x: bool = Field(default=False)
    fit_phase_y: bool = Field(default=False)
    max_iterations: int = Field(default=1000, gt=0)
    tolerance: float = Field(default=1e-8, gt=0)


class MCMCConfiguration(BaseModel):
    """Configuration for MCMC analysis."""

    n_walkers: int = Field(default=32, ge=2)
    n_steps: int = Field(default=5000, gt=0)
    burn_in: int | None = Field(default=None, ge=0)
    auto_burnin: bool = Field(default=True)


class RunMetadataSchema(BaseModel):
    """Schema for run_metadata.json.

    This file captures everything needed for reproducibility.
    """

    timestamp: datetime = Field(description="When analysis was run (ISO 8601)")
    software_version: str = Field(description="PeakFit version")
    git_commit: str | None = Field(default=None, description="Git commit hash")
    python_version: str = Field(description="Python interpreter version")
    platform: str = Field(description="OS platform")
    input_files: dict[str, InputFileInfo] = Field(
        default_factory=dict,
        description="Input files with checksums",
    )
    fitting_config: FittingConfiguration | None = Field(default=None)
    mcmc_config: MCMCConfiguration | None = Field(default=None)
    command_line: str = Field(default="", description="Command line arguments")
    run_duration_seconds: float | None = Field(default=None)

    model_config = {"extra": "allow"}


# =============================================================================
# Parameter Schema
# =============================================================================


class ParameterSchema(BaseModel):
    """Schema for a single parameter."""

    name: str = Field(description="Parameter identifier")
    value: float = Field(description="Best-fit value")
    std_error: float = Field(description="Standard error (symmetric uncertainty)")
    unit: str = Field(default="", description="Physical unit")
    category: str = Field(
        default="lineshape",
        description="Parameter category (lineshape, amplitude, exchange, etc.)",
    )
    ci_68: tuple[float, float] | None = Field(
        default=None,
        description="68% confidence interval [lower, upper]",
    )
    ci_95: tuple[float, float] | None = Field(
        default=None,
        description="95% confidence interval [lower, upper]",
    )
    min_bound: float | None = Field(default=None, description="Lower fitting bound")
    max_bound: float | None = Field(default=None, description="Upper fitting bound")
    is_fixed: bool = Field(default=False, description="Whether parameter was fixed")
    is_global: bool = Field(default=False, description="Whether shared across clusters")


class AmplitudeSchema(BaseModel):
    """Schema for an amplitude (intensity) value."""

    peak_name: str
    plane_index: int = Field(ge=0)
    z_value: float | None = Field(default=None, description="Z-dimension value")
    value: float
    std_error: float
    ci_68: tuple[float, float] | None = Field(default=None)


class CorrelationMatrixSchema(BaseModel):
    """Schema for parameter correlations."""

    parameter_names: list[str]
    matrix: list[list[float]] = Field(description="Correlation matrix as nested lists")


class ClusterResultSchema(BaseModel):
    """Schema for results of a single cluster."""

    cluster_id: int
    peak_names: list[str]
    lineshape_parameters: list[ParameterSchema]
    amplitudes: list[AmplitudeSchema] = Field(default_factory=list)
    correlation: CorrelationMatrixSchema | None = Field(default=None)


# =============================================================================
# Statistics Schema
# =============================================================================


class ResidualStatsSchema(BaseModel):
    """Schema for residual statistics."""

    n_points: int
    n_params: int
    dof: int = Field(description="Degrees of freedom")
    noise_level: float
    sum_squared: float = Field(description="Sum of squared normalized residuals")
    rms: float = Field(description="RMS of raw residuals")
    mean: float
    std: float


class FitStatisticsSchema(BaseModel):
    """Schema for fit statistics."""

    chi_squared: float
    reduced_chi_squared: float
    n_data: int
    n_params: int
    dof: int
    aic: float | None = Field(default=None, description="Akaike Information Criterion")
    bic: float | None = Field(default=None, description="Bayesian Information Criterion")
    log_likelihood: float | None = Field(default=None)
    fit_converged: bool = Field(default=True)
    n_function_evals: int = Field(default=0)
    fit_message: str = Field(default="")
    residuals: ResidualStatsSchema | None = Field(default=None)


class ModelComparisonSchema(BaseModel):
    """Schema for model comparison."""

    model_a: str
    model_b: str
    delta_aic: float | None = Field(
        default=None,
        description="AIC(B) - AIC(A), negative favors B",
    )
    delta_bic: float | None = Field(default=None)
    likelihood_ratio: float | None = Field(default=None)
    p_value: float | None = Field(default=None)
    preferred_model: str
    evidence_strength: str = Field(description="strong, moderate, weak, inconclusive")


# =============================================================================
# MCMC Diagnostics Schema
# =============================================================================


class ParameterDiagnosticSchema(BaseModel):
    """Schema for per-parameter MCMC diagnostics."""

    name: str
    rhat: float | None = Field(
        default=None,
        description="R-hat (should be ≤ 1.01)",
    )
    ess_bulk: float | None = Field(
        default=None,
        description="Bulk effective sample size",
    )
    ess_tail: float | None = Field(
        default=None,
        description="Tail effective sample size",
    )
    status: str = Field(
        default="unknown",
        description="Convergence status (excellent, good, acceptable, marginal, poor)",
    )
    warnings: list[str] = Field(default_factory=list)


class MCMCDiagnosticsSchema(BaseModel):
    """Schema for mcmc_diagnostics.json."""

    n_chains: int
    n_samples: int = Field(description="Samples per chain after burn-in")
    burn_in: int
    burn_in_method: str = Field(
        default="manual",
        description="How burn-in was determined (manual, auto, geweke, ess)",
    )
    total_samples: int
    overall_status: str = Field(description="Worst status among all parameters")
    converged: bool
    parameters: list[ParameterDiagnosticSchema]
    burn_in_details: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Fit Summary Schema (top-level)
# =============================================================================


class FitSummarySchema(BaseModel):
    """Schema for fit_summary.json - the main output file.

    This aggregates all results from a fitting run.
    """

    # Metadata
    metadata: RunMetadataSchema

    # Method
    method: str = Field(description="Fitting method used")
    experiment_type: str = Field(default="", description="CPMG, CEST, R1, etc.")

    # Counts
    n_clusters: int
    n_peaks: int

    # Results per cluster
    clusters: list[ClusterResultSchema]

    # Statistics (one per cluster)
    statistics: list[FitStatisticsSchema] = Field(default_factory=list)
    global_statistics: FitStatisticsSchema | None = Field(default=None)

    # MCMC diagnostics (one per cluster, if MCMC used)
    mcmc_diagnostics: list[MCMCDiagnosticsSchema] = Field(default_factory=list)

    # Model comparisons
    model_comparisons: list[ModelComparisonSchema] = Field(default_factory=list)

    # Z-axis information
    z_values: list[float] | None = Field(default=None)
    z_unit: str = Field(default="")

    model_config = {"extra": "forbid"}


# =============================================================================
# CSV Format Definitions
# =============================================================================


class CSVColumnDefinition(BaseModel):
    """Definition of a CSV column for documentation."""

    name: str
    description: str
    unit: str = ""
    data_type: str = "float"  # float, int, str, bool


# Long-format parameter CSV columns
PARAMETER_CSV_COLUMNS: list[CSVColumnDefinition] = [
    CSVColumnDefinition(
        name="cluster_id",
        description="Cluster identifier",
        data_type="int",
    ),
    CSVColumnDefinition(
        name="peak_name",
        description="Peak identifier from input peak list",
        data_type="str",
    ),
    CSVColumnDefinition(
        name="parameter",
        description="Parameter name",
        data_type="str",
    ),
    CSVColumnDefinition(
        name="category",
        description="Parameter category (lineshape, amplitude, etc.)",
        data_type="str",
    ),
    CSVColumnDefinition(
        name="value",
        description="Best-fit value",
        data_type="float",
    ),
    CSVColumnDefinition(
        name="std_error",
        description="Standard error (symmetric uncertainty)",
        data_type="float",
    ),
    CSVColumnDefinition(
        name="ci_68_lower",
        description="Lower bound of 68% CI",
        data_type="float",
    ),
    CSVColumnDefinition(
        name="ci_68_upper",
        description="Upper bound of 68% CI",
        data_type="float",
    ),
    CSVColumnDefinition(
        name="ci_95_lower",
        description="Lower bound of 95% CI",
        data_type="float",
    ),
    CSVColumnDefinition(
        name="ci_95_upper",
        description="Upper bound of 95% CI",
        data_type="float",
    ),
    CSVColumnDefinition(
        name="unit",
        description="Physical unit",
        data_type="str",
    ),
    CSVColumnDefinition(
        name="min_bound",
        description="Lower fitting bound",
        data_type="float",
    ),
    CSVColumnDefinition(
        name="max_bound",
        description="Upper fitting bound",
        data_type="float",
    ),
    CSVColumnDefinition(
        name="is_fixed",
        description="Whether parameter was fixed",
        data_type="bool",
    ),
    CSVColumnDefinition(
        name="is_global",
        description="Whether shared across clusters",
        data_type="bool",
    ),
]

# Amplitude CSV columns (separate file for intensities)
AMPLITUDE_CSV_COLUMNS: list[CSVColumnDefinition] = [
    CSVColumnDefinition(name="cluster_id", description="Cluster identifier", data_type="int"),
    CSVColumnDefinition(name="peak_name", description="Peak identifier", data_type="str"),
    CSVColumnDefinition(name="plane_index", description="Z-dimension index", data_type="int"),
    CSVColumnDefinition(name="z_value", description="Z-dimension value"),
    CSVColumnDefinition(name="value", description="Fitted amplitude"),
    CSVColumnDefinition(name="std_error", description="Amplitude uncertainty"),
    CSVColumnDefinition(name="ci_68_lower", description="Lower 68% CI"),
    CSVColumnDefinition(name="ci_68_upper", description="Upper 68% CI"),
]


def get_csv_header(columns: list[CSVColumnDefinition]) -> str:
    """Generate CSV header line from column definitions."""
    return ",".join(col.name for col in columns)


def get_csv_header_comment(columns: list[CSVColumnDefinition]) -> str:
    """Generate commented header documentation for CSV."""
    lines = ["# Column definitions:"]
    for col in columns:
        unit_str = f" ({col.unit})" if col.unit else ""
        lines.append(f"#   {col.name}: {col.description}{unit_str} [{col.data_type}]")
    return "\n".join(lines)
