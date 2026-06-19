"""Schema definitions for PeakFit JSON output files.

This module provides Pydantic models that define the structure of
JSON output files. These serve as both documentation and validation for
the output format used by current writers.
"""

from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

OUTPUT_SCHEMA_VERSION = "2.0.0"


def _normalize_optional_std_error(value: Any) -> float | None:
    """Normalize nullable/non-finite uncertainty values."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in {"", "nan", "none", "null", "na", "n/a"}:
            return None
        value = cleaned

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        msg = "std_error must be numeric or null"
        raise ValueError(msg) from exc

    if not np.isfinite(numeric):
        return None
    return numeric


# =============================================================================
# Run Metadata Schema
# =============================================================================


class RunMetadataSchema(BaseModel):
    """Run metadata embedded in summary/fit.json."""

    timestamp: datetime = Field(description="When analysis was run (ISO 8601)")
    software_version: str = Field(description="PeakFit version")
    git_commit: str | None = Field(default=None, description="Git commit hash")
    python_version: str = Field(description="Python interpreter version")
    platform: str = Field(description="OS platform")
    input_files: dict[str, dict[str, str]] = Field(
        default_factory=dict,
        description="Input file paths and checksums",
    )
    configuration: dict[str, Any] = Field(default_factory=dict)
    command_line: str = Field(default="", description="Command line arguments")
    run_duration_seconds: float | None = Field(default=None)

    model_config = ConfigDict(extra="allow")


# =============================================================================
# Parameter Schema
# =============================================================================


class ParameterSchema(BaseModel):
    """Schema for a single parameter."""

    name: str = Field(description="Parameter identifier")
    value: float = Field(description="Best-fit value")
    std_error: float | None = Field(
        default=None,
        description="Standard error (symmetric uncertainty), if available",
    )
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

    @field_validator("std_error", mode="before")
    @classmethod
    def normalize_std_error(cls, value: Any) -> float | None:
        """Normalize nullable/non-finite uncertainty values."""
        return _normalize_optional_std_error(value)


class CorrelationMatrixSchema(BaseModel):
    """Schema for parameter correlations."""

    parameter_names: list[str]
    matrix: list[list[float]] = Field(description="Correlation matrix as nested lists")


class ClusterResultSchema(BaseModel):
    """Schema for results of a single cluster."""

    cluster_id: int
    peak_names: list[str]
    lineshape_parameters: list[ParameterSchema] = Field(alias="parameters")
    correlation: CorrelationMatrixSchema | None = Field(default=None)

    model_config = ConfigDict(populate_by_name=True)


class ZAxisSchema(BaseModel):
    """Schema for z-axis metadata."""

    values: list[float]


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
    dof: int | None = Field(default=None, alias="degrees_of_freedom")
    aic: float | None = Field(default=None, description="Akaike Information Criterion")
    bic: float | None = Field(default=None, description="Bayesian Information Criterion")
    log_likelihood: float | None = Field(default=None)
    fit_converged: bool = Field(default=True)
    n_function_evals: int = Field(default=0)
    fit_message: str = Field(default="")
    residuals: ResidualStatsSchema | None = Field(default=None)

    model_config = ConfigDict(populate_by_name=True, extra="allow")


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
    """Schema for MCMC diagnostics embedded in summary/fit.json."""

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
    warnings: list[str] = Field(default_factory=list)
    burn_in_details: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Fit Summary Schema (top-level)
# =============================================================================


class FitSummarySchema(BaseModel):
    """Schema for summary/fit.json - the main output file.

    This aggregates all results from a fitting run.
    """

    # Versioning
    schema_version: str = Field(description="Output schema version")

    # Metadata
    metadata: RunMetadataSchema

    # Method
    method: str = Field(description="Fitting method used")

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

    # Z-axis information
    z_values: list[float] | None = Field(default=None)
    z_axis: ZAxisSchema | None = Field(default=None)

    model_config = ConfigDict(extra="allow", populate_by_name=True)
