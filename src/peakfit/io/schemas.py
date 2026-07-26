"""Validation models for the authoritative completed-fit JSON document."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

OUTPUT_SCHEMA_VERSION: Literal["4.0.0"] = "4.0.0"

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]


class _Schema(BaseModel):
    """Reject fields that are not part of the versioned completed-fit contract."""

    model_config = ConfigDict(extra="forbid")


class RunMetadataSchema(_Schema):
    """Operational metadata attached to one completed fit run."""

    timestamp: datetime
    software_version: str
    git_commit: str | None = None
    python_version: str
    platform: str
    input_files: dict[str, dict[str, str]] = Field(default_factory=dict)
    configuration: dict[str, Any] = Field(default_factory=dict)
    command_line: str = ""
    run_duration_seconds: FiniteFloat | None = None


class FinalParameterSchema(_Schema):
    """One immutable nonlinear parameter copied from a final outcome."""

    name: str = Field(min_length=1)
    value: FiniteFloat
    min_bound: FiniteFloat | None
    max_bound: FiniteFloat | None
    vary: bool
    unit: str
    standard_error: FiniteFloat | None


class AnalyticalStatisticsSchema(_Schema):
    """Statistics from the one frozen analytical evaluation."""

    chi_squared: FiniteFloat
    n_observations: NonNegativeInt
    n_nonlinear_parameters: NonNegativeInt
    n_amplitude_parameters: NonNegativeInt
    n_fitted_parameters: NonNegativeInt
    degrees_of_freedom: Annotated[int, Field(ge=1)]
    reduced_chi_squared: FiniteFloat
    amplitude_uncertainty_scale: FiniteFloat
    aic: FiniteFloat
    bic: FiniteFloat
    log_likelihood: FiniteFloat


class AnalyticalEvaluationSchema(_Schema):
    """The frozen numerical values available only for usable outcomes."""

    shapes: list[list[FiniteFloat]]
    amplitudes: list[list[FiniteFloat]]
    amplitude_standard_errors: list[FiniteFloat]
    amplitude_covariance: list[list[FiniteFloat]]
    scaled_amplitude_standard_errors: list[FiniteFloat]
    model_values: list[list[FiniteFloat]]
    raw_residuals: list[list[FiniteFloat]]
    normalized_residuals: list[FiniteFloat]
    statistics: AnalyticalStatisticsSchema

    @model_validator(mode="after")
    def validate_shapes(self) -> AnalyticalEvaluationSchema:
        """Reject incompatible analytical arrays instead of guessing their association."""
        n_peaks = len(self.shapes)
        if n_peaks == 0 or not self.shapes[0]:
            raise ValueError("analytical shapes must be a non-empty peak-by-point matrix")
        n_points = len(self.shapes[0])
        if any(len(row) != n_points for row in self.shapes):
            raise ValueError("analytical shapes must have a rectangular peak-by-point shape")

        if len(self.amplitudes) != n_peaks or not self.amplitudes[0]:
            raise ValueError("analytical amplitudes must match the peak count")
        n_series = len(self.amplitudes[0])
        if any(len(row) != n_series for row in self.amplitudes):
            raise ValueError("analytical amplitudes must have a rectangular peak-by-series shape")
        if len(self.amplitude_standard_errors) != n_peaks:
            raise ValueError("amplitude standard errors must match the peak count")
        if len(self.scaled_amplitude_standard_errors) != n_peaks:
            raise ValueError("scaled amplitude standard errors must match the peak count")
        if len(self.amplitude_covariance) != n_peaks or any(
            len(row) != n_peaks for row in self.amplitude_covariance
        ):
            raise ValueError("amplitude covariance must be a peak-by-peak matrix")

        for name, values in (
            ("model values", self.model_values),
            ("raw residuals", self.raw_residuals),
        ):
            if len(values) != n_points or any(len(row) != n_series for row in values):
                raise ValueError(f"{name} must have the point-by-series analytical shape")
        if len(self.normalized_residuals) != n_points * n_series:
            raise ValueError("normalized residuals must contain one value per observation")
        if self.statistics.n_observations != n_points * n_series:
            raise ValueError("n_observations must match the analytical point-by-series shape")
        return self


class OptimizerProvenanceSchema(_Schema):
    """Trustworthy diagnostics copied from the actual terminal optimizer call."""

    optimizer_kind: str | None = None
    success: StrictBool
    termination_message: str | None = None
    function_evaluations: NonNegativeInt | None = None
    jacobian_evaluations: NonNegativeInt | None = None
    iterations: NonNegativeInt | None = None
    optimality: FiniteFloat | None = None
    final_cost: FiniteFloat | None = None
    correction_revision: NonNegativeInt
    metadata: dict[str, Any] = Field(default_factory=dict)


class FinalClusterOutcomeSchema(_Schema):
    """One final cluster, always identified by its stable ``cluster_id``."""

    cluster_id: NonNegativeInt
    peak_names: list[str]
    classification: Literal["converged", "usable_non_converged", "unusable"]
    unusable_reason: str | None
    correction_revision: NonNegativeInt
    optimizer_provenance: OptimizerProvenanceSchema
    final_nonlinear_parameters: list[FinalParameterSchema]
    analytical_evaluation: AnalyticalEvaluationSchema | None

    @model_validator(mode="after")
    def validate_outcome_combination(self) -> FinalClusterOutcomeSchema:
        """Keep convergence, usability, provenance, and numerics internally consistent."""
        if self.optimizer_provenance.correction_revision != self.correction_revision:
            raise ValueError("optimizer provenance correction_revision must match the cluster")
        if self.classification == "unusable":
            if not self.unusable_reason:
                raise ValueError("unusable outcomes require an unusable_reason")
            if self.analytical_evaluation is not None:
                raise ValueError("unusable outcomes must not contain an analytical evaluation")
            if self.final_nonlinear_parameters:
                raise ValueError("unusable outcomes must not contain final nonlinear parameters")
        else:
            if self.unusable_reason is not None:
                raise ValueError("usable outcomes must not contain an unusable_reason")
            if self.analytical_evaluation is None:
                raise ValueError("usable outcomes require an analytical evaluation")
            if self.classification == "converged" and not self.optimizer_provenance.success:
                raise ValueError("a converged outcome requires optimizer provenance success")
            if self.classification == "usable_non_converged" and self.optimizer_provenance.success:
                raise ValueError("a usable non-converged outcome requires optimizer failure")
        return self


class FinalFitStatisticsSchema(_Schema):
    """Usable-only global statistics copied from the final outcome."""

    chi_squared: FiniteFloat
    reduced_chi_squared: FiniteFloat
    n_observations: NonNegativeInt
    n_fitted_parameters: NonNegativeInt
    degrees_of_freedom: Annotated[int, Field(ge=1)]
    aic: FiniteFloat | None
    bic: FiniteFloat | None
    log_likelihood: FiniteFloat | None
    function_evaluations: NonNegativeInt | None


class ZAxisSchema(_Schema):
    """The ordered series coordinate values used by the run."""

    values: list[FiniteFloat]


class FitSummarySchema(_Schema):
    """Version 4.0.0 completed-fit JSON document."""

    schema_version: Literal["4.0.0"]
    metadata: RunMetadataSchema
    terminal_correction_revision: NonNegativeInt
    noise: FiniteFloat
    final_nonlinear_parameters: list[FinalParameterSchema]
    clusters: list[FinalClusterOutcomeSchema]
    statistics: FinalFitStatisticsSchema
    z_axis: ZAxisSchema | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_unsupported_version(cls, value: Any) -> Any:
        """Name both versions when a development artifact predates this contract."""
        if isinstance(value, dict) and value.get("schema_version") != OUTPUT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported fit JSON schema version "
                f"{value.get('schema_version')!r}; supported version is {OUTPUT_SCHEMA_VERSION!r}."
            )
        return value

    @model_validator(mode="after")
    def validate_cluster_identity_and_revision(self) -> FitSummarySchema:
        """Require explicit, unique, presentation-ordered cluster identity."""
        ids = [cluster.cluster_id for cluster in self.clusters]
        if len(set(ids)) != len(ids):
            raise ValueError("cluster_id values must be unique")
        if ids != sorted(ids):
            raise ValueError("clusters must be serialized in ascending cluster_id order")
        for cluster in self.clusters:
            if cluster.correction_revision != self.terminal_correction_revision:
                raise ValueError("cluster correction_revision must match the terminal revision")
        return self


# ``ResultsLoader`` imports this public name.  In schema 4 it is the full final
# cluster outcome rather than the legacy estimates-only representation.
ClusterResultSchema = FinalClusterOutcomeSchema


__all__ = [
    "OUTPUT_SCHEMA_VERSION",
    "AnalyticalEvaluationSchema",
    "AnalyticalStatisticsSchema",
    "ClusterResultSchema",
    "FinalClusterOutcomeSchema",
    "FinalFitStatisticsSchema",
    "FinalParameterSchema",
    "FitSummarySchema",
    "OptimizerProvenanceSchema",
    "RunMetadataSchema",
    "ZAxisSchema",
]
