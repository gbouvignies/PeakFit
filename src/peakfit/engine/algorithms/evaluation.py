"""Shared analytical model evaluation for one peak cluster."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.linear_algebra import calculate_amplitudes_with_uncertainty
from peakfit.engine.results import (
    compute_chi_squared,
    compute_degrees_of_freedom,
    compute_reduced_chi_squared,
)

if TYPE_CHECKING:
    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.engine.results import FitResult
    from peakfit.shared.typing import FloatArray


class FitOutcomeClassification(StrEnum):
    """Relationship between optimizer convergence and numerical usability."""

    CONVERGED = "converged"
    USABLE_NON_CONVERGED = "usable_non_converged"
    UNUSABLE = "unusable"


@dataclass(frozen=True, slots=True)
class AnalyticalFitStatistics:
    """Statistics derived from one coherent analytical evaluation."""

    chi_squared: float
    n_observations: int
    n_nonlinear_parameters: int
    n_amplitude_parameters: int
    n_fitted_parameters: int
    degrees_of_freedom: int
    reduced_chi_squared: float
    amplitude_uncertainty_scale: float
    aic: float
    bic: float
    log_likelihood: float


@dataclass(frozen=True, slots=True)
class AnalyticalModelEvaluation:
    """One coherent analytical amplitude, model, residual, and statistics value."""

    cluster_id: int
    shapes: FloatArray
    amplitudes: FloatArray
    amplitude_standard_errors: FloatArray
    amplitude_covariance: FloatArray
    scaled_amplitude_standard_errors: FloatArray
    model_values: FloatArray
    raw_residuals: FloatArray
    normalized_residuals: FloatArray
    statistics: AnalyticalFitStatistics


@dataclass(frozen=True, slots=True)
class AnalyticalEvaluationFailure:
    """Explicit reason an analytical model could not be evaluated."""

    cluster_id: int
    reason: str


AnalyticalEvaluationResult = AnalyticalModelEvaluation | AnalyticalEvaluationFailure


@dataclass(frozen=True, slots=True)
class FitEvaluation:
    """Classification and analytical state for one optimizer result."""

    cluster_id: int
    classification: FitOutcomeClassification
    analytical: AnalyticalModelEvaluation | None
    unusable_reason: str | None = None

    @property
    def usable(self) -> bool:
        """Return whether the result may contribute numerical state."""
        return self.classification is not FitOutcomeClassification.UNUSABLE


class _InvalidEvaluationError(ValueError):
    """Internal control flow for an explicitly unusable analytical state."""


def evaluate_analytical_model(
    cluster: Cluster,
    params: Parameters,
    noise: float,
) -> AnalyticalEvaluationResult:
    """Solve amplitudes and derive the model and residuals from the same state."""
    try:
        return _evaluate_analytical_model(cluster, params, noise)
    except _InvalidEvaluationError as error:
        return _analytical_failure(cluster, str(error))


def _evaluate_analytical_model(
    cluster: Cluster,
    params: Parameters,
    noise: float,
) -> AnalyticalModelEvaluation:
    nonlinear_names = _cluster_nonlinear_parameter_names(cluster, params)
    nonfinite_names = [name for name in nonlinear_names if not np.isfinite(params[name].value)]
    if nonfinite_names:
        raise _InvalidEvaluationError(
            f"non-finite nonlinear parameters: {', '.join(nonfinite_names)}",
        )

    if not np.isfinite(noise) or noise <= 0.0:
        raise _InvalidEvaluationError(f"noise must be positive and finite, got {noise}")

    shapes, data = _evaluate_inputs(cluster, params)
    amplitudes, amplitude_errors, covariance = _solve_amplitudes(
        cluster,
        shapes,
        data,
        noise,
    )
    model_values, raw_residuals, normalized_residuals = _evaluate_model(
        cluster,
        shapes,
        amplitudes,
        data,
        noise,
    )
    chi_squared = compute_chi_squared(normalized_residuals)
    _require_finite_scalar(chi_squared, "non-finite chi-squared")
    statistics = _derive_statistics(
        cluster,
        params,
        nonlinear_names,
        noise,
        chi_squared,
    )
    scaled_amplitude_errors = amplitude_errors * statistics.amplitude_uncertainty_scale
    _require_finite_array(
        scaled_amplitude_errors,
        "non-finite scaled amplitude uncertainties",
    )

    return AnalyticalModelEvaluation(
        cluster_id=cluster.cluster_id,
        shapes=shapes,
        amplitudes=amplitudes,
        amplitude_standard_errors=amplitude_errors,
        amplitude_covariance=covariance,
        scaled_amplitude_standard_errors=scaled_amplitude_errors,
        model_values=model_values,
        raw_residuals=raw_residuals,
        normalized_residuals=normalized_residuals,
        statistics=statistics,
    )


def _evaluate_inputs(
    cluster: Cluster,
    params: Parameters,
) -> tuple[FloatArray, FloatArray]:
    try:
        shapes = np.asarray(cluster.evaluate(params), dtype=np.float64)
    except (
        AttributeError,
        IndexError,
        KeyError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        raise _InvalidEvaluationError(
            f"lineshape evaluation failed: {type(error).__name__}: {error}",
        ) from error

    try:
        data = np.asarray(cluster.corrected_data.real, dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as error:
        raise _InvalidEvaluationError(
            f"corrected data evaluation failed: {type(error).__name__}: {error}",
        ) from error
    _require_shape(
        shapes,
        (len(cluster.peaks), cluster.n_points),
        "lineshape values",
    )
    _require_finite_array(shapes, "non-finite lineshape values")
    _require_shape(data, (cluster.n_points, cluster.n_series), "corrected data")
    _require_finite_array(data, "non-finite corrected data")
    return shapes, data


def _solve_amplitudes(
    cluster: Cluster,
    shapes: FloatArray,
    data: FloatArray,
    noise: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    try:
        amplitudes, amplitude_errors, covariance = calculate_amplitudes_with_uncertainty(
            shapes,
            data,
            noise,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        raise _InvalidEvaluationError(
            f"analytical amplitude solve failed: {type(error).__name__}: {error}",
        ) from error

    n_peaks = len(cluster.peaks)
    _require_shape(
        amplitudes,
        (n_peaks, cluster.n_series),
        "analytical amplitude",
    )
    _require_finite_array(amplitudes, "non-finite analytical amplitudes")
    _require_shape(amplitude_errors, (n_peaks,), "amplitude standard-error")
    _require_shape(covariance, (n_peaks, n_peaks), "amplitude covariance")
    _require_finite_array(
        amplitude_errors,
        "non-finite amplitude uncertainty inputs",
    )
    _require_finite_array(covariance, "non-finite amplitude uncertainty inputs")
    return amplitudes, amplitude_errors, covariance


def _evaluate_model(
    cluster: Cluster,
    shapes: FloatArray,
    amplitudes: FloatArray,
    data: FloatArray,
    noise: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    with np.errstate(over="ignore", invalid="ignore"):
        model_values = shapes.T @ amplitudes
        raw_residuals = data - model_values
        normalized_residuals = raw_residuals.ravel() / noise

    expected_data_shape = (cluster.n_points, cluster.n_series)
    _require_shape(model_values, expected_data_shape, "model values")
    _require_finite_array(model_values, "non-finite model values")
    _require_shape(raw_residuals, expected_data_shape, "raw residual")
    _require_finite_array(raw_residuals, "non-finite analytical residuals")
    _require_shape(
        normalized_residuals,
        (cluster.n_observations,),
        "normalized residual",
    )
    _require_finite_array(normalized_residuals, "non-finite normalized residuals")
    return model_values, raw_residuals, normalized_residuals


def _derive_statistics(
    cluster: Cluster,
    params: Parameters,
    nonlinear_names: list[str],
    noise: float,
    chi_squared: float,
) -> AnalyticalFitStatistics:
    n_observations = cluster.n_observations
    n_nonlinear_parameters = sum(params[name].vary for name in nonlinear_names)
    n_amplitude_parameters = cluster.n_amplitude_params
    n_fitted_parameters = n_nonlinear_parameters + n_amplitude_parameters
    degrees_of_freedom = compute_degrees_of_freedom(
        n_observations,
        n_fitted_parameters,
    )
    reduced_chi_squared = compute_reduced_chi_squared(
        chi_squared,
        n_observations,
        n_fitted_parameters,
    )
    amplitude_uncertainty_scale = (
        float(np.sqrt(reduced_chi_squared)) if reduced_chi_squared > 1.0 else 1.0
    )
    log_likelihood = float(
        -0.5 * chi_squared
        - n_observations * np.log(noise)
        - 0.5 * n_observations * np.log(2.0 * np.pi)
    )
    aic = float(-2.0 * log_likelihood + 2.0 * n_fitted_parameters)
    bic = float(-2.0 * log_likelihood + n_fitted_parameters * np.log(n_observations))
    derived_values = (
        reduced_chi_squared,
        amplitude_uncertainty_scale,
        log_likelihood,
        aic,
        bic,
    )
    if not all(np.isfinite(value) for value in derived_values):
        raise _InvalidEvaluationError("non-finite derived statistics")
    return AnalyticalFitStatistics(
        chi_squared=chi_squared,
        n_observations=n_observations,
        n_nonlinear_parameters=n_nonlinear_parameters,
        n_amplitude_parameters=n_amplitude_parameters,
        n_fitted_parameters=n_fitted_parameters,
        degrees_of_freedom=degrees_of_freedom,
        reduced_chi_squared=reduced_chi_squared,
        amplitude_uncertainty_scale=amplitude_uncertainty_scale,
        aic=aic,
        bic=bic,
        log_likelihood=log_likelihood,
    )


def _require_shape(
    values: FloatArray,
    expected: tuple[int, ...],
    label: str,
) -> None:
    if values.shape != expected:
        raise _InvalidEvaluationError(
            f"{label} shape mismatch: expected {expected}, got {values.shape}"
        )


def _require_finite_array(values: FloatArray, reason: str) -> None:
    if not np.all(np.isfinite(values)):
        raise _InvalidEvaluationError(reason)


def _require_finite_scalar(value: float, reason: str) -> None:
    if not np.isfinite(value):
        raise _InvalidEvaluationError(reason)


def classify_optimizer_result(
    *,
    cluster: Cluster,
    result: FitResult,
    noise: float,
) -> FitEvaluation:
    """Classify optimizer convergence independently from numerical usability."""
    unusable_reason = _optimizer_result_failure_reason(cluster, result)
    if unusable_reason is not None:
        return _unusable_fit_evaluation(cluster.cluster_id, unusable_reason)

    analytical = evaluate_analytical_model(cluster, result.params, noise)
    if isinstance(analytical, AnalyticalEvaluationFailure):
        return _unusable_fit_evaluation(result.cluster_id, analytical.reason)

    classification = (
        FitOutcomeClassification.CONVERGED
        if result.success
        else FitOutcomeClassification.USABLE_NON_CONVERGED
    )
    return FitEvaluation(
        cluster_id=result.cluster_id,
        classification=classification,
        analytical=analytical,
    )


def _optimizer_result_failure_reason(
    cluster: Cluster,
    result: FitResult,
) -> str | None:
    if result.cluster_id != cluster.cluster_id:
        return f"cluster_id mismatch: expected {cluster.cluster_id}, got {result.cluster_id}"

    if result.n_amplitude_params != cluster.n_amplitude_params:
        return (
            "optimizer amplitude parameter count mismatch: "
            f"expected {cluster.n_amplitude_params}, "
            f"got {result.n_amplitude_params}"
        )

    optimizer_residual = np.asarray(result.residual, dtype=np.float64)
    expected_residual_shape = (cluster.n_observations,)
    if optimizer_residual.shape != expected_residual_shape:
        return (
            "optimizer residual shape mismatch: "
            f"expected {expected_residual_shape}, got {optimizer_residual.shape}"
        )
    if not np.all(np.isfinite(optimizer_residual)):
        return "non-finite optimizer residuals"
    if not np.isfinite(result.cost):
        return "non-finite optimizer cost"
    return None


def _unusable_fit_evaluation(cluster_id: int, reason: str) -> FitEvaluation:
    return FitEvaluation(
        cluster_id=cluster_id,
        classification=FitOutcomeClassification.UNUSABLE,
        analytical=None,
        unusable_reason=reason,
    )


def _analytical_failure(
    cluster: Cluster,
    reason: str,
) -> AnalyticalEvaluationFailure:
    return AnalyticalEvaluationFailure(cluster_id=cluster.cluster_id, reason=reason)


def _cluster_nonlinear_parameter_names(
    cluster: Cluster,
    params: Parameters,
) -> list[str]:
    peak_names = {peak.name for peak in cluster.peaks}
    cluster_prefix = f"cluster_{cluster.cluster_id}."
    names: list[str] = []
    for name, parameter in params.items():
        param_id = parameter.param_id
        if param_id is not None:
            if param_id.label == "I":
                continue
            belongs_to_cluster = param_id.cluster_id == cluster.cluster_id
            belongs_to_peak = param_id.peak_name in peak_names
        else:
            if re.search(r"\.I\d+$", name):
                continue
            belongs_to_cluster = name.startswith(cluster_prefix)
            belongs_to_peak = any(name.startswith(f"{peak_name}.") for peak_name in peak_names)
        if belongs_to_cluster or belongs_to_peak:
            names.append(name)
    return names


__all__ = [
    "AnalyticalEvaluationFailure",
    "AnalyticalEvaluationResult",
    "AnalyticalFitStatistics",
    "AnalyticalModelEvaluation",
    "FitEvaluation",
    "FitOutcomeClassification",
    "classify_optimizer_result",
    "evaluate_analytical_model",
]
