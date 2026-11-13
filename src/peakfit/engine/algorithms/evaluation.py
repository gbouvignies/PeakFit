"""Shared analytical model evaluation for one peak cluster."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.covariance_factors import (
    CovarianceFactors,
    covariance_factors,
    peak_derivatives,
)
from peakfit.engine.algorithms.linear_algebra import calculate_amplitudes_with_uncertainty
from peakfit.engine.algorithms.noise_correlation import modelled_correlation
from peakfit.engine.algorithms.varpro import at_active_bound
from peakfit.engine.results import (
    compute_chi_squared,
    compute_degrees_of_freedom,
    compute_reduced_chi_squared,
)

if TYPE_CHECKING:
    from peakfit.engine.algorithms.noise_correlation import NoiseModel
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


@dataclass(frozen=True, slots=True)
class AnalyticalModelEvaluation:
    """One coherent analytical amplitude, model, residual, and statistics value."""

    cluster_id: int
    shapes: FloatArray
    amplitudes: FloatArray
    amplitude_standard_errors: FloatArray
    amplitude_covariance: FloatArray
    model_values: FloatArray
    raw_residuals: FloatArray
    normalized_residuals: FloatArray
    statistics: AnalyticalFitStatistics
    # The factors of the amplitude covariance between planes (profile-uncertainty §3.1).
    covariance_factors: CovarianceFactors
    uncertainty_warning: str | None = None


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


# What a lineshape evaluation may raise on parameter values it cannot evaluate.
_LINESHAPE_ERRORS = (
    AttributeError,
    IndexError,
    KeyError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
    np.linalg.LinAlgError,
)


class _InvalidEvaluationError(ValueError):
    """Internal control flow for an explicitly unusable analytical state."""


def evaluate_analytical_model(
    cluster: Cluster,
    params: Parameters,
    noise: float,
    noise_model: NoiseModel,
) -> AnalyticalEvaluationResult:
    """Solve amplitudes and derive the model and residuals from the same state.

    Amplitude errors are the sandwich under the noise model's correlation when it is
    modelled, and assume independent points otherwise; nothing else depends on it.
    """
    try:
        return _evaluate_analytical_model(
            cluster, params, noise, modelled_correlation(noise_model, cluster.grid_indices)
        )
    except _InvalidEvaluationError as error:
        return _analytical_failure(cluster, str(error))


def _evaluate_analytical_model(
    cluster: Cluster,
    params: Parameters,
    noise: float,
    correlation: FloatArray | None,
) -> AnalyticalModelEvaluation:
    nonlinear_names = cluster.parameter_names(params)
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
        correlation,
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
        chi_squared,
    )
    uncertainty_warning = None
    if not np.all(np.isfinite(amplitude_errors)) or any(
        params[name].vary and not np.isfinite(params[name].stderr) for name in nonlinear_names
    ):
        uncertainty_warning = (
            "Unreliable local uncertainties: overlapping or unidentifiable parameters, "
            "active bounds, or uncertainty exceeding parameter bounds. "
            "Inspect residuals and alternative fits before using fitted positions."
        )

    return AnalyticalModelEvaluation(
        cluster_id=cluster.cluster_id,
        shapes=shapes,
        amplitudes=amplitudes,
        amplitude_standard_errors=amplitude_errors,
        amplitude_covariance=covariance,
        model_values=model_values,
        raw_residuals=raw_residuals,
        normalized_residuals=normalized_residuals,
        statistics=statistics,
        uncertainty_warning=uncertainty_warning,
        covariance_factors=_covariance_factors(
            cluster, params, nonlinear_names, shapes, amplitudes, covariance, noise, correlation
        ),
    )


def _covariance_factors(
    cluster: Cluster,
    params: Parameters,
    nonlinear_names: list[str],
    shapes: FloatArray,
    amplitudes: FloatArray,
    covariance: FloatArray,
    noise: float,
    correlation: FloatArray | None,
) -> CovarianceFactors:
    """The factors, withheld rather than failing the evaluation on a numerical error."""
    try:
        derivatives = peak_derivatives(cluster, params, _free(params, nonlinear_names))
    except _LINESHAPE_ERRORS:
        return CovarianceFactors(covariance, None, None, None)
    return covariance_factors(shapes, derivatives, amplitudes, covariance, noise, correlation)


def _free(params: Parameters, names: list[str]) -> list[str]:
    """The varied parameters not held at an active bound: the free lineshape θ."""
    varied = [name for name in names if params[name].vary]
    values, lower, upper = (
        np.array([getattr(params[name], field) for name in varied], dtype=np.float64)
        for field in ("value", "min", "max")
    )
    active = at_active_bound(values, lower, upper)
    return [name for name, held in zip(varied, active, strict=True) if not held]


def _evaluate_inputs(
    cluster: Cluster,
    params: Parameters,
) -> tuple[FloatArray, FloatArray]:
    try:
        shapes = np.asarray(cluster.evaluate(params), dtype=np.float64)
    except _LINESHAPE_ERRORS as error:
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
    correlation: FloatArray | None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    try:
        amplitudes, amplitude_errors, covariance = calculate_amplitudes_with_uncertainty(
            shapes,
            data,
            noise,
            correlation,
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
    # Expected NaNs mean that amplitudes/model remain evaluable but their
    # decomposition has no reliable joint covariance. Other nonfinite results
    # (including overflow) remain numerical failures.
    unavailable = (
        np.isnan(amplitude_errors).all()
        and np.isnan(covariance).all()
        and (
            shapes.shape[0] > shapes.shape[1]
            or np.linalg.cond(shapes) >= 1 / np.sqrt(np.finfo(float).eps)
        )
    )
    if not unavailable:
        _require_finite_array(amplitude_errors, "non-finite amplitude uncertainty inputs")
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
    if not np.isfinite(reduced_chi_squared):
        raise _InvalidEvaluationError("non-finite derived statistics")
    return AnalyticalFitStatistics(
        chi_squared=chi_squared,
        n_observations=n_observations,
        n_nonlinear_parameters=n_nonlinear_parameters,
        n_amplitude_parameters=n_amplitude_parameters,
        n_fitted_parameters=n_fitted_parameters,
        degrees_of_freedom=degrees_of_freedom,
        reduced_chi_squared=reduced_chi_squared,
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
    noise_model: NoiseModel,
) -> FitEvaluation:
    """Classify optimizer convergence independently from numerical usability."""
    unusable_reason = _optimizer_result_failure_reason(cluster, result)
    if unusable_reason is not None:
        return _unusable_fit_evaluation(cluster.cluster_id, unusable_reason)

    analytical = evaluate_analytical_model(cluster, result.params, noise, noise_model)
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
