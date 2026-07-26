"""Immutable authoritative results assembled when a fitting pipeline completes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.engine.algorithms.evaluation import (
    AnalyticalFitStatistics,
    AnalyticalModelEvaluation,
    FitEvaluation,
    FitOutcomeClassification,
)
from peakfit.engine.results import compute_reduced_chi_squared

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.params_scalar import Parameter, Parameters
    from peakfit.engine.results import FitResult
    from peakfit.fit.pipeline import PipelineCompletion
    from peakfit.shared.typing import FloatArray


@dataclass(frozen=True, slots=True)
class FinalParameter:
    """An immutable value copy of one final nonlinear fitting parameter."""

    name: str
    value: float
    min: float
    max: float
    vary: bool
    unit: str
    standard_error: float


@dataclass(frozen=True, slots=True)
class FinalAnalyticalEvaluation:
    """A frozen copy of ticket-03's one coherent analytical evaluation."""

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
class OptimizerProvenance:
    """Immutable diagnostics copied from one actual terminal optimizer invocation."""

    optimizer_kind: str | None
    converged: bool
    termination_message: str | None
    function_evaluations: int | None
    jacobian_evaluations: int | None
    iterations: int | None
    optimality: float | None
    final_cost: float
    correction_revision: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FinalClusterOutcome:
    """One immutable terminal result, associated only by ``cluster_id``."""

    cluster_id: int
    peak_names: tuple[str, ...]
    classification: FitOutcomeClassification
    correction_revision: int
    optimizer_provenance: OptimizerProvenance
    final_nonlinear_parameters: tuple[FinalParameter, ...]
    analytical_evaluation: FinalAnalyticalEvaluation | None
    unusable_reason: str | None

    @property
    def usable(self) -> bool:
        """Whether this outcome contributes completed scientific quantities."""
        return self.classification is not FitOutcomeClassification.UNUSABLE


@dataclass(frozen=True, slots=True)
class FinalFitStatistics:
    """Global quality statistics aggregated from usable terminal outcomes only."""

    chi_squared: float
    reduced_chi_squared: float
    n_observations: int
    n_fitted_parameters: int
    degrees_of_freedom: int
    aic: float | None
    bic: float | None
    log_likelihood: float | None
    function_evaluations: int | None


@dataclass(frozen=True, slots=True)
class FinalFitOutcome:
    """The sole authoritative completed scientific result of one fit run."""

    clusters: tuple[FinalClusterOutcome, ...]
    by_cluster_id: Mapping[int, FinalClusterOutcome]
    final_nonlinear_parameters: tuple[FinalParameter, ...]
    terminal_correction_revision: int
    noise: float
    n_optimizer_passes: int
    n_correction_updates: int
    overall_converged: bool
    statistics: FinalFitStatistics

    def cluster(self, cluster_id: int) -> FinalClusterOutcome:
        """Look up one terminal outcome without relying on presentation order."""
        return self.by_cluster_id[cluster_id]


def finalize_fit(pipeline_completion: PipelineCompletion) -> FinalFitOutcome:
    """Validate pipeline completion and freeze its single completed-fit outcome.

    ``pipeline_completion`` remains mutable orchestration state. This function is
    the only seam that turns its terminal results and ticket-03 evaluations into
    completed scientific truth.
    """
    clusters_by_id = _clusters_by_id(pipeline_completion.state.clusters)
    expected_ids = set(clusters_by_id)
    snapshot = pipeline_completion.correction_snapshot
    if snapshot is None:
        raise ValueError("Finalization requires a frozen terminal correction snapshot.")
    if snapshot.revision < 0:
        raise ValueError(
            f"Terminal correction revision must be non-negative, got {snapshot.revision}",
        )
    _require_exact_ids(
        expected_ids,
        set(snapshot.corrections),
        label="terminal correction snapshot",
    )

    noise = pipeline_completion.state.noise
    if noise is None or not np.isfinite(noise) or noise <= 0.0:
        raise ValueError(f"Finalization noise must be positive and finite, got {noise}")
    noise = float(noise)

    results_by_id = _by_cluster_id(pipeline_completion.results, label="terminal optimizer result")
    evaluations_by_id = _by_cluster_id(pipeline_completion.evaluations, label="terminal evaluation")
    _require_exact_ids(expected_ids, set(results_by_id), label="terminal optimizer result")
    _require_exact_ids(expected_ids, set(evaluations_by_id), label="terminal evaluation")

    final_parameters_by_name: dict[str, FinalParameter] = {}
    outcomes: list[FinalClusterOutcome] = []
    for cluster_id in sorted(expected_ids):
        cluster = clusters_by_id[cluster_id]
        result = results_by_id[cluster_id]
        evaluation = evaluations_by_id[cluster_id]
        _validate_terminal_pair(
            cluster=cluster,
            result=result,
            evaluation=evaluation,
            noise=noise,
            correction_revision=snapshot.revision,
        )
        if evaluation.usable:
            outcome, parameters = _usable_outcome(
                cluster=cluster,
                result=result,
                evaluation=evaluation,
                final_parameters=pipeline_completion.state.scalar_params,
                correction_revision=snapshot.revision,
            )
            for parameter in parameters:
                final_parameters_by_name[parameter.name] = parameter
        else:
            outcome = _unusable_outcome(
                cluster=cluster,
                result=result,
                evaluation=evaluation,
                correction_revision=snapshot.revision,
            )
        outcomes.append(outcome)

    frozen_outcomes = tuple(outcomes)
    by_cluster_id = MappingProxyType({outcome.cluster_id: outcome for outcome in frozen_outcomes})
    return FinalFitOutcome(
        clusters=frozen_outcomes,
        by_cluster_id=by_cluster_id,
        final_nonlinear_parameters=tuple(
            final_parameters_by_name[name] for name in sorted(final_parameters_by_name)
        ),
        terminal_correction_revision=snapshot.revision,
        noise=noise,
        n_optimizer_passes=pipeline_completion.n_optimizer_passes,
        n_correction_updates=pipeline_completion.n_correction_updates,
        overall_converged=all(
            outcome.classification is FitOutcomeClassification.CONVERGED
            for outcome in frozen_outcomes
        ),
        statistics=_global_statistics(frozen_outcomes),
    )


def _clusters_by_id(clusters: Sequence[Cluster]) -> dict[int, Cluster]:
    if not clusters:
        raise ValueError("Finalization requires at least one peak cluster.")
    by_id: dict[int, Cluster] = {}
    duplicates: set[int] = set()
    for cluster in clusters:
        if cluster.cluster_id in by_id:
            duplicates.add(cluster.cluster_id)
        by_id[cluster.cluster_id] = cluster
    if duplicates:
        raise ValueError(f"Duplicate expected cluster_id values: {sorted(duplicates)}")
    return by_id


def _by_cluster_id[T](values: Sequence[T], *, label: str) -> dict[int, T]:
    by_id: dict[int, T] = {}
    duplicates: set[int] = set()
    for value in values:
        cluster_id = value.cluster_id  # type: ignore[attr-defined]
        if cluster_id in by_id:
            duplicates.add(cluster_id)
        by_id[cluster_id] = value
    if duplicates:
        raise ValueError(f"Duplicate {label} cluster_id values: {sorted(duplicates)}")
    return by_id


def _require_exact_ids(expected: set[int], actual: set[int], *, label: str) -> None:
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unexpected or missing:
        details: list[str] = []
        if unexpected:
            details.append(f"Unexpected {label} cluster_id values: {unexpected}")
        if missing:
            details.append(f"Missing {label} cluster_id values: {missing}")
        raise ValueError("; ".join(details))


def _validate_terminal_pair(
    *,
    cluster: Cluster,
    result: FitResult,
    evaluation: FitEvaluation,
    noise: float,
    correction_revision: int,
) -> None:
    cluster_id = cluster.cluster_id
    if result.correction_revision != correction_revision:
        raise ValueError(
            "Terminal optimizer result has stale correction revision for "
            f"cluster_id {cluster_id}: expected {correction_revision}, "
            f"got {result.correction_revision}",
        )
    if result.noise is None or not np.isfinite(result.noise) or result.noise <= 0.0:
        raise ValueError(f"Terminal optimizer result has invalid noise for cluster_id {cluster_id}")
    if result.noise != noise:
        raise ValueError(
            "Terminal optimizer result noise does not match finalization noise for "
            f"cluster_id {cluster_id}: expected {noise}, got {result.noise}",
        )
    if evaluation.cluster_id != cluster_id:
        raise ValueError(
            f"Terminal evaluation cluster_id mismatch: expected {cluster_id}, "
            f"got {evaluation.cluster_id}",
        )
    if evaluation.usable:
        if evaluation.analytical is None:
            raise ValueError(
                f"Usable terminal evaluation is missing analytical values for {cluster_id}",
            )
        if evaluation.analytical.cluster_id != cluster_id:
            raise ValueError(
                f"Analytical evaluation cluster_id mismatch: expected {cluster_id}, "
                f"got {evaluation.analytical.cluster_id}",
            )
        if evaluation.unusable_reason is not None:
            raise ValueError(f"Usable terminal evaluation has an unusable reason for {cluster_id}")
        if evaluation.classification is FitOutcomeClassification.CONVERGED and not result.success:
            raise ValueError(
                "Converged terminal evaluation disagrees with optimizer for "
                f"cluster_id {cluster_id}",
            )
        if (
            evaluation.classification is FitOutcomeClassification.USABLE_NON_CONVERGED
            and result.success
        ):
            raise ValueError(
                "Usable non-converged evaluation disagrees with optimizer for "
                f"cluster_id {cluster_id}",
            )
    elif evaluation.analytical is not None or not evaluation.unusable_reason:
        raise ValueError(
            "Unusable terminal evaluation must not contain numerical values for "
            f"cluster_id {cluster_id}",
        )


def _usable_outcome(
    *,
    cluster: Cluster,
    result: FitResult,
    evaluation: FitEvaluation,
    final_parameters: Parameters,
    correction_revision: int,
) -> tuple[FinalClusterOutcome, tuple[FinalParameter, ...]]:
    assert evaluation.analytical is not None
    parameters = _final_cluster_parameters(cluster, result.params, final_parameters)
    return (
        FinalClusterOutcome(
            cluster_id=cluster.cluster_id,
            peak_names=tuple(peak.name for peak in cluster.peaks),
            classification=evaluation.classification,
            correction_revision=correction_revision,
            optimizer_provenance=_optimizer_provenance(result, correction_revision),
            final_nonlinear_parameters=parameters,
            analytical_evaluation=_freeze_evaluation(evaluation.analytical),
            unusable_reason=None,
        ),
        parameters,
    )


def _unusable_outcome(
    *,
    cluster: Cluster,
    result: FitResult,
    evaluation: FitEvaluation,
    correction_revision: int,
) -> FinalClusterOutcome:
    assert evaluation.unusable_reason is not None
    return FinalClusterOutcome(
        cluster_id=cluster.cluster_id,
        peak_names=tuple(peak.name for peak in cluster.peaks),
        classification=FitOutcomeClassification.UNUSABLE,
        correction_revision=correction_revision,
        optimizer_provenance=_optimizer_provenance(result, correction_revision),
        final_nonlinear_parameters=(),
        analytical_evaluation=None,
        unusable_reason=evaluation.unusable_reason,
    )


def _final_cluster_parameters(
    cluster: Cluster,
    terminal_parameters: Parameters,
    final_parameters: Parameters,
) -> tuple[FinalParameter, ...]:
    names = _cluster_nonlinear_parameter_names(cluster, terminal_parameters)
    frozen: list[FinalParameter] = []
    for name in names:
        if name not in final_parameters:
            raise ValueError(
                "Final merged nonlinear parameter is missing for "
                f"cluster_id {cluster.cluster_id}: {name}",
            )
        terminal = terminal_parameters[name]
        merged = final_parameters[name]
        if not np.isclose(terminal.value, merged.value, rtol=1e-12, atol=0.0):
            raise ValueError(
                "Final nonlinear parameter disagrees with terminal optimizer for "
                f"cluster_id {cluster.cluster_id}: {name}",
            )
        frozen.append(_freeze_parameter(merged))
    return tuple(sorted(frozen, key=lambda parameter: parameter.name))


def _cluster_nonlinear_parameter_names(
    cluster: Cluster,
    parameters: Parameters,
) -> list[str]:
    peak_names = {peak.name for peak in cluster.peaks}
    prefix = f"cluster_{cluster.cluster_id}."
    names: list[str] = []
    for name, parameter in parameters.items():
        identifier = parameter.param_id
        if identifier is not None:
            if identifier.label == "I":
                continue
            belongs_to_cluster = identifier.cluster_id == cluster.cluster_id
            belongs_to_peak = identifier.peak_name in peak_names
        else:
            if name.rsplit(".", maxsplit=1)[-1].startswith("I"):
                continue
            belongs_to_cluster = name.startswith(prefix)
            belongs_to_peak = any(name.startswith(f"{peak_name}.") for peak_name in peak_names)
        if belongs_to_cluster or belongs_to_peak:
            names.append(name)
    return names


def _freeze_parameter(parameter: Parameter) -> FinalParameter:
    return FinalParameter(
        name=parameter.name,
        value=float(parameter.value),
        min=float(parameter.min),
        max=float(parameter.max),
        vary=parameter.vary,
        unit=parameter.unit,
        standard_error=float(parameter.stderr),
    )


def _freeze_evaluation(evaluation: AnalyticalModelEvaluation) -> FinalAnalyticalEvaluation:
    return FinalAnalyticalEvaluation(
        cluster_id=evaluation.cluster_id,
        shapes=_freeze_array(evaluation.shapes),
        amplitudes=_freeze_array(evaluation.amplitudes),
        amplitude_standard_errors=_freeze_array(evaluation.amplitude_standard_errors),
        amplitude_covariance=_freeze_array(evaluation.amplitude_covariance),
        scaled_amplitude_standard_errors=_freeze_array(evaluation.scaled_amplitude_standard_errors),
        model_values=_freeze_array(evaluation.model_values),
        raw_residuals=_freeze_array(evaluation.raw_residuals),
        normalized_residuals=_freeze_array(evaluation.normalized_residuals),
        statistics=evaluation.statistics,
    )


def _freeze_array(values: FloatArray) -> FloatArray:
    frozen = np.array(values, copy=True)
    frozen.flags.writeable = False
    return frozen


def _optimizer_provenance(result: FitResult, correction_revision: int) -> OptimizerProvenance:
    iterations = result.metadata.get("global_iterations")
    has_varpro_diagnostics = result.optimizer_kind == "varpro"
    return OptimizerProvenance(
        optimizer_kind=result.optimizer_kind,
        converged=result.success,
        termination_message=result.message or None,
        function_evaluations=result.nfev if result.nfev > 0 else None,
        jacobian_evaluations=result.njev if has_varpro_diagnostics else None,
        iterations=iterations if isinstance(iterations, int) else None,
        optimality=result.optimality if has_varpro_diagnostics else None,
        final_cost=float(result.cost),
        correction_revision=correction_revision,
        metadata=_freeze_mapping(result.metadata),
    )


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_value(value) for key, value in values.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _freeze_array(value)
    if isinstance(value, dict):
        return _freeze_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return deepcopy(value)


def _global_statistics(outcomes: Sequence[FinalClusterOutcome]) -> FinalFitStatistics:
    usable = [outcome for outcome in outcomes if outcome.usable]
    statistics = [
        outcome.analytical_evaluation.statistics
        for outcome in usable
        if outcome.analytical_evaluation is not None
    ]
    total_chi_squared = float(sum(statistic.chi_squared for statistic in statistics))
    total_observations = sum(statistic.n_observations for statistic in statistics)
    total_parameters = sum(statistic.n_fitted_parameters for statistic in statistics)
    degrees_of_freedom = max(1, total_observations - total_parameters)
    log_likelihood = (
        float(sum(statistic.log_likelihood for statistic in statistics)) if statistics else None
    )
    aic = -2.0 * log_likelihood + 2.0 * total_parameters if log_likelihood is not None else None
    bic = (
        -2.0 * log_likelihood + total_parameters * float(np.log(total_observations))
        if log_likelihood is not None and total_observations > 0
        else None
    )
    evaluations = [outcome.optimizer_provenance.function_evaluations for outcome in usable]
    return FinalFitStatistics(
        chi_squared=total_chi_squared,
        reduced_chi_squared=compute_reduced_chi_squared(
            total_chi_squared,
            total_observations,
            total_parameters,
        ),
        n_observations=total_observations,
        n_fitted_parameters=total_parameters,
        degrees_of_freedom=degrees_of_freedom,
        aic=aic,
        bic=bic,
        log_likelihood=log_likelihood,
        function_evaluations=(
            sum(evaluation for evaluation in evaluations if evaluation is not None)
            if evaluations and all(evaluation is not None for evaluation in evaluations)
            else None
        ),
    )


__all__ = [
    "FinalAnalyticalEvaluation",
    "FinalClusterOutcome",
    "FinalFitOutcome",
    "FinalFitStatistics",
    "FinalParameter",
    "OptimizerProvenance",
    "finalize_fit",
]
