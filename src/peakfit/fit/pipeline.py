"""Direct fitting pipeline functions."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from peakfit.engine.algorithms.common import update_cluster_corrections
from peakfit.engine.algorithms.evaluation import (
    FitEvaluation,
    classify_optimizer_result,
)
from peakfit.engine.domain.config import (
    BasinHoppingConfig,
    FitConfig,
    OptimizerConfig,
    PeakFitConfig,
    VarProConfig,
)
from peakfit.engine.domain.constraints import apply_constraints
from peakfit.engine.domain.fit_steps import FitStep, apply_step_constraints, build_fit_steps
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.params_vector import FitParameters
from peakfit.engine.domain.state import FittingState
from peakfit.engine.fitting.optimizers import fit_with_optimizer
from peakfit.fit.final_outcome import FinalFitOutcome, finalize_fit
from peakfit.fit.simulation import FinalModelSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.peaks import Peak
    from peakfit.engine.results import FitResult
    from peakfit.shared.typing import FloatArray


@dataclass(frozen=True, slots=True)
class CorrectionSnapshot:
    """Read-only correction arrays used by a single optimizer pass."""

    revision: int
    corrections: Mapping[int, FloatArray]

    @classmethod
    def from_clusters(
        cls,
        clusters: Sequence[Cluster],
        revision: int,
    ) -> CorrectionSnapshot:
        """Copy corrections so later scheduling cannot alter this pass."""
        copied_corrections: dict[int, FloatArray] = {}
        for cluster in clusters:
            correction = cluster.corrections.copy()
            correction.flags.writeable = False
            copied_corrections[cluster.cluster_id] = correction
        return cls(revision=revision, corrections=MappingProxyType(copied_corrections))


@dataclass(frozen=True)
class PipelineCompletion:
    """Internal terminal attempts and state consumed once by finalization."""

    state: FittingState
    results: list[FitResult]
    evaluations: list[FitEvaluation] = field(default_factory=list)
    correction_snapshot: CorrectionSnapshot | None = None
    n_optimizer_passes: int = 0
    n_correction_updates: int = 0


@dataclass(frozen=True)
class PipelineResult:
    """Completed pipeline result with one scientific authority and continuation state."""

    final_outcome: FinalFitOutcome
    continuation_state: FittingState
    simulation_snapshot: FinalModelSnapshot


@dataclass(frozen=True, slots=True)
class ClusterFitTask:
    """Optimizer task associated with one run-local cluster identity."""

    cluster_id: int
    correction_revision: int
    cluster: Cluster
    params: Parameters
    noise: float
    optimizer: str
    config: OptimizerConfig

    def __post_init__(self) -> None:
        """Reject disagreement between the task key and cluster payload."""
        if self.cluster_id != self.cluster.cluster_id:
            msg = (
                "Cluster task cluster_id does not match its cluster: "
                f"expected {self.cluster_id}, got {self.cluster.cluster_id}"
            )
            raise ValueError(msg)
        if self.correction_revision < 0:
            msg = f"Correction revision must be non-negative, got {self.correction_revision}"
            raise ValueError(msg)


def fit_cluster_worker(
    cluster: Cluster,
    params: Parameters,
    noise: float,
    config: OptimizerConfig,
    optimizer: str = "varpro",
) -> FitResult:
    """Fit one cluster using the selected optimizer."""
    return fit_with_optimizer(optimizer, params, cluster, noise, config)


def fit_single_cluster_task(
    task: ClusterFitTask,
) -> FitResult:
    """Execute fitting for a single cluster task."""
    result = fit_cluster_worker(
        task.cluster,
        task.params,
        task.noise,
        task.config,
        task.optimizer,
    )
    if result.cluster_id != task.cluster_id:
        msg = (
            "Optimizer result cluster_id does not match submitted task: "
            f"expected {task.cluster_id}, got {result.cluster_id}"
        )
        raise ValueError(msg)
    if result.correction_revision is None:
        result.correction_revision = task.correction_revision
    elif result.correction_revision != task.correction_revision:
        msg = (
            "Optimizer result correction_revision does not match submitted task: "
            f"expected {task.correction_revision}, got {result.correction_revision}"
        )
        raise ValueError(msg)
    if result.optimizer_kind is None:
        result.optimizer_kind = task.optimizer
    elif result.optimizer_kind != task.optimizer:
        msg = (
            "Optimizer result optimizer_kind does not match submitted task: "
            f"expected {task.optimizer}, got {result.optimizer_kind}"
        )
        raise ValueError(msg)
    if result.noise is None:
        result.noise = task.noise
    elif result.noise != task.noise:
        msg = (
            "Optimizer result noise does not match submitted task: "
            f"expected {task.noise}, got {result.noise}"
        )
        raise ValueError(msg)
    return result


def run_pipeline(
    config: FitConfig | PeakFitConfig,
    clusters: Sequence[Cluster],
    data_noise: float,
    base_params: Parameters,
    peaks: Sequence[Peak],
    *,
    optimizer: str = "varpro",
    executor: Callable[[Callable[..., Any], list[Any]], Iterable[Any]] | None = None,
) -> PipelineResult:
    """Execute fitting steps and return the final pipeline result."""
    final_result = None
    for item in run_pipeline_iter(
        config,
        clusters,
        data_noise,
        base_params,
        peaks,
        optimizer=optimizer,
        executor=executor,
    ):
        if isinstance(item, PipelineResult):
            final_result = item

    if final_result is None:
        raise RuntimeError("Pipeline iterator did not return a final PipelineResult.")

    return final_result


def run_pipeline_iter(
    config: FitConfig | PeakFitConfig,
    clusters: Sequence[Cluster],
    data_noise: float,
    base_params: Parameters,
    peaks: Sequence[Peak],
    *,
    optimizer: str = "varpro",
    executor: Callable[[Callable[..., Any], list[Any]], Iterable[Any]] | None = None,
) -> Iterator[Any]:
    """Yield fit progress items and finally a PipelineResult."""
    cluster_ids = _validate_cluster_ids(clusters)
    fit_config = _normalize_config(config)
    steps = build_fit_steps(
        steps=fit_config.fitting.steps,
        refine_iterations=fit_config.fitting.refine_iterations,
    )
    passes = [(step, iteration) for step in steps for iteration in range(step.iterations)]
    if not passes:
        raise ValueError("Fitting requires at least one optimizer pass.")
    mapper = executor or map
    final_params = base_params
    current_fit_results: list[FitResult] = []
    current_evaluations: list[FitEvaluation] = []
    optimizer_config = _build_optimizer_config(fit_config, optimizer)
    correction_revision = 0
    correction_updates = 0
    terminal_snapshot: CorrectionSnapshot | None = None

    for pass_index, (step, iteration) in enumerate(passes):
        if step.iterations > 1:
            iter_msg = f"Iteration {iteration + 1}/{step.iterations}"
            yield ("status", f"[bold blue]{iter_msg}[/]")

        correction_snapshot = CorrectionSnapshot.from_clusters(clusters, correction_revision)
        terminal_snapshot = correction_snapshot
        tasks = _prepare_cluster_tasks(
            config=fit_config,
            clusters=clusters,
            correction_snapshot=correction_snapshot,
            current_params=final_params,
            step=step,
            data_noise=data_noise,
            optimizer=optimizer,
            optimizer_config=optimizer_config,
        )
        task_clusters_by_id = {task.cluster_id: task.cluster for task in tasks}
        step_results_map: dict[int, FitResult] = {}
        results_iter = mapper(fit_single_cluster_task, tasks)
        yield from _process_execution_results(
            results_iter,
            step_results_map,
            expected_cluster_ids=cluster_ids,
            correction_revision=correction_snapshot.revision,
            optimizer_kind=optimizer,
            noise=data_noise,
        )

        current_fit_results = [step_results_map[cluster_id] for cluster_id in cluster_ids]
        current_evaluations = [
            classify_optimizer_result(
                cluster=task_clusters_by_id[result.cluster_id],
                result=result,
                noise=data_noise,
            )
            for result in current_fit_results
        ]
        evaluations_by_id = {
            evaluation.cluster_id: evaluation for evaluation in current_evaluations
        }
        usable_cluster_ids = {
            evaluation.cluster_id for evaluation in current_evaluations if evaluation.usable
        }
        for result in current_fit_results:
            if evaluations_by_id[result.cluster_id].usable:
                final_params.update(result.params)

        if pass_index < len(passes) - 1:
            yield ("status", "[dim]Correcting data with neighbors...[/]")
            update_cluster_corrections(
                final_params,
                clusters,
                contributing_cluster_ids=usable_cluster_ids,
            )
            correction_updates += 1
            correction_revision += 1

    if terminal_snapshot is None:
        raise RuntimeError("Fitting did not create a terminal correction snapshot.")
    _validate_terminal_result_revisions(
        current_fit_results,
        current_evaluations,
        terminal_snapshot.revision,
    )

    fit_params = FitParameters.from_parameters(final_params, list(peaks))
    state = FittingState(
        clusters=sorted(clusters, key=lambda cluster: cluster.cluster_id),
        params=fit_params,
        scalar_params=final_params,
        noise=data_noise,
    )
    completion = PipelineCompletion(
        state=state,
        results=current_fit_results,
        evaluations=current_evaluations,
        correction_snapshot=terminal_snapshot,
        n_optimizer_passes=len(passes),
        n_correction_updates=correction_updates,
    )
    final_outcome = finalize_fit(completion)
    simulation_snapshot = FinalModelSnapshot.capture(
        final_outcome,
        completion.state,
        terminal_snapshot,
    )
    yield PipelineResult(
        final_outcome=final_outcome,
        continuation_state=completion.state,
        simulation_snapshot=simulation_snapshot,
    )


def _validate_cluster_ids(clusters: Sequence[Cluster]) -> tuple[int, ...]:
    """Require one unique run-local identifier for every input cluster."""
    cluster_ids: set[int] = set()
    duplicates: set[int] = set()
    for cluster in clusters:
        if cluster.cluster_id in cluster_ids:
            duplicates.add(cluster.cluster_id)
        cluster_ids.add(cluster.cluster_id)

    if duplicates:
        msg = f"Duplicate cluster_id values in pipeline input: {sorted(duplicates)}"
        raise ValueError(msg)
    return tuple(sorted(cluster_ids))


def _normalize_config(config: FitConfig | PeakFitConfig) -> PeakFitConfig:
    if isinstance(config, FitConfig):
        return PeakFitConfig(fitting=config)
    return config


def _prepare_cluster_tasks(
    *,
    config: PeakFitConfig,
    clusters: Sequence[Cluster],
    correction_snapshot: CorrectionSnapshot,
    current_params: Parameters,
    step: FitStep,
    data_noise: float,
    optimizer: str,
    optimizer_config: OptimizerConfig,
) -> list[ClusterFitTask]:
    """Prepare task arguments for cluster fitting."""
    tasks = []
    for cluster in clusters:
        task_cluster = copy(cluster)
        task_cluster.corrections = correction_snapshot.corrections[cluster.cluster_id]
        cluster_params = Parameters.from_peaks(cluster.peaks, fixed=False)
        if config.parameters:
            cluster_params = apply_constraints(cluster_params, config.parameters)
        cluster_params = apply_step_constraints(cluster_params, step)

        for pid in cluster_params:
            if pid in current_params:
                cluster_params[pid].value = current_params[pid].value

        tasks.append(
            ClusterFitTask(
                cluster_id=cluster.cluster_id,
                correction_revision=correction_snapshot.revision,
                cluster=task_cluster,
                params=cluster_params,
                noise=data_noise,
                optimizer=optimizer,
                config=optimizer_config,
            )
        )
    return tasks


def _process_execution_results(
    results_iter: Iterable[FitResult],
    results_map: dict[int, FitResult],
    *,
    expected_cluster_ids: Sequence[int],
    correction_revision: int,
    optimizer_kind: str,
    noise: float,
) -> Iterator[FitResult]:
    """Yield fit results from an executor iterator."""
    duplicates: set[int] = set()
    for result in results_iter:
        if result.correction_revision is None:
            result.correction_revision = correction_revision
        elif result.correction_revision != correction_revision:
            msg = (
                "Optimizer result correction_revision does not match pass snapshot: "
                f"expected {correction_revision}, got {result.correction_revision}"
            )
            raise ValueError(msg)
        if result.optimizer_kind is None:
            result.optimizer_kind = optimizer_kind
        elif result.optimizer_kind != optimizer_kind:
            msg = (
                "Optimizer result optimizer_kind does not match pass: "
                f"expected {optimizer_kind}, got {result.optimizer_kind}"
            )
            raise ValueError(msg)
        if result.noise is None:
            result.noise = noise
        elif result.noise != noise:
            msg = (
                f"Optimizer result noise does not match pass: expected {noise}, got {result.noise}"
            )
            raise ValueError(msg)
        if result.cluster_id in results_map:
            duplicates.add(result.cluster_id)
            continue
        results_map[result.cluster_id] = result
        yield result

    if duplicates:
        msg = f"Duplicate optimizer result cluster_id values: {sorted(duplicates)}"
        raise ValueError(msg)

    expected_ids = set(expected_cluster_ids)
    unexpected = sorted(results_map.keys() - expected_ids)
    if unexpected:
        msg = f"Unexpected optimizer result cluster_id values: {unexpected}"
        raise ValueError(msg)

    missing = sorted(expected_ids - results_map.keys())
    if missing:
        msg = f"Missing optimizer result cluster_id values: {missing}"
        raise ValueError(msg)


def _validate_terminal_result_revisions(
    results: Sequence[FitResult],
    evaluations: Sequence[FitEvaluation],
    terminal_revision: int,
) -> None:
    """Require every usable terminal result to name the frozen correction revision."""
    evaluations_by_id = {evaluation.cluster_id: evaluation for evaluation in evaluations}
    stale_usable_ids = sorted(
        result.cluster_id
        for result in results
        if evaluations_by_id[result.cluster_id].usable
        and result.correction_revision != terminal_revision
    )
    if stale_usable_ids:
        msg = (
            "Usable terminal optimizer results must reference correction revision "
            f"{terminal_revision}; stale cluster_id values: {stale_usable_ids}"
        )
        raise ValueError(msg)


def _build_optimizer_config(config: PeakFitConfig, optimizer: str) -> OptimizerConfig:
    if optimizer == "varpro":
        return VarProConfig(
            ftol=config.fitting.tolerance,
            xtol=config.fitting.tolerance,
            max_nfev=config.fitting.max_iterations,
        )
    if optimizer == "basin_hopping":
        return BasinHoppingConfig(seed=config.fitting.optimizer_seed)
    raise ValueError(f"Unknown optimizer: {optimizer}")


__all__ = [
    "CorrectionSnapshot",
    "PipelineCompletion",
    "PipelineResult",
    "fit_single_cluster_task",
    "run_pipeline",
    "run_pipeline_iter",
]
