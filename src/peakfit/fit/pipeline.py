"""Direct fitting pipeline functions."""

from __future__ import annotations

from dataclasses import dataclass, field
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

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.peaks import Peak
    from peakfit.engine.results import FitResult


@dataclass(frozen=True)
class PipelineResult:
    """Aggregate output of a pipeline run."""

    state: FittingState
    results: list[FitResult]
    evaluations: list[FitEvaluation] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ClusterFitTask:
    """Optimizer task associated with one run-local cluster identity."""

    cluster_id: int
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
    mapper = executor or map
    final_params = base_params
    current_fit_results: list[FitResult] = []
    current_evaluations: list[FitEvaluation] = []
    optimizer_config = _build_optimizer_config(fit_config, optimizer)
    clusters_by_id = {cluster.cluster_id: cluster for cluster in clusters}

    for step in steps:
        for iteration in range(step.iterations):
            if step.iterations > 1:
                iter_msg = f"Iteration {iteration + 1}/{step.iterations}"
                yield ("status", f"[bold blue]{iter_msg}[/]")

            tasks = _prepare_cluster_tasks(
                config=fit_config,
                clusters=clusters,
                current_params=final_params,
                step=step,
                data_noise=data_noise,
                optimizer=optimizer,
                optimizer_config=optimizer_config,
            )
            step_results_map: dict[int, FitResult] = {}
            results_iter = mapper(fit_single_cluster_task, tasks)
            yield from _process_execution_results(
                results_iter,
                step_results_map,
                expected_cluster_ids=cluster_ids,
            )

            current_fit_results = [step_results_map[cluster_id] for cluster_id in cluster_ids]
            current_evaluations = [
                classify_optimizer_result(
                    cluster=clusters_by_id[result.cluster_id],
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

            if step.iterations > 1 and iteration < step.iterations - 1:
                yield ("status", "[dim]Correcting data with neighbors...[/]")

            update_cluster_corrections(
                final_params,
                clusters,
                contributing_cluster_ids=usable_cluster_ids,
            )

    fit_params = FitParameters.from_parameters(final_params, list(peaks))
    state = FittingState(
        clusters=sorted(clusters, key=lambda cluster: cluster.cluster_id),
        params=fit_params,
        scalar_params=final_params,
        noise=data_noise,
    )
    yield PipelineResult(
        state=state,
        results=current_fit_results,
        evaluations=current_evaluations,
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
    current_params: Parameters,
    step: FitStep,
    data_noise: float,
    optimizer: str,
    optimizer_config: OptimizerConfig,
) -> list[ClusterFitTask]:
    """Prepare task arguments for cluster fitting."""
    tasks = []
    for cluster in clusters:
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
                cluster=cluster,
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
) -> Iterator[FitResult]:
    """Yield fit results from an executor iterator."""
    duplicates: set[int] = set()
    for result in results_iter:
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
    "PipelineResult",
    "fit_single_cluster_task",
    "run_pipeline",
    "run_pipeline_iter",
]
