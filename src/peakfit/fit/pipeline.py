"""Direct fitting pipeline functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from peakfit.engine.algorithms.common import update_cluster_corrections
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
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.engine.results import FitResult


@dataclass(frozen=True)
class PipelineResult:
    """Aggregate output of a pipeline run."""

    state: FittingState
    results: list[FitResult]


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
    args: tuple[int, Cluster, Parameters, float, str, OptimizerConfig],
) -> tuple[int, FitResult]:
    """Execute fitting for a single cluster task."""
    task_idx, cluster, params, noise, optimizer, config = args
    result = fit_cluster_worker(cluster, params, noise, config, optimizer)
    return task_idx, result


def run_pipeline(
    config: FitConfig | PeakFitConfig,
    clusters: Sequence[Cluster],
    data_noise: float,
    base_params: Parameters,
    peaks: Sequence[Peak],
    spectra: Spectra,
    *,
    optimizer: str = "varpro",
    executor: Callable[[Callable[..., Any], list[Any]], Iterable[Any]] | None = None,
    progress_callback: Callable[[str, Any], None] | None = None,
) -> PipelineResult:
    """Execute fitting steps and return the final pipeline result."""
    final_result = None
    for item in run_pipeline_iter(
        config,
        clusters,
        data_noise,
        base_params,
        peaks,
        spectra,
        optimizer=optimizer,
        executor=executor,
        progress_callback=progress_callback,
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
    spectra: Spectra,
    *,
    optimizer: str = "varpro",
    executor: Callable[[Callable[..., Any], list[Any]], Iterable[Any]] | None = None,
    progress_callback: Callable[[str, Any], None] | None = None,
) -> Iterator[Any]:
    """Yield fit progress items and finally a PipelineResult."""
    fit_config = _normalize_config(config)
    steps = build_fit_steps(
        steps=fit_config.fitting.steps,
        refine_iterations=fit_config.fitting.refine_iterations,
    )
    mapper = executor or map
    final_params = base_params
    current_fit_results: list[FitResult] = []
    optimizer_config = _build_optimizer_config(fit_config, optimizer)

    for step_idx, step in enumerate(steps):
        step_msg = f"Step {step_idx + 1}/{len(steps)}: {step.name}"

        if progress_callback:
            progress_callback("step_start", step_msg)

        for iteration in range(step.iterations):
            if step.iterations > 1:
                iter_msg = f"Iteration {iteration + 1}/{step.iterations}"
                yield ("status", f"[bold blue]{iter_msg}[/]")
                if progress_callback:
                    progress_callback("iteration_start", iter_msg)

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
            yield from _process_execution_results(results_iter, step_results_map, progress_callback)

            current_fit_results = [step_results_map[i] for i in range(len(clusters))]
            for res in current_fit_results:
                final_params.update(res.params)

            if step.iterations > 1 and iteration < step.iterations - 1:
                yield ("status", "[dim]Correcting data with neighbors...[/]")

            update_cluster_corrections(final_params, clusters)

        if progress_callback:
            progress_callback("step_complete", step_idx)

    fit_params = FitParameters.from_parameters(final_params, list(peaks))
    state = FittingState(
        clusters=list(clusters),
        params=fit_params,
        scalar_params=final_params,
        noise=data_noise,
    )
    yield PipelineResult(state=state, results=current_fit_results)


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
) -> list[tuple[int, Cluster, Parameters, float, str, OptimizerConfig]]:
    """Prepare task arguments for cluster fitting."""
    tasks = []
    for idx, cluster in enumerate(clusters):
        cluster_params = Parameters.from_peaks(cluster.peaks, fixed=False)
        if config.parameters:
            cluster_params = apply_constraints(cluster_params, config.parameters)
        cluster_params = apply_step_constraints(cluster_params, step)

        for pid in cluster_params:
            if pid in current_params:
                cluster_params[pid].value = current_params[pid].value

        tasks.append((idx, cluster, cluster_params, data_noise, optimizer, optimizer_config))
    return tasks


def _process_execution_results(
    results_iter: Iterable[Any],
    results_map: dict[int, FitResult],
    progress_callback: Callable[[str, Any], None] | None,
) -> Iterator[FitResult]:
    """Yield fit results from an executor iterator."""
    for task_res in results_iter:
        task_idx, result = task_res
        results_map[task_idx] = result
        yield result

        if progress_callback:
            progress_callback(
                "cluster_end",
                {"idx": task_idx, "success": result.success, "result": result},
            )


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
