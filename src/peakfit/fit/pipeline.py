"""Fitting pipeline orchestration (fit slice).

Encapsulates the execution of fitting protocols, including parallel execution,
parameter synchronization, and cluster corrections. The FitPipeline can be
invoked from the fit workflow without depending on IO or UI layers.
"""

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
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.params_vector import FitParameters
from peakfit.engine.domain.protocol import (
    apply_step_constraints,
    build_fit_steps,
)
from peakfit.engine.domain.state import FittingState
from peakfit.engine.fitting.optimizers import fit_with_optimizer
from peakfit.shared.reporter import NullReporter, Reporter

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.peaks import Peak
    from peakfit.engine.domain.protocol import FitStep
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
    """Execute fitting for a single cluster task.

    Args:
        args: Tuple containing (task_idx, cluster, params, noise, optimizer, config)

    Returns:
    -------
        Tuple of (task_idx, fit_result)
    """
    (
        task_idx,
        cluster,
        params,
        noise,
        optimizer,
        config,
    ) = args

    result = fit_cluster_worker(cluster, params, noise, config, optimizer)
    return task_idx, result


class FitPipeline:
    """Core fitting pipeline independent of IO/UI concerns."""

    def __init__(self, config: FitConfig | PeakFitConfig, reporter: Reporter | None = None) -> None:
        if isinstance(config, FitConfig):
            self._config: PeakFitConfig = PeakFitConfig(fitting=config)
        else:
            self._config = config
        self._reporter = reporter or NullReporter()

    def run(
        self,
        clusters: Sequence[Cluster],
        data_noise: float,
        base_params: Parameters,
        peaks: Sequence[Peak],
        spectra: Spectra,
        *,
        optimizer: str = "varpro",
        executor: Callable[[Callable[..., Any], list[Any]], Any] | None = None,
        progress_callback: Callable[[str, Any], None] | None = None,
    ) -> PipelineResult:
        """Execute the fitting protocol across all clusters (Blocking Wrapper)."""
        # Collect all results
        iterator = self.run_iter(
            clusters,
            data_noise,
            base_params,
            peaks,
            spectra,
            optimizer=optimizer,
            executor=executor,
            progress_callback=progress_callback,
        )

        final_result = None
        for item in iterator:
            if isinstance(item, PipelineResult):
                final_result = item

        if final_result is None:
            raise RuntimeError("Pipeline iterator did not return a final PipelineResult.")

        return final_result

    def run_iter(
        self,
        clusters: Sequence[Cluster],
        data_noise: float,
        base_params: Parameters,
        peaks: Sequence[Peak],
        spectra: Spectra,
        *,
        optimizer: str = "varpro",
        executor: Callable[[Callable[..., Any], list[Any]], Any] | None = None,
        progress_callback: Callable[[str, Any], None] | None = None,
    ) -> Any:
        """Execute the fitting protocol as a generator.

        Yields:
        ------
             FitResult: incrementally as tasks complete.
             PipelineResult: once at the very end.
        """
        steps = self._get_steps(self._config)

        # Default to serial execution if no executor provided
        mapper = executor or map

        final_params = base_params
        current_fit_results: list[FitResult] = []

        optimizer_config = self._build_optimizer_config(optimizer)

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

                # Prepare tasks
                tasks = self._prepare_cluster_tasks(
                    clusters, final_params, step, data_noise, optimizer, optimizer_config
                )

                step_results_map: dict[int, FitResult] = {}

                # Execute tasks and yield results
                results_iter = mapper(fit_single_cluster_task, tasks)

                # Process execution results
                yield from self._process_execution_results(
                    results_iter, step_results_map, progress_callback
                )

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

        # Yield the final result object at the end
        yield PipelineResult(state=state, results=current_fit_results)

    def _prepare_cluster_tasks(
        self,
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
            if self._config.parameters:
                cluster_params = apply_constraints(cluster_params, self._config.parameters)
            cluster_params = apply_step_constraints(cluster_params, step)

            for pid in cluster_params:
                if pid in current_params:
                    cluster_params[pid].value = current_params[pid].value

            tasks.append((idx, cluster, cluster_params, data_noise, optimizer, optimizer_config))
        return tasks

    def _process_execution_results(
        self,
        results_iter: Any,
        results_map: dict[int, FitResult],
        progress_callback: Callable[[str, Any], None] | None,
    ) -> Any:
        """Process and yield results from executor iterator."""
        for task_res in results_iter:
            task_idx, result = task_res
            results_map[task_idx] = result

            # Yield result for real-time UI updates
            yield result

            if progress_callback:
                progress_callback(
                    "cluster_end",
                    {"idx": task_idx, "success": result.success, "result": result},
                )

    def _get_steps(self, config: PeakFitConfig) -> list[FitStep]:
        return build_fit_steps(
            steps=config.fitting.steps,
            refine_iterations=config.fitting.refine_iterations,
        )

    def _build_optimizer_config(self, optimizer: str) -> OptimizerConfig:
        if optimizer == "varpro":
            return VarProConfig(
                ftol=self._config.fitting.tolerance,
                xtol=self._config.fitting.tolerance,
                max_nfev=self._config.fitting.max_iterations,
            )
        if optimizer == "basin_hopping":
            return BasinHoppingConfig(seed=self._config.fitting.optimizer_seed)
        raise ValueError(f"Unknown optimizer: {optimizer}")


__all__ = ["FitPipeline", "PipelineResult", "fit_single_cluster_task"]
