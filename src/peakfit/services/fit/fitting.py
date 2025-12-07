"""Fitting execution logic for the fit pipeline.

This module handles the core fitting loop including parameter initialization,
cluster fitting, refinement iterations, progress tracking, and constraint application.

Supports both legacy single-pass fitting and modern multi-step protocols with
parameter constraints.
"""

from __future__ import annotations

import os
import time as time_module
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from threadpoolctl import threadpool_limits

from peakfit.core.domain.peaks import create_params
from peakfit.core.fitting.computation import update_cluster_corrections
from peakfit.core.fitting.constraints import ParameterConfig, apply_constraints
from peakfit.core.fitting.parameters import Parameters
from peakfit.core.fitting.protocol import (
    FitProtocol,
    FitStep,
    apply_step_constraints,
)
from peakfit.core.fitting.strategies import get_strategy
from peakfit.core.shared.constants import (
    BASIN_HOPPING_NITER,
    DIFF_EVOLUTION_MAXITER,
    LEAST_SQUARES_FTOL,
    LEAST_SQUARES_MAX_NFEV,
)
from peakfit.core.shared.events import Event, EventType, FitProgressEvent
from peakfit.core.shared.reporter import NullReporter, Reporter
from peakfit.ui import LiveClusterDisplay, console, log, log_section, subsection_header

if TYPE_CHECKING:
    from peakfit.core.domain.cluster import Cluster
    from peakfit.core.shared.events import EventDispatcher
    from peakfit.services.fit.pipeline import FitArguments


__all__ = [
    "fit_all_clusters",
    "fit_all_clusters_with_protocol",
]


# Default optimizer uses VarPro with analytical Jacobian for best performance
DEFAULT_OPTIMIZER = "varpro"


@dataclass(slots=True)
class ClusterFitResult:
    """Result from fitting a single cluster."""

    params: Parameters
    cost: float
    time: float
    success: bool
    n_evaluations: int
    message: str = "Converged"
    metadata: dict[str, Any] | None = None

    @property
    def converged(self) -> bool:
        """Alias for success."""
        return self.success


def fit_all_clusters(
    clargs: FitArguments,
    clusters: list[Cluster],
    protocol: FitProtocol,
    *,
    optimizer: str = DEFAULT_OPTIMIZER,
    optimizer_seed: int | None = None,
    max_iterations: int | None = None,
    tolerance: float | None = None,
    verbose: bool = False,
    dispatcher: EventDispatcher | None = None,
    parameter_config: ParameterConfig | None = None,
    workers: int = -1,
    reporter: Reporter | None = None,
    headless: bool = False,
) -> Parameters:
    """Fit all clusters with the requested optimization strategy.

    Args:
        clargs: Command line arguments (only used for noise level check now)
        clusters: List of clusters to fit
        protocol: Multi-step fitting protocol
        optimizer: Name of optimizer to use (default: "varpro" with analytical Jacobian)
        optimizer_seed: Seed for stochastic optimizers (basin-hopping, differential evolution)
        max_iterations: Iteration budget for optimizers
        tolerance: Convergence tolerance for optimizers
        verbose: Whether to show verbose output
        dispatcher: Optional event dispatcher for progress tracking
        parameter_config: Optional parameter constraints configuration
        workers: Number of parallel workers (-1 for all CPUs, 1 for sequential)

    Returns
    -------
        Fitted parameters for all clusters
    """
    if clargs.noise is None:
        msg = "Noise must be specified before fitting clusters"
        raise ValueError(msg)

    runner = FitRunner(
        clusters=clusters,
        protocol=protocol,
        optimizer=optimizer,
        noise=float(clargs.noise),
        optimizer_seed=optimizer_seed,
        max_nfev=max_iterations,
        tolerance=tolerance,
        parameter_config=parameter_config,
        verbose=verbose,
        dispatcher=dispatcher,
        workers=workers,
        headless=headless,
        reporter=reporter,
    )
    return runner.run()


def fit_all_clusters_with_protocol(
    clusters: list[Cluster],
    noise: float,
    protocol: FitProtocol,
    *,
    optimizer: str = DEFAULT_OPTIMIZER,
    optimizer_seed: int | None = None,
    max_iterations: int | None = None,
    tolerance: float | None = None,
    parameter_config: ParameterConfig | None = None,
    verbose: bool = False,
    dispatcher: EventDispatcher | None = None,
    workers: int = 1,
    headless: bool = False,
    reporter: Reporter | None = None,
) -> Parameters:
    """Fit clusters using a multi-step protocol.

    This is the modern API for fitting with full control over the process.

    Args:
        clusters: List of clusters to fit
        noise: Noise level
        protocol: Multi-step fitting protocol
        optimizer: Name of optimizer to use (default: "varpro" with analytical Jacobian)
        optimizer_seed: Seed for stochastic optimizers (basin-hopping, differential evolution)
        max_iterations: Iteration budget for optimizers
        tolerance: Convergence tolerance for optimizers
        parameter_config: Optional parameter constraints
        verbose: Whether to show verbose output
        dispatcher: Optional event dispatcher
        workers: Number of parallel workers (-1 for all CPUs, 1 for sequential)
        headless: Disable live display/UI elements during fitting

    Returns
    -------
        Fitted parameters for all clusters
    """
    runner = FitRunner(
        clusters=clusters,
        protocol=protocol,
        optimizer=optimizer,
        noise=noise,
        optimizer_seed=optimizer_seed,
        max_nfev=max_iterations,
        tolerance=tolerance,
        parameter_config=parameter_config,
        verbose=verbose,
        dispatcher=dispatcher,
        headless=headless,
        workers=workers,
        reporter=reporter,
    )
    return runner.run()


class FitRunner:
    """Orchestrates the fitting process for a set of clusters."""

    def __init__(
        self,
        clusters: list[Cluster],
        protocol: FitProtocol,
        noise: float,
        optimizer: str = DEFAULT_OPTIMIZER,
        optimizer_seed: int | None = None,
        max_nfev: int | None = None,
        tolerance: float | None = None,
        parameter_config: ParameterConfig | None = None,
        verbose: bool = False,
        dispatcher: EventDispatcher | None = None,
        headless: bool = False,
        workers: int = 1,
        reporter: Reporter | None = None,
    ) -> None:
        self.clusters = clusters
        self.protocol = protocol
        self.noise = noise
        self.optimizer = optimizer
        self.optimizer_seed = optimizer_seed
        self.max_nfev = max_nfev
        self.tolerance = tolerance
        self.parameter_config = parameter_config
        self.verbose = verbose
        self.dispatcher = dispatcher
        self.headless = headless
        self.workers = workers
        self.reporter = reporter or NullReporter()

        self.strategy = self._create_strategy()
        self.params_all = Parameters()
        self.total_iterations = sum(step.iterations for step in protocol.steps)
        self.global_iteration = 0

    def _create_strategy(self) -> Any:
        """Create optimization strategy with appropriate settings."""
        strategy_kwargs: dict[str, Any] = {}

        if self.optimizer in ("leastsq", "varpro"):
            tol = self.tolerance or LEAST_SQUARES_FTOL
            max_nfev = self.max_nfev or LEAST_SQUARES_MAX_NFEV
            strategy_kwargs = {
                "ftol": tol,
                "xtol": tol,
                "max_nfev": max_nfev,
                "verbose": 2 if self.verbose else 0,
            }

            if self.optimizer == "varpro":
                strategy_kwargs["gtol"] = tol

        if self.optimizer in ("basin-hopping", "basin_hopping"):
            bh_iterations = BASIN_HOPPING_NITER
            if self.max_nfev is not None:
                bh_iterations = min(self.max_nfev, BASIN_HOPPING_NITER)
            strategy_kwargs = {
                "n_iterations": bh_iterations,
                "seed": self.optimizer_seed,
            }

        if self.optimizer in ("differential-evolution", "differential_evolution"):
            de_iterations = DIFF_EVOLUTION_MAXITER
            if self.max_nfev is not None:
                de_iterations = min(self.max_nfev, DIFF_EVOLUTION_MAXITER)
            strategy_kwargs = {
                "max_iterations": de_iterations,
                "seed": self.optimizer_seed,
            }

        return get_strategy(self.optimizer, **strategy_kwargs)

    def run(self) -> Parameters:
        """Execute the fitting protocol."""
        use_parallel = self.workers != 1

        # Determine actual number of workers for logging
        actual_workers = (os.cpu_count() or 1) if self.workers == -1 else self.workers

        self.reporter.action(
            f"Fitting {len(self.clusters)} clusters across {self.total_iterations} iteration(s)"
        )

        with threadpool_limits(limits=1, user_api="blas"):
            for step_idx, step in enumerate(self.protocol.steps):
                self._log_step_header(step, step_idx, len(self.protocol.steps))

                for iteration in range(step.iterations):
                    self.global_iteration += 1

                    ctx = _IterationContext(
                        step=step,
                        step_idx=step_idx,
                        step_name=step.name or f"Step {step_idx + 1}",
                        iteration=iteration,
                        global_iteration=self.global_iteration,
                        total_iterations=self.total_iterations,
                        is_first=(iteration == 0 and step_idx == 0),
                    )

                    self._log_iteration_header(ctx)

                    if use_parallel:
                        self._run_parallel(ctx, actual_workers)
                    else:
                        self._run_sequential(ctx)

        self.reporter.success("Cluster fitting complete")

        return self.params_all

    def _run_sequential(self, ctx: _IterationContext) -> None:
        """Run iteration sequentially with live display."""
        log(f"Sequential fitting: {len(self.clusters)} clusters")
        params_init = {key: self.params_all[key].value for key in self.params_all}

        if self.headless:
            for cluster_idx, cluster in enumerate(self.clusters, 1):
                result = self._fit_and_report_cluster(
                    cluster=cluster,
                    cluster_idx=cluster_idx,
                    step=ctx.step,
                    params_init=params_init,
                    use_live_display=False,
                )
                self.params_all.update(result.params)
                self.reporter.info(
                    f"Cluster {cluster_idx}/{len(self.clusters)} cost={result.cost:.2e}"
                )
        else:
            display = LiveClusterDisplay.from_clusters(self.clusters)
            display.set_step_name(ctx.step_name)
            display.set_workers(1)

            with display:
                for cluster_idx, cluster in enumerate(self.clusters, 1):
                    display.mark_running(cluster_idx)

                    result = self._fit_and_report_cluster(
                        cluster=cluster,
                        cluster_idx=cluster_idx,
                        step=ctx.step,
                        params_init=params_init,
                        use_live_display=True,
                    )

                    display.mark_completed(
                        cluster_idx,
                        cost=result.cost,
                        n_evaluations=result.n_evaluations,
                        time_sec=result.time,
                        success=result.success,
                        message=result.message if not result.success else None,
                    )

                    self.params_all.update(result.params)

    def _run_parallel(self, ctx: _IterationContext, actual_workers: int) -> None:
        """Run iteration using joblib parallelism."""
        from joblib import Parallel, delayed

        log(f"Parallel fitting: {len(self.clusters)} clusters with {actual_workers} workers")
        params_init = {key: self.params_all[key].value for key in self.params_all}

        if self.headless:
            parallel_results = Parallel(n_jobs=self.workers, backend="loky")(
                delayed(_fit_single_cluster)(
                    cluster=cluster,
                    step=ctx.step,
                    strategy=self.strategy,
                    noise_value=self.noise,
                    parameter_config=self.parameter_config,
                    params_init=params_init,
                )
                for cluster in self.clusters
            )

            results = [r for r in parallel_results if isinstance(r, ClusterFitResult)]
        else:
            display = LiveClusterDisplay.from_clusters(self.clusters)
            display.set_step_name(ctx.step_name)
            display.set_workers(actual_workers)

            with display:
                for idx in range(1, len(self.clusters) + 1):
                    display.mark_running(idx)

                parallel_results = Parallel(n_jobs=self.workers, backend="loky")(
                    delayed(_fit_single_cluster)(
                        cluster=cluster,
                        step=ctx.step,
                        strategy=self.strategy,
                        noise_value=self.noise,
                        parameter_config=self.parameter_config,
                        params_init=params_init,
                    )
                    for cluster in self.clusters
                )

                # Filter valid results
                results = [r for r in parallel_results if isinstance(r, ClusterFitResult)]

                for cluster_idx, result in enumerate(results, 1):
                    display.mark_completed(
                        cluster_idx,
                        cost=result.cost,
                        n_evaluations=result.n_evaluations,
                        time_sec=result.time,
                        success=result.success,
                        message=result.message if not result.success else None,
                    )

        # Collect results and update parameters
        self._collect_results(results)

    def _collect_results(self, results: list[ClusterFitResult]) -> None:
        """Update parameters and dispatch completion events for batch results."""
        total_time = 0.0
        successful = 0
        cluster_count = len(self.clusters)

        for cluster_idx, (cluster, result) in enumerate(
            zip(self.clusters, results, strict=True), 1
        ):
            total_time += result.time
            if result.success:
                successful += 1

            self._dispatch_cluster_completed(
                cluster_idx=cluster_idx,
                cluster_count=cluster_count,
                cost=result.cost,
                success=result.success,
                cluster_time=result.time,
                metadata=result.metadata,
            )

            # Log specific details (since live display is gone)
            peak_names = [p.name for p in cluster.peaks]
            log(
                f"Cluster {cluster_idx}: {', '.join(peak_names)} - "
                f"cost={result.cost:.2e}, iter={result.n_evaluations}, time={result.time:.2f}s"
            )

            self.params_all.update(result.params)

        log(
            f"Completed {cluster_count} clusters ({successful} converged, CPU time: {total_time:.1f}s)"
        )

    def _fit_and_report_cluster(
        self,
        cluster: Cluster,
        cluster_idx: int,
        step: FitStep,
        params_init: dict[str, float],
        use_live_display: bool = True,
    ) -> ClusterFitResult:
        """Fit a single cluster, handle logging and event dispatching."""
        peak_names = [peak.name for peak in cluster.peaks]
        peaks_str = ", ".join(peak_names)
        cluster_count = len(self.clusters)

        if self.verbose and not use_live_display:
            _print_cluster_header(cluster_idx, cluster_count, peaks_str, len(peak_names))

        log("")
        log(f"Cluster {cluster_idx}/{cluster_count}: {peaks_str}")
        log(f"  - Peaks: {len(cluster.peaks)}")

        self._dispatch_cluster_started(cluster_idx, cluster_count, peak_names)

        result = _fit_single_cluster(
            cluster=cluster,
            step=step,
            strategy=self.strategy,
            noise_value=self.noise,
            parameter_config=self.parameter_config,
            params_init=params_init,
        )

        self._dispatch_cluster_completed(
            cluster_idx, cluster_count, result.cost, result.success, result.time, result.metadata
        )

        if self.verbose and not use_live_display:
            _print_cluster_result(result)
        _log_cluster_result(result)

        return result

    def _log_step_header(self, step: FitStep, step_idx: int, total_steps: int) -> None:
        """Log step header information."""
        if not self.verbose:
            return

        step_name = step.name or f"Step {step_idx + 1}"
        console.print()
        console.print(
            f"[header]═══ Protocol Step {step_idx + 1}/{total_steps}: {step_name} ═══[/header]"
        )

        if step.description:
            console.print(f"  [dim]{step.description}[/dim]")
        if step.fix:
            console.print(f"  [warning]Fix:[/warning] {', '.join(step.fix)}")
        if step.vary:
            console.print(f"  [success]Vary:[/success] {', '.join(step.vary)}")

        console.print(f"  [key]Iterations:[/key] {step.iterations}")
        console.print()

    def _log_iteration_header(self, ctx: _IterationContext) -> None:
        """Log iteration header and apply updates."""
        use_label = ctx.step.name and ctx.step.name.lower() != "default"

        if ctx.is_first:
            subsection_header("Initial Fit" if not use_label else ctx.label)
        else:
            subsection_header(
                f"Refining Parameters: {ctx.label}" if use_label else "Refining Parameters"
            )
            log_section(f"Protocol: {ctx.label}")
            update_cluster_corrections(self.params_all, self.clusters)

    def _dispatch_cluster_started(
        self, cluster_idx: int, cluster_count: int, peak_names: list[str]
    ) -> None:
        if self.dispatcher is None:
            return

        self.dispatcher.dispatch(
            Event(
                event_type=EventType.CLUSTER_STARTED,
                data={
                    "cluster_index": cluster_idx,
                    "total_clusters": cluster_count,
                    "iteration": self.global_iteration,
                    "total_iterations": self.total_iterations,
                    "peak_names": peak_names,
                },
            )
        )

    def _dispatch_cluster_completed(
        self,
        cluster_idx: int,
        cluster_count: int,
        cost: float,
        success: bool,
        cluster_time: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.dispatcher is None:
            return

        self.dispatcher.dispatch(
            Event(
                event_type=EventType.CLUSTER_COMPLETED,
                data={
                    "cluster_index": cluster_idx,
                    "total_clusters": cluster_count,
                    "iteration": self.global_iteration,
                    "total_iterations": self.total_iterations,
                    "cost": cost,
                    "success": success,
                    "time_sec": cluster_time,
                    "metadata": metadata,
                },
            )
        )
        self.dispatcher.dispatch(
            FitProgressEvent(
                event_type=EventType.FIT_PROGRESS,
                data={
                    "cluster_index": cluster_idx,
                    "iteration": self.global_iteration,
                    "cost": cost,
                    "success": success,
                    "metadata": metadata,
                },
                current_cluster=cluster_idx,
                total_clusters=cluster_count,
                current_iteration=self.global_iteration,
                total_iterations=self.total_iterations,
            )
        )


@dataclass
class _IterationContext:
    """Context for a single fitting iteration."""

    step: FitStep
    step_idx: int
    step_name: str
    iteration: int
    global_iteration: int
    total_iterations: int
    is_first: bool

    @property
    def label(self) -> str:
        """Get human-readable iteration label."""
        if self.step.iterations > 1:
            return f"{self.step_name} (Iteration {self.iteration + 1}/{self.step.iterations})"
        return self.step_name


def _fit_single_cluster(
    cluster: Cluster,
    step: FitStep,
    strategy: Any,
    noise_value: float,
    parameter_config: ParameterConfig | None,
    params_init: dict[str, float],
) -> ClusterFitResult:
    """Fit a single cluster (used by both sequential and parallel execution)."""
    cluster_start = time_module.time()

    # Create and configure parameters
    params = create_params(cluster.peaks, fixed=False)
    if parameter_config is not None:
        params = apply_constraints(params, parameter_config)
    params = apply_step_constraints(params, step)

    # Initialize from previous iterations
    for key in params:
        if key in params_init:
            params[key].value = params_init[key]

    # Run optimization
    result = strategy.optimize(params, cluster, noise_value)
    cluster_time = time_module.time() - cluster_start

    n_evals = int(result.n_evaluations) if hasattr(result, "n_evaluations") else 0
    message = getattr(result, "message", "Converged" if result.success else "Did not converge")

    return ClusterFitResult(
        params=result.params,
        cost=result.cost,
        time=cluster_time,
        success=result.success,
        n_evaluations=n_evals,
        message=message,
        metadata=getattr(result, "metadata", None),
    )


def _print_cluster_header(
    cluster_idx: int,
    cluster_count: int,
    peaks_str: str,
    n_peaks: int,
) -> None:
    """Print cluster header."""
    if cluster_idx > 1:
        console.print()
    console.print(
        f"[header]Cluster {cluster_idx}/{cluster_count}[/header] [dim]│[/dim] "
        f"{peaks_str} [dim][{n_peaks} peak{'s' if n_peaks != 1 else ''}][/dim]"
    )


def _print_cluster_result(result: ClusterFitResult) -> None:
    """Print cluster fitting result."""
    if result.success:
        console.print(
            f"[success]✓[/success] Converged [dim]│[/dim] "
            f"χ² = [metric]{result.cost:.2e}[/metric] [dim]│[/dim] "
            f"{result.n_evaluations} evaluations [dim]│[/dim] "
            f"{result.time:.1f}s"
        )
    else:
        console.print(
            f"[warning]⚠[/warning] {result.message} [dim]│[/dim] "
            f"χ² = [metric]{result.cost:.2e}[/metric] [dim]│[/dim] "
            f"{result.n_evaluations} evaluations [dim]│[/dim] "
            f"{result.time:.1f}s"
        )


def _log_cluster_result(result: ClusterFitResult) -> None:
    """Log cluster fitting result."""
    if result.success:
        log("  - Status: Converged", level="info")
    else:
        log(f"  - Status: {result.message}", level="warning")

    log(f"  - Cost: {result.cost:.3e}")
    log(f"  - Function evaluations: {result.n_evaluations}")
    log(f"  - Time: {result.time:.1f}s")
    if result.metadata:
        _log_metadata(result.metadata)


def _log_metadata(metadata: dict[str, Any]) -> None:
    """Log supplemental optimizer metadata."""
    numeric_fields = {
        "initial_cost": "Initial cost",
        "final_cost": "Final cost",
        "wall_time_sec": "Wall time (s)",
    }
    for key, label in numeric_fields.items():
        if key in metadata:
            value = metadata[key]
            if isinstance(value, (int, float)):
                display = f"{value:.3e}" if isinstance(value, float) else f"{value}"
                log(f"  - {label}: {display}")

    for key in ("global_iterations", "local_minimizations", "polished", "seed"):
        if key in metadata:
            log(f"  - {key.replace('_', ' ').title()}: {metadata[key]}")
