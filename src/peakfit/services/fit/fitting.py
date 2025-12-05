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
    create_protocol_from_config,
)
from peakfit.core.fitting.strategies import get_strategy
from peakfit.core.shared.constants import (
    LEAST_SQUARES_FTOL,
    LEAST_SQUARES_MAX_NFEV,
    LEAST_SQUARES_XTOL,
)
from peakfit.core.shared.events import Event, EventType, FitProgressEvent
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

    @property
    def converged(self) -> bool:
        """Alias for success."""
        return self.success


def fit_all_clusters(
    clargs: FitArguments,
    clusters: list[Cluster],
    *,
    optimizer: str = DEFAULT_OPTIMIZER,
    verbose: bool = False,
    dispatcher: EventDispatcher | None = None,
    parameter_config: ParameterConfig | None = None,
    protocol: FitProtocol | None = None,
    workers: int = -1,
) -> Parameters:
    """Fit all clusters with the requested optimization strategy.

    Supports both legacy mode (using refine_nb) and modern protocol mode.

    Args:
        clargs: Command line arguments
        clusters: List of clusters to fit
        optimizer: Name of optimizer to use (default: "varpro" with analytical Jacobian)
        verbose: Whether to show verbose output
        dispatcher: Optional event dispatcher for progress tracking
        parameter_config: Optional parameter constraints configuration
        protocol: Optional multi-step fitting protocol
        workers: Number of parallel workers (-1 for all CPUs, 1 for sequential)

    Returns
    -------
        Fitted parameters for all clusters
    """
    if clargs.noise is None:
        msg = "Noise must be specified before fitting clusters"
        raise ValueError(msg)

    strategy = _create_strategy(optimizer, verbose)

    # Use protocol if provided, otherwise create from legacy options
    if protocol is None:
        protocol = create_protocol_from_config(
            steps=None,
            refine_iterations=clargs.refine_nb,
            fixed=clargs.fixed,
        )

    return _fit_with_protocol(
        clusters=clusters,
        protocol=protocol,
        strategy=strategy,
        noise_value=float(clargs.noise),
        parameter_config=parameter_config,
        verbose=verbose,
        dispatcher=dispatcher,
        workers=workers,
    )


def fit_all_clusters_with_protocol(
    clusters: list[Cluster],
    noise: float,
    protocol: FitProtocol,
    *,
    optimizer: str = DEFAULT_OPTIMIZER,
    parameter_config: ParameterConfig | None = None,
    verbose: bool = False,
    dispatcher: EventDispatcher | None = None,
    workers: int = 1,
) -> Parameters:
    """Fit clusters using a multi-step protocol.

    This is the modern API for fitting with full control over the process.

    Args:
        clusters: List of clusters to fit
        noise: Noise level
        protocol: Multi-step fitting protocol
        optimizer: Name of optimizer to use (default: "varpro" with analytical Jacobian)
        parameter_config: Optional parameter constraints
        verbose: Whether to show verbose output
        dispatcher: Optional event dispatcher
        workers: Number of parallel workers (-1 for all CPUs, 1 for sequential)

    Returns
    -------
        Fitted parameters for all clusters
    """
    strategy = _create_strategy(optimizer, verbose)

    return _fit_with_protocol(
        clusters=clusters,
        protocol=protocol,
        strategy=strategy,
        noise_value=noise,
        parameter_config=parameter_config,
        verbose=verbose,
        dispatcher=dispatcher,
        workers=workers,
    )


def _create_strategy(optimizer: str, verbose: bool) -> Any:
    """Create optimization strategy with appropriate settings."""
    strategy_kwargs: dict[str, Any] = {}

    if optimizer in ("leastsq", "varpro"):
        strategy_kwargs = {
            "ftol": LEAST_SQUARES_FTOL,
            "xtol": LEAST_SQUARES_XTOL,
            "max_nfev": LEAST_SQUARES_MAX_NFEV,
            "verbose": 2 if verbose else 0,
        }

    return get_strategy(optimizer, **strategy_kwargs)


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


def _fit_with_protocol(
    clusters: list[Cluster],
    protocol: FitProtocol,
    strategy: Any,
    noise_value: float,
    parameter_config: ParameterConfig | None,
    verbose: bool,
    dispatcher: EventDispatcher | None,
    workers: int = 1,
) -> Parameters:
    """Run protocol-based fitting internally."""
    params_all = Parameters()
    cluster_count = len(clusters)
    total_iterations = sum(step.iterations for step in protocol.steps)
    use_parallel = workers != 1

    with threadpool_limits(limits=1, user_api="blas"):
        global_iteration = 0

        for step_idx, step in enumerate(protocol.steps):
            step_name = step.name or f"Step {step_idx + 1}"

            if verbose:
                _log_step_header(step, step_idx, len(protocol.steps))

            for iteration in range(step.iterations):
                global_iteration += 1
                ctx = _IterationContext(
                    step=step,
                    step_idx=step_idx,
                    step_name=step_name,
                    iteration=iteration,
                    global_iteration=global_iteration,
                    total_iterations=total_iterations,
                    is_first=(iteration == 0 and step_idx == 0),
                )

                _log_iteration_header(ctx, params_all, clusters)
                _run_iteration(
                    ctx,
                    clusters,
                    params_all,
                    strategy,
                    noise_value,
                    parameter_config,
                    cluster_count,
                    dispatcher,
                    workers,
                    use_parallel,
                    verbose,
                )

    return params_all


def _log_iteration_header(
    ctx: _IterationContext, params_all: Parameters, clusters: list[Cluster]
) -> None:
    """Log the header for a fitting iteration."""
    # Don't print "default" step name - use descriptive labels instead
    use_label = ctx.step.name and ctx.step.name.lower() != "default"

    if ctx.is_first:
        subsection_header("Initial Fit" if not use_label else ctx.label)
    else:
        subsection_header(
            f"Refining Parameters: {ctx.label}" if use_label else "Refining Parameters"
        )
        log_section(f"Protocol: {ctx.label}")
        update_cluster_corrections(params_all, clusters)


def _run_iteration(
    ctx: _IterationContext,
    clusters: list[Cluster],
    params_all: Parameters,
    strategy: Any,
    noise_value: float,
    parameter_config: ParameterConfig | None,
    cluster_count: int,
    dispatcher: EventDispatcher | None,
    workers: int,
    use_parallel: bool,
    verbose: bool,
) -> None:
    """Run a single fitting iteration (parallel or sequential)."""
    common_args = {
        "clusters": clusters,
        "params_all": params_all,
        "step": ctx.step,
        "strategy": strategy,
        "noise_value": noise_value,
        "parameter_config": parameter_config,
        "cluster_count": cluster_count,
        "iteration_idx": ctx.global_iteration,
        "total_iterations": ctx.total_iterations,
        "dispatcher": dispatcher,
    }

    if use_parallel:
        _fit_iteration_parallel(**common_args, workers=workers)
    else:
        _fit_iteration_sequential(**common_args, verbose=verbose)


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
    )


def _fit_iteration_sequential(
    clusters: list[Cluster],
    params_all: Parameters,
    step: FitStep,
    strategy: Any,
    noise_value: float,
    parameter_config: ParameterConfig | None,
    cluster_count: int,
    iteration_idx: int,
    total_iterations: int,
    dispatcher: EventDispatcher | None,
    verbose: bool = False,
) -> None:
    """Fit all clusters sequentially in a single iteration with live display."""
    params_init = {key: params_all[key].value for key in params_all}

    log(f"Sequential fitting: {cluster_count} clusters")

    # Create the live display
    display = LiveClusterDisplay.from_clusters(clusters)
    display.set_step_name(step.name or "Fitting")
    display.set_workers(1)

    with display:
        for cluster_idx, cluster in enumerate(clusters, 1):
            # Mark as running
            display.mark_running(cluster_idx)

            result = _fit_and_report_cluster(
                cluster=cluster,
                cluster_idx=cluster_idx,
                cluster_count=cluster_count,
                step=step,
                strategy=strategy,
                noise_value=noise_value,
                parameter_config=parameter_config,
                params_init=params_init,
                iteration_idx=iteration_idx,
                total_iterations=total_iterations,
                dispatcher=dispatcher,
                verbose=verbose,
                use_live_display=True,
            )

            # Mark as completed
            display.mark_completed(
                cluster_idx,
                cost=result.cost,
                n_evaluations=result.n_evaluations,
                time_sec=result.time,
                success=result.success,
                message=result.message if not result.success else None,
            )

            params_all.update(result.params)


def _fit_and_report_cluster(
    cluster: Cluster,
    cluster_idx: int,
    cluster_count: int,
    step: FitStep,
    strategy: Any,
    noise_value: float,
    parameter_config: ParameterConfig | None,
    params_init: dict[str, float],
    iteration_idx: int,
    total_iterations: int,
    dispatcher: EventDispatcher | None,
    verbose: bool,
    *,
    use_live_display: bool = True,
) -> ClusterFitResult:
    """Fit a single cluster and report progress.

    Args:
        cluster: Cluster to fit
        cluster_idx: Index of this cluster (1-based)
        cluster_count: Total number of clusters
        step: Current fitting step
        strategy: Optimization strategy
        noise_value: Noise value for chi-squared calculation
        parameter_config: Optional parameter constraints
        params_init: Initial parameter values
        iteration_idx: Current iteration index
        total_iterations: Total number of iterations
        dispatcher: Event dispatcher for progress events
        verbose: Whether to log verbose output
        use_live_display: Whether live display is active (suppresses console prints)

    Returns
    -------
        ClusterFitResult with fitted parameters and statistics
    """
    peak_names = [peak.name for peak in cluster.peaks]
    peaks_str = ", ".join(peak_names)

    # Only print to console if verbose AND not using live display
    if verbose and not use_live_display:
        _print_cluster_header(cluster_idx, cluster_count, peaks_str, len(peak_names))

    log("")
    log(f"Cluster {cluster_idx}/{cluster_count}: {peaks_str}")
    log(f"  - Peaks: {len(cluster.peaks)}")

    _dispatch_cluster_started(
        dispatcher, cluster_idx, cluster_count, iteration_idx, total_iterations, peak_names
    )

    result = _fit_single_cluster(
        cluster=cluster,
        step=step,
        strategy=strategy,
        noise_value=noise_value,
        parameter_config=parameter_config,
        params_init=params_init,
    )

    _dispatch_cluster_completed(
        dispatcher,
        cluster_idx,
        cluster_count,
        iteration_idx,
        total_iterations,
        result.cost,
        result.success,
        result.time,
    )

    # Only print to console if verbose AND not using live display
    if verbose and not use_live_display:
        _print_cluster_result(result)
    _log_cluster_result(result)

    return result


def _fit_iteration_parallel(
    clusters: list[Cluster],
    params_all: Parameters,
    step: FitStep,
    strategy: Any,
    noise_value: float,
    parameter_config: ParameterConfig | None,
    cluster_count: int,
    iteration_idx: int,
    total_iterations: int,
    dispatcher: EventDispatcher | None,
    workers: int,
) -> None:
    """Fit all clusters in parallel using joblib with live status display."""
    from joblib import Parallel, delayed

    params_init = {key: params_all[key].value for key in params_all}

    # Determine actual number of workers
    actual_workers = (os.cpu_count() or 1) if workers == -1 else workers

    log(f"Parallel fitting: {cluster_count} clusters with {actual_workers} workers")

    # Create the live display
    display = LiveClusterDisplay.from_clusters(clusters)
    display.set_step_name(step.name or "Fitting")
    display.set_workers(actual_workers)

    with display:
        # Mark all clusters as running since they'll be processed in parallel
        for idx in range(1, cluster_count + 1):
            display.mark_running(idx)

        # Run parallel fitting
        # Note: loky backend doesn't support streaming results, so we get all at once
        parallel_results = Parallel(n_jobs=workers, backend="loky")(
            delayed(_fit_single_cluster)(
                cluster=cluster,
                step=step,
                strategy=strategy,
                noise_value=noise_value,
                parameter_config=parameter_config,
                params_init=params_init,
            )
            for cluster in clusters
        )

        # Type assertion: joblib returns list matching delayed function return type
        results = [r for r in parallel_results if isinstance(r, ClusterFitResult)]

        # Update display with all results
        for cluster_idx, result in enumerate(results, 1):
            display.mark_completed(
                cluster_idx,
                cost=result.cost,
                n_evaluations=result.n_evaluations,
                time_sec=result.time,
                success=result.success,
                message=result.message if not result.success else None,
            )

    # Collect and update parameters
    _collect_parallel_results(
        clusters,
        results,
        params_all,
        dispatcher,
        cluster_count,
        iteration_idx,
        total_iterations,
    )


def _collect_parallel_results(
    clusters: list[Cluster],
    results: list[ClusterFitResult],
    params_all: Parameters,
    dispatcher: EventDispatcher | None,
    cluster_count: int,
    iteration_idx: int,
    total_iterations: int,
) -> None:
    """Collect results from parallel fitting and update parameters."""
    total_time = 0.0
    successful = 0

    for cluster_idx, (cluster, result) in enumerate(zip(clusters, results, strict=True), 1):
        total_time += result.time
        if result.success:
            successful += 1

        peak_names = [peak.name for peak in cluster.peaks]

        _dispatch_cluster_completed(
            dispatcher,
            cluster_idx,
            cluster_count,
            iteration_idx,
            total_iterations,
            result.cost,
            result.success,
            result.time,
        )

        log(
            f"Cluster {cluster_idx}: {', '.join(peak_names)} - "
            f"cost={result.cost:.2e}, iter={result.n_evaluations}, time={result.time:.2f}s"
        )

        params_all.update(result.params)

    # Log summary (display already shows visual feedback)
    log(f"Completed {cluster_count} clusters ({successful} converged, CPU time: {total_time:.1f}s)")


def _log_step_header(step: FitStep, step_idx: int, total_steps: int) -> None:
    """Log step header information."""
    step_name = step.name or f"Step {step_idx + 1}"
    console.print()
    console.print(
        f"[bold magenta]═══ Protocol Step {step_idx + 1}/{total_steps}: "
        f"{step_name} ═══[/bold magenta]"
    )

    if step.description:
        console.print(f"  [dim]{step.description}[/dim]")

    if step.fix:
        console.print(f"  [yellow]Fix:[/yellow] {', '.join(step.fix)}")
    if step.vary:
        console.print(f"  [green]Vary:[/green] {', '.join(step.vary)}")

    console.print(f"  [cyan]Iterations:[/cyan] {step.iterations}")
    console.print()


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
        f"[bold cyan]Cluster {cluster_idx}/{cluster_count}[/bold cyan] [dim]│[/dim] "
        f"{peaks_str} [dim][{n_peaks} peak{'s' if n_peaks != 1 else ''}][/dim]"
    )


def _print_cluster_result(result: ClusterFitResult) -> None:
    """Print cluster fitting result."""
    if result.success:
        console.print(
            f"[green]✓[/green] Converged [dim]│[/dim] "
            f"χ² = [cyan]{result.cost:.2e}[/cyan] [dim]│[/dim] "
            f"{result.n_evaluations} evaluations [dim]│[/dim] "
            f"{result.time:.1f}s"
        )
    else:
        console.print(
            f"[yellow]⚠[/yellow] {result.message} [dim]│[/dim] "
            f"χ² = [cyan]{result.cost:.2e}[/cyan] [dim]│[/dim] "
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


def _dispatch_cluster_started(
    dispatcher: EventDispatcher | None,
    cluster_idx: int,
    cluster_count: int,
    iteration_idx: int,
    total_iterations: int,
    peak_names: list[str],
) -> None:
    """Dispatch cluster started event."""
    if dispatcher is None:
        return

    dispatcher.dispatch(
        Event(
            event_type=EventType.CLUSTER_STARTED,
            data={
                "cluster_index": cluster_idx,
                "total_clusters": cluster_count,
                "iteration": iteration_idx,
                "total_iterations": total_iterations,
                "peak_names": peak_names,
            },
        )
    )


def _dispatch_cluster_completed(
    dispatcher: EventDispatcher | None,
    cluster_idx: int,
    cluster_count: int,
    iteration_idx: int,
    total_iterations: int,
    cost: float,
    success: bool,
    cluster_time: float,
) -> None:
    """Dispatch cluster completed events."""
    if dispatcher is None:
        return

    dispatcher.dispatch(
        Event(
            event_type=EventType.CLUSTER_COMPLETED,
            data={
                "cluster_index": cluster_idx,
                "total_clusters": cluster_count,
                "iteration": iteration_idx,
                "total_iterations": total_iterations,
                "cost": cost,
                "success": success,
                "time_sec": cluster_time,
            },
        )
    )
    dispatcher.dispatch(
        FitProgressEvent(
            event_type=EventType.FIT_PROGRESS,
            data={
                "cluster_index": cluster_idx,
                "iteration": iteration_idx,
                "cost": cost,
                "success": success,
            },
            current_cluster=cluster_idx,
            total_clusters=cluster_count,
            current_iteration=iteration_idx,
            total_iterations=total_iterations,
        )
    )
