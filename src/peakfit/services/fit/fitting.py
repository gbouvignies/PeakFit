"""Fitting execution logic for the fit pipeline.

This module handles the core fitting loop including parameter initialization,
cluster fitting, refinement iterations, progress tracking, and constraint application.

Supports both legacy single-pass fitting and modern multi-step protocols with
parameter constraints.
"""

from __future__ import annotations

import time as time_module
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
from peakfit.ui import console, log, log_section, subsection_header

if TYPE_CHECKING:
    from peakfit.core.domain.cluster import Cluster
    from peakfit.core.shared.events import EventDispatcher
    from peakfit.services.fit.pipeline import FitArguments


__all__ = [
    "fit_all_clusters",
    "fit_all_clusters_with_protocol",
]


def fit_all_clusters(
    clargs: FitArguments,
    clusters: list[Cluster],
    *,
    optimizer: str,
    verbose: bool,
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
        optimizer: Name of optimizer to use
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

    noise_value = float(clargs.noise)
    strategy_kwargs: dict[str, Any] = {}

    if optimizer == "leastsq":
        strategy_kwargs = {
            "ftol": LEAST_SQUARES_FTOL,
            "xtol": LEAST_SQUARES_XTOL,
            "max_nfev": LEAST_SQUARES_MAX_NFEV,
            "verbose": 2 if verbose else 0,
        }

    strategy = get_strategy(optimizer, **strategy_kwargs)

    # Use protocol if provided, otherwise create from legacy options
    if protocol is None:
        protocol = create_protocol_from_config(
            steps=None,
            refine_iterations=clargs.refine_nb,
            fixed=clargs.fixed,
        )

    # Execute protocol-based fitting
    return _fit_with_protocol(
        clusters=clusters,
        protocol=protocol,
        strategy=strategy,
        noise_value=noise_value,
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
    optimizer: str = "leastsq",
    parameter_config: ParameterConfig | None = None,
    verbose: bool = False,
    dispatcher: EventDispatcher | None = None,
) -> Parameters:
    """Fit clusters using a multi-step protocol.

    This is the modern API for fitting with full control over the process.

    Args:
        clusters: List of clusters to fit
        noise: Noise level
        protocol: Multi-step fitting protocol
        optimizer: Name of optimizer to use
        parameter_config: Optional parameter constraints
        verbose: Whether to show verbose output
        dispatcher: Optional event dispatcher

    Returns
    -------
        Fitted parameters for all clusters
    """
    strategy_kwargs: dict[str, Any] = {}

    if optimizer == "leastsq":
        strategy_kwargs = {
            "ftol": LEAST_SQUARES_FTOL,
            "xtol": LEAST_SQUARES_XTOL,
            "max_nfev": LEAST_SQUARES_MAX_NFEV,
            "verbose": 2 if verbose else 0,
        }

    strategy = get_strategy(optimizer, **strategy_kwargs)

    return _fit_with_protocol(
        clusters=clusters,
        protocol=protocol,
        strategy=strategy,
        noise_value=noise,
        parameter_config=parameter_config,
        verbose=verbose,
        dispatcher=dispatcher,
    )


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
    """Internal implementation of protocol-based fitting.

    Args:
        clusters: Clusters to fit
        protocol: Fitting protocol
        strategy: Optimization strategy
        noise_value: Noise level
        parameter_config: Parameter constraints
        verbose: Verbose output
        dispatcher: Event dispatcher
        workers: Number of parallel workers (-1 for all CPUs, 1 for sequential)

    Returns
    -------
        Fitted parameters
    """
    params_all = Parameters()
    cluster_count = len(clusters)
    total_iterations = sum(step.iterations for step in protocol.steps)

    # Determine if parallel execution is enabled
    use_parallel = workers != 1

    with threadpool_limits(limits=1, user_api="blas"):
        global_iteration = 0

        for step_idx, step in enumerate(protocol.steps):
            step_name = step.name or f"Step {step_idx + 1}"

            if verbose:
                _log_step_header(step, step_idx, len(protocol.steps))

            for iteration in range(step.iterations):
                global_iteration += 1
                iteration_idx = global_iteration

                # Log iteration header
                if step.iterations > 1:
                    iter_label = f"{step_name} (Iteration {iteration + 1}/{step.iterations})"
                else:
                    iter_label = step_name

                if iteration == 0 and step_idx == 0:
                    subsection_header("Initial Fit" if not step.name else iter_label)
                else:
                    subsection_header(f"Refining Parameters: {iter_label}")
                    log_section(f"Protocol: {iter_label}")
                    update_cluster_corrections(params_all, clusters)

                # Fit all clusters (parallel or sequential)
                if use_parallel:
                    _fit_iteration_parallel(
                        clusters=clusters,
                        params_all=params_all,
                        step=step,
                        strategy=strategy,
                        noise_value=noise_value,
                        parameter_config=parameter_config,
                        cluster_count=cluster_count,
                        iteration_idx=iteration_idx,
                        total_iterations=total_iterations,
                        dispatcher=dispatcher,
                        workers=workers,
                    )
                else:
                    _fit_iteration_sequential(
                        clusters=clusters,
                        params_all=params_all,
                        step=step,
                        strategy=strategy,
                        noise_value=noise_value,
                        parameter_config=parameter_config,
                        cluster_count=cluster_count,
                        iteration_idx=iteration_idx,
                        total_iterations=total_iterations,
                        dispatcher=dispatcher,
                    )

    return params_all


def _fit_single_cluster(
    cluster: Cluster,
    cluster_idx: int,
    cluster_count: int,
    step: FitStep,
    strategy: Any,
    noise_value: float,
    parameter_config: ParameterConfig | None,
    params_init: dict[str, float],
) -> tuple[Parameters, float, float, bool]:
    """Fit a single cluster (used by both sequential and parallel execution).

    Args:
        cluster: Cluster to fit
        cluster_idx: Index of this cluster (1-based)
        cluster_count: Total number of clusters
        step: Current protocol step
        strategy: Optimization strategy
        noise_value: Noise level
        parameter_config: Parameter constraints
        params_init: Initial parameter values from previous iterations

    Returns
    -------
        Tuple of (fitted_params, cost, time, success)
    """
    cluster_start = time_module.time()

    # Create parameters for this cluster
    params = create_params(cluster.peaks, fixed=False)

    # Apply parameter constraints from config
    if parameter_config is not None:
        params = apply_constraints(params, parameter_config)

    # Apply step-level fix/vary patterns
    params = apply_step_constraints(params, step)

    # Update with values from previous iterations
    for key in params:
        if key in params_init:
            params[key].value = params_init[key]

    # Run optimization
    result = strategy.optimize(params, cluster, noise_value)

    cluster_time = time_module.time() - cluster_start

    return result.params, result.cost, cluster_time, result.success


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
) -> None:
    """Fit all clusters sequentially in a single iteration.

    Args:
        clusters: List of clusters to fit
        params_all: Global parameters to update
        step: Current protocol step
        strategy: Optimization strategy
        noise_value: Noise level
        parameter_config: Parameter constraints
        cluster_count: Total number of clusters
        iteration_idx: Current iteration (1-based)
        total_iterations: Total number of iterations
        dispatcher: Optional event dispatcher
    """
    # Extract current parameter values for initialization
    params_init = {key: params_all[key].value for key in params_all}

    for cluster_idx, cluster in enumerate(clusters, 1):
        peak_names = [peak.name for peak in cluster.peaks]
        peaks_str = ", ".join(peak_names)
        n_peaks = len(cluster.peaks)

        _print_cluster_header(cluster_idx, cluster_count, peaks_str, n_peaks)

        log("")
        log(f"Cluster {cluster_idx}/{cluster_count}: {peaks_str}")
        log(f"  - Peaks: {len(cluster.peaks)}")

        _dispatch_cluster_started(
            dispatcher, cluster_idx, cluster_count, iteration_idx, total_iterations, peak_names
        )

        params, cost, cluster_time, success = _fit_single_cluster(
            cluster=cluster,
            cluster_idx=cluster_idx,
            cluster_count=cluster_count,
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
            cost,
            success,
            cluster_time,
        )

        # Create a result-like object for printing
        class _ResultProxy:
            def __init__(self, cost: float, success: bool, n_evals: int = 0) -> None:
                self.cost = cost
                self.success = success
                self.n_evaluations = n_evals
                self.message = "Converged" if success else "Did not converge"

        result_proxy = _ResultProxy(cost, success)
        _print_cluster_result(result_proxy, cluster_time)
        _log_cluster_result(result_proxy, cluster_time)

        params_all.update(params)


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
    """Fit all clusters in parallel using joblib.

    Args:
        clusters: List of clusters to fit
        params_all: Global parameters to update
        step: Current protocol step
        strategy: Optimization strategy
        noise_value: Noise level
        parameter_config: Parameter constraints
        cluster_count: Total number of clusters
        iteration_idx: Current iteration (1-based)
        total_iterations: Total number of iterations
        dispatcher: Optional event dispatcher
        workers: Number of parallel workers (-1 for all CPUs)
    """
    from joblib import Parallel, delayed

    # Extract current parameter values for initialization
    params_init = {key: params_all[key].value for key in params_all}

    # Print header for parallel execution
    console.print(f"[cyan]Fitting {cluster_count} clusters in parallel...[/cyan]")
    log(f"Parallel fitting: {cluster_count} clusters with {workers} workers")

    # Run all cluster fits in parallel
    results = Parallel(n_jobs=workers, backend="loky")(
        delayed(_fit_single_cluster)(
            cluster=cluster,
            cluster_idx=cluster_idx,
            cluster_count=cluster_count,
            step=step,
            strategy=strategy,
            noise_value=noise_value,
            parameter_config=parameter_config,
            params_init=params_init,
        )
        for cluster_idx, cluster in enumerate(clusters, 1)
    )

    # Collect results and update parameters
    total_time = 0.0
    successful = 0
    for cluster_idx, (cluster, (params, cost, cluster_time, success)) in enumerate(
        zip(clusters, results, strict=True), 1
    ):
        total_time += cluster_time
        if success:
            successful += 1

        peak_names = [peak.name for peak in cluster.peaks]

        _dispatch_cluster_completed(
            dispatcher,
            cluster_idx,
            cluster_count,
            iteration_idx,
            total_iterations,
            cost,
            success,
            cluster_time,
        )

        # Log each result
        log(f"Cluster {cluster_idx}: {', '.join(peak_names)} - cost={cost:.2e}, time={cluster_time:.2f}s")

        params_all.update(params)

    # Print summary
    console.print(
        f"[green]✓[/green] Completed {cluster_count} clusters "
        f"({successful}/{cluster_count} converged, total CPU time: {total_time:.1f}s)"
    )


def _fit_iteration_with_constraints(
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
) -> None:
    """Fit all clusters in a single iteration with constraints.

    Args:
        clusters: List of clusters to fit
        params_all: Global parameters to update
        step: Current protocol step
        strategy: Optimization strategy
        noise_value: Noise level
        parameter_config: Parameter constraints
        cluster_count: Total number of clusters
        iteration_idx: Current iteration (1-based)
        total_iterations: Total number of iterations
        dispatcher: Optional event dispatcher

    Note:
        This function is deprecated. Use _fit_iteration_sequential or
        _fit_iteration_parallel instead.
    """
    _fit_iteration_sequential(
        clusters=clusters,
        params_all=params_all,
        step=step,
        strategy=strategy,
        noise_value=noise_value,
        parameter_config=parameter_config,
        cluster_count=cluster_count,
        iteration_idx=iteration_idx,
        total_iterations=total_iterations,
        dispatcher=dispatcher,
    )


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


def _print_cluster_result(result: Any, cluster_time: float) -> None:
    """Print cluster fitting result."""
    evaluations = result.n_evaluations

    if result.success:
        console.print(
            f"[green]✓[/green] Converged [dim]│[/dim] "
            f"χ² = [cyan]{result.cost:.2e}[/cyan] [dim]│[/dim] "
            f"{evaluations} evaluations [dim]│[/dim] "
            f"{cluster_time:.1f}s"
        )
    else:
        console.print(
            f"[yellow]⚠[/yellow] {result.message} [dim]│[/dim] "
            f"χ² = [cyan]{result.cost:.2e}[/cyan] [dim]│[/dim] "
            f"{evaluations} evaluations [dim]│[/dim] "
            f"{cluster_time:.1f}s"
        )


def _log_cluster_result(result: Any, cluster_time: float) -> None:
    """Log cluster fitting result."""
    if result.success:
        log("  - Status: Converged", level="info")
    else:
        log(f"  - Status: {result.message}", level="warning")

    log(f"  - Cost: {result.cost:.3e}")
    log(f"  - Function evaluations: {result.n_evaluations}")
    log(f"  - Time: {cluster_time:.1f}s")


def _update_params(params: Parameters, params_all: Parameters) -> Parameters:
    """Update parameters with global parameters."""
    for key in params:
        if key in params_all:
            params[key].value = params_all[key].value
            # Preserve vary status from step constraints, not from params_all
            # params[key].vary = params_all[key].vary
    return params


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
