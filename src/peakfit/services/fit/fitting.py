"""Fitting execution logic for the fit pipeline.

This module handles the core fitting loop including parameter initialization,
cluster fitting, refinement iterations, and progress tracking.
"""

from __future__ import annotations

import time as time_module
from typing import TYPE_CHECKING, Any

from threadpoolctl import threadpool_limits

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import create_params
from peakfit.core.fitting.computation import update_cluster_corrections
from peakfit.core.fitting.parameters import Parameters
from peakfit.core.fitting.strategies import get_strategy
from peakfit.core.shared.constants import (
    LEAST_SQUARES_FTOL,
    LEAST_SQUARES_MAX_NFEV,
    LEAST_SQUARES_XTOL,
)
from peakfit.core.shared.events import Event, EventDispatcher, EventType, FitProgressEvent
from peakfit.ui import console, log, log_section, subsection_header

if TYPE_CHECKING:
    from peakfit.services.fit.pipeline import FitArguments


__all__ = [
    "fit_all_clusters",
]


def fit_all_clusters(
    clargs: FitArguments,
    clusters: list[Cluster],
    *,
    optimizer: str,
    verbose: bool,
    dispatcher: EventDispatcher | None = None,
) -> Parameters:
    """Fit all clusters with the requested optimization strategy.

    Args:
        clargs: Command line arguments
        clusters: List of clusters to fit
        optimizer: Name of optimizer to use
        verbose: Whether to show verbose output
        dispatcher: Optional event dispatcher for progress tracking

    Returns:
        Fitted parameters for all clusters
    """
    if clargs.noise is None:
        msg = "Noise must be specified before fitting clusters"
        raise ValueError(msg)

    noise_value = float(clargs.noise)
    total_iterations = clargs.refine_nb + 1
    strategy_kwargs: dict[str, Any] = {}

    if optimizer == "leastsq":
        strategy_kwargs = {
            "ftol": LEAST_SQUARES_FTOL,
            "xtol": LEAST_SQUARES_XTOL,
            "max_nfev": LEAST_SQUARES_MAX_NFEV,
            "verbose": 2 if verbose else 0,
        }

    strategy = get_strategy(optimizer, **strategy_kwargs)
    params_all = Parameters()
    cluster_count = len(clusters)

    with threadpool_limits(limits=1, user_api="blas"):
        for iteration in range(total_iterations):
            iteration_idx = iteration + 1
            if iteration == 0:
                subsection_header("Initial Fit")
            else:
                subsection_header(f"Refining Parameters (Iteration {iteration_idx})")
                log_section(f"Refinement Iteration {iteration_idx}")
                update_cluster_corrections(params_all, clusters)

            _fit_iteration(
                clusters=clusters,
                params_all=params_all,
                clargs=clargs,
                strategy=strategy,
                noise_value=noise_value,
                cluster_count=cluster_count,
                iteration_idx=iteration_idx,
                total_iterations=total_iterations,
                dispatcher=dispatcher,
            )

    return params_all


def _fit_iteration(
    clusters: list[Cluster],
    params_all: Parameters,
    clargs: FitArguments,
    strategy: Any,
    noise_value: float,
    cluster_count: int,
    iteration_idx: int,
    total_iterations: int,
    dispatcher: EventDispatcher | None,
) -> None:
    """Fit all clusters in a single iteration.

    Args:
        clusters: List of clusters to fit
        params_all: Global parameters to update
        clargs: Command line arguments
        strategy: Optimization strategy
        noise_value: Noise level
        cluster_count: Total number of clusters
        iteration_idx: Current iteration (1-based)
        total_iterations: Total number of iterations
        dispatcher: Optional event dispatcher
    """
    for cluster_idx, cluster in enumerate(clusters, 1):
        cluster_start = time_module.time()
        peak_names = [peak.name for peak in cluster.peaks]
        peaks_str = ", ".join(peak_names)
        n_peaks = len(cluster.peaks)

        _print_cluster_header(cluster_idx, cluster_count, peaks_str, n_peaks)

        log("")
        log(f"Cluster {cluster_idx}/{cluster_count}: {peaks_str}")
        log(f"  - Peaks: {len(cluster.peaks)}")

        params = create_params(cluster.peaks, fixed=clargs.fixed)
        params = _update_params(params, params_all)

        vary_names = params.get_vary_names()
        log(f"  - Varying parameters: {len(vary_names)}")

        _dispatch_cluster_started(
            dispatcher, cluster_idx, cluster_count, iteration_idx, total_iterations, peak_names
        )

        result = strategy.optimize(params, cluster, noise_value)
        params = result.params

        cluster_time = time_module.time() - cluster_start

        _dispatch_cluster_completed(
            dispatcher,
            cluster_idx,
            cluster_count,
            iteration_idx,
            total_iterations,
            result.cost,
            result.success,
            cluster_time,
        )

        _print_cluster_result(result, cluster_time)
        _log_cluster_result(result, cluster_time)

        params_all.update(params)


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
            params[key] = params_all[key]
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
