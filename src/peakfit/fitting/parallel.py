"""Parallel fitting support for PeakFit.

This module provides parallel processing capabilities to fit multiple clusters
concurrently, significantly improving performance on multi-core systems.
"""

import multiprocessing as mp
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from threadpoolctl import threadpool_limits

from peakfit.data.clustering import Cluster
from peakfit.fitting.parameters import Parameters


def _get_mp_context() -> mp.context.BaseContext:
    """Get multiprocessing context for parallel processing.

    Uses 'fork' on Unix systems for better performance.

    Returns:
        Multiprocessing context
    """
    # Use fork when available
    try:
        return mp.get_context("fork")
    except ValueError:
        # fork not available (Windows), fall back to spawn
        return mp.get_context("spawn")


def _fit_single_cluster(
    cluster: Cluster,
    noise: float,
    *,
    fixed: bool,
    params_dict: dict[str, Any],
) -> dict[str, Any]:
    """Fit a single cluster (worker function for parallel processing).

    Args:
        cluster: Cluster to fit
        noise: Noise level
        fixed: Whether to fix positions
        params_dict: Dictionary of global parameter values

    Returns:
        Dictionary with fitted parameter values and fit statistics
    """
    from peakfit.fitting.optimizer import fit_cluster

    # Use scipy-based fitting
    return fit_cluster(cluster, noise, fixed=fixed, params_init=params_dict)


def fit_clusters_parallel(
    clusters: Sequence[Cluster],
    noise: float,
    *,
    fixed: bool = False,
    n_workers: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Parameters:
    """Fit multiple clusters in parallel.

    Args:
        clusters: List of clusters to fit
        noise: Noise level
        fixed: Whether to fix peak positions
        n_workers: Number of parallel workers (default: number of CPUs)
        progress_callback: Optional callback for progress updates

    Returns:
        Combined parameters from all clusters
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(clusters))

    # Initial empty global parameters
    params_dict: dict[str, Any] = {}

    # Worker function with fixed arguments
    worker = partial(
        _fit_single_cluster,
        noise=noise,
        fixed=fixed,
        params_dict=params_dict,
    )

    # Run parallel fitting
    if n_workers > 1 and len(clusters) > 1:
        ctx = _get_mp_context()
        with ctx.Pool(n_workers) as pool:
            results = pool.map(worker, clusters)
    else:
        # Fall back to sequential for single cluster or worker
        results = [worker(cluster) for cluster in clusters]

    # Combine results into single Parameters object
    params_all = Parameters()
    for result in results:
        for name, param_info in result["params"].items():
            if name not in params_all:
                params_all.add(
                    name,
                    value=param_info["value"],
                    vary=param_info["vary"],
                    min=param_info["min"],
                    max=param_info["max"],
                )
            # Propagate standard errors from fitting
            if "stderr" in param_info and param_info["stderr"] is not None:
                params_all[name].stderr = param_info["stderr"]

        if progress_callback is not None:
            progress_callback(result)

    return params_all


def _optimal_worker_count(n_clusters: int) -> int:
    """Calculate optimal number of workers for parallel fitting.

    Balances parallelism benefits against GIL contention overhead.
    Too many threads = excessive GIL contention, reducing efficiency.
    Optimal is ~10-16 workers for thread-based parallelism.

    Args:
        n_clusters: Number of clusters to fit

    Returns:
        Optimal number of workers
    """
    cpu_count = mp.cpu_count()

    # For thread-based parallelism, GIL contention limits effective parallelism
    # Even with Numba GIL release, scipy's Python wrapper still contends
    # MacBook Pro: 10 workers, 309% CPU = 3.1 effective cores = optimal
    # Threadripper: 48 workers, 3312% CPU = 33 effective cores = too much contention

    # Optimal is ~8-16 workers regardless of CPU count
    # More threads = more GIL contention = diminishing returns
    max_effective_workers = 12

    # But don't use more workers than clusters
    optimal = min(max_effective_workers, n_clusters)

    # And don't exceed CPU count (though this is rarely the limit)
    return min(optimal, cpu_count)


def _set_blas_threads(n_threads: int = 1) -> dict[str, str | None]:
    """Set BLAS libraries to single-threaded mode to avoid oversubscription.

    When using Python thread parallelism, internal BLAS threading causes
    massive performance degradation due to thread oversubscription.
    This sets OpenBLAS, MKL, and OpenMP to single-threaded mode.

    Args:
        n_threads: Number of threads per BLAS operation (default 1)

    Returns:
        Original environment values (for restoration)
    """
    import os

    # Save original values
    original = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
    }

    # Set to single-threaded
    n_str = str(n_threads)
    os.environ["OMP_NUM_THREADS"] = n_str
    os.environ["OPENBLAS_NUM_THREADS"] = n_str
    os.environ["MKL_NUM_THREADS"] = n_str
    os.environ["VECLIB_MAXIMUM_THREADS"] = n_str
    os.environ["NUMEXPR_NUM_THREADS"] = n_str

    return original


def _restore_blas_threads(original: dict[str, str | None]) -> None:
    """Restore original BLAS thread settings."""
    import os

    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def fit_clusters_parallel_refined(
    clusters: Sequence[Cluster],
    noise: float,
    refine_iterations: int = 1,
    *,
    fixed: bool = False,
    n_workers: int | None = None,
    verbose: bool = False,
) -> Parameters:
    """Fit clusters with parallel processing and refinement iterations.

    This function performs iterative fitting with cross-talk correction,
    parallelizing the cluster fitting within each iteration.

    Uses thread-based parallelism with NumPy releasing the GIL for
    numerical computations. While threads can't achieve perfect parallel
    scaling due to GIL contention in Python code, they avoid overhead
    of process spawning.

    Args:
        clusters: List of clusters to fit
        noise: Noise level
        refine_iterations: Number of refinement passes
        fixed: Whether to fix peak positions
        n_workers: Number of parallel workers (default: CPU count, capped at 16)
        verbose: Print progress information

    Returns:
        Final fitted parameters for all clusters
    """
    from peakfit.fitting.computation import update_cluster_corrections

    if n_workers is None:
        # Default: use CPU count but cap at 16 to avoid excessive GIL contention
        n_workers = min(16, mp.cpu_count(), len(clusters))
    else:
        # User specified: respect it but warn if very high
        n_workers = min(n_workers, len(clusters))

    params_all = Parameters()

    # Use threadpoolctl to limit BLAS threads at runtime
    # This prevents OpenBLAS/MKL from spawning threads that fight with Python threads
    with threadpool_limits(limits=1, user_api="blas"):
        for iteration in range(refine_iterations + 1):
            if verbose:
                from peakfit.ui.style import console

                if refine_iterations > 0:
                    console.print(
                        f"[cyan]Refinement iteration {iteration + 1}/{refine_iterations + 1}[/]"
                    )

            # Update corrections if not first iteration
            if iteration > 0:
                update_cluster_corrections(params_all, clusters)

            # Prepare global parameters dict for workers
            params_dict = {name: params_all[name].value for name in params_all}

            # Worker function with current global parameters
            worker = partial(
                _fit_single_cluster,
                noise=noise,
                fixed=fixed,
                params_dict=params_dict,
            )

            # Fit all clusters in parallel using threads
            if n_workers > 1 and len(clusters) > 1:
                if verbose:
                    from peakfit.ui.style import console

                    console.print(
                        f"[dim]Fitting {len(clusters)} clusters with {n_workers} workers...[/]"
                    )
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    results = list(executor.map(worker, clusters))
            else:
                results = [worker(cluster) for cluster in clusters]

            # Update global parameters with results
            for result in results:
                for name, param_info in result["params"].items():
                    if name not in params_all:
                        params_all.add(
                            name,
                            value=param_info["value"],
                            vary=param_info["vary"],
                            min=param_info["min"],
                            max=param_info["max"],
                        )
                    else:
                        params_all[name].value = param_info["value"]
                    # Propagate standard errors from fitting
                    if "stderr" in param_info and param_info["stderr"] is not None:
                        params_all[name].stderr = param_info["stderr"]

    return params_all
