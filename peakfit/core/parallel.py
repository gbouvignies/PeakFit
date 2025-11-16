"""Parallel fitting support for PeakFit.

This module provides parallel processing capabilities to fit multiple clusters
concurrently, significantly improving performance on multi-core systems.
"""

import multiprocessing as mp
from collections.abc import Sequence
from functools import partial
from typing import Any

import lmfit as lf

from peakfit.clustering import Cluster
from peakfit.computing import residuals
from peakfit.peak import create_params


def _fit_single_cluster(
    cluster: Cluster,
    noise: float,
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
    # Create parameters for this cluster
    params = create_params(cluster.peaks, fixed=fixed)

    # Update with global parameter values
    for key in params:
        if key in params_dict:
            params[key].value = params_dict[key]

    # Perform fit
    mini = lf.Minimizer(residuals, params, fcn_args=(cluster, noise))
    result = mini.least_squares(verbose=0)

    # Extract results
    fitted_params = {}
    for name in result.params:
        param = result.params[name]
        fitted_params[name] = {
            "value": param.value,
            "stderr": param.stderr,
            "vary": param.vary,
            "min": param.min,
            "max": param.max,
        }

    return {
        "params": fitted_params,
        "success": result.success,
        "chisqr": result.chisqr if hasattr(result, "chisqr") else 0.0,
        "redchi": result.redchi if hasattr(result, "redchi") else 0.0,
        "nfev": result.nfev if hasattr(result, "nfev") else 0,
        "message": result.message if hasattr(result, "message") else "",
    }


def fit_clusters_parallel(
    clusters: Sequence[Cluster],
    noise: float,
    fixed: bool = False,
    n_workers: int | None = None,
    progress_callback: Any = None,
) -> lf.Parameters:
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
        with mp.Pool(n_workers) as pool:
            results = pool.map(worker, clusters)
    else:
        # Fall back to sequential for single cluster or worker
        results = [worker(cluster) for cluster in clusters]

    # Combine results into single Parameters object
    params_all = lf.Parameters()
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
                if param_info["stderr"] is not None:
                    params_all[name].stderr = param_info["stderr"]

        if progress_callback is not None:
            progress_callback(result)

    return params_all


def fit_clusters_parallel_refined(
    clusters: Sequence[Cluster],
    noise: float,
    refine_iterations: int = 1,
    fixed: bool = False,
    n_workers: int | None = None,
    verbose: bool = False,
) -> lf.Parameters:
    """Fit clusters with parallel processing and refinement iterations.

    This function performs iterative fitting with cross-talk correction,
    parallelizing the cluster fitting within each iteration.

    Args:
        clusters: List of clusters to fit
        noise: Noise level
        refine_iterations: Number of refinement passes
        fixed: Whether to fix peak positions
        n_workers: Number of parallel workers
        verbose: Print progress information

    Returns:
        Final fitted parameters for all clusters
    """
    from peakfit.computing import update_cluster_corrections

    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(clusters))

    params_all = lf.Parameters()

    for iteration in range(refine_iterations + 1):
        if verbose:
            if iteration == 0:
                print(f"Fitting {len(clusters)} clusters with {n_workers} workers...")
            else:
                print(f"Refinement iteration {iteration}/{refine_iterations}...")

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

        # Fit all clusters in parallel
        if n_workers > 1 and len(clusters) > 1:
            with mp.Pool(n_workers) as pool:
                results = pool.map(worker, clusters)
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
                if param_info["stderr"] is not None:
                    params_all[name].stderr = param_info["stderr"]

        if verbose:
            successes = sum(1 for r in results if r["success"])
            print(f"  {successes}/{len(results)} clusters converged")

    return params_all
