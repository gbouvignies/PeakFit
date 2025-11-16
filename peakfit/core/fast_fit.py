"""Fast fitting using direct scipy optimization (bypassing lmfit overhead)."""

from typing import Any

import lmfit as lf
import numpy as np
from scipy.optimize import least_squares

from peakfit.clustering import Cluster
from peakfit.peak import create_params


def params_to_arrays(params: lf.Parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Convert lmfit Parameters to numpy arrays.

    Returns:
        x0: Initial values for varying parameters
        lower: Lower bounds
        upper: Upper bounds
        names: Parameter names (in order)
    """
    names = []
    x0 = []
    lower = []
    upper = []

    for name in params:
        param = params[name]
        if param.vary:
            names.append(name)
            x0.append(param.value)
            lower.append(param.min if param.min is not None else -np.inf)
            upper.append(param.max if param.max is not None else np.inf)

    return np.array(x0), np.array(lower), np.array(upper), names


def arrays_to_params(
    x: np.ndarray,
    names: list[str],
    params_template: lf.Parameters
) -> lf.Parameters:
    """Update lmfit Parameters with optimized values.

    Args:
        x: Optimized parameter values
        names: Parameter names
        params_template: Template Parameters object

    Returns:
        Updated Parameters object
    """
    params = params_template.copy()
    for i, name in enumerate(names):
        params[name].value = x[i]
    return params


def residuals_fast(
    x: np.ndarray,
    names: list[str],
    params_template: lf.Parameters,
    cluster: Cluster,
    noise: float,
) -> np.ndarray:
    """Fast residual function for scipy.optimize.

    Args:
        x: Current parameter values (varying only)
        names: Parameter names
        params_template: Template Parameters with fixed values
        cluster: Cluster being fit
        noise: Noise level

    Returns:
        Residual vector
    """
    # Update parameters with current values
    for i, name in enumerate(names):
        params_template[name].value = x[i]

    # Calculate shapes (this uses the optimized JIT functions)
    shapes = np.array(
        [peak.evaluate(cluster.positions, params_template) for peak in cluster.peaks]
    )

    # Least squares for amplitudes
    amplitudes = np.linalg.lstsq(shapes.T, cluster.corrected_data, rcond=None)[0]

    # Residual
    return (cluster.corrected_data - shapes.T @ amplitudes).ravel() / noise


def fit_cluster_fast(
    cluster: Cluster,
    noise: float,
    fixed: bool = False,
    params_init: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a single cluster using direct scipy optimization.

    Args:
        cluster: Cluster to fit
        noise: Noise level
        fixed: Whether to fix positions
        params_init: Optional initial parameter values

    Returns:
        Dictionary with fitted parameter values and statistics
    """
    # Create parameters
    params = create_params(cluster.peaks, fixed=fixed)

    # Update with initial values if provided
    if params_init:
        for key in params:
            if key in params_init:
                params[key].value = params_init[key]

    # Convert to arrays
    x0, lower, upper, names = params_to_arrays(params)

    if len(x0) == 0:
        # No varying parameters, return as-is
        fitted_params = {}
        for name in params:
            param = params[name]
            fitted_params[name] = {
                "value": param.value,
                "stderr": None,
                "vary": param.vary,
                "min": param.min,
                "max": param.max,
            }
        return {
            "params": fitted_params,
            "success": True,
            "chisqr": 0.0,
            "redchi": 0.0,
            "nfev": 0,
            "message": "No varying parameters",
        }

    # Direct scipy optimization
    result = least_squares(
        residuals_fast,
        x0,
        args=(names, params, cluster, noise),
        bounds=(lower, upper),
        method="trf",
        ftol=1e-7,
        xtol=1e-7,
        max_nfev=1000,
        verbose=0,
    )

    # Update parameters with optimized values
    for i, name in enumerate(names):
        params[name].value = result.x[i]

    # Calculate chi-square
    residual = residuals_fast(result.x, names, params, cluster, noise)
    chisqr = float(np.sum(residual**2))
    ndata = len(residual)
    nvarys = len(x0)
    redchi = chisqr / max(1, ndata - nvarys)

    # Extract results
    fitted_params = {}
    for name in params:
        param = params[name]
        fitted_params[name] = {
            "value": param.value,
            "stderr": None,  # Not computed for speed
            "vary": param.vary,
            "min": param.min,
            "max": param.max,
        }

    return {
        "params": fitted_params,
        "success": result.success,
        "chisqr": chisqr,
        "redchi": redchi,
        "nfev": result.nfev,
        "message": result.message,
    }


def fit_clusters_fast(
    clusters: list[Cluster],
    noise: float,
    refine_iterations: int = 1,
    fixed: bool = False,
    verbose: bool = False,
) -> lf.Parameters:
    """Fit all clusters using fast scipy optimization.

    Args:
        clusters: List of clusters
        noise: Noise level
        refine_iterations: Number of refinement passes
        fixed: Whether to fix positions
        verbose: Print progress

    Returns:
        Combined fitted parameters
    """
    from peakfit.computing import update_cluster_corrections

    params_all = lf.Parameters()
    params_dict: dict[str, Any] = {}

    for iteration in range(refine_iterations + 1):
        if verbose:
            if iteration == 0:
                print(f"Fitting {len(clusters)} clusters...")
            else:
                print(f"Refinement iteration {iteration}/{refine_iterations}...")

        # Update corrections if not first iteration
        if iteration > 0:
            update_cluster_corrections(params_all, clusters)

        # Fit all clusters sequentially
        successes = 0
        for i, cluster in enumerate(clusters):
            result = fit_cluster_fast(cluster, noise, fixed, params_dict)

            if result["success"]:
                successes += 1

            # Update global parameters
            for name, param_info in result["params"].items():
                params_dict[name] = param_info["value"]
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

        if verbose:
            print(f"  {successes}/{len(clusters)} clusters converged")

    return params_all
