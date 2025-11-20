"""Fast fitting using direct scipy optimization.

This module provides high-performance fitting functions that directly
interface with scipy.optimize.least_squares.

Key features:
- Direct scipy optimization without external wrappers
- Direct parameter array manipulation
- Uses custom Parameters class for lightweight parameter management
- Supports bounded optimization with TRF algorithm
"""

import warnings
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from peakfit.clustering import Cluster
from peakfit.core.constants import LEAST_SQUARES_FTOL, LEAST_SQUARES_MAX_NFEV, LEAST_SQUARES_XTOL
from peakfit.core.fitting import Parameters
from peakfit.peak import create_params


class FastFitError(Exception):
    """Exception raised for errors in fast fitting."""


class ConvergenceWarning(UserWarning):
    """Warning for convergence issues."""


def params_to_arrays(params: Parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Convert Parameters to numpy arrays.

    Returns:
        x0: Initial values for varying parameters
        lower: Lower bounds
        upper: Upper bounds
        names: Parameter names (in order)
    """
    names = params.get_vary_names()
    x0 = params.get_vary_values()
    lower = np.array([params[name].min for name in names])
    upper = np.array([params[name].max for name in names])

    return x0, lower, upper, names


def arrays_to_params(x: np.ndarray, names: list[str], params_template: Parameters) -> Parameters:
    """Update Parameters with optimized values.

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
    params_template: Parameters,
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
    shapes = np.array([peak.evaluate(cluster.positions, params_template) for peak in cluster.peaks])

    # Least squares for amplitudes
    amplitudes = np.linalg.lstsq(shapes.T, cluster.corrected_data, rcond=None)[0]

    # Residual
    return (cluster.corrected_data - shapes.T @ amplitudes).ravel() / noise


def fit_cluster_dict(
    cluster: Cluster,
    noise: float,
    *,
    fixed: bool = False,
    params_init: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a single cluster using direct scipy optimization (dict interface).

    This function bypasses lmfit overhead by directly interfacing with
    scipy.optimize.least_squares, providing significant performance gains.

    Note: This function returns a dictionary for backward compatibility.
    For new code, prefer using fit_cluster from peakfit.core.fitting.

    Args:
        cluster: Cluster to fit
        noise: Noise level (must be positive)
        fixed: Whether to fix positions during optimization
        params_init: Optional initial parameter values from previous fits

    Returns:
        Dictionary with fitted parameter values and statistics:
            - params: Dict of parameter info (value, stderr, vary, min, max)
            - success: Whether optimization converged
            - chisqr: Chi-squared statistic
            - redchi: Reduced chi-squared
            - nfev: Number of function evaluations
            - message: Optimizer message

    Raises:
        FastFitError: If cluster has no peaks or invalid data
        ValueError: If noise is non-positive
    """
    # Validate inputs
    if noise <= 0:
        msg = f"Noise must be positive, got {noise}"
        raise ValueError(msg)

    if not cluster.peaks:
        msg = "Cluster has no peaks to fit"
        raise FastFitError(msg)

    if not hasattr(cluster, "corrected_data") or cluster.corrected_data is None:
        msg = "Cluster has no data to fit"
        raise FastFitError(msg)

    # Create parameters
    try:
        params = create_params(cluster.peaks, fixed=fixed)
    except Exception as e:
        msg = f"Failed to create parameters: {e}"
        raise FastFitError(msg) from e

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

    # Validate bounds
    if np.any(lower >= upper):
        msg = "Invalid parameter bounds: lower bound >= upper bound"
        raise FastFitError(msg)

    if np.any((x0 < lower) | (x0 > upper)):
        msg = "Initial values outside bounds"
        raise FastFitError(msg)

    # Direct scipy optimization
    try:
        result = least_squares(
            residuals_fast,
            x0,
            args=(names, params, cluster, noise),
            bounds=(lower, upper),
            method="trf",
            ftol=LEAST_SQUARES_FTOL,
            xtol=LEAST_SQUARES_XTOL,
            max_nfev=LEAST_SQUARES_MAX_NFEV,
            verbose=0,
        )
    except Exception as e:
        msg = f"Optimization failed: {e}"
        raise FastFitError(msg) from e

    # Check for convergence issues
    if not result.success:
        warnings.warn(
            f"Optimization did not converge: {result.message}",
            ConvergenceWarning,
            stacklevel=2,
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

    # Check for potential issues
    if redchi > 100:
        warnings.warn(
            f"Poor fit quality: reduced chi-squared = {redchi:.2f}",
            ConvergenceWarning,
            stacklevel=2,
        )

    # Compute standard errors from Jacobian
    # cov = inv(J.T @ J) * s2, where s2 = chi^2 / (n - p)
    stderr_dict: dict[str, float] = {}
    try:
        if result.jac is not None and ndata > nvarys:
            jac = result.jac
            # Compute covariance matrix
            jtj = jac.T @ jac
            try:
                cov = np.linalg.inv(jtj) * redchi
                stderr = np.sqrt(np.diag(cov))
                for i, name in enumerate(names):
                    stderr_dict[name] = float(stderr[i])
            except np.linalg.LinAlgError:
                # Singular matrix, can't compute errors - skip stderr computation
                pass
    except (ValueError, RuntimeError):
        # If error computation fails due to numerical issues, continue without errors
        pass

    # Extract results
    fitted_params = {}
    for name in params:
        param = params[name]
        fitted_params[name] = {
            "value": param.value,
            "stderr": stderr_dict.get(name, 0.0),
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
    *,
    fixed: bool = False,
    verbose: bool = False,
) -> Parameters:
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

    params_all = Parameters()
    params_dict: dict[str, Any] = {}

    for iteration in range(refine_iterations + 1):
        if verbose:
            if iteration == 0:
                pass
            else:
                pass

        # Update corrections if not first iteration
        if iteration > 0:
            update_cluster_corrections(params_all, clusters)

        # Fit all clusters sequentially
        successes = 0
        for _i, cluster in enumerate(clusters):
            result = fit_cluster_dict(cluster, noise, fixed=fixed, params_init=params_dict)

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
                # Propagate standard errors from fitting
                if "stderr" in param_info and param_info["stderr"] is not None:
                    params_all[name].stderr = param_info["stderr"]

        if verbose:
            pass

    return params_all


# Backward compatibility alias (deprecated)
fit_cluster_fast = fit_cluster_dict
