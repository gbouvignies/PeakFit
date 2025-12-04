"""Least-squares optimization for NMR peak fitting.

This module provides fitting functions that directly interface with
scipy.optimize.least_squares for efficient parameter optimization.

Key features:
- Direct scipy.optimize.least_squares integration
- Bounded optimization using Trust Region Reflective (TRF) algorithm
- Sequential cluster fitting with refinement iterations
- Parameter array manipulation utilities
- Robust error handling and convergence warnings
"""

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import least_squares

from peakfit.core.fitting.parameters import Parameters
from peakfit.core.fitting.results import FitResult
from peakfit.core.results.statistics import compute_chi_squared, compute_reduced_chi_squared
from peakfit.core.shared.constants import (
    LEAST_SQUARES_FTOL,
    LEAST_SQUARES_MAX_NFEV,
    LEAST_SQUARES_XTOL,
)

if TYPE_CHECKING:
    from peakfit.core.domain.cluster import Cluster


class ScipyOptimizerError(Exception):
    """Exception raised for errors in scipy optimization."""


class ConvergenceWarning(UserWarning):
    """Warning for convergence issues."""


def params_to_arrays(params: Parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Convert Parameters to numpy arrays.

    Args:
        params: Parameters object to convert

    Returns
    -------
        Tuple of (x0, lower, upper, names) where:
            - x0: Initial values for varying parameters
            - lower: Lower bounds
            - upper: Upper bounds
            - names: Parameter names (in order)
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

    Returns
    -------
        Updated Parameters object
    """
    params = params_template.copy()
    for i, name in enumerate(names):
        params[name].value = x[i]
    return params


def _residuals_for_optimizer(
    x: np.ndarray, params: Parameters, cluster: "Cluster", noise: float
) -> np.ndarray:
    """Compute residuals for scipy.optimize.least_squares.

    This function delegates to the main residuals computation function.

    Args:
        x: Array of varying parameter values
        params: Parameters object (will be updated in-place)
        cluster: Cluster being fitted
        noise: Noise level for normalization

    Returns
    -------
        Flattened residual array normalized by noise
    """
    from peakfit.core.fitting.computation import residuals

    # Update parameters with current values
    params.set_vary_values(x)

    return residuals(params, cluster, noise)


def compute_residuals(
    x: np.ndarray,
    names: list[str],
    params_template: Parameters,
    cluster: "Cluster",
    noise: float,
) -> np.ndarray:
    """Compute residuals for scipy.optimize.least_squares.

    This is a thin wrapper around the canonical residuals function
    that updates parameters before computing.

    Args:
        x: Current parameter values (varying only)
        names: Parameter names
        params_template: Template Parameters with fixed values
        cluster: Cluster being fit
        noise: Noise level

    Returns
    -------
        Residual vector normalized by noise
    """
    from peakfit.core.fitting.computation import residuals

    # Update parameters with current values
    for i, name in enumerate(names):
        params_template[name].value = x[i]

    return residuals(params_template, cluster, noise)


def compute_jacobian(
    x: np.ndarray,
    names: list[str],
    params_template: Parameters,
    cluster: "Cluster",
    noise: float,
) -> np.ndarray:
    """Compute Jacobian for scipy.optimize.least_squares using VarPro.

    Uses the Kaufman approximation for the Variable Projection Jacobian:
    J approx - (I - P_S) (dS/dtheta) c
    where P_S is the projector onto the subspace spanned by the shape functions S,
    and c are the linear amplitudes.

    Args:
        x: Current parameter values (varying only)
        names: Parameter names
        params_template: Template Parameters with fixed values
        cluster: Cluster being fit
        noise: Noise level

    Returns
    -------
        Jacobian matrix of shape (n_residuals, n_params)
    """
    from peakfit.core.fitting.computation import calculate_amplitudes

    # Update parameters with current values
    for i, name in enumerate(names):
        params_template[name].value = x[i]

    # 1. Evaluate shapes and derivatives
    shapes_list = []
    derivs_list = []

    for peak in cluster.peaks:
        # evaluate_derivatives returns (val, dict_derivs)
        val, derivs = peak.evaluate_derivatives(cluster.positions, params_template)
        shapes_list.append(val)
        derivs_list.append(derivs)

    shapes = np.array(shapes_list)  # (n_peaks, n_points)
    n_points = shapes.shape[1]

    # 2. Compute optimal amplitudes
    # amplitudes: (n_peaks,) or (n_peaks, n_planes)
    amplitudes = calculate_amplitudes(shapes, cluster.corrected_data)

    # Handle dimensions for multi-plane data
    if amplitudes.ndim == 1:
        amplitudes = amplitudes[:, np.newaxis]  # (n_peaks, 1)
    n_planes = amplitudes.shape[1]

    # 3. Construct V matrix (unprojected Jacobian components)
    # V_tensor: (n_points, n_planes, n_params)
    n_params = len(names)
    V_tensor = np.zeros((n_points, n_planes, n_params))
    param_name_to_idx = {name: i for i, name in enumerate(names)}

    for i, peak in enumerate(cluster.peaks):
        amp = amplitudes[i]  # (n_planes,)
        peak_derivs = derivs_list[i]

        for p_name, d_val in peak_derivs.items():
            if p_name in param_name_to_idx:
                idx = param_name_to_idx[p_name]
                # d_val is (n_points,)
                # V[:, :, idx] += np.outer(d_val, amp)
                # Manual outer product to avoid broadcasting issues if shapes mismatch
                V_tensor[:, :, idx] += d_val[:, np.newaxis] * amp[np.newaxis, :]

    # 4. Project V onto orthogonal complement of S (VarPro)
    # Flatten planes and params to treat as multiple RHS for linear solve
    # V_flat: (n_points, n_planes * n_params)
    V_flat = V_tensor.reshape(n_points, -1)

    # X = (S^T S)^-1 S^T V_flat
    # This projects V_flat onto the subspace S
    X = calculate_amplitudes(shapes, V_flat)

    # P_perp V = V - S X
    # S is shapes.T in matrix notation (n_points, n_peaks)
    projection = shapes.T @ X
    P_perp_V_flat = V_flat - projection

    # 5. Reshape and scale
    # We need to flatten (n_points, n_planes) to match residuals.ravel()
    # P_perp_V_flat is (n_points, n_planes * n_params)
    # Reshape to (n_points, n_planes, n_params)
    P_perp_V_tensor = P_perp_V_flat.reshape(n_points, n_planes, n_params)

    # Flatten first two dims: (n_points * n_planes, n_params)
    J = -P_perp_V_tensor.reshape(-1, n_params)

    return J / noise


def fit_cluster(
    params: Parameters,
    cluster: "Cluster",
    noise: float,
    max_nfev: int = 1000,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    verbose: int = 0,
) -> FitResult:
    """Fit a single cluster using scipy.optimize.least_squares.

    Args:
        params: Initial parameters
        cluster: Cluster to fit
        noise: Noise level (must be positive)
        max_nfev: Maximum number of function evaluations
        ftol: Tolerance for termination by change of cost function
        xtol: Tolerance for termination by change of variables
        gtol: Tolerance for termination by gradient norm
        verbose: Verbosity level (0=silent, 1=termination, 2=iteration)

    Returns
    -------
        FitResult containing optimized parameters and fit statistics

    Raises
    ------
        ValueError: If noise is non-positive
        ScipyOptimizerError: If cluster has no peaks
    """
    # Validate inputs
    if noise <= 0:
        msg = f"Noise must be positive, got {noise}"
        raise ValueError(msg)

    if not cluster.peaks:
        msg = "Cluster has no peaks to fit"
        raise ScipyOptimizerError(msg)

    # Get initial values and bounds for varying parameters
    x0 = params.get_vary_values()
    lower, upper = params.get_vary_bounds()

    # Calculate number of amplitude parameters for DOF
    n_peaks = len(cluster.peaks)
    n_planes = cluster.corrected_data.shape[0] if cluster.corrected_data.ndim > 1 else 1
    n_amplitude_params = n_peaks * n_planes

    # Run optimization
    result = least_squares(
        _residuals_for_optimizer,
        x0,
        args=(params, cluster, noise),
        bounds=(lower, upper),
        method="trf",  # Trust Region Reflective - good for bounded problems
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
        verbose=verbose,
    )

    # Update parameters with final values
    params.set_vary_values(result.x)

    return FitResult(
        params=params,
        residual=result.fun,
        cost=result.cost,
        nfev=result.nfev,
        njev=result.njev if hasattr(result, "njev") else 0,
        success=result.success,
        message=result.message,
        optimality=result.optimality if hasattr(result, "optimality") else 0.0,
        n_amplitude_params=n_amplitude_params,
    )


def fit_cluster_dict(
    cluster: "Cluster",
    noise: float,
    *,
    fixed: bool = False,
    params_init: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a single cluster using direct scipy optimization (dict interface).

    This function bypasses wrapper overhead by directly interfacing with
    scipy.optimize.least_squares, providing significant performance gains.

    Note: This function returns a dictionary for backward compatibility.
    For new code, prefer using fit_cluster which returns a FitResult.

    Args:
        cluster: Cluster to fit
        noise: Noise level (must be positive)
        fixed: Whether to fix positions during optimization
        params_init: Optional initial parameter values from previous fits

    Returns
    -------
        Dictionary with fitted parameter values and statistics:
            - params: Dict of parameter info (value, stderr, vary, min, max)
            - success: Whether optimization converged
            - chisqr: Chi-squared statistic
            - redchi: Reduced chi-squared
            - nfev: Number of function evaluations
            - message: Optimizer message

    Raises
    ------
        ScipyOptimizerError: If cluster has no peaks or invalid data
        ValueError: If noise is non-positive
    """
    # Validate inputs
    if noise <= 0:
        msg = f"Noise must be positive, got {noise}"
        raise ValueError(msg)

    if not cluster.peaks:
        msg = "Cluster has no peaks to fit"
        raise ScipyOptimizerError(msg)

    if not hasattr(cluster, "corrected_data") or cluster.corrected_data is None:
        msg = "Cluster has no data to fit"
        raise ScipyOptimizerError(msg)

    # Create parameters
    from peakfit.core.domain.peaks import create_params

    try:
        params = create_params(cluster.peaks, fixed=fixed)
    except (ValueError, TypeError) as e:
        # Create params may raise errors if peaks are invalid
        msg = f"Failed to create parameters: {e}"
        raise ScipyOptimizerError(msg) from e

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
        raise ScipyOptimizerError(msg)

    if np.any((x0 < lower) | (x0 > upper)):
        msg = "Initial values outside bounds"
        raise ScipyOptimizerError(msg)

    # Direct scipy optimization
    try:
        result = least_squares(
            compute_residuals,
            x0,
            jac=compute_jacobian,
            args=(names, params, cluster, noise),
            bounds=(lower, upper),
            method="trf",
            ftol=LEAST_SQUARES_FTOL,
            xtol=LEAST_SQUARES_XTOL,
            max_nfev=LEAST_SQUARES_MAX_NFEV,
            verbose=0,
        )
    except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as e:
        # SciPy may raise ValueError for invalid inputs, TypeError or RuntimeError,
        # or low-level NumPy linear algebra errors.
        msg = f"Optimization failed: {e}"
        raise ScipyOptimizerError(msg) from e

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
    residual = compute_residuals(result.x, names, params, cluster, noise)
    chisqr = compute_chi_squared(residual)
    ndata = len(residual)
    # Degrees of freedom must account for both nonlinearly optimized parameters
    # (nvarys) and analytically computed amplitudes (n_peaks * n_planes).
    # For linear least-squares: each amplitude per plane is a fitted parameter.
    nvarys = len(x0)
    n_peaks = len(cluster.peaks)
    n_planes = cluster.corrected_data.shape[0] if cluster.corrected_data.ndim > 1 else 1
    n_amplitude_params = n_peaks * n_planes
    n_total_fitted = nvarys + n_amplitude_params
    redchi = compute_reduced_chi_squared(chisqr, ndata, n_total_fitted)

    # Check for potential issues
    if redchi > 100:
        warnings.warn(
            f"Poor fit quality: reduced chi-squared = {redchi:.2f}",
            ConvergenceWarning,
            stacklevel=2,
        )

    # Compute standard errors from Jacobian
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
                # Singular matrix, can't compute errors
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


def fit_clusters(
    clusters: list["Cluster"],
    noise: float,
    refine_iterations: int = 1,
    *,
    fixed: bool = False,
    verbose: bool = False,
) -> Parameters:
    """Fit all clusters using direct scipy optimization.

    Args:
        clusters: List of clusters
        noise: Noise level
        refine_iterations: Number of refinement passes
        fixed: Whether to fix positions
        verbose: Print progress

    Returns
    -------
        Combined fitted parameters
    """
    from peakfit.core.fitting.computation import update_cluster_corrections

    params_all = Parameters()
    params_dict: dict[str, Any] = {}

    for iteration in range(refine_iterations + 1):
        # Update corrections if not first iteration
        if iteration > 0:
            update_cluster_corrections(params_all, clusters)

        # Fit all clusters sequentially
        for cluster in clusters:
            result = fit_cluster_dict(cluster, noise, fixed=fixed, params_init=params_dict)

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

    return params_all
