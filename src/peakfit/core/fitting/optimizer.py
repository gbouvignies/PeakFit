"""Least-squares optimization for NMR peak fitting.

This module provides fitting functions that directly interface with
scipy.optimize.least_squares for efficient parameter optimization.
"""

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from peakfit.core.fitting.computation import residuals
from peakfit.core.fitting.jacobian import compute_jacobian
from peakfit.core.fitting.parameters import Parameters
from peakfit.core.fitting.results import FitResult

if TYPE_CHECKING:
    from peakfit.core.domain.cluster import Cluster


class ScipyOptimizerError(Exception):
    """Exception raised for errors in scipy optimization."""


def compute_residuals(
    x: np.ndarray,
    names: list[str],
    params_template: Parameters,
    cluster: "Cluster",
    noise: float,
) -> np.ndarray:
    """Compute residuals for scipy.optimize.least_squares.

    Args:
        x: Current parameter values
        names: Parameter names
        params_template: Template Parameters
        cluster: Cluster being fit
        noise: Noise level

    Returns
    -------
    np.ndarray
        Residual vector normalized by noise
    """
    # Update parameters with current values
    for i, name in enumerate(names):
        params_template[name].value = x[i]

    return residuals(params_template, cluster, noise)


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
        verbose: Verbosity level

    Returns
    -------
    FitResult
        FitResult containing optimized parameters and fit statistics
    """
    if noise <= 0:
        raise ValueError(f"Noise must be positive, got {noise}")

    if not cluster.peaks:
        raise ScipyOptimizerError("Cluster has no peaks to fit")

    # Get initial values and bounds
    x0 = params.get_vary_values()
    lower, upper = params.get_vary_bounds()
    vary_names = params.get_vary_names()

    # Calculate number of amplitude parameters for DOF
    n_amplitude_params = cluster.n_amplitude_params

    # Run optimization
    result = least_squares(
        compute_residuals,
        x0,
        jac=compute_jacobian,
        args=(vary_names, params, cluster, noise),
        bounds=(lower, upper),
        method="trf",
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
        verbose=verbose,
    )

    # Update parameters
    params.set_vary_values(result.x)

    return FitResult(
        params=params,
        residual=result.fun,
        cost=result.cost,
        nfev=result.nfev,
        njev=getattr(result, "njev", 0),
        success=result.success,
        message=result.message,
        optimality=getattr(result, "optimality", 0.0),
        n_amplitude_params=n_amplitude_params,
    )


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
    Parameters
        Combined fitted parameters
    """
    from peakfit.core.domain.peaks import create_params
    from peakfit.core.fitting.computation import update_cluster_corrections

    params_all = Parameters()

    # Initialize with all parameters
    for cluster in clusters:
        try:
            cluster_params = create_params(cluster.peaks, fixed=fixed)
            for name, param in cluster_params.items():
                params_all.add(
                    name, value=param.value, vary=param.vary, min=param.min, max=param.max
                )
        except (ValueError, TypeError):
            continue

    for iteration in range(refine_iterations + 1):
        if iteration > 0:
            update_cluster_corrections(params_all, clusters)

        for cluster in clusters:
            try:
                # Synchronize current global parameters to cluster
                cluster_params = create_params(cluster.peaks, fixed=fixed)
                for name in cluster_params:
                    if name in params_all:
                        cluster_params[name].value = params_all[name].value

                result = fit_cluster(cluster_params, cluster, noise)

                # Update global parameters from result
                for name, param in result.params.items():
                    target = (
                        params_all[name]
                        if name in params_all
                        else params_all.add(name, value=param.value)
                    )

                    target.value = param.value
                    target.stderr = param.stderr
                    target.vary = param.vary
                    target.min = param.min
                    target.max = param.max
                    target.computed = param.computed

            except ScipyOptimizerError:
                if verbose:
                    print("Skipping cluster with error")
                continue

    return params_all
