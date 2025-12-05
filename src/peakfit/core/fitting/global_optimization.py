"""Global optimization methods for NMR peak fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy import optimize

from peakfit.core.fitting.computation import residuals
from peakfit.core.fitting.parameters import Parameters  # noqa: TC001
from peakfit.core.results.statistics import (
    compute_chi_squared,
    compute_reduced_chi_squared,
)
from peakfit.core.shared.constants import (
    BASIN_HOPPING_LOCAL_MAXITER,
    BASIN_HOPPING_NITER,
    BASIN_HOPPING_STEPSIZE,
    BASIN_HOPPING_TEMPERATURE,
    DIFF_EVOLUTION_MAXITER,
    DIFF_EVOLUTION_MUTATION,
    DIFF_EVOLUTION_POPSIZE,
    DIFF_EVOLUTION_RECOMBINATION,
)

if TYPE_CHECKING:
    from peakfit.core.domain.cluster import Cluster
    from peakfit.core.shared.typing import FloatArray


@dataclass
class GlobalFitResult:
    """Extended fit result with global optimization info."""

    params: Parameters
    residual: FloatArray
    cost: float
    nfev: int
    success: bool
    message: str
    global_iterations: int = 0
    local_minimizations: int = 0
    global_minimum_found: bool = False
    basin_hopping_temperature: float = 1.0
    covar: FloatArray | None = None
    n_amplitude_params: int = 0  # Number of analytically computed amplitude parameters

    @property
    def chisqr(self) -> float:
        """Chi-squared value."""
        return compute_chi_squared(self.residual)

    @property
    def redchi(self) -> float:
        """Reduced chi-squared.

        Degrees of freedom includes both nonlinearly optimized parameters
        (vary=True) and analytically computed amplitude parameters.
        """
        ndata = len(self.residual)
        nvarys = len(self.params.get_vary_names())
        n_total_fitted = nvarys + self.n_amplitude_params
        return compute_reduced_chi_squared(self.chisqr, ndata, n_total_fitted)


def residuals_global(x: FloatArray, params: Parameters, cluster: Cluster, noise: float) -> float:
    """Compute sum of squared residuals for global optimization.

    Args:
        x: Parameter values (varying only)
        params: Parameters object
        cluster: Cluster to fit
        noise: Noise level

    Returns
    -------
        Sum of squared residuals
    """
    params.set_vary_values(x)
    res = residuals(params, cluster, noise)
    return float(np.sum(res**2))


def _compute_numerical_hessian(
    func: Callable[[FloatArray], float],
    x: FloatArray,
    bounds: list[tuple[float, float]],
    epsilon: float = 1e-8,
) -> FloatArray:
    """Compute numerical Hessian matrix.

    Args:
        func: Objective function
        x: Parameter values
        bounds: Parameter bounds
        epsilon: Finite difference step

    Returns
    -------
        Hessian matrix
    """
    n = len(x)
    hessian = np.zeros((n, n))

    f0 = func(x)

    for i in range(n):
        # Compute step size respecting bounds
        hi = epsilon * max(1.0, abs(x[i]))
        if x[i] + hi > bounds[i][1]:
            hi = -hi
        if x[i] + hi < bounds[i][0]:
            hi = epsilon  # Try positive direction

        xi_plus = x.copy()
        xi_plus[i] += hi
        fi_plus = func(xi_plus)

        xi_minus = x.copy()
        xi_minus[i] -= hi
        fi_minus = func(xi_minus)

        # Diagonal element (second derivative)
        hessian[i, i] = (fi_plus - 2 * f0 + fi_minus) / hi**2

        # Off-diagonal elements
        for j in range(i + 1, n):
            hj = epsilon * max(1.0, abs(x[j]))
            if x[j] + hj > bounds[j][1]:
                hj = -hj
            if x[j] + hj < bounds[j][0]:
                hj = epsilon

            xij_pp = x.copy()
            xij_pp[i] += hi
            xij_pp[j] += hj

            xij_pm = x.copy()
            xij_pm[i] += hi
            xij_pm[j] -= hj

            xij_mp = x.copy()
            xij_mp[i] -= hi
            xij_mp[j] += hj

            xij_mm = x.copy()
            xij_mm[i] -= hi
            xij_mm[j] -= hj

            fij_pp = func(xij_pp)
            fij_pm = func(xij_pm)
            fij_mp = func(xij_mp)
            fij_mm = func(xij_mm)

            hessian[i, j] = (fij_pp - fij_pm - fij_mp + fij_mm) / (4 * hi * hj)
            hessian[j, i] = hessian[i, j]

    return hessian


def fit_basin_hopping(
    params: Parameters,
    cluster: Cluster,
    noise: float,
    n_iterations: int = BASIN_HOPPING_NITER,
    temperature: float = BASIN_HOPPING_TEMPERATURE,
    step_size: float = BASIN_HOPPING_STEPSIZE,
) -> GlobalFitResult:
    """Fit cluster using basin-hopping global optimization.

    Basin-hopping is effective for avoiding local minima in complex
    fitting problems with multiple overlapping peaks.

    Args:
        params: Initial parameters
        cluster: Cluster to fit
        noise: Noise level
        n_iterations: Number of basin-hopping iterations
        temperature: Temperature parameter for acceptance
        step_size: Step size for random perturbations

    Returns
    -------
        GlobalFitResult with optimized parameters
    """
    x0 = params.get_vary_values()
    bounds = params.get_vary_bounds_list()

    # Calculate number of amplitude parameters for DOF
    # Calculate number of amplitude parameters for DOF
    n_amplitude_params = cluster.n_amplitude_params

    # Objective function
    def objective(x: FloatArray) -> float:
        return residuals_global(x, params, cluster, noise)

    # Custom step-taking that respects bounds
    class BoundedStep:
        def __init__(self, step_size: float) -> None:
            self.step_size = step_size

        def __call__(self, x: FloatArray) -> FloatArray:
            rng = np.random.default_rng()
            x_new = x + rng.uniform(-self.step_size, self.step_size, len(x))
            # Clip to bounds
            for i, (lb, ub) in enumerate(bounds):
                x_new[i] = np.clip(x_new[i], lb, ub)
            return x_new

    # Run basin-hopping
    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": bounds,
        "options": {"maxiter": BASIN_HOPPING_LOCAL_MAXITER},
    }

    result = optimize.basinhopping(
        objective,
        x0,
        niter=n_iterations,
        T=temperature,
        take_step=BoundedStep(step_size),
        minimizer_kwargs=minimizer_kwargs,
        disp=False,
    )

    # Update parameters with result
    params.set_vary_values(result.x)

    # Compute final statistics
    final_residuals = residuals(params, cluster, noise)

    # Compute covariance from final local minimum
    covar = None
    try:
        # Use numerical Hessian for covariance
        hessian = _compute_numerical_hessian(objective, result.x, bounds)
        covar = np.linalg.inv(hessian) * 2.0  # Factor of 2 for least-squares
        std_errors = np.sqrt(np.diag(covar))
        params.set_errors(std_errors)
    except (np.linalg.LinAlgError, ValueError):
        pass

    return GlobalFitResult(
        params=params,
        residual=final_residuals,
        cost=result.fun,
        nfev=result.nfev,
        success=result.lowest_optimization_result.success,
        message=str(result.message),
        global_iterations=n_iterations,
        local_minimizations=result.nit,
        global_minimum_found=result.lowest_optimization_result.success,
        basin_hopping_temperature=temperature,
        covar=covar,
        n_amplitude_params=n_amplitude_params,
    )


def fit_differential_evolution(
    params: Parameters,
    cluster: Cluster,
    noise: float,
    max_iterations: int = DIFF_EVOLUTION_MAXITER,
    population_size: int = DIFF_EVOLUTION_POPSIZE,
    mutation: tuple[float, float] = DIFF_EVOLUTION_MUTATION,
    recombination: float = DIFF_EVOLUTION_RECOMBINATION,
    polish: bool = True,
) -> GlobalFitResult:
    """Fit cluster using differential evolution.

    Differential evolution is a population-based optimizer that's
    good for finding global optima in high-dimensional spaces.

    Args:
        params: Initial parameters
        cluster: Cluster to fit
        noise: Noise level
        max_iterations: Maximum generations
        population_size: Population multiplier
        mutation: Mutation constant range
        recombination: Recombination constant
        polish: Whether to polish with L-BFGS-B

    Returns
    -------
        GlobalFitResult with optimized parameters
    """
    bounds = params.get_vary_bounds_list()

    # Calculate number of amplitude parameters for DOF
    # Calculate number of amplitude parameters for DOF
    n_amplitude_params = cluster.n_amplitude_params

    def objective(x: FloatArray) -> float:
        return residuals_global(x, params, cluster, noise)

    result = optimize.differential_evolution(
        objective,
        bounds,
        maxiter=max_iterations,
        popsize=population_size,
        mutation=mutation,
        recombination=recombination,
        polish=polish,
        disp=False,
        workers=1,  # Single worker for now
    )

    # Update parameters
    params.set_vary_values(result.x)

    # Compute statistics
    final_residuals = residuals(params, cluster, noise)

    # Compute covariance
    covar = None
    try:
        hessian = _compute_numerical_hessian(objective, result.x, bounds)
        covar = np.linalg.inv(hessian) * 2.0
        std_errors = np.sqrt(np.diag(covar))
        params.set_errors(std_errors)
    except (np.linalg.LinAlgError, ValueError):
        pass

    return GlobalFitResult(
        params=params,
        residual=final_residuals,
        cost=result.fun,
        nfev=result.nfev,
        success=result.success,
        message=result.message,
        global_iterations=result.nit,
        local_minimizations=1 if polish else 0,
        global_minimum_found=result.success,
        covar=covar,
        n_amplitude_params=n_amplitude_params,
    )
