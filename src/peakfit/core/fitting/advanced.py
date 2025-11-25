"""Advanced optimization methods for NMR peak fitting.

Provides global optimization algorithms and improved uncertainty estimation
beyond basic least-squares fitting.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize

from peakfit.core.fitting.computation import residuals
from peakfit.core.fitting.parameters import Parameters
from peakfit.core.shared.constants import (
    BASIN_HOPPING_LOCAL_MAXITER,
    BASIN_HOPPING_NITER,
    BASIN_HOPPING_STEPSIZE,
    BASIN_HOPPING_TEMPERATURE,
    DIFF_EVOLUTION_MAXITER,
    DIFF_EVOLUTION_MUTATION,
    DIFF_EVOLUTION_POPSIZE,
    DIFF_EVOLUTION_RECOMBINATION,
    MCMC_N_STEPS,
    MCMC_N_WALKERS,
    PROFILE_LIKELIHOOD_DELTA_CHI2,
    PROFILE_LIKELIHOOD_NPOINTS,
)
from peakfit.core.shared.typing import FloatArray

if TYPE_CHECKING:
    from peakfit.core.domain.cluster import Cluster


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

    @property
    def chisqr(self) -> float:
        """Chi-squared value."""
        return float(np.sum(self.residual**2))

    @property
    def redchi(self) -> float:
        """Reduced chi-squared."""
        ndata = len(self.residual)
        nvarys = len(self.params.get_vary_names())
        if ndata > nvarys:
            return self.chisqr / (ndata - nvarys)
        return self.chisqr


@dataclass
class UncertaintyResult:
    """Comprehensive uncertainty estimates for fitted parameters."""

    parameter_names: list[str]
    values: FloatArray
    std_errors: FloatArray  # From covariance matrix
    confidence_intervals_68: FloatArray  # 68% CI (1 sigma)
    confidence_intervals_95: FloatArray  # 95% CI (2 sigma)
    correlation_matrix: FloatArray
    profile_likelihood_ci: FloatArray | None = None  # From profile likelihood
    mcmc_samples: FloatArray | None = None  # MCMC samples (flattened, post-burn-in)
    mcmc_percentiles: FloatArray | None = None  # 16th, 50th, 84th percentiles
    mcmc_chains: FloatArray | None = (
        None  # Full chains INCLUDING burn-in (n_walkers, n_steps_total, n_params)
    )
    mcmc_diagnostics: "ConvergenceDiagnostics | None" = (
        None  # Convergence diagnostics (computed on post-burn-in)
    )
    burn_in_info: dict | None = None  # Burn-in determination information


def residuals_global(x: FloatArray, params: Parameters, cluster: "Cluster", noise: float) -> float:
    """Compute sum of squared residuals for global optimization.

    Args:
        x: Parameter values (varying only)
        params: Parameters object
        cluster: Cluster to fit
        noise: Noise level

    Returns:
        Sum of squared residuals
    """
    params.set_vary_values(x)
    res = residuals(params, cluster, noise)
    return float(np.sum(res**2))


def fit_basin_hopping(
    params: Parameters,
    cluster: "Cluster",
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

    Returns:
        GlobalFitResult with optimized parameters
    """
    x0 = params.get_vary_values()
    bounds = params.get_vary_bounds_list()

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
    )


def fit_differential_evolution(
    params: Parameters,
    cluster: "Cluster",
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

    Returns:
        GlobalFitResult with optimized parameters
    """
    bounds = params.get_vary_bounds_list()

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
    )


def compute_profile_likelihood(
    params: Parameters,
    cluster: "Cluster",
    noise: float,
    param_name: str,
    n_points: int = PROFILE_LIKELIHOOD_NPOINTS,
    delta_chi2: float = PROFILE_LIKELIHOOD_DELTA_CHI2,  # 95% CI for 1 parameter
) -> tuple[FloatArray, FloatArray, tuple[float, float]]:
    """Compute profile likelihood confidence interval.

    Profile likelihood provides more accurate confidence intervals than
    the covariance matrix, especially for non-linear parameters.

    Args:
        params: Fitted parameters
        cluster: Cluster data
        noise: Noise level
        param_name: Parameter to profile
        n_points: Number of profile points
        delta_chi2: Chi-squared threshold (3.84 for 95% CI)

    Returns:
        Tuple of (parameter_values, chi_squared_values, confidence_interval)
    """
    # Get current best-fit chi-squared
    best_chi2 = float(np.sum(residuals(params, cluster, noise) ** 2))
    param = params[param_name]
    best_value = param.value

    # Determine profile range
    if param.stderr > 0:
        # Use 3 sigma range
        range_min = max(param.min, best_value - 3 * param.stderr)
        range_max = min(param.max, best_value + 3 * param.stderr)
    else:
        # Use 10% of allowed range
        span = param.max - param.min
        range_min = max(param.min, best_value - 0.1 * span)
        range_max = min(param.max, best_value + 0.1 * span)

    profile_values = np.linspace(range_min, range_max, n_points)
    chi2_values = np.zeros(n_points)

    # Compute profile
    for i, val in enumerate(profile_values):
        # Fix parameter at this value
        params_copy = params.copy()
        params_copy[param_name].value = val
        params_copy[param_name].vary = False

        # Re-optimize other parameters
        x0 = params_copy.get_vary_values()
        if len(x0) > 0:
            bounds_list = params_copy.get_vary_bounds_list()

            def objective(x: FloatArray, params: Parameters = params_copy) -> float:
                return residuals_global(x, params, cluster, noise)

            result = optimize.minimize(objective, x0, method="L-BFGS-B", bounds=bounds_list)
            params_copy.set_vary_values(result.x)

        chi2_values[i] = float(np.sum(residuals(params_copy, cluster, noise) ** 2))

    # Find confidence interval
    threshold = best_chi2 + delta_chi2
    below_threshold = chi2_values <= threshold

    if np.any(below_threshold):
        indices = np.where(below_threshold)[0]
        ci_low = profile_values[indices[0]]
        ci_high = profile_values[indices[-1]]
    else:
        ci_low = range_min
        ci_high = range_max

    return profile_values, chi2_values, (ci_low, ci_high)


def estimate_uncertainties_mcmc(
    params: Parameters,
    cluster: "Cluster",
    noise: float,
    n_walkers: int = MCMC_N_WALKERS,
    n_steps: int = MCMC_N_STEPS,
    burn_in: int | None = None,
) -> UncertaintyResult:
    """Estimate parameter uncertainties using MCMC sampling.

    Uses emcee for Markov Chain Monte Carlo sampling to get
    full posterior distributions for parameters.

    This function now computes comprehensive convergence diagnostics following
    the Bayesian Analysis Reporting Guidelines (BARG):
    - R-hat (Gelman-Rubin statistic) for convergence assessment
    - Effective Sample Size (ESS) for sample quality
    - Full chain data for diagnostic plotting
    - Adaptive burn-in determination using R-hat monitoring

    Args:
        params: Fitted parameters (starting point)
        cluster: Cluster data
        noise: Noise level
        n_walkers: Number of MCMC walkers/chains
        n_steps: Number of MCMC steps per walker
        burn_in: Steps to discard as burn-in. If None, automatically determined
            using R-hat convergence monitoring (recommended).

    Returns:
        UncertaintyResult with comprehensive uncertainty estimates and diagnostics

    Raises:
        ImportError: If emcee is not installed
    """
    try:
        import emcee
    except ImportError as e:
        msg = "emcee required for MCMC. Install with: pip install emcee"
        raise ImportError(msg) from e

    x0 = params.get_vary_values()
    bounds = params.get_vary_bounds_list()
    ndim = len(x0)

    # Log-likelihood function
    def log_likelihood(x: FloatArray) -> float:
        # Check bounds
        for i, (lb, ub) in enumerate(bounds):
            if not lb <= x[i] <= ub:
                return -np.inf

        params.set_vary_values(x)
        res = residuals(params, cluster, noise)
        return -0.5 * np.sum(res**2)

    # Initialize walkers near best-fit
    rng = np.random.default_rng()
    pos = x0 + 1e-4 * rng.standard_normal((n_walkers, ndim))

    # Clip to bounds
    for i, (lb, ub) in enumerate(bounds):
        pos[:, i] = np.clip(pos[:, i], lb + 1e-10, ub - 1e-10)

    # Run MCMC
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_likelihood)
    sampler.run_mcmc(pos, n_steps, progress=False)

    # Get FULL chains including burn-in for diagnostic plotting
    # emcee returns shape (n_steps, n_walkers, n_params)
    # Transpose to (n_walkers, n_steps, n_params) for our plotting functions
    chains_full = sampler.get_chain(discard=0, flat=False).swapaxes(0, 1)

    # Determine burn-in period (adaptive or manual)
    burn_in_info = {}
    if burn_in is None:
        # Automatic burn-in determination using R-hat monitoring
        from peakfit.core.diagnostics.burnin import determine_burnin_rhat, validate_burnin

        burn_in, burn_in_diagnostics = determine_burnin_rhat(
            chains_full,
            rhat_threshold=1.05,
            window_size=100,
            min_samples=100,
            check_interval=50,
        )

        # Validate burn-in and get warnings if needed
        is_valid, warning_msg = validate_burnin(burn_in, n_steps, max_fraction=0.5)

        burn_in_info = {
            "burn_in": burn_in,
            "method": "adaptive",
            "diagnostics": burn_in_diagnostics,
            "validation_warning": warning_msg,
        }
    else:
        # Manual burn-in specified
        from peakfit.core.diagnostics.burnin import validate_burnin

        is_valid, warning_msg = validate_burnin(burn_in, n_steps, max_fraction=0.5)

        burn_in_info = {
            "burn_in": burn_in,
            "method": "manual",
            "validation_warning": warning_msg,
        }

    # Get post-burn-in chains for convergence diagnostics
    # Shape after transpose: (n_walkers, n_steps_after_burnin, n_params)
    chains_post_burnin = sampler.get_chain(discard=burn_in, flat=False).swapaxes(0, 1)

    # Get flattened samples after burn-in for statistics
    samples = sampler.get_chain(discard=burn_in, flat=True)

    # Compute convergence diagnostics on post-burn-in samples
    from peakfit.core.diagnostics import diagnose_convergence

    diagnostics = diagnose_convergence(chains_post_burnin, params.get_vary_names())

    # Compute statistics
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    std_errors = np.std(samples, axis=0)

    # Confidence intervals
    ci_68 = np.array([[percentiles[0, i], percentiles[2, i]] for i in range(ndim)])
    ci_95 = np.array(
        [
            [np.percentile(samples[:, i], 2.5), np.percentile(samples[:, i], 97.5)]
            for i in range(ndim)
        ]
    )

    # Correlation matrix
    corr_matrix = np.corrcoef(samples.T)

    # Update parameter errors
    params.set_errors(std_errors)

    return UncertaintyResult(
        parameter_names=params.get_vary_names(),
        values=percentiles[1],  # Median
        std_errors=std_errors,
        confidence_intervals_68=ci_68,
        confidence_intervals_95=ci_95,
        correlation_matrix=corr_matrix,
        mcmc_samples=samples,
        mcmc_percentiles=percentiles,
        mcmc_chains=chains_full,  # Full chains including burn-in for plotting
        mcmc_diagnostics=diagnostics,
        burn_in_info=burn_in_info,
    )


def _compute_numerical_hessian(
    func: callable,
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

    Returns:
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
