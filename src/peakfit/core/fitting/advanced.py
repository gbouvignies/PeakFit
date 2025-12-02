"""Advanced optimization methods for NMR peak fitting.

Provides global optimization algorithms and improved uncertainty estimation
beyond basic least-squares fitting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Pool
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize

from peakfit.core.fitting.computation import residuals
from peakfit.core.fitting.parameters import Parameters
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
    MCMC_N_STEPS,
    MCMC_N_WALKERS,
    PROFILE_LIKELIHOOD_DELTA_CHI2,
    PROFILE_LIKELIHOOD_NPOINTS,
)
from peakfit.core.shared.typing import FloatArray

if TYPE_CHECKING:
    from peakfit.core.diagnostics.convergence import ConvergenceDiagnostics
    from peakfit.core.domain.cluster import Cluster


# Module-level globals for MCMC parallelization
# These are set before running MCMC and used by the log-likelihood function
# This pattern avoids pickling large data structures on each likelihood call
# (see https://emcee.readthedocs.io/en/stable/tutorials/parallel/)
_mcmc_params: Parameters | None = None
_mcmc_cluster: Cluster | None = None
_mcmc_noise: float = 0.0
_mcmc_bounds: list[tuple[float, float]] = []

# Module-level globals for amplitude computation parallelization
# Same pattern as MCMC - avoids pickling overhead
_amp_params: Parameters | None = None
_amp_cluster: Cluster | None = None


def _init_mcmc_worker(
    params: Parameters,
    cluster: Cluster,
    noise: float,
    bounds: list[tuple[float, float]],
) -> None:
    """Initialize globals in worker processes for MCMC parallelization.

    This function is called once per worker process when the Pool is created.
    It sets the module-level globals that _log_likelihood_global needs.
    """
    global _mcmc_params, _mcmc_cluster, _mcmc_noise, _mcmc_bounds
    _mcmc_params = params
    _mcmc_cluster = cluster
    _mcmc_noise = noise
    _mcmc_bounds = bounds


def _log_likelihood_global(x: FloatArray) -> float:
    """Log-likelihood function for MCMC using global state.

    This function uses module-level globals to avoid pickling overhead
    when using multiprocessing with emcee.
    """
    global _mcmc_params, _mcmc_cluster, _mcmc_noise, _mcmc_bounds

    if _mcmc_params is None or _mcmc_cluster is None:
        return float(-np.inf)

    # Check bounds
    for i, (lb, ub) in enumerate(_mcmc_bounds):
        if not lb <= x[i] <= ub:
            return float(-np.inf)

    # Create a copy of params for thread safety
    params_copy = _mcmc_params.copy()
    params_copy.set_vary_values(x)
    res = residuals(params_copy, _mcmc_cluster, _mcmc_noise)
    return float(-0.5 * np.sum(res**2))


def _compute_amplitudes_for_sample(
    sample: FloatArray,
    params: Parameters,
    cluster: Cluster,
) -> FloatArray:
    """Compute amplitudes for a single MCMC sample.

    Args:
        sample: Parameter values for this sample
        params: Parameters object (will be modified)
        cluster: Cluster data

    Returns
    -------
        Flattened amplitudes array
    """
    from peakfit.core.fitting.computation import calculate_shape_heights

    params.set_vary_values(sample)
    _shapes, amps = calculate_shape_heights(params, cluster)
    return amps.ravel()


def _init_amp_worker(params: Parameters, cluster: Cluster) -> None:
    """Initialize globals in worker processes for amplitude computation.

    This function is called once per worker process when the Pool is created.
    It sets the module-level globals that _compute_amplitudes_for_sample_global needs.

    Following emcee best practices: data is passed once at pool creation,
    not pickled on every function call.
    """
    global _amp_params, _amp_cluster
    _amp_params = params
    _amp_cluster = cluster


def _compute_amplitudes_for_sample_global(sample: FloatArray) -> FloatArray:
    """Compute amplitudes for a single MCMC sample using global state.

    This function uses module-level globals to avoid pickling overhead
    when using multiprocessing. The pattern follows emcee's recommendation:
    https://emcee.readthedocs.io/en/stable/tutorials/parallel/

    Args:
        sample: Parameter values for this sample

    Returns
    -------
        Flattened amplitudes array
    """
    global _amp_params, _amp_cluster
    from peakfit.core.fitting.computation import calculate_shape_heights

    if _amp_params is None or _amp_cluster is None:
        return np.array([])

    # Create a copy of params for thread safety
    params_copy = _amp_params.copy()
    params_copy.set_vary_values(sample)
    _shapes, amps = calculate_shape_heights(params_copy, _amp_cluster)
    return amps.ravel()


def _compute_amplitude_chains_parallel(
    chains: FloatArray,
    params: Parameters,
    cluster: Cluster,
    n_amp_params: int,
    workers: int,
) -> FloatArray:
    """Compute amplitude chains for all MCMC samples, optionally in parallel.

    This is a performance-critical function that processes all MCMC samples
    to compute amplitudes via linear least-squares. For large chains
    (e.g., 64 walkers × 5000 steps = 320,000 samples), this can be slow.

    Args:
        chains: MCMC chains, shape (n_walkers, n_steps, n_lineshape_params)
        params: Parameters object (template)
        cluster: Cluster data
        n_amp_params: Number of amplitude parameters per sample
        workers: Number of parallel workers (1 = sequential, -1 = all CPUs)

    Returns
    -------
        Amplitude chains, shape (n_walkers, n_steps, n_amp_params)
    """
    n_walkers, n_steps, _ = chains.shape

    # Flatten chains for easier processing
    flat_chains = chains.reshape(-1, chains.shape[-1])
    n_samples = flat_chains.shape[0]

    if workers == 1:
        # Sequential processing - simple loop
        amp_flat = np.zeros((n_samples, n_amp_params))
        params_copy = params.copy()
        for i in range(n_samples):
            amp_flat[i] = _compute_amplitudes_for_sample(flat_chains[i], params_copy, cluster)
    else:
        # Parallel processing using multiprocessing.Pool with initializer
        # Following emcee best practices: data is passed once via initializer,
        # not pickled on every function call
        # (see https://emcee.readthedocs.io/en/stable/tutorials/parallel/)
        import os

        from threadpoolctl import threadpool_limits

        n_processes = workers if workers > 0 else os.cpu_count() or 1

        # Limit BLAS threads to 1 during parallel execution to avoid
        # oversubscription (each worker would spawn its own BLAS threads)
        with (
            threadpool_limits(limits=1, user_api="blas"),
            Pool(
                processes=n_processes,
                initializer=_init_amp_worker,
                initargs=(params.copy(), cluster),
            ) as pool,
        ):
            results = pool.map(_compute_amplitudes_for_sample_global, flat_chains)
        amp_flat = np.array(results)

    # Reshape back to (n_walkers, n_steps, n_amp_params)
    return amp_flat.reshape(n_walkers, n_steps, n_amp_params)


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


@dataclass
class UncertaintyResult:
    """Comprehensive uncertainty estimates for fitted parameters.

    All parameters (lineshape and amplitudes) are treated uniformly.
    Amplitudes are computed via linear least-squares for each MCMC sample
    and included alongside lineshape parameters in the chains and statistics.
    """

    # Combined parameter names: lineshape params + amplitude params ({peak}.I[plane])
    parameter_names: list[str]
    values: FloatArray  # Best-fit values for all parameters
    std_errors: FloatArray  # Standard errors for all parameters
    confidence_intervals_68: FloatArray  # 68% CI (1 sigma) for all parameters
    confidence_intervals_95: FloatArray  # 95% CI (2 sigma) for all parameters
    correlation_matrix: FloatArray  # Correlation matrix (lineshape params only)

    # MCMC chain data - includes both lineshape and amplitude parameters
    mcmc_samples: FloatArray | None = None  # Flattened samples (n_samples, n_all_params)
    mcmc_percentiles: FloatArray | None = None  # 16th, 50th, 84th percentiles
    mcmc_chains: FloatArray | None = None  # Full chains (n_walkers, n_steps, n_all_params)
    mcmc_diagnostics: ConvergenceDiagnostics | None = None  # Convergence diagnostics
    burn_in_info: dict | None = None  # Burn-in determination information

    # Metadata for distinguishing parameter types
    n_lineshape_params: int = 0  # Number of lineshape parameters
    amplitude_names: list[str] | None = None  # Peak names (for grouping amplitudes)
    n_planes: int = 1  # Number of planes (for amplitude indexing)

    # Legacy fields for profile likelihood (not affected by this refactoring)
    profile_likelihood_ci: FloatArray | None = None


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
    n_peaks = len(cluster.peaks)
    n_planes = cluster.corrected_data.shape[0] if cluster.corrected_data.ndim > 1 else 1
    n_amplitude_params = n_peaks * n_planes

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
    n_peaks = len(cluster.peaks)
    n_planes = cluster.corrected_data.shape[0] if cluster.corrected_data.ndim > 1 else 1
    n_amplitude_params = n_peaks * n_planes

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


def compute_profile_likelihood(
    params: Parameters,
    cluster: Cluster,
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

    Returns
    -------
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
        ci_low = float(profile_values[indices[0]])
        ci_high = float(profile_values[indices[-1]])
    else:
        ci_low = range_min
        ci_high = range_max

    return profile_values, chi2_values, (ci_low, ci_high)


def estimate_uncertainties_mcmc(
    params: Parameters,
    cluster: Cluster,
    noise: float,
    n_walkers: int = MCMC_N_WALKERS,
    n_steps: int = MCMC_N_STEPS,
    burn_in: int | None = None,
    workers: int = 1,
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
        workers: Number of parallel workers for likelihood evaluation.
            Use 1 for sequential (default), -1 for all CPUs.

    Returns
    -------
        UncertaintyResult with comprehensive uncertainty estimates and diagnostics

    """
    global _mcmc_params, _mcmc_cluster, _mcmc_noise, _mcmc_bounds

    import emcee

    x0 = params.get_vary_values()
    bounds = params.get_vary_bounds_list()
    ndim = len(x0)

    # Initialize walkers near best-fit
    rng = np.random.default_rng()
    pos = x0 + 1e-4 * rng.standard_normal((n_walkers, ndim))

    # Clip to bounds
    for i, (lb, ub) in enumerate(bounds):
        pos[:, i] = np.clip(pos[:, i], lb + 1e-10, ub - 1e-10)

    # Determine if parallel execution should be used
    use_parallel = workers != 1

    if use_parallel:
        import os

        from threadpoolctl import threadpool_limits

        # Use Pool initializer to set globals in worker processes
        # This pattern ensures each worker has access to the shared data
        # (see https://emcee.readthedocs.io/en/stable/tutorials/parallel/)
        n_processes = workers if workers > 0 else os.cpu_count() or 1

        # Limit BLAS threads to 1 during parallel execution to avoid
        # oversubscription (each worker would spawn its own BLAS threads)
        with (
            threadpool_limits(limits=1, user_api="blas"),
            Pool(
                processes=n_processes,
                initializer=_init_mcmc_worker,
                initargs=(params.copy(), cluster, noise, bounds),
            ) as pool,
        ):
            sampler = emcee.EnsembleSampler(n_walkers, ndim, _log_likelihood_global, pool=pool)
            sampler.run_mcmc(pos, n_steps, progress=False)
    else:
        # Sequential execution with local closure (simpler, no globals needed)
        def log_likelihood(x: FloatArray) -> float:
            # Check bounds
            for i, (lb, ub) in enumerate(bounds):
                if not lb <= x[i] <= ub:
                    return float(-np.inf)

            params.set_vary_values(x)
            res = residuals(params, cluster, noise)
            return float(-0.5 * np.sum(res**2))

        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_likelihood)
        sampler.run_mcmc(pos, n_steps, progress=False)

    # Get FULL chains including burn-in for diagnostic plotting
    # emcee returns shape (n_steps, n_walkers, n_params)
    # Transpose to (n_walkers, n_steps, n_params) for our plotting functions
    chains_full = np.asarray(sampler.get_chain(discard=0, flat=False))
    if chains_full.ndim == 3:
        chains_full = chains_full.swapaxes(0, 1)

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
        _, warning_msg = validate_burnin(burn_in, n_steps, max_fraction=0.5)

        burn_in_info = {
            "burn_in": burn_in,
            "method": "adaptive",
            "diagnostics": burn_in_diagnostics,
            "validation_warning": warning_msg,
        }
    else:
        # Manual burn-in specified
        from peakfit.core.diagnostics.burnin import validate_burnin

        _is_valid, warning_msg = validate_burnin(burn_in, n_steps, max_fraction=0.5)

        burn_in_info = {
            "burn_in": burn_in,
            "method": "manual",
            "validation_warning": warning_msg,
        }

    # Get post-burn-in chains for convergence diagnostics
    # Shape after transpose: (n_walkers, n_steps_after_burnin, n_params)
    chains_post_burnin = np.asarray(sampler.get_chain(discard=burn_in, flat=False))
    if chains_post_burnin.ndim == 3:
        chains_post_burnin = chains_post_burnin.swapaxes(0, 1)

    # Get flattened samples after burn-in for statistics
    lineshape_samples = np.asarray(sampler.get_chain(discard=burn_in, flat=True))

    lineshape_names = params.get_vary_names()

    # Correlation matrix for lineshape parameters only
    # (amplitudes are conditionally independent given lineshape params)
    corr_matrix = np.corrcoef(lineshape_samples.T)

    # Compute amplitudes for each MCMC sample (via fast linear least-squares)
    n_peaks = len(cluster.peaks)
    n_planes = cluster.corrected_data.shape[1] if cluster.corrected_data.ndim > 1 else 1
    _n_walkers_chain, _n_steps_chain, n_lineshape = chains_full.shape
    n_amp_params = n_peaks * n_planes

    # Compute amplitude chains: shape (n_walkers, n_steps, n_peaks * n_planes)
    # Optimize by flattening and using parallel processing
    amp_chains = _compute_amplitude_chains_parallel(
        chains_full, params, cluster, n_amp_params, workers
    )

    # Generate amplitude parameter names using ParameterId for consistency
    from peakfit.core.fitting.parameters import PSEUDO_AXIS, ParameterId

    amp_names = [
        ParameterId.amplitude(peak.name, PSEUDO_AXIS, i_plane).name
        for peak in cluster.peaks
        for i_plane in range(n_planes)
    ]
    amplitude_peak_names = [p.name for p in cluster.peaks]

    # Combine lineshape and amplitude chains into unified chains
    # Shape: (n_walkers, n_steps, n_lineshape + n_amp_params)
    combined_chains = np.concatenate([chains_full, amp_chains], axis=2)
    combined_names = lineshape_names + amp_names

    # Combined chains post burn-in for diagnostics
    combined_chains_post_burnin = combined_chains[:, burn_in:, :]

    # Compute convergence diagnostics on ALL parameters (lineshape + amplitudes)
    from peakfit.core.diagnostics import diagnose_convergence

    diagnostics = diagnose_convergence(combined_chains_post_burnin, combined_names)

    # Flatten combined samples for statistics (post burn-in)
    combined_samples = combined_chains_post_burnin.reshape(-1, len(combined_names))

    # Compute unified statistics for all parameters
    percentiles = np.percentile(combined_samples, [16, 50, 84], axis=0)
    std_errors = np.std(combined_samples, axis=0)

    n_all_params = len(combined_names)
    ci_68 = np.array([[percentiles[0, i], percentiles[2, i]] for i in range(n_all_params)])
    ci_95 = np.array(
        [
            [
                np.percentile(combined_samples[:, i], 2.5),
                np.percentile(combined_samples[:, i], 97.5),
            ]
            for i in range(n_all_params)
        ]
    )

    # Update lineshape parameter errors
    params.set_errors(std_errors[:n_lineshape])

    # Restore best-fit values for lineshape params
    params.set_vary_values(percentiles[1, :n_lineshape])

    # Inject amplitude parameters as computed parameters
    # This allows uniform treatment in statistics and reporting
    # Note: We use explicit bounds (-inf, inf) because MCMC-derived amplitudes
    # can be negative (e.g., in CEST experiments with inverted signals)
    from peakfit.core.fitting.parameters import ParameterType

    for i, amp_name in enumerate(amp_names):
        params.add(
            amp_name,
            value=float(percentiles[1, n_lineshape + i]),  # Median value
            min=-np.inf,  # Allow negative amplitudes from MCMC
            max=np.inf,
            vary=False,
            param_type=ParameterType.AMPLITUDE,
            computed=True,
        )
        params[amp_name].stderr = float(std_errors[n_lineshape + i])

    return UncertaintyResult(
        parameter_names=combined_names,
        values=percentiles[1],  # Median for all params
        std_errors=std_errors,
        confidence_intervals_68=ci_68,
        confidence_intervals_95=ci_95,
        correlation_matrix=corr_matrix,  # Lineshape params only
        mcmc_samples=combined_samples,
        mcmc_percentiles=percentiles,
        mcmc_chains=combined_chains,  # Unified chains
        mcmc_diagnostics=diagnostics,  # Diagnostics for ALL parameters
        burn_in_info=burn_in_info,
        # Metadata for parameter type distinction
        n_lineshape_params=n_lineshape,
        amplitude_names=amplitude_peak_names,
        n_planes=n_planes,
    )


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
