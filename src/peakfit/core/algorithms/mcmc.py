r"""Bayesian uncertainty estimation using Affine Invariant MCMC.

This module implements Markov Chain Monte Carlo (MCMC) sampling to estimate
parameter uncertainties and posterior distributions. It uses the `emcee` library,
which implements the Affine Invariant Ensemble Sampler (Goodman & Weare).

Sampling Strategy
-----------------
*   **Nonlinear Parameters** (positions, linewidths): Sampled directly using MCMC.
    *   **Priors**: Uniform (Flat) within the bounds defined in `Parameters`.
    *   **Likelihood**: Gaussian, assuming independent identically distributed (i.i.d.) noise.
        $ \ln L \propto -\frac{1}{2} \sum (y_{obs} - y_{model})^2 $

*   **Linear Parameters** (amplitudes): Solved marginal optimization.
    *   For each MCMC sample of nonlinear parameters, amplitudes are computed analytically
        using Linear Least Squares ($R\alpha = Q^T y$).
    *   This is known as "concentrating out" the linear parameters, effectively sampling
        from the profile likelihood of the nonlinear parameters.

This hybrid approach drastically reduces the dimensionality of the sampling space
(by $N_{peaks} \times N_{planes}$), improving convergence and mixing.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
import os
from typing import TYPE_CHECKING, Any

import emcee
import numpy as np

from peakfit.core.fitting.computation import residuals
from peakfit.core.fitting.parameters import Parameters  # noqa: TC001
from peakfit.core.shared.constants import MCMC_N_STEPS, MCMC_N_WALKERS

if TYPE_CHECKING:
    from collections.abc import Callable

    from peakfit.core.diagnostics.convergence import ConvergenceDiagnostics
    from peakfit.core.domain.cluster import Cluster
    from peakfit.core.shared.typing import FloatArray


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


@dataclass
class UncertaintyResult:
    """Comprehensive uncertainty estimates for fitted parameters.

    Results include both sampled nonlinear parameters and analytically projected
    amplitudes, providing a unified view of the posterior.

    Attributes
    ----------
    parameter_names : list[str]
        Names of all parameters (lineshape + amplitudes).
    values : FloatArray
        Best-fit values (median of posterior) for all parameters.
    std_errors : FloatArray
        Standard errors (std dev of posterior) for all parameters.
    confidence_intervals_68 : FloatArray
        68% Credible Intervals (16th-84th percentile), shape (n_params, 2).
    confidence_intervals_95 : FloatArray
        95% Credible Intervals (2.5th-97.5th percentile), shape (n_params, 2).
    correlation_matrix : FloatArray
        Correlation matrix for separate lineshape parameters.
    mcmc_samples : FloatArray | None
        Flattened posterior samples. Shape: `(n_samples, n_params)`.
    mcmc_percentiles : FloatArray | None
        16th, 50th, 84th percentiles. Shape: `(3, n_params)`.
    mcmc_chains : FloatArray | None
        Full MCMC chains. Shape: `(n_walkers, n_steps, n_params)`.
    mcmc_diagnostics : ConvergenceDiagnostics | None
        Convergence metrics (R-hat, ESS) for all parameters.
    burn_in_info : dict[str, Any] | None
        Metadata about the burn-in phase (e.g., number of steps discarded).
    n_lineshape_params : int
        Count of nonlinear parameters sampled directly.
    amplitude_names : list[str] | None
        Names of peaks (used for grouping amplitudes).
    n_planes : int
        Number of spectral planes (used for amplitude indexing).
    profile_likelihood_ci : FloatArray | None
        Legacy field for profile likelihood results.
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
    burn_in_info: dict[str, Any] | None = None  # Burn-in determination information

    # Metadata for distinguishing parameter types
    n_lineshape_params: int = 0  # Number of lineshape parameters
    amplitude_names: list[str] | None = None  # Peak names (for grouping amplitudes)
    n_planes: int = 1  # Number of planes (for amplitude indexing)

    # Legacy fields for profile likelihood (not affected by this refactoring)
    profile_likelihood_ci: FloatArray | None = None


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


def estimate_uncertainties_mcmc(
    params: Parameters,
    cluster: Cluster,
    noise: float,
    n_walkers: int = MCMC_N_WALKERS,
    n_steps: int = MCMC_N_STEPS,
    burn_in: int | None = None,
    workers: int = 1,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> UncertaintyResult:
    r"""Estimate parameter uncertainties using Affine Invariant MCMC (emcee).

    Samples the posterior distribution of the nonlinear parameters (positions,
    linewidths) while treating amplitudes as nuisance parameters that are
    marginalized out (via analytical optimization at each step).

    Priors
    ------
    - **Nonlinear Parameters**: Uniform (Flat) distribution within the min/max
        bounds specified in the `params` object.
    - **Amplitudes**: Uninformative (Uniform on $(-\infty, \infty)$), implied
        by the linear least squares solution.

    Parameters
    ----------
    params : Parameters
        Fitted parameters serving as the starting point (mean of the initial ball).
    cluster : Cluster
        The spectral cluster containing data and peaks.
    noise : float
        Noise level (sigma) of the data. Used in the Gaussian likelihood function.
    n_walkers : int, optional
        Number of MCMC walkers (chains). Should be at least 2 * n_params.
        Default is defined in constants.
    n_steps : int, optional
        Number of steps to run for each walker. Default is defined in constants.
    burn_in : int, optional
        Number of initial steps to discard. If None, it is automatically
        determined using the R-hat statistic (recommended).
    workers : int, optional
        Number of parallel processes to use.
        * 1: Sequential execution (default).
        * -1: Use all available CPU cores.
    progress_callback : Callable[[int, int, str], None], optional
        Function to be called with (step, total_steps, status_message) for UI updates.

    Returns
    -------
    UncertaintyResult
        Data object containing the full chains, statistical summaries (CI, geometric means),
        and convergence diagnostics (R-hat, ESS).
    """
    global _mcmc_params, _mcmc_cluster, _mcmc_noise, _mcmc_bounds

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
            # Run MCMC with progress callback
            for i, _sample in enumerate(sampler.sample(pos, iterations=n_steps, progress=False)):
                if progress_callback:
                    progress_callback(i + 1, n_steps, "Running MCMC...")
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
        # Run MCMC with progress callback
        for i, _sample in enumerate(sampler.sample(pos, iterations=n_steps, progress=False)):
            if progress_callback:
                progress_callback(i + 1, n_steps, "Running MCMC...")

    # Signal processing phase
    if progress_callback:
        progress_callback(n_steps, n_steps, "Processing results...")

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
