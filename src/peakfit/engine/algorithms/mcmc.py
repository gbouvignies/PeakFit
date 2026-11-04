"""Bayesian uncertainty estimation using Affine Invariant MCMC.

This module implements Markov Chain Monte Carlo (MCMC) sampling to estimate
parameter uncertainties. It samples a cluster's ``SamplingProblem`` (the fit's
profile likelihood, see docs/mcmc-formulation.md) and records the conditional
optimum amplitudes at each step as "blobs".
"""

import os
from dataclasses import dataclass
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any

import emcee
import numpy as np
from threadpoolctl import threadpool_limits

from peakfit.engine.diagnostics.convergence import diagnose_convergence
from peakfit.engine.domain.param_id import PSEUDO_AXIS, ParameterId
from peakfit.shared.constants import MCMC_N_STEPS, MCMC_N_WALKERS

if TYPE_CHECKING:
    from collections.abc import Callable

    from peakfit.engine.diagnostics.convergence import ConvergenceDiagnostics
    from peakfit.engine.scientific.sampling import SamplingProblem
    from peakfit.shared.typing import FloatArray


@dataclass
class _MCMCState:
    """Mutable worker state used for multiprocessing.

    We keep a single module-level object so worker initialization can update
    attributes without using the `global` statement.
    """

    problem: SamplingProblem | None = None


_mcmc_state = _MCMCState()


@dataclass
class UncertaintyResult:
    """Comprehensive uncertainty estimates for fitted parameters."""

    parameter_names: list[str]
    values: FloatArray
    std_errors: FloatArray
    confidence_intervals_68: FloatArray
    confidence_intervals_95: FloatArray
    correlation_matrix: FloatArray
    mcmc_samples: FloatArray | None = None
    mcmc_percentiles: FloatArray | None = None
    mcmc_chains: FloatArray | None = None
    mcmc_diagnostics: ConvergenceDiagnostics | None = None
    burn_in_info: dict[str, Any] | None = None
    n_lineshape_params: int = 0
    amplitude_names: list[str] | None = None
    n_series: int = 1
    profile_likelihood_ci: FloatArray | None = None


def _init_mcmc_worker(problem: SamplingProblem) -> None:
    """Initialize the sampled problem in worker processes."""
    _mcmc_state.problem = problem


def _log_likelihood_blobs(x: FloatArray) -> tuple[float, FloatArray]:
    """The profile log-likelihood at θ, with the conditional optimum amplitudes."""
    problem = _mcmc_state.problem
    if problem is None:
        return float(-np.inf), np.array([])
    conditional = problem.conditional(x)
    if conditional is None:
        blob_size = len(problem.cluster.peaks) * problem.cluster.n_series
        return float(-np.inf), np.full(blob_size, np.nan)
    return conditional.log_likelihood, conditional.mean.ravel()


def _prepare_mcmc_sampling(problem: SamplingProblem, n_walkers: int) -> FloatArray:
    """Initial walker positions around the fit's θ̂."""
    vary_names = list(problem.names)
    x0 = problem.start
    bounds = list(zip(problem.lower, problem.upper, strict=True))
    ndim = len(x0)

    # Walkers configuration
    min_walkers = 2 * ndim + 2
    n_walkers = max(n_walkers, min_walkers)

    # Initialize walkers
    rng = np.random.default_rng()
    stderrs = np.array([problem.parameters[n].stderr for n in vary_names])
    values_abs = np.abs(x0)

    # Use small perturbation if stderr is missing
    # Heuristic: 10% of stderr OR 0.01% of value OR 1e-6
    scales = np.where(stderrs > 0, 0.1 * stderrs, np.maximum(1e-4 * values_abs, 1e-6))

    # Generate random positions around x0
    pos = x0 + scales * rng.standard_normal((n_walkers, ndim))

    # Clip to bounds
    for i, (lb, ub) in enumerate(bounds):
        pos[:, i] = np.clip(pos[:, i], lb, ub)

    return pos


def _run_emcee(
    pos: FloatArray,
    n_steps: int,
    workers: int,
    progress_callback: Callable[[int, int, str, float | None], None] | None,
    problem: SamplingProblem,
) -> emcee.EnsembleSampler:
    """Run the emcee sampler."""
    n_walkers, ndim = pos.shape
    moves = [(emcee.moves.StretchMove(), 0.8), (emcee.moves.DEMove(), 0.2)]
    if progress_callback:
        progress_callback(0, n_steps, "Initializing sampler...", None)

    if workers != 1:
        n_processes = workers if workers > 0 else (os.cpu_count() or 1)
        if progress_callback:
            progress_callback(0, n_steps, f"Starting worker pool ({n_processes} workers)...", None)
        # Limit low-level threads to avoid oversubscription
        with (
            threadpool_limits(limits=1, user_api="blas"),
            Pool(
                processes=n_processes,
                initializer=_init_mcmc_worker,
                initargs=(problem,),
            ) as pool,
        ):
            sampler = emcee.EnsembleSampler(
                n_walkers, ndim, _log_likelihood_blobs, pool=pool, moves=moves
            )
            if progress_callback:
                progress_callback(0, n_steps, "Running MCMC...", None)
            _run_sampling_loop(sampler, pos, n_steps, progress_callback)
    else:
        # Serial execution
        if progress_callback:
            progress_callback(0, n_steps, "Preparing serial sampler...", None)
        _init_mcmc_worker(problem)
        sampler = emcee.EnsembleSampler(n_walkers, ndim, _log_likelihood_blobs, moves=moves)
        if progress_callback:
            progress_callback(0, n_steps, "Running MCMC...", None)
        _run_sampling_loop(sampler, pos, n_steps, progress_callback)

    return sampler


def _run_sampling_loop(
    sampler: emcee.EnsembleSampler,
    pos: FloatArray,
    n_steps: int,
    progress_callback: Callable[[int, int, str, float | None], None] | None,
) -> None:
    """Execute the sampling loop with progress reporting."""
    for i, _ in enumerate(sampler.sample(pos, iterations=n_steps, progress=False)):
        step = i + 1
        emit = step <= min(50, n_steps) or step % 10 == 0 or step == n_steps
        if progress_callback and emit:
            accept_frac = np.mean(sampler.acceptance_fraction)
            progress_callback(step, n_steps, f"Running MCMC (acc={accept_frac:.2f})", accept_frac)


def _process_mcmc_chains(
    sampler: emcee.EnsembleSampler,
    burn_in: int | None,
) -> tuple[FloatArray, dict[str, Any]]:
    """Extract chains, handle blobs, and apply burn-in/thinning."""
    blob_amplitudes_2d_ndim = 2

    # Post-processing
    # get_chain returns (n_steps, n_walkers, n_params)
    chain_lineshape = np.swapaxes(sampler.get_chain(flat=False), 0, 1)

    # Extract blobs (amplitudes)
    # blobs are list of arrays
    try:
        raw_blobs = sampler.get_blobs(flat=False)
        blob_amplitudes = np.swapaxes(np.array(raw_blobs), 0, 1)
        # Handle case where only 1 step or shape mismatch
        if blob_amplitudes.ndim == blob_amplitudes_2d_ndim:
            blob_amplitudes = blob_amplitudes[..., np.newaxis]
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve amplitude blobs: {e}") from e

    # Calculate auto-correlation time
    try:
        tau = float(np.max(sampler.get_autocorr_time(tol=0)))
        burn_in_val = int(2 * tau) if burn_in is None else burn_in
        thin = int(max(1, 0.5 * tau))
    except emcee.autocorr.AutocorrError:
        burn_in_val = 500 if burn_in is None else burn_in
        thin = 1
        tau = np.nan

    valid_start_idx = int(np.ceil(burn_in_val / thin))

    # Apply burn-in and thinning
    # Note: If burn-in > total steps, this will fail or return empty.
    # We should clamp or check.
    if valid_start_idx >= chain_lineshape.shape[1]:
        # Emergency fallback: keep last 10%
        valid_start_idx = int(chain_lineshape.shape[1] * 0.9)
        burn_in_val = valid_start_idx * thin  # approx

    full_lineshape = chain_lineshape[:, ::thin, :]
    full_amplitudes = blob_amplitudes[:, ::thin, :]

    # Combine shape: (n_walkers, n_steps, n_params)
    combined_chains = np.concatenate([full_lineshape, full_amplitudes], axis=2)
    kept_chains = combined_chains[:, valid_start_idx:, :]

    burn_in_info = {"burn_in": burn_in_val, "thin": thin, "tau": tau}
    return kept_chains, burn_in_info


def _compute_statistics(
    kept_chains: FloatArray, parameter_names: list[str]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Compute summary statistics from MCMC chains."""
    flat_kept = np.asarray(kept_chains.reshape(-1, len(parameter_names)), dtype=np.float64)
    percentiles = np.asarray(np.percentile(flat_kept, [16, 50, 84], axis=0), dtype=np.float64)
    std_errors = np.asarray(np.std(flat_kept, axis=0), dtype=np.float64)

    ci_68 = np.asarray(np.column_stack((percentiles[0], percentiles[2])), dtype=np.float64)
    ci_95 = np.asarray(
        np.column_stack(
            (np.percentile(flat_kept, 2.5, axis=0), np.percentile(flat_kept, 97.5, axis=0))
        ),
        dtype=np.float64,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.asarray(np.corrcoef(flat_kept.T), dtype=np.float64)
    if corr.ndim == 0:
        corr = np.array([[corr]])
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)

    return flat_kept, percentiles, std_errors, ci_68, ci_95, corr


def estimate_uncertainties_mcmc(
    problem: SamplingProblem,
    n_walkers: int = MCMC_N_WALKERS,
    n_steps: int = MCMC_N_STEPS,
    burn_in: int | None = None,
    workers: int = 1,
    progress_callback: Callable[[int, int, str, float | None], None] | None = None,
) -> UncertaintyResult:
    """Estimate parameter uncertainties using Affine Invariant MCMC (emcee)."""
    if progress_callback:
        progress_callback(0, n_steps, "Preparing walker initialization...", None)

    # 1. Prepare
    pos = _prepare_mcmc_sampling(problem, n_walkers)
    vary_names = list(problem.names)
    ndim = len(vary_names)
    if progress_callback:
        progress_callback(
            0,
            n_steps,
            f"Initialized {pos.shape[0]} walkers across {ndim} nonlinear parameters",
            None,
        )

    # 2. Run Sampler
    sampler = _run_emcee(pos, n_steps, workers, progress_callback, problem)

    # 3. Process Chains
    kept_chains, burn_in_info = _process_mcmc_chains(sampler, burn_in)

    # 4. Construct Parameter Names
    lineshape_names = vary_names

    # Consistent with the point-by-series cluster data contract.
    cluster = problem.cluster
    n_series = cluster.n_series

    # Amplitudes are solved, never parameters; they are named <peak>.F1.I<plane_index>.
    amp_names = [
        ParameterId(peak_name=p.name, axis=PSEUDO_AXIS, label="I", index=i).name
        for p in cluster.peaks
        for i in range(n_series)
    ]

    combined_names = lineshape_names + amp_names

    # 5. Diagnostics
    diagnostics = diagnose_convergence(kept_chains, combined_names)

    # 6. Statistics
    (
        flat_kept,
        percentiles,
        std_errors,
        ci_68,
        ci_95,
        corr,
    ) = _compute_statistics(kept_chains, combined_names)

    return UncertaintyResult(
        parameter_names=combined_names,
        values=percentiles[1],
        std_errors=std_errors,
        confidence_intervals_68=ci_68,
        confidence_intervals_95=ci_95,
        correlation_matrix=corr,
        mcmc_samples=flat_kept,
        mcmc_percentiles=percentiles,
        mcmc_chains=kept_chains,
        mcmc_diagnostics=diagnostics,
        burn_in_info=burn_in_info,
        n_lineshape_params=ndim,
        amplitude_names=[p.name for p in cluster.peaks],
        n_series=n_series,
    )
