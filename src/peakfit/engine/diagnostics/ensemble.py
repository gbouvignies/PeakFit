"""Diagnostics for an affine-invariant ensemble sampler.

See ``docs/mcmc-formulation.md`` §7. Walkers in an ensemble are coupled, so nothing
here treats them as independent chains: there is no R̂. Chains are arrays of shape
(steps, walkers, parameters), as emcee returns them.
"""

from typing import TYPE_CHECKING

import emcee
import numpy as np

from peakfit.engine.diagnostics.bounds import AT_BOUND_FRACTION

if TYPE_CHECKING:
    from peakfit.shared.typing import FloatArray

# Kept steps per τ_max for τ, and the errors built on it, to be reliable.
MIN_TAUS = 50
# A drift between the two halves of the kept chain larger than this many combined
# Monte Carlo standard errors fails the stationarity check.
STATIONARITY_ERRORS = 3.0
# A walker accepting under this fraction of the ensemble's median rate is stuck.
STUCK_FRACTION = 0.1


def autocorrelation_times(chain: FloatArray) -> tuple[FloatArray, bool]:
    """Integrated autocorrelation time per parameter, and whether it is reliable.

    Reliable means the chain is at least ``MIN_TAUS`` τ long for every parameter
    (emcee's own criterion). A parameter that never moves has an undefined τ (NaN).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        try:
            tau = emcee.autocorr.integrated_time(chain, tol=MIN_TAUS)
        except emcee.autocorr.AutocorrError as error:
            return np.asarray(error.tau, dtype=float), False
    tau = np.asarray(tau, dtype=float)
    return tau, bool(np.all(np.isfinite(tau)))


def effective_sample_size(chain: FloatArray, tau: FloatArray) -> FloatArray:
    """Approximate ESS: walkers × steps / τ, per parameter."""
    steps, walkers = chain.shape[:2]
    with np.errstate(invalid="ignore", divide="ignore"):
        return walkers * steps / tau


def quantile_errors(samples: FloatArray, quantiles: FloatArray, ess: FloatArray) -> FloatArray:
    """Monte Carlo standard error of each quantile, per parameter: (quantiles, params).

    The binomial error of a quantile's probability, √(q(1−q)/ESS), mapped to the
    parameter scale through the sample quantiles on either side.
    """
    q = np.asarray(quantiles, dtype=float)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.sqrt(q * (1.0 - q) / ess[None, :])
    lower = np.clip(q - delta, 0.0, 1.0)
    upper = np.clip(q + delta, 0.0, 1.0)
    columns = range(samples.shape[1])
    low = np.array(
        [[np.quantile(samples[:, j], lower[i, j]) for j in columns] for i in range(q.size)]
    )
    high = np.array(
        [[np.quantile(samples[:, j], upper[i, j]) for j in columns] for i in range(q.size)]
    )
    return (high - low) / 2.0


def stationary(chain: FloatArray, tau: FloatArray) -> np.ndarray:
    """Whether each parameter's median agrees between the two halves of the chain.

    Both halves pool all walkers. The Monte Carlo error of a median is estimated from
    the second half, the better estimate of the stationary distribution, and applies
    to both halves under the hypothesis of stationarity. The first half's own spread
    would include any drift and so hide it. The halves agree when their medians
    differ by at most ``STATIONARITY_ERRORS`` combined errors (√2 × the error).
    """
    half = chain.shape[0] // 2
    first = chain[:half].reshape(-1, chain.shape[2])
    second = chain[half:]
    flat = second.reshape(-1, chain.shape[2])
    error = quantile_errors(flat, np.array([0.5]), effective_sample_size(second, tau))[0]
    drift = np.abs(np.median(first, axis=0) - np.median(flat, axis=0))
    with np.errstate(invalid="ignore"):
        return drift <= STATIONARITY_ERRORS * np.sqrt(2.0) * error


def stuck_walkers(acceptance: FloatArray) -> np.ndarray:
    """Indices of walkers accepting under ``STUCK_FRACTION`` of the median rate."""
    return np.flatnonzero(acceptance < STUCK_FRACTION * np.median(acceptance))


def bound_pileup(
    samples: FloatArray, lower: FloatArray, upper: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Fraction of samples At bound at the lower and upper bound, per parameter.

    The same rule as the fit's review warning (``bounds.at_bound``): within
    ``AT_BOUND_FRACTION`` of a finite interval. A one-sided or unbounded parameter
    has no interval to scale the tolerance, so it never piles up.
    """
    span = upper - lower
    finite = np.isfinite(span)
    tolerance = np.where(finite, AT_BOUND_FRACTION * span, 0.0)
    at_lower = finite & (samples <= lower + tolerance)
    at_upper = finite & (samples >= upper - tolerance)
    return at_lower.mean(axis=0), at_upper.mean(axis=0)


def correlation(samples: FloatArray) -> FloatArray:
    """Sample correlation matrix; undefined entries (a constant parameter) stay NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.atleast_2d(np.corrcoef(samples.T))
    np.fill_diagonal(matrix, np.where(np.std(samples, axis=0) > 0, 1.0, np.nan))
    return matrix


def exchange_fraction(first: FloatArray, second: FloatArray, reference: float) -> float:
    """Fraction of samples whose order of two coordinates differs from ``reference``.

    ``reference`` is the signed separation (second − first) at the fit's θ̂. A zero
    reference has no order to keep, so every sample counts as exchanged.
    """
    if reference == 0:
        return 1.0
    return float(np.mean(np.sign(second - first) != np.sign(reference)))


__all__ = [
    "MIN_TAUS",
    "autocorrelation_times",
    "bound_pileup",
    "correlation",
    "effective_sample_size",
    "exchange_fraction",
    "quantile_errors",
    "stationary",
    "stuck_walkers",
]
