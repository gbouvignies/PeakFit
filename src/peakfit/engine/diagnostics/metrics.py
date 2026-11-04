"""Pure computation functions for MCMC diagnostics.

This module contains only stateless, pure functions that compute
diagnostic metrics from MCMC chains. No plotting or I/O.

All functions are designed to be:
- Pure (no side effects)
- Stateless (no global state)
- Reusable (consumed by both plotting and MCMC workflows)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.diagnostics import ensemble

if TYPE_CHECKING:
    from peakfit.shared.typing import FloatArray


@dataclass(frozen=True)
class TraceMetrics:
    """Summary of one parameter over an ensemble's kept chain.

    There is no R̂: ensemble walkers are coupled, not independent chains
    (``docs/mcmc-formulation.md`` §7).
    """

    mean: float
    std: float
    median: float
    q05: float
    q95: float
    q16: float
    q84: float
    tau: float  # integrated autocorrelation time, in steps
    ess: float  # approximate: walkers × steps / τ
    stationary: bool  # the two halves of the chain agree

    @property
    def ci_68(self) -> tuple[float, float]:
        """68% credible interval (approximately 1 sigma)."""
        return (self.q16, self.q84)


@dataclass(frozen=True)
class AutocorrelationResult:
    """Autocorrelation analysis results.

    Attributes:
    ----------
        lags: Array of lag values (0 to max_lag)
        autocorr: Autocorrelation values at each lag
        integrated_autocorr_time: Estimated integrated autocorrelation time
        effective_decorrelation_lag: Lag where autocorrelation drops below threshold
    """

    lags: FloatArray
    autocorr: FloatArray
    integrated_autocorr_time: float
    effective_decorrelation_lag: int


def compute_all_trace_metrics(chains: FloatArray) -> list[TraceMetrics]:
    """Summaries of every parameter of a kept chain of shape (walkers, steps, params)."""
    chain = np.swapaxes(chains, 0, 1)  # the ensemble diagnostics' (steps, walkers, params)
    tau, _ = ensemble.autocorrelation_times(chain)
    ess = ensemble.effective_sample_size(chain, tau)
    stationary = ensemble.stationary(chain, tau)
    flat = chain.reshape(-1, chain.shape[2])
    q05, q16, median, q84, q95 = np.quantile(flat, [0.05, 0.16, 0.5, 0.84, 0.95], axis=0)
    return [
        TraceMetrics(
            mean=float(np.mean(flat[:, j])),
            std=float(np.std(flat[:, j])),
            median=float(median[j]),
            q05=float(q05[j]),
            q95=float(q95[j]),
            q16=float(q16[j]),
            q84=float(q84[j]),
            tau=float(tau[j]),
            ess=float(ess[j]),
            stationary=bool(stationary[j]),
        )
        for j in range(flat.shape[1])
    ]


def compute_autocorrelation(
    chain: FloatArray,
    max_lag: int | None = None,
    threshold: float = 0.1,
) -> AutocorrelationResult:
    """Compute autocorrelation function for a single chain.

    Uses FFT for efficient computation.

    Args:
        chain: 1D array of samples from single chain
        max_lag: Maximum lag to compute (default: len(chain) // 2)
        threshold: Threshold for determining effective decorrelation lag

    Returns:
    -------
        AutocorrelationResult with lags and autocorrelation values
    """
    n = len(chain)
    if max_lag is None:
        max_lag = min(n // 2, 100)
    max_lag = min(max_lag, n - 1)

    # Center the data
    chain_centered = chain - np.mean(chain)
    var = np.var(chain)

    if var == 0:
        # Constant chain - no autocorrelation
        return AutocorrelationResult(
            lags=np.arange(max_lag + 1, dtype=np.float64),
            autocorr=np.ones(max_lag + 1),
            integrated_autocorr_time=float(n),
            effective_decorrelation_lag=max_lag,
        )

    # Compute autocorrelation using FFT for efficiency
    n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    fft_data = np.fft.rfft(chain_centered, n=n_fft)
    autocorr_fft = np.fft.irfft(fft_data * np.conj(fft_data), n=n_fft)
    autocorr = autocorr_fft[:n] / (var * n)
    autocorr = autocorr[: max_lag + 1]

    lags = np.arange(max_lag + 1)

    # Integrated autocorrelation time
    # Sum until autocorrelation goes negative or below threshold
    cumsum = np.cumsum(autocorr)
    first_negative = int(np.argmax(autocorr < 0))
    if first_negative == 0:
        first_negative = int(max_lag) + 1
    index = min(first_negative - 1, int(max_lag))
    iat = 1 + 2 * float(cumsum[index])

    # Find effective decorrelation lag (where autocorr drops below threshold)
    below_threshold = np.where(np.abs(autocorr) < threshold)[0]
    effective_lag = int(below_threshold[1]) if len(below_threshold) > 1 else max_lag

    return AutocorrelationResult(
        lags=lags.astype(np.float64),
        autocorr=autocorr,
        integrated_autocorr_time=float(iat),
        effective_decorrelation_lag=effective_lag,
    )
