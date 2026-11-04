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

if TYPE_CHECKING:
    from peakfit.shared.typing import FloatArray


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
