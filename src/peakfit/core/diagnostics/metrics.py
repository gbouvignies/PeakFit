"""Pure computation functions for MCMC diagnostics.

This module contains only stateless, pure functions that compute
diagnostic metrics from MCMC chains. No plotting or I/O.

All functions are designed to be:
- Pure (no side effects)
- Stateless (no global state)
- Reusable (consumed by both plotting and analysis services)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from peakfit.core.shared.typing import FloatArray


@dataclass(frozen=True)
class TraceMetrics:
    """Computed metrics for a single parameter's trace.

    Attributes:
        mean: Mean value of samples
        std: Standard deviation of samples
        median: Median value of samples
        q05: 5th percentile (lower bound of 90% CI)
        q95: 95th percentile (upper bound of 90% CI)
        q16: 16th percentile (lower bound of 68% CI / 1 sigma)
        q84: 84th percentile (upper bound of 68% CI / 1 sigma)
        ess: Effective sample size
        rhat: Gelman-Rubin statistic
    """

    mean: float
    std: float
    median: float
    q05: float
    q95: float
    q16: float
    q84: float
    ess: float
    rhat: float

    @property
    def is_converged(self) -> bool:
        """Check if R-hat indicates convergence (< 1.01).

        Following BARG recommendations (Kruschke 2021).
        """
        return self.rhat <= 1.01

    @property
    def ci_68(self) -> tuple[float, float]:
        """68% credible interval (approximately 1 sigma)."""
        return (self.q16, self.q84)

    @property
    def ci_90(self) -> tuple[float, float]:
        """90% credible interval."""
        return (self.q05, self.q95)


@dataclass(frozen=True)
class AutocorrelationResult:
    """Autocorrelation analysis results.

    Attributes:
        lags: Array of lag values (0 to max_lag)
        autocorr: Autocorrelation values at each lag
        integrated_autocorr_time: Estimated integrated autocorrelation time
        effective_decorrelation_lag: Lag where autocorrelation drops below threshold
    """

    lags: FloatArray
    autocorr: FloatArray
    integrated_autocorr_time: float
    effective_decorrelation_lag: int


@dataclass(frozen=True)
class CorrelationPair:
    """A pair of correlated parameters.

    Attributes:
        index_i: Index of first parameter
        index_j: Index of second parameter
        correlation: Pearson correlation coefficient
    """

    index_i: int
    index_j: int
    correlation: float

    @property
    def is_strong(self) -> bool:
        """Check if correlation is strong (|r| >= 0.5)."""
        return abs(self.correlation) >= 0.5


def compute_trace_metrics(
    chains: FloatArray,
    param_index: int,
) -> TraceMetrics:
    """Compute summary metrics for a single parameter across all chains.

    Args:
        chains: Array of shape (n_chains, n_samples, n_params)
        param_index: Index of parameter to analyze

    Returns:
        TraceMetrics with computed statistics
    """
    # Extract parameter samples from all chains
    samples = chains[:, :, param_index].flatten()

    # Basic statistics
    mean = float(np.mean(samples))
    std = float(np.std(samples))
    median = float(np.median(samples))
    percentiles = np.atleast_1d(np.percentile(samples, [5, 16, 84, 95]))
    q05 = float(percentiles[0])
    q16 = float(percentiles[1])
    q84 = float(percentiles[2])
    q95 = float(percentiles[3])

    # Effective sample size
    ess = _compute_ess(chains[:, :, param_index])

    # R-hat (Gelman-Rubin)
    rhat = _compute_rhat(chains[:, :, param_index])

    return TraceMetrics(
        mean=mean,
        std=std,
        median=median,
        q05=q05,
        q95=q95,
        q16=q16,
        q84=q84,
        ess=ess,
        rhat=rhat,
    )


def compute_all_trace_metrics(
    chains: FloatArray,
    parameter_names: list[str] | None = None,
) -> list[TraceMetrics]:
    """Compute trace metrics for all parameters.

    Args:
        chains: Array of shape (n_chains, n_samples, n_params)
        parameter_names: Optional list of parameter names (for future use)

    Returns:
        List of TraceMetrics, one per parameter
    """
    n_params = chains.shape[2]
    return [compute_trace_metrics(chains, i) for i in range(n_params)]


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
    first_negative = np.argmax(autocorr < 0)
    if first_negative == 0:
        first_negative = max_lag + 1
    iat = 1 + 2 * cumsum[min(first_negative - 1, max_lag)]

    # Find effective decorrelation lag (where autocorr drops below threshold)
    below_threshold = np.where(np.abs(autocorr) < threshold)[0]
    if len(below_threshold) > 1:
        effective_lag = int(below_threshold[1])  # Skip lag 0
    else:
        effective_lag = max_lag

    return AutocorrelationResult(
        lags=lags.astype(np.float64),
        autocorr=autocorr,
        integrated_autocorr_time=float(iat),
        effective_decorrelation_lag=effective_lag,
    )


def compute_correlation_matrix(samples: FloatArray) -> FloatArray:
    """Compute correlation matrix for samples.

    Args:
        samples: Array of shape (n_samples, n_params)

    Returns:
        Correlation matrix of shape (n_params, n_params)
    """
    return np.corrcoef(samples.T)


def find_correlated_pairs(
    samples: FloatArray,
    min_correlation: float = 0.5,
) -> list[CorrelationPair]:
    """Find strongly correlated parameter pairs.

    Args:
        samples: Array of shape (n_samples, n_params)
        min_correlation: Minimum |correlation| to report

    Returns:
        List of CorrelationPair, sorted by |correlation| (strongest first)
    """
    n_params = samples.shape[1]
    corr_matrix = compute_correlation_matrix(samples)

    pairs = []
    for i in range(n_params):
        for j in range(i + 1, n_params):
            corr = corr_matrix[i, j]
            if abs(corr) >= min_correlation:
                pairs.append(
                    CorrelationPair(
                        index_i=i,
                        index_j=j,
                        correlation=float(corr),
                    )
                )

    # Sort by absolute correlation (strongest first)
    pairs.sort(key=lambda x: abs(x.correlation), reverse=True)
    return pairs


def compute_posterior_statistics(
    samples: FloatArray,
) -> dict[str, FloatArray]:
    """Compute summary statistics for posterior samples.

    Args:
        samples: Array of shape (n_samples, n_params)

    Returns:
        Dictionary with arrays for each statistic:
        - medians: Median for each parameter
        - ci_16, ci_84: 68% credible interval bounds
        - ci_025, ci_975: 95% credible interval bounds
    """
    return {
        "medians": np.median(samples, axis=0),
        "ci_16": np.percentile(samples, 16, axis=0),
        "ci_84": np.percentile(samples, 84, axis=0),
        "ci_025": np.percentile(samples, 2.5, axis=0),
        "ci_975": np.percentile(samples, 97.5, axis=0),
    }


def _compute_ess(chains: FloatArray) -> float:
    """Compute effective sample size across chains.

    Uses a simple variance ratio estimation.

    Args:
        chains: Array of shape (n_chains, n_samples)

    Returns:
        Effective sample size estimate
    """
    n_chains, n_samples = chains.shape

    # Simple ESS estimation using variance ratio
    within_chain_var = np.mean(np.var(chains, axis=1))
    total_var = np.var(chains)

    if total_var == 0:
        return float(n_chains * n_samples)

    # Approximate ESS
    ess = n_chains * n_samples * within_chain_var / total_var
    return float(min(ess, n_chains * n_samples))


def _compute_rhat(chains: FloatArray) -> float:
    """Compute Gelman-Rubin R-hat statistic.

    Args:
        chains: Array of shape (n_chains, n_samples)

    Returns:
        R-hat value (should be <= 1.01 for convergence)
    """
    n_chains, n_samples = chains.shape

    if n_chains < 2:
        return float("nan")

    # Chain means
    chain_means = np.mean(chains, axis=1)

    # Between-chain variance (B in Gelman-Rubin notation)
    between_chain_var = n_samples * np.var(chain_means, ddof=1)

    # Within-chain variance (W in Gelman-Rubin notation)
    within_chain_var = np.mean(np.var(chains, axis=1, ddof=1))

    if within_chain_var == 0:
        return float("nan")

    # Pooled variance estimate
    var_hat = ((n_samples - 1) * within_chain_var + between_chain_var) / n_samples

    # R-hat
    rhat = np.sqrt(var_hat / within_chain_var)
    return float(rhat)
