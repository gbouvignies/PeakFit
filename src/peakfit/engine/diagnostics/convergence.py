"""Convergence diagnostics for MCMC chains."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

_RHAT_STRICT_THRESHOLD = 1.01
_RHAT_POOR_THRESHOLD = 1.05
_MIN_CHAINS_FOR_RHAT = 2

if TYPE_CHECKING:
    from peakfit.shared.typing import FloatArray


@dataclass
class ConvergenceDiagnostics:
    """Results of MCMC convergence diagnostics."""

    parameter_names: list[str]
    rhat: FloatArray
    ess_bulk: FloatArray
    ess_tail: FloatArray
    n_chains: int
    n_samples: int

    @property
    def converged(self) -> bool:
        """Check if all parameters have converged (R-hat <= 1.01, ESS >= 100*chains)."""
        rhat_ok = np.all(self.rhat <= _RHAT_STRICT_THRESHOLD)
        ess_ok = np.all(self.ess_bulk >= 100 * self.n_chains)
        return bool(rhat_ok and ess_ok)


def compute_rhat(chains: FloatArray) -> float:
    """Compute split R-hat statistic (Gelman & Rubin 1992)."""
    n_chains, n_samples = chains.shape
    if n_chains < _MIN_CHAINS_FOR_RHAT:
        return np.nan

    half = n_samples // 2
    # Discard non-even sample logic if needed, but split is robust this way
    split_chains = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    n_split_samples = split_chains.shape[1]

    chain_means = np.mean(split_chains, axis=1)
    chain_vars = np.var(split_chains, axis=1, ddof=1)

    w = np.mean(chain_vars)
    b = n_split_samples * np.var(chain_means, ddof=1)

    var_plus = ((n_split_samples - 1) / n_split_samples) * w + (1 / n_split_samples) * b
    return float(np.sqrt(var_plus / w) if w > 0 else np.nan)


def compute_ess(chains: FloatArray, method: str = "bulk") -> float:
    """Compute Effective Sample Size (ESS)."""
    n_chains, n_samples = chains.shape
    if n_chains < 1:
        return np.nan

    if method == "tail":
        median = np.median(chains)
        chains = np.abs(chains - median)

    chain_means = np.mean(chains, axis=1, keepdims=True)
    centered_chains = chains - chain_means

    # FFT Autocorrelation
    n_fft = 2 ** int(np.ceil(np.log2(2 * n_samples - 1)))
    ess_per_chain: list[float] = []

    for chain in centered_chains:
        var = np.var(chain, ddof=1)
        if var == 0:
            ess_per_chain.append(float(n_samples))
            continue

        fft_chain = np.fft.rfft(chain, n=n_fft)
        autocorr = np.fft.irfft(fft_chain * np.conj(fft_chain), n=n_fft)[:n_samples] / (
            var * n_samples
        )

        # Geyer's monotone sequence
        rho: list[float] = []
        for lag in range(1, len(autocorr) - 1, 2):
            rho_pair = autocorr[lag] + autocorr[lag + 1]
            if rho_pair > 0:
                rho.append(rho_pair)
            else:
                break

        tau = 1 + sum(rho)
        ess_per_chain.append(n_samples / tau)

    return float(sum(ess_per_chain))


def diagnose_convergence(chains: FloatArray, parameter_names: list[str]) -> ConvergenceDiagnostics:
    """Compute R-hat and ESS for all parameters."""
    n_chains, n_samples, n_params = chains.shape
    rhat_vals = np.zeros(n_params)
    ess_bulk = np.zeros(n_params)
    ess_tail = np.zeros(n_params)

    # Vectorization of ESS/Rhat is non-trivial due to per-chain FFT/sequence logic
    # Keeping the loop is acceptable for clarity and robustness
    for i in range(n_params):
        rhat_vals[i] = compute_rhat(chains[:, :, i])
        ess_bulk[i] = compute_ess(chains[:, :, i], "bulk")
        ess_tail[i] = compute_ess(chains[:, :, i], "tail")

    return ConvergenceDiagnostics(
        parameter_names=parameter_names,
        rhat=rhat_vals,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        n_chains=n_chains,
        n_samples=n_samples,
    )
