"""Convergence diagnostics for MCMC chains.

Implements R-hat (Gelman-Rubin statistic) and Effective Sample Size (ESS)
following the recommendations from the Bayesian Analysis Reporting Guidelines (BARG)
by Kruschke (2021, Nature Human Behaviour, https://doi.org/10.1038/s41562-021-01177-7).

References:
    - Gelman & Rubin (1992): "Inference from Iterative Simulation Using Multiple Sequences"
    - Vehtari et al. (2021): "Rank-normalization, folding, and localization:
      An improved R-hat for assessing convergence of MCMC"
    - Stan Development Team: "Brief Guide to Stan's Warnings"
"""

from dataclasses import dataclass

import numpy as np

from peakfit.core.shared.typing import FloatArray


@dataclass
class ConvergenceDiagnostics:
    """Results of MCMC convergence diagnostics.

    Attributes:
        parameter_names: Names of parameters
        rhat: R-hat values for each parameter (should be ≤ 1.01)
        ess_bulk: Bulk effective sample size for each parameter (recommended ≥ 100 * chains)
        ess_tail: Tail effective sample size for each parameter (recommended ≥ 100 * chains)
        n_chains: Number of chains
        n_samples: Number of samples per chain (after burn-in)
        converged: Boolean indicating if all parameters converged
    """

    parameter_names: list[str]
    rhat: FloatArray
    ess_bulk: FloatArray
    ess_tail: FloatArray
    n_chains: int
    n_samples: int

    @property
    def converged(self) -> bool:
        """Check if all parameters have converged.

        Convergence criteria:
        - R-hat ≤ 1.01 for all parameters
        - ESS_bulk ≥ 100 * n_chains for all parameters

        Returns:
            True if converged, False otherwise
        """
        rhat_ok = np.all(self.rhat <= 1.01)
        ess_ok = np.all(self.ess_bulk >= 100 * self.n_chains)
        return bool(rhat_ok and ess_ok)

    def get_warnings(self) -> list[str]:
        """Get list of convergence warnings.

        Returns:
            List of warning messages for parameters that haven't converged
        """
        warnings = []

        for i, name in enumerate(self.parameter_names):
            if self.rhat[i] > 1.01:
                warnings.append(
                    f"{name}: R-hat = {self.rhat[i]:.4f} (should be ≤ 1.01). "
                    "Chains have not mixed well. Consider increasing n_steps or n_walkers."
                )

            if self.rhat[i] > 1.05:
                warnings.append(
                    f"{name}: R-hat = {self.rhat[i]:.4f} is very high (> 1.05). "
                    "Convergence is poor. Results should not be trusted."
                )

            recommended_ess = 100 * self.n_chains
            if self.ess_bulk[i] < recommended_ess:
                warnings.append(
                    f"{name}: ESS_bulk = {self.ess_bulk[i]:.0f} "
                    f"(recommended ≥ {recommended_ess:.0f}). "
                    "Increase n_steps for more stable estimates."
                )

            if self.ess_bulk[i] < 10 * self.n_chains:
                warnings.append(
                    f"{name}: ESS_bulk = {self.ess_bulk[i]:.0f} is very low "
                    f"(< {10 * self.n_chains:.0f}). "
                    "Posterior estimates are highly uncertain."
                )

        return warnings


def compute_rhat(chains: FloatArray) -> float:
    """Compute split R-hat statistic for a single parameter.

    R-hat measures the ratio of the between-chain variance to the within-chain variance.
    Values close to 1 indicate convergence. Values > 1.01 suggest lack of convergence.

    This implements the split R-hat from Gelman & Rubin (1992), which is more
    conservative than the original version.

    Args:
        chains: Array of shape (n_chains, n_samples) containing MCMC samples
                for a single parameter

    Returns:
        R-hat value (should be ≤ 1.01 for convergence)

    References:
        Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation
        using multiple sequences. Statistical Science, 7(4), 457-472.
    """
    n_chains, n_samples = chains.shape

    if n_chains < 2:
        # Cannot compute R-hat with less than 2 chains
        return np.nan

    # Split each chain in half to get 2 * n_chains
    split_chains = np.concatenate(
        [chains[:, : n_samples // 2], chains[:, n_samples // 2 :]], axis=0
    )
    n_split_samples = split_chains.shape[1]

    # Compute within-chain and between-chain variances
    chain_means = np.mean(split_chains, axis=1)
    chain_vars = np.var(split_chains, axis=1, ddof=1)

    # (overall_mean is unnecessary; removed to keep lint clean)

    # Within-chain variance (w)
    w = np.mean(chain_vars)

    # Between-chain variance (b)
    b = n_split_samples * np.var(chain_means, ddof=1)

    # Estimate of marginal posterior variance
    var_plus = ((n_split_samples - 1) / n_split_samples) * w + (1 / n_split_samples) * b

    # R-hat
    if w > 0:
        rhat = np.sqrt(var_plus / w)
    else:
        rhat = np.nan

    return float(rhat)


def compute_ess(chains: FloatArray, method: str = "bulk") -> float:
    """Compute Effective Sample Size (ESS) for a single parameter.

    ESS estimates the number of independent samples in the MCMC chains.
    Higher ESS means better mixing and more reliable posterior estimates.

    Args:
        chains: Array of shape (n_chains, n_samples) containing MCMC samples
        method: Either "bulk" (for main distribution) or "tail" (for extremes)

    Returns:
        Effective sample size (recommended ≥ 100 * n_chains)

    References:
        Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021).
        Rank-normalization, folding, and localization: An improved R-hat for assessing
        convergence of MCMC. Bayesian Analysis, 16(2), 667-718.
    """
    n_chains, n_samples = chains.shape

    if n_chains < 1:
        return np.nan

    # For tail ESS, we focus on extreme quantiles
    if method == "tail":
        # Fold the distribution around the median for tail analysis
        median = np.median(chains)
        chains = np.abs(chains - median)

    # Compute autocorrelation for each chain
    # ESS = N / (1 + 2 * sum of autocorrelations)
    max_lag = min(n_samples - 1, 1000)  # Limit max lag for computational efficiency

    # Compute mean across chains
    chain_means = np.mean(chains, axis=1, keepdims=True)
    centered_chains = chains - chain_means

    # Compute autocorrelation using FFT for efficiency
    ess_per_chain = []

    for chain in centered_chains:
        # Variance
        var = np.var(chain, ddof=1)

        if var == 0:
            ess_per_chain.append(float(n_samples))
            continue

        # Compute autocorrelation using FFT
        n_fft = 2 ** int(np.ceil(np.log2(2 * n_samples - 1)))
        fft_chain = np.fft.rfft(chain, n=n_fft)
        autocorr_fft = np.fft.irfft(fft_chain * np.conj(fft_chain), n=n_fft)
        autocorr = autocorr_fft[:n_samples] / (var * n_samples)

        # Sum autocorrelations until they become negative (Geyer's initial monotone sequence)
        rho = []
        for lag in range(1, max_lag, 2):
            if lag + 1 < len(autocorr):
                # Sum of pairs (Geyer's monotone sequence)
                rho_pair = autocorr[lag] + autocorr[lag + 1]
                if rho_pair > 0:
                    rho.append(rho_pair)
                else:
                    break

        # ESS for this chain
        tau = 1 + sum(rho)  # Integrated autocorrelation time
        ess_per_chain.append(n_samples / tau)

    # Total ESS across all chains
    total_ess = sum(ess_per_chain)

    return float(total_ess)


def diagnose_convergence(
    chains: FloatArray,
    parameter_names: list[str],
) -> ConvergenceDiagnostics:
    """Compute comprehensive convergence diagnostics for MCMC chains.

    Args:
        chains: Array of shape (n_chains, n_samples, n_params) containing all MCMC samples
        parameter_names: List of parameter names

    Returns:
        ConvergenceDiagnostics object with R-hat and ESS for all parameters

    Example:
        >>> chains = sampler.get_chain()  # Shape: (n_walkers, n_steps, n_params)
        >>> diagnostics = diagnose_convergence(chains, ["mu", "sigma"])
        >>> print(f"R-hat for mu: {diagnostics.rhat[0]:.4f}")
        >>> if not diagnostics.converged:
        ...     for warning in diagnostics.get_warnings():
        ...         print(f"Warning: {warning}")
    """
    n_chains, n_samples, n_params = chains.shape

    rhat_vals = np.zeros(n_params)
    ess_bulk_vals = np.zeros(n_params)
    ess_tail_vals = np.zeros(n_params)

    for i in range(n_params):
        param_chains = chains[:, :, i]

        # Compute R-hat
        rhat_vals[i] = compute_rhat(param_chains)

        # Compute ESS (bulk and tail)
        ess_bulk_vals[i] = compute_ess(param_chains, method="bulk")
        ess_tail_vals[i] = compute_ess(param_chains, method="tail")

    return ConvergenceDiagnostics(
        parameter_names=parameter_names,
        rhat=rhat_vals,
        ess_bulk=ess_bulk_vals,
        ess_tail=ess_tail_vals,
        n_chains=n_chains,
        n_samples=n_samples,
    )


def format_diagnostics_table(diagnostics: ConvergenceDiagnostics) -> str:
    """Format convergence diagnostics as a readable table.

    Args:
        diagnostics: ConvergenceDiagnostics object

    Returns:
        Formatted string table with diagnostics
    """
    lines = []
    lines.append("=" * 80)
    lines.append("MCMC Convergence Diagnostics")
    lines.append("=" * 80)
    lines.append(f"Chains: {diagnostics.n_chains}")
    lines.append(f"Samples per chain: {diagnostics.n_samples}")
    lines.append(f"Total samples: {diagnostics.n_chains * diagnostics.n_samples}")
    lines.append("")

    # Recommendations
    lines.append("Convergence criteria (BARG guidelines):")
    lines.append("  • R-hat ≤ 1.01 (excellent), ≤ 1.05 (acceptable)")
    lines.append(f"  • ESS_bulk ≥ {100 * diagnostics.n_chains:.0f} (recommended for stable CI)")
    lines.append(f"  • ESS_bulk ≥ {10 * diagnostics.n_chains:.0f} (minimum for rough estimates)")
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"{'Parameter':<25} {'R-hat':>10} {'ESS_bulk':>12} {'ESS_tail':>12}  {'Status'}")
    lines.append("-" * 80)

    for i, name in enumerate(diagnostics.parameter_names):
        rhat = diagnostics.rhat[i]
        ess_b = diagnostics.ess_bulk[i]
        ess_t = diagnostics.ess_tail[i]

        # Determine status
        if np.isnan(rhat):
            status = "N/A"
        elif rhat <= 1.01 and ess_b >= 100 * diagnostics.n_chains:
            status = "✓ Good"
        elif rhat <= 1.05 and ess_b >= 10 * diagnostics.n_chains:
            status = "⚠ Marginal"
        else:
            status = "✗ Poor"

        lines.append(f"{name:<25} {rhat:>10.4f} {ess_b:>12.1f} {ess_t:>12.1f}  {status}")

    lines.append("=" * 80)

    # Add warnings if any
    warnings = diagnostics.get_warnings()
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend([f"  • {warning}" for warning in warnings])
        lines.append("")

    # Overall assessment
    if diagnostics.converged:
        lines.append("Overall: ✓ All parameters have converged")
    else:
        lines.append("Overall: ⚠ Some parameters have not converged")
        lines.append("         Consider running longer chains (increase n_steps)")

    lines.append("=" * 80)

    return "\n".join(lines)
