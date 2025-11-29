"""Adaptive burn-in determination for MCMC chains.

Implements principled methods for automatically determining burn-in period
based on convergence diagnostics, following modern Bayesian workflow best practices.

References
----------
    - Gelman et al. (2013): Bayesian Data Analysis (3rd ed.)
    - Vehtari et al. (2021): Rank-normalization, folding, and localization
    - Stan Development Team: Warmup and adaptation guidelines
"""

import warnings

import numpy as np

from peakfit.core.diagnostics.convergence import compute_rhat
from peakfit.core.shared.typing import FloatArray


def determine_burnin_rhat(
    chains: FloatArray,
    rhat_threshold: float = 1.05,
    window_size: int = 100,
    min_samples: int = 100,
    check_interval: int = 50,
) -> tuple[int, dict]:
    """Determine burn-in by monitoring when R-hat stabilizes below threshold.

    This method computes R-hat on progressively longer chain segments until
    all parameters show convergence (R-hat ≤ threshold). The burn-in is set
    to the point where convergence is first achieved.

    This is more principled than fixed burn-in and follows BARG guidelines
    by using convergence diagnostics to determine adaptation period.

    Args:
        chains: Array of shape (n_chains, n_steps, n_params)
        rhat_threshold: R-hat threshold for convergence (1.05 recommended, 1.01 stricter)
        window_size: Minimum number of samples to use for R-hat calculation
        min_samples: Minimum samples before checking convergence
        check_interval: How often to check convergence (in steps)

    Returns
    -------
        Tuple of (burn_in, diagnostics_dict) where:
        - burn_in: Number of initial samples to discard
        - diagnostics_dict: Information about convergence detection

    Example:
        >>> chains = sampler.get_chain(flat=False).swapaxes(0, 1)  # (n_walkers, n_steps, n_params)
        >>> burn_in, info = determine_burnin_rhat(chains)
        >>> print(f"Burn-in: {burn_in} steps ({burn_in / chains.shape[1] * 100:.1f}% of chain)")
        >>> print(f"Method: {info['method']}, Max R-hat at convergence: {info['max_rhat']:.4f}")
    """
    n_chains, n_steps, n_params = chains.shape

    # Validation
    if n_chains < 2:
        warnings.warn(
            "Need at least 2 chains for R-hat-based burn-in. Using default.",
            stacklevel=2,
        )
        return _default_burnin(n_steps), {"method": "default", "reason": "insufficient_chains"}

    # Start checking after minimum samples
    start_check = max(min_samples, window_size)

    if start_check >= n_steps:
        warnings.warn(
            f"Chain too short ({n_steps} steps) for reliable burn-in detection. "
            f"Using conservative default.",
            stacklevel=2,
        )
        return _default_burnin(n_steps), {
            "method": "default",
            "reason": "chain_too_short",
        }

    # Check convergence at intervals
    rhat_history = []
    for i in range(start_check, n_steps, check_interval):
        # Use samples from start to current point
        samples = chains[:, :i, :]

        # Compute R-hat for all parameters
        rhat_values = np.array([compute_rhat(samples[:, :, j]) for j in range(n_params)])

        # Filter out NaN values (can happen with constant parameters)
        valid_rhat = rhat_values[~np.isnan(rhat_values)]

        if len(valid_rhat) == 0:
            continue

        max_rhat = np.max(valid_rhat)
        rhat_history.append((i, max_rhat))

        # Check if all parameters have converged
        if np.all(valid_rhat <= rhat_threshold):
            # Found convergence! Use this as burn-in
            # Be slightly conservative: round up to nearest check_interval
            burn_in = i

            diagnostics = {
                "method": "rhat_monitoring",
                "threshold": rhat_threshold,
                "convergence_step": i,
                "max_rhat": float(max_rhat),
                "n_checks": len(rhat_history),
                "rhat_history": rhat_history,
            }

            return burn_in, diagnostics

    # If we never converged, use conservative default and warn
    warnings.warn(
        f"R-hat did not converge below {rhat_threshold} within {n_steps} steps. "
        f"Using conservative burn-in. Consider running longer chains.",
        stacklevel=2,
    )

    burn_in = _default_burnin(n_steps)
    final_rhat = rhat_history[-1][1] if rhat_history else np.nan

    diagnostics = {
        "method": "default_fallback",
        "threshold": rhat_threshold,
        "max_rhat": float(final_rhat),
        "n_checks": len(rhat_history),
        "reason": "no_convergence",
        "rhat_history": rhat_history,
    }

    return burn_in, diagnostics


def determine_burnin_running_mean(
    chains: FloatArray,
    stability_threshold: float = 0.01,
    window_size: int = 50,
    min_samples: int = 100,
) -> tuple[int, dict]:
    """Determine burn-in by finding when running mean stabilizes.

    This method tracks the running mean of each parameter and identifies
    when relative changes drop below a threshold, indicating the chain
    has reached stationarity.

    Args:
        chains: Array of shape (n_chains, n_steps, n_params)
        stability_threshold: Relative change threshold for stability (0.01 = 1%)
        window_size: Window for computing running statistics
        min_samples: Minimum samples before checking stability

    Returns
    -------
        Tuple of (burn_in, diagnostics_dict)
    """
    n_chains, n_steps, n_params = chains.shape

    if n_steps < min_samples + window_size:
        return _default_burnin(n_steps), {
            "method": "default",
            "reason": "chain_too_short",
        }

    # Compute running mean across all chains
    # Flatten chains for each parameter
    stability_points = []

    for param_idx in range(n_params):
        # Combine all chains for this parameter
        param_data = chains[:, :, param_idx].flatten()  # All chains concatenated

        # Compute running mean
        cumsum = np.cumsum(param_data)
        running_mean = cumsum / np.arange(1, len(param_data) + 1)

        # Find where relative change drops below threshold
        start_idx = min_samples
        for i in range(start_idx, len(running_mean) - window_size):
            # Compute relative change over window
            mean_now = running_mean[i]
            mean_later = running_mean[i + window_size]

            if mean_now != 0:
                rel_change = abs(mean_later - mean_now) / abs(mean_now)
                if rel_change < stability_threshold:
                    # Convert back to per-chain steps
                    stability_points.append(i // n_chains)
                    break

    if stability_points:
        # Use median of stability points across parameters
        burn_in = int(np.median(stability_points))
        diagnostics = {
            "method": "running_mean_stability",
            "threshold": stability_threshold,
            "stability_points": stability_points,
            "burn_in": burn_in,
        }
        return burn_in, diagnostics
    else:
        # Fallback
        burn_in = _default_burnin(n_steps)
        return burn_in, {
            "method": "default_fallback",
            "reason": "no_stability_detected",
        }


def _default_burnin(n_steps: int) -> int:
    """Conservative default burn-in: first 20% or 500 steps, whichever is smaller.

    Args:
        n_steps: Total number of MCMC steps

    Returns
    -------
        Conservative burn-in estimate
    """
    # Use smaller of 20% or 500 steps
    # This is conservative but safe
    return min(n_steps // 5, 500)


def validate_burnin(
    burn_in: int,
    n_steps: int,
    max_fraction: float = 0.5,
) -> tuple[bool, str | None]:
    """Validate that burn-in is reasonable.

    Args:
        burn_in: Proposed burn-in period
        n_steps: Total number of steps
        max_fraction: Maximum allowed fraction of chain to discard

    Returns
    -------
        Tuple of (is_valid, warning_message)
    """
    if burn_in < 0:
        return False, "Burn-in cannot be negative"

    if burn_in >= n_steps:
        return False, f"Burn-in ({burn_in}) must be less than total steps ({n_steps})"

    fraction = burn_in / n_steps

    if fraction > max_fraction:
        warning = (
            f"Large burn-in detected ({fraction * 100:.1f}% of chain). "
            f"Consider:\n"
            f"  • Improving initial parameter estimates\n"
            f"  • Increasing total number of steps\n"
            f"  • Checking for multimodality or poor mixing"
        )
        return True, warning

    if fraction > 0.3:
        warning = (
            f"Moderate burn-in ({fraction * 100:.1f}% of chain). "
            f"This is acceptable but you may want to run longer chains for efficiency."
        )
        return True, warning

    return True, None


def format_burnin_report(
    burn_in: int,
    n_steps: int,
    n_chains: int,
    diagnostics: dict,
) -> str:
    """Format a human-readable report about burn-in determination.

    Args:
        burn_in: Determined burn-in period
        n_steps: Total steps
        n_chains: Number of chains
        diagnostics: Diagnostics dictionary from burn-in determination

    Returns
    -------
        Formatted string report
    """
    lines = []

    # Header
    method = diagnostics.get("method", "unknown")
    lines.append(f"Burn-in: {burn_in} steps ({burn_in / n_steps * 100:.1f}% of chain)")

    # Method-specific details
    if method == "rhat_monitoring":
        lines.append("  Method: R-hat convergence monitoring")
        lines.append(f"  Threshold: R̂ ≤ {diagnostics['threshold']:.3f}")
        lines.append(f"  Max R̂ at convergence: {diagnostics['max_rhat']:.4f}")
        lines.append(f"  Convergence achieved at step: {diagnostics['convergence_step']}")

    elif method == "running_mean_stability":
        lines.append("  Method: Running mean stabilization")
        lines.append(f"  Stability threshold: {diagnostics['threshold']:.2%}")

    elif method in ("default", "default_fallback"):
        lines.append("  Method: Conservative default (20% or 500 steps)")
        reason = diagnostics.get("reason", "unknown")
        lines.append(f"  Reason: {reason.replace('_', ' ')}")

        if "max_rhat" in diagnostics and not np.isnan(diagnostics["max_rhat"]):
            lines.append(f"  Final R̂: {diagnostics['max_rhat']:.4f}")

    # Effective samples
    effective_samples = n_chains * (n_steps - burn_in)
    lines.append(
        f"  Effective samples: {n_chains} chains × {n_steps - burn_in} steps = {effective_samples:,}"
    )

    return "\n".join(lines)
