"""Autocorrelation plots for MCMC chains."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from peakfit.engine.diagnostics.metrics import compute_autocorrelation

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from peakfit.engine.diagnostics.metrics import AutocorrelationResult
    from peakfit.shared.typing import FloatArray


_MAX_CHAINS_FOR_LEGEND = 10
_MAX_LABELED_CHAINS = 3
_AUTOCORR_EFFECTIVE_LAG_THRESHOLD = 0.1


def plot_autocorrelation(
    chains: FloatArray,
    parameter_names: list[str],
    thin: int,
    max_lag: int = 100,
    max_params: int = 20,
) -> Figure:
    """Create autocorrelation plots for MCMC chains.

    Autocorrelation plots show how correlated samples are with previous samples.
    Good mixing shows autocorrelation dropping quickly to zero.

    Args:
        chains: Array of shape (n_chains, n_samples, n_params)
        parameter_names: List of parameter names
        thin: Thinning factor
        max_lag: Maximum lag to compute
        max_params: Maximum number of parameters to plot

    Returns:
    -------
        Matplotlib Figure object
    """
    n_chains, n_samples, n_params = chains.shape
    n_params_plot = min(n_params, max_params)
    max_lag = min(max_lag, n_samples - 1)

    # Create figure
    n_cols = min(3, n_params_plot)
    n_rows = int(np.ceil(n_params_plot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))

    if n_params_plot == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_params_plot):
        ax = axes[i]
        param_name = parameter_names[i]

        # Compute autocorrelation for each chain
        autocorr_results: list[AutocorrelationResult] = []
        for chain_idx in range(n_chains):
            chain_data = chains[chain_idx, :, i]
            result = compute_autocorrelation(chain_data, max_lag)
            autocorr_results.append(result)

            # Plot
            ax.plot(
                result.lags,
                result.autocorr,
                alpha=0.6,
                linewidth=1,
                label=(
                    f"Chain {chain_idx + 1}" if i == 0 and chain_idx < _MAX_LABELED_CHAINS else None
                ),
            )

        # Add reference lines
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.axhline(0.1, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axhline(-0.1, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

        # Mark effective decorrelation lag from mean autocorrelation
        mean_autocorr = np.mean([r.autocorr for r in autocorr_results], axis=0)
        lag_below_threshold = np.where(np.abs(mean_autocorr) < _AUTOCORR_EFFECTIVE_LAG_THRESHOLD)[0]
        if len(lag_below_threshold) > 1:
            effective_lag = int(lag_below_threshold[1])
            ax.axvline(effective_lag, color="red", linestyle=":", alpha=0.5, linewidth=1)
            ax.text(
                effective_lag,
                0.9,
                f"~{effective_lag} steps",
                fontsize=8,
                color="red",
                rotation=90,
                va="top",
            )

        ax.set_title(param_name, fontsize=10)
        ax.set_xlabel("Lag", fontsize=9)
        ax.set_ylabel("Autocorrelation", fontsize=9)
        ax.set_ylim(-0.2, 1.1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        if i == 0 and n_chains <= _MAX_CHAINS_FOR_LEGEND:
            ax.legend(fontsize=8, loc="upper right")

    # Hide unused subplots
    for i in range(n_params_plot, len(axes)):
        axes[i].set_visible(False)

    n_samples_stored = n_samples
    total_steps = n_samples_stored * thin
    fig.suptitle(
        f"Autocorrelation: {n_chains} chains × {total_steps} iterations (thinned by {thin})",
        fontsize=14,
        fontweight="bold",
    )

    # Add interpretation guide
    guide_text = (
        "Good mixing: autocorrelation drops quickly to ~0 (within 10-20 lags) | "
        "Red line = effective decorrelation lag\n"
        "Slow decay (>100 lags) indicates high autocorrelation -> low ESS"
    )
    fig.text(
        0.5,
        0.02,
        guide_text,
        ha="center",
        fontsize=8,
        style="italic",
        color="gray",
        wrap=True,
    )

    plt.tight_layout(rect=(0, 0.04, 1, 0.96))
    return fig
