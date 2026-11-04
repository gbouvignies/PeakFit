"""Trace plots for MCMC chains."""

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from peakfit.engine.scientific.sampler import ParameterSummary
    from peakfit.shared.typing import FloatArray


_MAX_CHAINS_FOR_LEGEND = 10
_MAX_LABELED_CHAINS = 3


def plot_trace(
    chains: FloatArray,
    parameter_names: list[str],
    parameters: Sequence[ParameterSummary],
    burn_in: int = 0,
    max_params: int = 20,
) -> Figure:
    """Create trace plots showing MCMC chain evolution.

    Trace plots show the parameter value at each iteration for each chain.
    Good mixing appears as chains that:
    - Overlap and explore the same space
    - Show no trends or drifts
    - Look like "white noise" around a stable mean

    Args:
        chains: Array of shape (n_chains, n_samples, n_params)
        parameter_names: List of parameter names
        parameters: The recorded summaries, whose τ, ESS and stationarity title traces
        burn_in: Number of burn-in steps to mark (shown in gray)
        max_params: Maximum number of parameters to plot

    Returns:
    -------
        Matplotlib Figure object
    """
    n_chains, n_samples, n_params = chains.shape
    n_params_plot = min(n_params, max_params, len(parameter_names))

    # Create figure
    n_cols = min(3, n_params_plot)
    n_rows = int(np.ceil(n_params_plot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))

    if n_params_plot == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    if len(axes) < n_params_plot:
        # Should not happen if n_rows/n_cols logic is correct
        raise ValueError(f"Not enough axes created: {len(axes)} < {n_params_plot}")

    # Color palette for chains
    color_map = plt.get_cmap("tab10")
    colors = color_map(np.linspace(0, 1, min(n_chains, 10)))

    steps = np.arange(n_samples)

    for i in range(n_params_plot):
        ax = axes[i]
        param_name = parameter_names[i]

        for chain_idx in range(n_chains):
            chain_data = chains[chain_idx, :, i]
            _plot_single_parameter_trace(ax, chain_data, steps, burn_in, colors, chain_idx)

        if burn_in > 0:
            ax.axvline(burn_in, color="red", linestyle="--", alpha=0.5, linewidth=1)

        _add_trace_annotation(ax, param_name, parameters[i])

        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # Legend only on first subplot
        if i == 0 and n_chains <= _MAX_CHAINS_FOR_LEGEND:
            ax.legend(fontsize=8, loc="best")

    # Hide unused subplots
    for i in range(n_params_plot, len(axes)):
        axes[i].set_visible(False)

    # Add title
    title_text = f"MCMC Trace Plots: {n_chains} walkers × {n_samples} steps"
    if burn_in > 0:
        title_text += f"\nGray region = Burn-in ({burn_in} steps) | Red dashed line = Cutoff"
    fig.suptitle(title_text, fontsize=12, fontweight="bold")

    # Add interpretation guide
    guide_text = (
        "Good mixing: walkers overlap, with no trends or drifts after burn-in. "
        "R̂ is not assessed: ensemble walkers are coupled."
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


def _plot_single_parameter_trace(
    ax: Axes,
    chain_data: FloatArray,
    steps: Any,
    burn_in: int,
    colors: Any,
    chain_idx: int,
) -> None:
    """Plot trace for a single parameter and single chain."""
    if burn_in > 0:
        ax.plot(
            steps[:burn_in],
            chain_data[:burn_in],
            color="gray",
            alpha=0.3,
            linewidth=0.5,
        )
        # Overlap by one point to ensure connectivity
        start_idx = max(0, burn_in - 1)
        ax.plot(
            steps[start_idx:],
            chain_data[start_idx:],
            color=colors[chain_idx % len(colors)],
            alpha=0.7,
            linewidth=0.5,
            label=f"Chain {chain_idx + 1}" if chain_idx < _MAX_LABELED_CHAINS else None,
        )
    else:
        ax.plot(
            steps,
            chain_data,
            color=colors[chain_idx % len(colors)],
            alpha=0.7,
            linewidth=0.5,
            label=f"Chain {chain_idx + 1}" if chain_idx < _MAX_LABELED_CHAINS else None,
        )


def _add_trace_annotation(ax: Axes, param_name: str, metric: ParameterSummary) -> None:
    """Title a trace with its recorded autocorrelation time and approximate ESS."""
    title = param_name
    if np.isfinite(metric.tau):
        drift = "" if metric.stationary else ", halves disagree"
        title = f"{param_name} (τ={metric.tau:.1f}, ESS≈{metric.ess:.0f}{drift})"
    ax.set_title(title, fontsize=10)
