"""Trace plots for MCMC chains."""

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from peakfit.engine.diagnostics.metrics import compute_all_trace_metrics

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from peakfit.engine.diagnostics.convergence import ConvergenceDiagnostics
    from peakfit.engine.diagnostics.metrics import TraceMetrics
    from peakfit.shared.typing import FloatArray


_MAX_CHAINS_FOR_LEGEND = 10
_MAX_LABELED_CHAINS = 3

_RHAT_EXCELLENT_THRESHOLD = 1.01
_RHAT_ACCEPTABLE_THRESHOLD = 1.05


def plot_trace(
    chains: FloatArray,
    parameter_names: list[str],
    burn_in: int = 0,
    metrics: list[TraceMetrics] | None = None,
    diagnostics: ConvergenceDiagnostics | None = None,
    max_params: int = 20,
    thin: int = 1,
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
        burn_in: Number of burn-in samples to mark (shown in gray)
        metrics: Pre-computed TraceMetrics (computed if not provided)
        diagnostics: Optional ConvergenceDiagnostics for R-hat values
        max_params: Maximum number of parameters to plot
        thin: Thinning factor applied to chains (for x-axis scaling)

    Returns:
    -------
        Matplotlib Figure object
    """
    n_chains, n_samples, n_params = chains.shape
    n_params_plot = min(n_params, max_params, len(parameter_names))

    if metrics is None and diagnostics is None:
        chains_post_burnin = chains[:, burn_in:, :] if burn_in > 0 else chains
        metrics = compute_all_trace_metrics(chains_post_burnin)

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

    # X-axis values (scaled by thinning)
    steps = np.arange(n_samples) * thin
    burn_in_steps = burn_in * thin

    for i in range(n_params_plot):
        ax = axes[i]
        param_name = parameter_names[i]

        for chain_idx in range(n_chains):
            chain_data = chains[chain_idx, :, i]
            _plot_single_parameter_trace(ax, chain_data, steps, burn_in, colors, chain_idx)

        if burn_in > 0:
            ax.axvline(burn_in_steps, color="red", linestyle="--", alpha=0.5, linewidth=1)

        # Add R-hat annotation
        metric = metrics[i] if metrics is not None and i < len(metrics) else None
        _add_trace_annotation(ax, param_name, metric, diagnostics, i)

        ax.set_xlabel(f"Iteration (thin={thin})", fontsize=9)
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
    total_steps = n_samples * thin
    title_text = f"MCMC Trace Plots: {n_chains} chains × {total_steps} iterations"

    details = []
    if thin > 1:
        details.append(f"thinned by {thin} → {n_samples} samples stored")
    else:
        details.append(f"{n_samples} samples stored")

    title_text += f" ({', '.join(details)})"

    if burn_in > 0:
        burn_in_steps = burn_in * thin
        title_text += (
            f"\nGray region = Burn-in ({burn_in_steps} steps / {burn_in} samples) "
            "| Red dashed line = Cutoff"
        )
    fig.suptitle(title_text, fontsize=12, fontweight="bold")

    # Add interpretation guide
    guide_text = (
        "Good convergence: chains overlap, no trends or drifts\n"
        "✓ R̂ ≤ 1.01 excellent | ⚠ 1.01 < R̂ ≤ 1.05 acceptable | "
        "✗ R̂ > 1.05 poor (BARG: Kruschke 2021)"
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


def _add_trace_annotation(
    ax: Axes,
    param_name: str,
    metric: TraceMetrics | None,
    diagnostic: ConvergenceDiagnostics | None,
    idx: int,
) -> None:
    """Add title and R-hat/ESS annotation to trace plot."""
    title = param_name
    rhat = None
    ess = None

    if metric is not None:
        rhat = metric.rhat
        ess = metric.ess
    elif diagnostic is not None and idx < len(diagnostic.rhat):
        rhat = diagnostic.rhat[idx]
        ess = diagnostic.ess_bulk[idx]

    stats_parts = []
    if rhat is not None and not np.isnan(rhat):
        if rhat <= _RHAT_EXCELLENT_THRESHOLD:
            status = "✓"
        elif rhat <= _RHAT_ACCEPTABLE_THRESHOLD:
            status = "⚠"
        else:
            status = "✗"
        stats_parts.append(f"{status} R̂={rhat:.3f}")

    if ess is not None and not np.isnan(ess):
        stats_parts.append(f"ESS={ess:.0f}")

    if stats_parts:
        title = f"{param_name} ({', '.join(stats_parts)})"

    ax.set_title(title, fontsize=10)
