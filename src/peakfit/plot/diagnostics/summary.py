"""Posterior summary plots."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from peakfit.engine.diagnostics.metrics import compute_posterior_statistics

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from peakfit.engine.diagnostics.convergence import ConvergenceDiagnostics
    from peakfit.shared.typing import FloatArray


def plot_posterior_summary(
    samples: FloatArray,
    parameter_names: list[str],
    max_params: int = 30,
) -> Figure:
    """Create a compact summary plot of posterior distributions.

    Shows marginal distributions for all parameters in a single figure.

    Args:
        samples: Array of shape (n_total_samples, n_params)
        parameter_names: List of parameter names
        max_params: Maximum number of parameters to display

    Returns:
    -------
        Matplotlib Figure object
    """
    n_total_samples, n_params = samples.shape
    n_params_plot = min(n_params, max_params)

    # Compute statistics using metrics module
    stats = compute_posterior_statistics(samples[:, :n_params_plot])

    fig, ax = plt.subplots(figsize=(10, max(6, n_params_plot * 0.3)))

    y_positions = np.arange(n_params_plot)

    # 95 percent CI
    ax.barh(
        y_positions,
        stats["ci_975"] - stats["ci_025"],
        left=stats["ci_025"],
        height=0.5,
        alpha=0.3,
        color="steelblue",
        label="95% CI",
    )

    # 68 percent CI
    ax.barh(
        y_positions,
        stats["ci_84"] - stats["ci_16"],
        left=stats["ci_16"],
        height=0.5,
        alpha=0.6,
        color="steelblue",
        label="68% CI",
    )

    # Median
    ax.scatter(
        stats["medians"],
        y_positions,
        color="red",
        s=50,
        zorder=10,
        marker="|",
        linewidths=2,
        label="Median",
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(parameter_names[:n_params_plot], fontsize=9)
    ax.set_xlabel("Parameter Value", fontsize=11)
    ax.set_title(
        f"Posterior Summary ({n_total_samples:,} samples)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, axis="x", alpha=0.3)

    guide_text = "Red line = median | Dark blue = 68% CI (±1σ) | Light blue = 95% CI"
    fig.text(
        0.5,
        0.02,
        guide_text,
        ha="center",
        fontsize=8,
        style="italic",
        color="gray",
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_marginal_distributions(
    samples: FloatArray,
    parameter_names: list[str],
    n_chains: int,
    n_samples: int,
    thin: int,
    truths: FloatArray | None = None,
    diagnostics: ConvergenceDiagnostics | None = None,
    max_params_per_page: int = 12,
) -> list[Figure]:
    """Create clear 1D marginal distribution plots with full parameter names.

    Shows histograms of posterior distributions for each parameter with:
    - Full parameter names (no truncation)
    - Median and 68% credible intervals
    - R-hat and ESS values if diagnostics provided
    - Best-fit values if provided

    Args:
        samples: Array of shape (n_total_samples, n_params)
        parameter_names: List of parameter names (full names)
        n_chains: Number of chains
        n_samples: Number of stored samples per chain
        thin: Thinning factor
        truths: Optional best-fit values
        diagnostics: Optional convergence diagnostics
        max_params_per_page: Maximum parameters per page

    Returns:
    -------
        List of matplotlib Figure objects (one per page)
    """
    _n_total_samples, n_params = samples.shape
    n_pages = int(np.ceil(n_params / max_params_per_page))
    figures: list[Figure] = []

    for page in range(n_pages):
        start_idx = page * max_params_per_page
        end_idx = min((page + 1) * max_params_per_page, n_params)
        n_params_page = end_idx - start_idx

        # Create figure with subplots
        n_cols = min(3, n_params_page)
        n_rows = int(np.ceil(n_params_page / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

        if n_params_page == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, param_idx in enumerate(range(start_idx, end_idx)):
            ax = axes[i]
            param_name = parameter_names[param_idx]
            param_samples = samples[:, param_idx]

            # Plot histogram
            ax.hist(
                param_samples,
                bins=40,
                density=True,
                alpha=0.7,
                color="steelblue",
                edgecolor="black",
                linewidth=0.5,
            )

            # Calculate statistics
            median = np.median(param_samples)
            percentiles = np.atleast_1d(np.percentile(param_samples, [16, 84]))
            ci_16 = float(percentiles[0])
            ci_84 = float(percentiles[1])

            # Mark median and credible intervals
            ax.axvline(median, color="red", linestyle="-", linewidth=2, label="Median", zorder=10)
            ax.axvline(ci_16, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="68% CI")
            ax.axvline(ci_84, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
            ax.axvspan(ci_16, ci_84, alpha=0.15, color="red")

            # Mark best-fit if provided
            if truths is not None:
                ax.axvline(
                    truths[param_idx],
                    color="green",
                    linestyle=":",
                    linewidth=2,
                    label="Best-fit",
                    zorder=10,
                )

            # Add statistics text box
            stats_text = f"Median: {median:.6f}\n68% CI: [{ci_16:.6f}, {ci_84:.6f}]"
            if diagnostics is not None:
                rhat = diagnostics.rhat[param_idx]
                ess_bulk = diagnostics.ess_bulk[param_idx]
                stats_text += f"\nR̂: {rhat:.4f}"
                stats_text += f"\nESS: {ess_bulk:.0f}"

            ax.text(
                0.98,
                0.97,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                horizontalalignment="right",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

            # Labels and title
            ax.set_xlabel("Parameter Value", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(param_name, fontsize=11, fontweight="bold")

            # Format x-axis to prevent overlap
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.3, linestyle=":")

        # Hide unused subplots
        for i in range(n_params_page, len(axes)):
            axes[i].set_visible(False)

        # Add page title
        total_steps = n_samples * thin
        page_title = (
            f"Marginal Distributions: {n_chains} chains × {total_steps} iterations "
            f"(thinned by {thin})"
        )
        if n_pages > 1:
            page_title += f" — Page {page + 1}/{n_pages}"
        fig.suptitle(page_title, fontsize=14, fontweight="bold")

        plt.tight_layout(rect=(0, 0, 1, 0.97))
        figures.append(fig)

    return figures
