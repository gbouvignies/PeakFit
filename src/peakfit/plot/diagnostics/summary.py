"""Posterior summary plots."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from peakfit.shared.typing import FloatArray


def plot_marginal_distributions(
    samples: FloatArray,
    parameter_names: list[str],
    n_chains: int,
    n_samples: int,
    truths: FloatArray | None = None,
    max_params_per_page: int = 12,
) -> list[Figure]:
    """Create clear 1D marginal distribution plots with full parameter names.

    Shows histograms of posterior distributions for each parameter with:
    - Full parameter names (no truncation)
    - Median and 68% credible intervals
    - Best-fit values if provided

    Args:
        samples: Array of shape (n_total_samples, n_params)
        parameter_names: List of parameter names (full names)
        n_chains: Number of chains
        n_samples: Number of kept steps per walker
        truths: Optional best-fit values
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
        page_title = f"Marginal Distributions: {n_chains} walkers × {n_samples} steps"
        if n_pages > 1:
            page_title += f" — Page {page + 1}/{n_pages}"
        fig.suptitle(page_title, fontsize=14, fontweight="bold")

        plt.tight_layout(rect=(0, 0, 1, 0.97))
        figures.append(fig)

    return figures
