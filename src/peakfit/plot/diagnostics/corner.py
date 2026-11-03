"""Corner and correlation plots."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from peakfit.shared.typing import FloatArray

_SCATTER_SAMPLE_THRESHOLD = 2000
_STRONG_CORRELATION_THRESHOLD = 0.5
_MIN_PARTS_FOR_SHORT_NAME = 3


def plot_correlation_pairs(  # noqa: PLR0912
    samples: FloatArray,
    parameter_names: list[str],
    n_chains: int,
    n_samples: int,
    thin: int,
    truths: FloatArray | None = None,
    min_correlation: float = 0.5,
    max_pairs_per_page: int = 6,
    max_pairs_total: int = 24,
) -> list[Figure]:
    """Create 2D scatter plots for strongly correlated parameter pairs.

    Only plots pairs with |correlation| > min_correlation to focus on
    important relationships.

    Args:
        samples: Array of shape (n_total_samples, n_params)
        parameter_names: List of parameter names
        n_chains: Number of chains
        n_samples: Number of stored samples per chain
        thin: Thinning factor
        truths: Optional best-fit values
        min_correlation: Minimum |correlation| to plot
        max_pairs_per_page: Maximum pairs per page
        max_pairs_total: Maximum number of pairs to plot overall

    Returns:
    -------
        List of matplotlib Figure objects (one per page), empty if no strong correlations
    """
    n_total_samples, n_params = samples.shape

    # Find strongly correlated pairs
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_matrix = np.corrcoef(samples.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr_matrix, 1.0)
    strong_pairs: list[tuple[int, int, float]] = []

    for i in range(n_params):
        for j in range(i + 1, n_params):
            corr = corr_matrix[i, j]
            if abs(corr) >= min_correlation:
                strong_pairs.append((i, j, corr))

    # Sort by absolute correlation (strongest first)
    strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    if not strong_pairs:
        # No strong correlations - return empty list
        return []

    if max_pairs_total > 0:
        strong_pairs = strong_pairs[:max_pairs_total]

    # Create pages
    n_pages = int(np.ceil(len(strong_pairs) / max_pairs_per_page))
    figures: list[Figure] = []

    for page in range(n_pages):
        start_idx = page * max_pairs_per_page
        end_idx = min((page + 1) * max_pairs_per_page, len(strong_pairs))
        pairs_page = strong_pairs[start_idx:end_idx]

        # Create figure
        n_cols = 2
        n_rows = int(np.ceil(len(pairs_page) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))

        if len(pairs_page) == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (i, j, corr) in enumerate(pairs_page):
            ax = axes[idx]

            # Use hexbin for large samples, scatter for small
            if n_total_samples < _SCATTER_SAMPLE_THRESHOLD:
                ax.scatter(
                    samples[:, j],
                    samples[:, i],
                    s=2,
                    alpha=0.4,
                    color="steelblue",
                    rasterized=True,
                )
            else:
                ax.hexbin(
                    samples[:, j],
                    samples[:, i],
                    gridsize=40,
                    cmap="Blues",
                    mincnt=1,
                    rasterized=True,
                )

            # Mark best-fit if provided
            if truths is not None:
                ax.plot(
                    truths[j],
                    truths[i],
                    "g+",
                    markersize=15,
                    markeredgewidth=3,
                    label="Best-fit",
                )
                ax.legend(fontsize=9)

            # Labels with full parameter names
            ax.set_xlabel(parameter_names[j], fontsize=10)
            ax.set_ylabel(parameter_names[i], fontsize=10)
            ax.set_title(f"Correlation: r = {corr:.3f}", fontsize=11, fontweight="bold")

            # Format x-axis to prevent overlap
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

            ax.grid(alpha=0.3, linestyle=":")

        # Hide unused subplots
        for idx in range(len(pairs_page), len(axes)):
            axes[idx].set_visible(False)

        # Add page title
        total_steps = n_samples * thin
        page_title = (
            f"Correlation Plots: {n_chains} chains × {total_steps} iterations "
            f"(thinned by {thin}, |r| ≥ {min_correlation})"
        )
        if n_pages > 1:
            page_title += f" — Page {page + 1}/{n_pages}"
        fig.suptitle(page_title, fontsize=14, fontweight="bold")

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        figures.append(fig)

    return figures
