"""Corner and correlation plots."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from peakfit.shared.typing import FloatArray

_SCATTER_SAMPLE_THRESHOLD = 2000
_STRONG_CORRELATION_THRESHOLD = 0.5
_MIN_PARTS_FOR_SHORT_NAME = 3


def plot_corner(
    samples: FloatArray,
    parameter_names: list[str],
    truths: FloatArray | None = None,
    max_params: int = 15,
) -> Figure:
    """Create corner plot showing marginal and joint distributions.

    Corner plots show:
    - Diagonal: 1D marginal posterior distributions
    - Off-diagonal: 2D joint distributions
    - Correlations between parameters

    Args:
        samples: Array of shape (n_total_samples, n_params)
        parameter_names: List of parameter names
        truths: Optional array of best-fit values
        max_params: Maximum number of parameters

    Returns:
    -------
        Matplotlib Figure object
    """
    n_total_samples, n_params = samples.shape
    n_params_plot = min(n_params, max_params)

    short_names = _shorten_parameter_names(parameter_names, n_params_plot)

    # Create figure
    fig, axes_raw = plt.subplots(
        n_params_plot,
        n_params_plot,
        figsize=(min(16, 2.2 * n_params_plot), min(16, 2.2 * n_params_plot)),
    )

    axes = np.atleast_2d(np.array(axes_raw, dtype=object)).reshape(
        n_params_plot,
        n_params_plot,
    )

    # Plot each panel
    for i in range(n_params_plot):
        for j in range(n_params_plot):
            ax = axes[i, j]
            if not _plot_corner_panel(
                ax=ax,
                i=i,
                j=j,
                samples=samples,
                truths=truths,
                n_total_samples=n_total_samples,
            ):
                continue

            _style_corner_axis(
                ax=ax,
                i=i,
                j=j,
                n_params_plot=n_params_plot,
                short_names=short_names,
            )

    fig.suptitle(
        f"Corner Plot: Posterior Distributions ({n_total_samples:,} samples)",
        fontsize=14,
        fontweight="bold",
    )

    guide_text = (
        "Diagonal: marginal posteriors (red = median, dashed = 68% CI) | "
        "Off-diagonal: joint distributions | Green + = best-fit"
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

    plt.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=1.5, w_pad=1.5)
    return fig


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


def _plot_diagonal_hist(
    ax: Axes,
    samples_col: FloatArray,
    truth_val: float | None = None,
) -> None:
    """Plot 1D histogram for diagonal panel."""
    ax.hist(
        samples_col,
        bins=30,
        density=True,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )

    median = float(np.median(samples_col))
    percentile_values = np.atleast_1d(np.percentile(samples_col, [16, 84]))
    ci_16 = float(percentile_values[0])
    ci_84 = float(percentile_values[-1])

    ax.axvline(median, color="red", linestyle="-", linewidth=1.5)
    ax.axvline(ci_16, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(ci_84, color="red", linestyle="--", linewidth=1, alpha=0.7)

    if truth_val is not None:
        ax.axvline(truth_val, color="green", linestyle=":", linewidth=2)


def _plot_off_diagonal_scatter(
    ax: Axes,
    samples_j: FloatArray,
    samples_i: FloatArray,
    truth_j: float | None,
    truth_i: float | None,
    n_total_samples: int,
) -> None:
    """Plot 2D scatter or hexbin for off-diagonal panel."""
    if n_total_samples < _SCATTER_SAMPLE_THRESHOLD:
        ax.scatter(
            samples_j,
            samples_i,
            s=1,
            alpha=0.3,
            color="steelblue",
            rasterized=True,
        )
    else:
        ax.hexbin(
            samples_j,
            samples_i,
            gridsize=30,
            cmap="Blues",
            mincnt=1,
            rasterized=True,
        )

    if truth_j is not None and truth_i is not None:
        ax.plot(truth_j, truth_i, "g+", markersize=10, markeredgewidth=2)

    corr = np.corrcoef(samples_j, samples_i)[0, 1]
    if abs(corr) > _STRONG_CORRELATION_THRESHOLD:
        ax.text(
            0.05,
            0.95,
            f"r={corr:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )


def _shorten_parameter_names(parameter_names: list[str], n_params: int) -> list[str]:
    """Shorten parameter names for compact display in corner plots."""
    short_names: list[str] = []
    for name in parameter_names[:n_params]:
        if "_" not in name:
            short_names.append(name[:12])
            continue

        parts = name.split("_")
        if len(parts) >= _MIN_PARTS_FOR_SHORT_NAME:
            short_names.append("_".join(parts[-2:]))
        else:
            short_names.append(parts[-1])
    return short_names


def _plot_corner_panel(
    *,
    ax: Axes,
    i: int,
    j: int,
    samples: FloatArray,
    truths: FloatArray | None,
    n_total_samples: int,
) -> bool:
    """Plot a single corner panel.

    Returns:
        True if the axis should be visible, False if it is upper-triangular and hidden.
    """
    if j > i:
        ax.set_visible(False)
        return False

    if i == j:
        truth_val = truths[i] if truths is not None else None
        _plot_diagonal_hist(ax, samples[:, i], truth_val)
        return True

    truth_j = truths[j] if truths is not None else None
    truth_i = truths[i] if truths is not None else None
    _plot_off_diagonal_scatter(
        ax,
        samples[:, j],
        samples[:, i],
        truth_j,
        truth_i,
        n_total_samples,
    )
    return True


def _style_corner_axis(
    *,
    ax: Axes,
    i: int,
    j: int,
    n_params_plot: int,
    short_names: list[str],
) -> None:
    """Apply consistent labels/ticks styling for a corner plot axis."""
    if i == n_params_plot - 1:
        ax.set_xlabel(short_names[j], fontsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=6)
    else:
        ax.set_xticklabels([])

    if j == 0 and i > 0:
        ax.set_ylabel(short_names[i], fontsize=8)
        plt.setp(ax.yaxis.get_majorticklabels(), fontsize=6)
    else:
        ax.set_yticklabels([])

    ax.tick_params(labelsize=6)
