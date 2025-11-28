"""Visualization functions for MCMC diagnostics.

This module contains only plotting functions that consume pre-computed
metrics from core/diagnostics/metrics.py and convergence.py.

All functions return matplotlib Figure objects for flexible usage.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from peakfit.core.diagnostics.convergence import ConvergenceDiagnostics
from peakfit.core.diagnostics.metrics import (
    AutocorrelationResult,
    TraceMetrics,
    compute_all_trace_metrics,
    compute_autocorrelation,
    compute_posterior_statistics,
)
from peakfit.core.shared.typing import FloatArray

if TYPE_CHECKING:
    pass


def plot_trace(
    chains: FloatArray,
    parameter_names: list[str],
    burn_in: int = 0,
    metrics: list[TraceMetrics] | None = None,
    diagnostics: ConvergenceDiagnostics | None = None,
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
        burn_in: Number of burn-in samples to mark (shown in gray)
        metrics: Pre-computed TraceMetrics (computed if not provided)
        diagnostics: Optional ConvergenceDiagnostics for R-hat values
        max_params: Maximum number of parameters to plot

    Returns:
        Matplotlib Figure object
    """
    n_chains, n_samples, n_params = chains.shape
    n_params_plot = min(n_params, max_params)

    # Compute metrics if not provided
    if metrics is None and diagnostics is None:
        chains_post_burnin = chains[:, burn_in:, :] if burn_in > 0 else chains
        metrics = compute_all_trace_metrics(chains_post_burnin)

    # Create figure
    n_cols = min(3, n_params_plot)
    n_rows = int(np.ceil(n_params_plot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))

    if n_params_plot == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Color palette for chains
    color_map = plt.get_cmap("tab10")
    colors = color_map(np.linspace(0, 1, min(n_chains, 10)))

    for i in range(n_params_plot):
        ax = axes[i]
        param_name = parameter_names[i]

        # Plot each chain
        for chain_idx in range(n_chains):
            chain_data = chains[chain_idx, :, i]

            # Plot burn-in period in gray if specified
            if burn_in > 0:
                ax.plot(
                    range(burn_in),
                    chain_data[:burn_in],
                    color="gray",
                    alpha=0.3,
                    linewidth=0.5,
                )
                ax.plot(
                    range(burn_in, n_samples),
                    chain_data[burn_in:],
                    color=colors[chain_idx % len(colors)],
                    alpha=0.7,
                    linewidth=0.5,
                    label=f"Chain {chain_idx + 1}" if i == 0 and chain_idx < 3 else None,
                )
            else:
                ax.plot(
                    chain_data,
                    color=colors[chain_idx % len(colors)],
                    alpha=0.7,
                    linewidth=0.5,
                    label=f"Chain {chain_idx + 1}" if i == 0 and chain_idx < 3 else None,
                )

        # Mark burn-in boundary
        if burn_in > 0:
            ax.axvline(burn_in, color="red", linestyle="--", alpha=0.5, linewidth=1)

        # Add R-hat annotation
        title = param_name
        if metrics is not None and i < len(metrics):
            rhat = metrics[i].rhat
            if not np.isnan(rhat):
                status = "✓" if rhat <= 1.01 else "⚠" if rhat <= 1.05 else "✗"
                title = f"{param_name} ({status} R̂={rhat:.3f})"
        elif diagnostics is not None and i < len(diagnostics.rhat):
            rhat = diagnostics.rhat[i]
            if not np.isnan(rhat):
                status = "✓" if rhat <= 1.01 else "⚠" if rhat <= 1.05 else "✗"
                title = f"{param_name} ({status} R̂={rhat:.3f})"

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # Legend only on first subplot
        if i == 0 and n_chains <= 10:
            ax.legend(fontsize=8, loc="best")

    # Hide unused subplots
    for i in range(n_params_plot, len(axes)):
        axes[i].set_visible(False)

    # Add title
    title_text = f"MCMC Trace Plots ({n_chains} chains, {n_samples} samples)"
    if burn_in > 0:
        title_text += f"\nGray = burn-in ({burn_in} samples) | Red line = burn-in cutoff"
    fig.suptitle(title_text, fontsize=13, fontweight="bold")

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


def plot_marginal_distributions(
    samples: FloatArray,
    parameter_names: list[str],
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
        truths: Optional best-fit values
        diagnostics: Optional convergence diagnostics
        max_params_per_page: Maximum parameters per page

    Returns:
        List of matplotlib Figure objects (one per page)
    """
    n_total_samples, n_params = samples.shape
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
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.3, linestyle=":")

        # Hide unused subplots
        for i in range(n_params_page, len(axes)):
            axes[i].set_visible(False)

        # Add page title
        page_title = f"Marginal Posterior Distributions ({n_total_samples:,} samples)"
        if n_pages > 1:
            page_title += f" — Page {page + 1}/{n_pages}"
        fig.suptitle(page_title, fontsize=16, fontweight="bold")

        plt.tight_layout(rect=(0, 0, 1, 0.97))
        figures.append(fig)

    return figures


def plot_correlation_pairs(
    samples: FloatArray,
    parameter_names: list[str],
    truths: FloatArray | None = None,
    min_correlation: float = 0.5,
    max_pairs_per_page: int = 6,
) -> list[Figure]:
    """Create 2D scatter plots for strongly correlated parameter pairs.

    Only plots pairs with |correlation| > min_correlation to focus on
    important relationships.

    Args:
        samples: Array of shape (n_total_samples, n_params)
        parameter_names: List of parameter names
        truths: Optional best-fit values
        min_correlation: Minimum |correlation| to plot
        max_pairs_per_page: Maximum pairs per page

    Returns:
        List of matplotlib Figure objects (one per page), empty if no strong correlations
    """
    n_total_samples, n_params = samples.shape

    # Find strongly correlated pairs
    corr_matrix = np.corrcoef(samples.T)
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
            if n_total_samples < 2000:
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
            ax.grid(alpha=0.3, linestyle=":")

        # Hide unused subplots
        for idx in range(len(pairs_page), len(axes)):
            axes[idx].set_visible(False)

        # Add page title
        page_title = f"Parameter Correlations (|r| ≥ {min_correlation})"
        if n_pages > 1:
            page_title += f" — Page {page + 1}/{n_pages}"
        page_title += f"\n{len(strong_pairs)} strongly correlated pair(s) found"
        fig.suptitle(page_title, fontsize=16, fontweight="bold")

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        figures.append(fig)

    return figures


def plot_autocorrelation(
    chains: FloatArray,
    parameter_names: list[str],
    max_lag: int = 100,
    max_params: int = 20,
) -> Figure:
    """Create autocorrelation plots for MCMC chains.

    Autocorrelation plots show how correlated samples are with previous samples.
    Good mixing shows autocorrelation dropping quickly to zero.

    Args:
        chains: Array of shape (n_chains, n_samples, n_params)
        parameter_names: List of parameter names
        max_lag: Maximum lag to compute
        max_params: Maximum number of parameters to plot

    Returns:
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
                label=f"Chain {chain_idx + 1}" if i == 0 and chain_idx < 3 else None,
            )

        # Add reference lines
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.axhline(0.1, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axhline(-0.1, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

        # Mark effective decorrelation lag from mean autocorrelation
        mean_autocorr = np.mean([r.autocorr for r in autocorr_results], axis=0)
        lag_below_threshold = np.where(np.abs(mean_autocorr) < 0.1)[0]
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

        if i == 0 and n_chains <= 10:
            ax.legend(fontsize=8, loc="upper right")

    # Hide unused subplots
    for i in range(n_params_plot, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f"Autocorrelation Plots ({n_chains} chains)",
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
        Matplotlib Figure object
    """
    n_total_samples, n_params = samples.shape
    n_params_plot = min(n_params, max_params)

    # Shorten parameter names for display
    short_names = []
    for name in parameter_names[:n_params_plot]:
        if "_" in name:
            parts = name.split("_")
            if len(parts) >= 2:
                short_names.append("_".join(parts[-2:]) if len(parts) > 2 else parts[-1])
            else:
                short_names.append(parts[-1])
        else:
            short_names.append(name[:12])

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

            if j > i:
                ax.set_visible(False)
                continue

            if i == j:
                # Diagonal: 1D histogram
                ax.hist(
                    samples[:, i],
                    bins=30,
                    density=True,
                    alpha=0.7,
                    color="steelblue",
                    edgecolor="black",
                    linewidth=0.5,
                )

                median = np.median(samples[:, i])
                percentile_values = np.atleast_1d(np.percentile(samples[:, i], [16, 84]))
                ci_16 = float(percentile_values[0])
                ci_84 = float(percentile_values[-1])

                ax.axvline(median, color="red", linestyle="-", linewidth=1.5)
                ax.axvline(ci_16, color="red", linestyle="--", linewidth=1, alpha=0.7)
                ax.axvline(ci_84, color="red", linestyle="--", linewidth=1, alpha=0.7)

                if truths is not None:
                    ax.axvline(truths[i], color="green", linestyle=":", linewidth=2)
            else:
                # Off-diagonal: 2D scatter or density
                if n_total_samples < 2000:
                    ax.scatter(
                        samples[:, j],
                        samples[:, i],
                        s=1,
                        alpha=0.3,
                        color="steelblue",
                        rasterized=True,
                    )
                else:
                    ax.hexbin(
                        samples[:, j],
                        samples[:, i],
                        gridsize=30,
                        cmap="Blues",
                        mincnt=1,
                        rasterized=True,
                    )

                if truths is not None:
                    ax.plot(truths[j], truths[i], "g+", markersize=10, markeredgewidth=2)

                corr = np.corrcoef(samples[:, j], samples[:, i])[0, 1]
                if abs(corr) > 0.5:
                    ax.text(
                        0.05,
                        0.95,
                        f"r={corr:.2f}",
                        transform=ax.transAxes,
                        fontsize=8,
                        verticalalignment="top",
                        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
                    )

            # Axis labels
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
        Matplotlib Figure object
    """
    n_total_samples, n_params = samples.shape
    n_params_plot = min(n_params, max_params)

    # Compute statistics using metrics module
    stats = compute_posterior_statistics(samples[:, :n_params_plot])

    fig, ax = plt.subplots(figsize=(10, max(6, n_params_plot * 0.3)))

    y_positions = np.arange(n_params_plot)

    # 95% CI
    ax.barh(
        y_positions,
        stats["ci_975"] - stats["ci_025"],
        left=stats["ci_025"],
        height=0.5,
        alpha=0.3,
        color="steelblue",
        label="95% CI",
    )

    # 68% CI
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


def save_diagnostic_plots(
    chains: FloatArray,
    parameter_names: list[str],
    output_path: Path,
    burn_in: int = 0,
    diagnostics: ConvergenceDiagnostics | None = None,
    truths: FloatArray | None = None,
) -> None:
    """Generate and save all diagnostic plots to a PDF file.

    Args:
        chains: Array of shape (n_chains, n_samples, n_params)
        parameter_names: List of parameter names
        output_path: Path to save PDF
        burn_in: Number of burn-in samples
        diagnostics: Optional diagnostics for annotations
        truths: Optional best-fit values
    """
    # Remove burn-in before flattening for marginal/correlation plots
    chains_post_burnin = chains[:, burn_in:, :] if burn_in > 0 else chains
    samples_flat = chains_post_burnin.reshape(-1, chains_post_burnin.shape[2])

    with PdfPages(output_path) as pdf:
        # Page 1: Trace plots
        fig_trace = plot_trace(chains, parameter_names, burn_in, diagnostics=diagnostics)
        pdf.savefig(fig_trace, bbox_inches="tight")
        plt.close(fig_trace)

        # Page 2: Corner plot
        fig_corner = plot_corner(samples_flat, parameter_names, truths)
        pdf.savefig(fig_corner, bbox_inches="tight")
        plt.close(fig_corner)

        # Page 3: Autocorrelation plots
        fig_autocorr = plot_autocorrelation(chains, parameter_names)
        pdf.savefig(fig_autocorr, bbox_inches="tight")
        plt.close(fig_autocorr)

        # Page 4: Posterior summary
        fig_summary = plot_posterior_summary(samples_flat, parameter_names)
        pdf.savefig(fig_summary, bbox_inches="tight")
        plt.close(fig_summary)
