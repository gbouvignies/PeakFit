"""Diagnostic plots for MCMC chains.

Implements publication-quality diagnostic visualizations following BARG guidelines:
- Trace plots showing chain evolution
- Corner plots (pair plots) showing marginal and joint distributions
- Autocorrelation plots for assessing mixing

These plots help assess MCMC convergence and identify correlations between parameters.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from peakfit.diagnostics.convergence import ConvergenceDiagnostics
from peakfit.typing import FloatArray


def plot_trace(
    chains: FloatArray,
    parameter_names: list[str],
    burn_in: int = 0,
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
        burn_in: Number of burn-in samples to mark (will be shown in gray)
        diagnostics: Optional diagnostics to annotate R-hat values
        max_params: Maximum number of parameters to plot (to avoid huge figures)

    Returns:
        Matplotlib Figure object
    """
    n_chains, n_samples, n_params = chains.shape
    n_params_plot = min(n_params, max_params)

    # Create figure with subplots
    n_cols = min(3, n_params_plot)
    n_rows = int(np.ceil(n_params_plot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))

    if n_params_plot == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Color palette for chains
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_chains, 10)))

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

        # Add R-hat annotation if available
        title = param_name
        if diagnostics is not None and i < len(diagnostics.rhat):
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

    fig.suptitle(
        f"MCMC Trace Plots ({n_chains} chains, {n_samples} samples)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig


def plot_corner(
    samples: FloatArray,
    parameter_names: list[str],
    truths: FloatArray | None = None,
    max_params: int = 15,
) -> Figure:
    """Create corner plot (pair plot) showing marginal and joint distributions.

    Corner plots show:
    - Diagonal: 1D marginal posterior distributions
    - Off-diagonal: 2D joint distributions (scatter or contours)
    - Correlations between parameters

    This is crucial for NMR fitting where chemical shifts, linewidths, and
    intensities can be correlated.

    Args:
        samples: Array of shape (n_total_samples, n_params) - flattened across all chains
        parameter_names: List of parameter names
        truths: Optional array of true/best-fit values to mark on plots
        max_params: Maximum number of parameters (corner plots get unwieldy with many params)

    Returns:
        Matplotlib Figure object
    """
    n_total_samples, n_params = samples.shape
    n_params_plot = min(n_params, max_params)

    # Create figure
    fig, axes = plt.subplots(
        n_params_plot,
        n_params_plot,
        figsize=(min(15, 2 * n_params_plot), min(15, 2 * n_params_plot)),
    )

    if n_params_plot == 1:
        axes = np.array([[axes]])

    # Plot each panel
    for i in range(n_params_plot):
        for j in range(n_params_plot):
            ax = axes[i, j]

            if j > i:
                # Upper triangle: hide
                ax.set_visible(False)
                continue

            if i == j:
                # Diagonal: 1D histogram (marginal distribution)
                ax.hist(
                    samples[:, i],
                    bins=30,
                    density=True,
                    alpha=0.7,
                    color="steelblue",
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Add median and credible intervals
                median = np.median(samples[:, i])
                ci_16, ci_84 = np.percentile(samples[:, i], [16, 84])

                ax.axvline(median, color="red", linestyle="-", linewidth=1.5, label="Median")
                ax.axvline(ci_16, color="red", linestyle="--", linewidth=1, alpha=0.7)
                ax.axvline(ci_84, color="red", linestyle="--", linewidth=1, alpha=0.7)

                # Mark truth if provided
                if truths is not None:
                    ax.axvline(truths[i], color="green", linestyle=":", linewidth=2, label="Best-fit")

                # Labels
                if i == 0:
                    ax.legend(fontsize=7, loc="upper right")

            else:
                # Off-diagonal: 2D scatter or density
                # Use scatter for smaller sample sizes, hexbin for larger
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
                    # Use hexbin for large sample sizes (more efficient)
                    ax.hexbin(
                        samples[:, j],
                        samples[:, i],
                        gridsize=30,
                        cmap="Blues",
                        mincnt=1,
                        rasterized=True,
                    )

                # Mark truths if provided
                if truths is not None:
                    ax.plot(truths[j], truths[i], "g+", markersize=10, markeredgewidth=2)

                # Compute correlation
                corr = np.corrcoef(samples[:, j], samples[:, i])[0, 1]
                if abs(corr) > 0.5:
                    # Annotate strong correlations
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
                ax.set_xlabel(parameter_names[j], fontsize=9)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
            else:
                ax.set_xticklabels([])

            if j == 0 and i > 0:
                ax.set_ylabel(parameter_names[i], fontsize=9)
                plt.setp(ax.yaxis.get_majorticklabels(), fontsize=7)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Corner Plot: Posterior Distributions and Correlations\n({n_total_samples:,} samples)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig


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
        for chain_idx in range(n_chains):
            chain_data = chains[chain_idx, :, i]

            # Compute autocorrelation
            autocorr = compute_autocorrelation(chain_data, max_lag)

            # Plot
            ax.plot(
                range(max_lag + 1),
                autocorr,
                alpha=0.6,
                linewidth=1,
                label=f"Chain {chain_idx + 1}" if i == 0 and chain_idx < 3 else None,
            )

        # Add reference lines
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.axhline(0.1, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axhline(-0.1, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

        # Estimate effective sample size from autocorrelation
        # (where autocorrelation first drops below threshold)
        mean_autocorr = np.mean(
            [compute_autocorrelation(chains[c, :, i], max_lag) for c in range(n_chains)], axis=0
        )
        lag_below_threshold = np.where(np.abs(mean_autocorr) < 0.1)[0]
        if len(lag_below_threshold) > 1:
            effective_lag = lag_below_threshold[1]  # Skip lag 0
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
    plt.tight_layout()

    return fig


def compute_autocorrelation(data: FloatArray, max_lag: int) -> FloatArray:
    """Compute autocorrelation function for a time series.

    Args:
        data: 1D array of samples
        max_lag: Maximum lag to compute

    Returns:
        Array of autocorrelation values from lag 0 to max_lag
    """
    n = len(data)
    max_lag = min(max_lag, n - 1)

    # Center the data
    data_centered = data - np.mean(data)
    var = np.var(data)

    if var == 0:
        return np.ones(max_lag + 1)

    # Compute using FFT for efficiency
    n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    fft_data = np.fft.rfft(data_centered, n=n_fft)
    autocorr_fft = np.fft.irfft(fft_data * np.conj(fft_data), n=n_fft)
    autocorr = autocorr_fft[:n] / (var * n)

    return autocorr[: max_lag + 1]


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
        truths: Optional best-fit values for corner plot
    """
    # Flatten chains for corner plot
    samples_flat = chains.reshape(-1, chains.shape[2])

    with PdfPages(output_path) as pdf:
        # Page 1: Trace plots
        fig_trace = plot_trace(chains, parameter_names, burn_in, diagnostics)
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


def plot_posterior_summary(
    samples: FloatArray,
    parameter_names: list[str],
    max_params: int = 30,
) -> Figure:
    """Create a compact summary plot of posterior distributions.

    Shows marginal distributions for all parameters in a single figure,
    useful for quick overview of many parameters.

    Args:
        samples: Array of shape (n_total_samples, n_params)
        parameter_names: List of parameter names
        max_params: Maximum number of parameters to display

    Returns:
        Matplotlib Figure object
    """
    n_total_samples, n_params = samples.shape
    n_params_plot = min(n_params, max_params)

    fig, ax = plt.subplots(figsize=(10, max(6, n_params_plot * 0.3)))

    # Compute statistics for each parameter
    medians = np.median(samples, axis=0)
    ci_16 = np.percentile(samples, 16, axis=0)
    ci_84 = np.percentile(samples, 84, axis=0)
    ci_025 = np.percentile(samples, 2.5, axis=0)
    ci_975 = np.percentile(samples, 97.5, axis=0)

    # Plot
    y_positions = np.arange(n_params_plot)

    # 95% CI
    ax.barh(
        y_positions,
        ci_975[:n_params_plot] - ci_025[:n_params_plot],
        left=ci_025[:n_params_plot],
        height=0.5,
        alpha=0.3,
        color="steelblue",
        label="95% CI",
    )

    # 68% CI (1 sigma)
    ax.barh(
        y_positions,
        ci_84[:n_params_plot] - ci_16[:n_params_plot],
        left=ci_16[:n_params_plot],
        height=0.5,
        alpha=0.6,
        color="steelblue",
        label="68% CI",
    )

    # Median
    ax.scatter(
        medians[:n_params_plot],
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
        f"Posterior Summary: Median and Credible Intervals\n({n_total_samples:,} samples)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    return fig
