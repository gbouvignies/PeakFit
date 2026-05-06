"""Diagnostic plotting package."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from peakfit.plot.diagnostics.autocorr import plot_autocorrelation
from peakfit.plot.diagnostics.corner import plot_corner, plot_correlation_pairs
from peakfit.plot.diagnostics.summary import (
    plot_marginal_distributions,
    plot_posterior_summary,
)
from peakfit.plot.diagnostics.trace import plot_trace

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.diagnostics.convergence import ConvergenceDiagnostics
    from peakfit.shared.typing import FloatArray


def save_diagnostic_plots(
    chains: FloatArray,
    parameter_names: list[str],
    output_path: Path,
    burn_in: int = 0,
    thin: int = 1,
    diagnostics: ConvergenceDiagnostics | None = None,
    truths: FloatArray | None = None,
) -> None:
    """Generate and save all diagnostic plots to a PDF file.

    Args:
        chains: Array of shape (n_chains, n_samples, n_params)
        parameter_names: List of parameter names
        output_path: Path to save PDF
        burn_in: Number of burn-in samples
        thin: Thinning factor
        diagnostics: Optional diagnostics for annotations
        truths: Optional best-fit values
    """
    # Remove burn-in before flattening for marginal/correlation plots
    chains_post_burnin = chains[:, burn_in:, :] if burn_in > 0 else chains
    samples_flat = chains_post_burnin.reshape(-1, chains_post_burnin.shape[2])

    with PdfPages(output_path) as pdf:
        # Page 1: Trace plots
        fig_trace = plot_trace(chains, parameter_names, burn_in, thin=thin, diagnostics=diagnostics)
        pdf.savefig(fig_trace, bbox_inches="tight")
        plt.close(fig_trace)

        # Page 2: Corner plot
        fig_corner = plot_corner(samples_flat, parameter_names, truths)
        pdf.savefig(fig_corner, bbox_inches="tight")
        plt.close(fig_corner)

        # Page 3: Autocorrelation plots
        fig_autocorr = plot_autocorrelation(chains, parameter_names, thin=thin)
        pdf.savefig(fig_autocorr, bbox_inches="tight")
        plt.close(fig_autocorr)

        # Page 4: Posterior summary
        fig_summary = plot_posterior_summary(samples_flat, parameter_names)
        pdf.savefig(fig_summary, bbox_inches="tight")
        plt.close(fig_summary)


__all__ = [
    "plot_autocorrelation",
    "plot_corner",
    "plot_correlation_pairs",
    "plot_marginal_distributions",
    "plot_posterior_summary",
    "plot_trace",
    "save_diagnostic_plots",
]
