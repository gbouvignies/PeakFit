"""Plotting module for PeakFit.

This module contains visualization functions that can be imported directly.
Plotting is also integrated into the Typer CLI (peakfit plot).

Submodules:
    - diagnostics: MCMC diagnostic plots (trace, corner, autocorrelation)
    - common: Shared plotting utilities
    - plots.spectra: Spectrum and peak plots
"""

from peakfit.plotting.diagnostics import (
    plot_autocorrelation,
    plot_corner,
    plot_posterior_summary,
    plot_trace,
    save_diagnostic_plots,
)

__all__ = [
    "plot_autocorrelation",
    "plot_corner",
    "plot_posterior_summary",
    "plot_trace",
    "save_diagnostic_plots",
]
