"""Plotting module for PeakFit.

This module provides visualization functions for NMR data analysis:
- MCMC diagnostic plots (trace, corner, autocorrelation)
- Profile plots (intensity, CEST, CPMG)
- Interactive spectrum viewer
"""

from peakfit.plotting.diagnostics import (
    plot_autocorrelation,
    plot_corner,
    plot_correlation_pairs,
    plot_marginal_distributions,
    plot_posterior_summary,
    plot_trace,
    save_diagnostic_plots,
)
from peakfit.plotting.profiles import (
    intensity_to_r2eff,
    make_cest_figure,
    make_cpmg_figure,
    make_intensity_ensemble,
    make_intensity_figure,
    ncyc_to_nu_cpmg,
)

__all__ = [
    # Diagnostics
    "plot_autocorrelation",
    "plot_corner",
    "plot_correlation_pairs",
    "plot_marginal_distributions",
    "plot_posterior_summary",
    "plot_trace",
    "save_diagnostic_plots",
    # Profiles
    "intensity_to_r2eff",
    "make_cest_figure",
    "make_cpmg_figure",
    "make_intensity_ensemble",
    "make_intensity_figure",
    "ncyc_to_nu_cpmg",
]
