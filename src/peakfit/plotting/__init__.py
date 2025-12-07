"""Plotting module for PeakFit.

This module provides visualization functions for NMR data analysis:
- MCMC diagnostic plots (trace, corner, autocorrelation)
- Profile plots (intensity, CEST, CPMG)
- Interactive spectrum viewer

Deprecated: prefer :mod:`peakfit.contrib.plotting`. This alias stays for
backward compatibility.
"""

from __future__ import annotations

import os
import warnings

_warn_deprecated = os.environ.get("PEAKFIT_WARN_DEPRECATED", "").lower() not in {
    "",
    "0",
    "false",
}
if _warn_deprecated:
    warnings.warn(
        "'peakfit.plotting' is deprecated; use 'peakfit.contrib.plotting' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

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
    "intensity_to_r2eff",
    "make_cest_figure",
    "make_cpmg_figure",
    "make_intensity_ensemble",
    "make_intensity_figure",
    "ncyc_to_nu_cpmg",
    "plot_autocorrelation",
    "plot_corner",
    "plot_correlation_pairs",
    "plot_marginal_distributions",
    "plot_posterior_summary",
    "plot_trace",
    "save_diagnostic_plots",
]
