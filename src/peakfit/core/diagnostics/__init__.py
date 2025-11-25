"""MCMC diagnostics module for PeakFit.

This module provides comprehensive diagnostics and visualizations for MCMC sampling,
following the Bayesian Analysis Reporting Guidelines (BARG) by Kruschke (2021).

The module includes:
- Convergence diagnostics (R-hat, ESS)
- Diagnostic plots (trace, corner, autocorrelation)
- Interpretation guidelines for assessing MCMC quality

References:
    Kruschke, J. K. (2021). Bayesian analysis reporting guidelines.
    Nature Human Behaviour, 5(10), 1282-1291.
    https://doi.org/10.1038/s41562-021-01177-7
"""

from peakfit.core.diagnostics.convergence import (
    ConvergenceDiagnostics,
    compute_ess,
    compute_rhat,
    diagnose_convergence,
    format_diagnostics_table,
)
from peakfit.core.diagnostics.plots import (
    plot_autocorrelation,
    plot_corner,
    plot_correlation_pairs,
    plot_marginal_distributions,
    plot_posterior_summary,
    plot_trace,
    save_diagnostic_plots,
)

__all__ = [
    "ConvergenceDiagnostics",
    "compute_ess",
    "compute_rhat",
    "diagnose_convergence",
    "format_diagnostics_table",
    "plot_autocorrelation",
    "plot_corner",
    "plot_correlation_pairs",
    "plot_marginal_distributions",
    "plot_posterior_summary",
    "plot_trace",
    "save_diagnostic_plots",
]
