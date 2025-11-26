"""MCMC diagnostics module for PeakFit.

This module provides comprehensive diagnostics and visualizations for MCMC sampling,
following the Bayesian Analysis Reporting Guidelines (BARG) by Kruschke (2021).

The module includes:
- Convergence diagnostics (R-hat, ESS)
- Pure computation metrics (no plotting dependencies)
- Diagnostic plots (trace, corner, autocorrelation) - deprecated, use plotting.diagnostics

For new code, prefer:
- `core.diagnostics.metrics` for pure computation
- `plotting.diagnostics` for visualization

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

# Pure computation metrics (no matplotlib dependency)
from peakfit.core.diagnostics.metrics import (
    AutocorrelationResult,
    CorrelationPair,
    TraceMetrics,
    compute_all_trace_metrics,
    compute_autocorrelation,
    compute_correlation_matrix,
    compute_posterior_statistics,
    compute_trace_metrics,
    find_correlated_pairs,
)

# Backward compatibility - these will be deprecated
# Prefer using plotting.diagnostics for new code
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
    # Convergence diagnostics
    "ConvergenceDiagnostics",
    "compute_ess",
    "compute_rhat",
    "diagnose_convergence",
    "format_diagnostics_table",
    # Pure computation metrics
    "AutocorrelationResult",
    "CorrelationPair",
    "TraceMetrics",
    "compute_all_trace_metrics",
    "compute_autocorrelation",
    "compute_correlation_matrix",
    "compute_posterior_statistics",
    "compute_trace_metrics",
    "find_correlated_pairs",
    # Visualization (backward compatibility - prefer plotting.diagnostics)
    "plot_autocorrelation",
    "plot_corner",
    "plot_correlation_pairs",
    "plot_marginal_distributions",
    "plot_posterior_summary",
    "plot_trace",
    "save_diagnostic_plots",
]
