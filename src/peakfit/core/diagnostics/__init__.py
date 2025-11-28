"""MCMC diagnostics module for PeakFit.

This module provides comprehensive diagnostics for MCMC sampling,
following the Bayesian Analysis Reporting Guidelines (BARG) by Kruschke (2021).

The module includes:
- Convergence diagnostics (R-hat, ESS)
- Pure computation metrics (no plotting dependencies)

For visualization, use `plotting.diagnostics` module.

References
----------
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

__all__ = [
    "AutocorrelationResult",
    "ConvergenceDiagnostics",
    "CorrelationPair",
    "TraceMetrics",
    "compute_all_trace_metrics",
    "compute_autocorrelation",
    "compute_correlation_matrix",
    "compute_ess",
    "compute_posterior_statistics",
    "compute_rhat",
    "compute_trace_metrics",
    "diagnose_convergence",
    "find_correlated_pairs",
    "format_diagnostics_table",
]
