"""MCMC workflow slice (analysis, diagnostics, CLI-facing APIs)."""

from peakfit.mcmc.analysis import (
    PeaksNotFoundError,
    format_mcmc_cluster_result,
    run_mcmc_analysis,
)

__all__ = [
    "PeaksNotFoundError",
    "format_mcmc_cluster_result",
    "run_mcmc_analysis",
]
