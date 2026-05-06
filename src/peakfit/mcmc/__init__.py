"""MCMC workflow slice (analysis, diagnostics, CLI-facing APIs)."""

from peakfit.mcmc.analysis import (
    MCMCAnalysisService,
    PeaksNotFoundError,
    format_mcmc_cluster_result,
)

__all__ = [
    "MCMCAnalysisService",
    "PeaksNotFoundError",
    "format_mcmc_cluster_result",
]
