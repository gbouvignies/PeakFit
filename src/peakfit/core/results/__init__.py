"""Result models for PeakFit output system.

This module provides unified data models for representing fitting results,
statistics, and diagnostics in a serialization-friendly format.

Design Principles:
    - All numeric arrays use numpy for computation, but serialize to lists/base64
    - Units are always explicitly specified in field metadata
    - Asymmetric uncertainties (from MCMC posteriors) are first-class citizens
    - Clear separation between point estimates and full distributions
"""

from peakfit.core.results.builder import FitResultsBuilder
from peakfit.core.results.diagnostics import ConvergenceStatus, MCMCDiagnostics, ParameterDiagnostic
from peakfit.core.results.estimates import AmplitudeEstimate, ClusterEstimates, ParameterEstimate
from peakfit.core.results.fit_results import FitMethod, FitResults, OutputVerbosity
from peakfit.core.results.statistics import FitStatistics, ModelComparison, ResidualStatistics

__all__ = [
    "AmplitudeEstimate",
    "ClusterEstimates",
    "ConvergenceStatus",
    "FitMethod",
    "FitResults",
    "FitResultsBuilder",
    "FitStatistics",
    "MCMCDiagnostics",
    "ModelComparison",
    "OutputVerbosity",
    "ParameterDiagnostic",
    "ParameterEstimate",
    "ResidualStatistics",
]
