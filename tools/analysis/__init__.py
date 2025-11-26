"""Performance analysis and profiling tools for PeakFit development.

This package contains development utilities that are not part of the
production PeakFit package. They require PeakFit to be installed to run.

Modules:
    - benchmarks: Performance benchmarking utilities
    - profiling: Execution timing and profiling
"""

from analysis.benchmarks import (
    BenchmarkResult,
    benchmark_fitting_methods,
    benchmark_function,
    benchmark_lineshape_backends,
    compare_backends_report,
    create_synthetic_cluster,
    profile_fit_cluster,
)
from analysis.profiling import (
    Profiler,
    ProfileReport,
    TimingResult,
    compare_fitting_methods,
    estimate_optimal_workers,
)

__all__ = [
    "BenchmarkResult",
    "ProfileReport",
    "Profiler",
    "TimingResult",
    "benchmark_fitting_methods",
    "benchmark_function",
    "benchmark_lineshape_backends",
    "compare_backends_report",
    "compare_fitting_methods",
    "create_synthetic_cluster",
    "estimate_optimal_workers",
    "profile_fit_cluster",
]
