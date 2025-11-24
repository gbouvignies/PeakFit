"""Performance analysis and profiling tools for PeakFit."""

from peakfit.analysis.benchmarks import (
    BenchmarkResult,
    benchmark_fitting_methods,
    benchmark_function,
    benchmark_lineshape_backends,
    compare_backends_report,
    create_synthetic_cluster,
    profile_fit_cluster,
)
from peakfit.analysis.profiling import (
    Profiler,
    ProfileReport,
    TimingResult,
    compare_fitting_methods,
    estimate_optimal_workers,
)

__all__ = [
    # Benchmarking
    "BenchmarkResult",
    "benchmark_fitting_methods",
    "benchmark_function",
    "benchmark_lineshape_backends",
    "compare_backends_report",
    "create_synthetic_cluster",
    "profile_fit_cluster",
    # Profiling
    "Profiler",
    "ProfileReport",
    "TimingResult",
    "compare_fitting_methods",
    "estimate_optimal_workers",
]
