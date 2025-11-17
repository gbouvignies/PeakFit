"""Benchmarking utilities for PeakFit performance analysis.

Provides tools for profiling and comparing different optimization backends,
lineshape calculations, and fitting strategies.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.typing import FloatArray

if TYPE_CHECKING:
    from peakfit.clustering import Cluster
    from peakfit.core.fitting import Parameters


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    total_time: float  # seconds
    iterations: int
    mean_time: float  # seconds per iteration
    std_time: float  # standard deviation
    min_time: float
    max_time: float
    times: list[float] = field(default_factory=list)
    extra_info: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"<BenchmarkResult {self.name}: {self.mean_time*1000:.3f} ± "
            f"{self.std_time*1000:.3f} ms ({self.iterations} iterations)>"
        )


def benchmark_function(
    func: Callable[[], Any],
    name: str = "benchmark",
    n_iterations: int = 100,
    warmup: int = 5,
) -> BenchmarkResult:
    """Benchmark a function's execution time.

    Args:
        func: Function to benchmark (should take no arguments)
        name: Name for the benchmark
        n_iterations: Number of timed iterations
        warmup: Number of warmup iterations (not timed)

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Timed runs
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    times_array = np.array(times)

    return BenchmarkResult(
        name=name,
        total_time=float(np.sum(times_array)),
        iterations=n_iterations,
        mean_time=float(np.mean(times_array)),
        std_time=float(np.std(times_array)),
        min_time=float(np.min(times_array)),
        max_time=float(np.max(times_array)),
        times=times,
    )


def benchmark_lineshape_backends(
    n_points: int = 1000,
    n_iterations: int = 100,
) -> dict[str, BenchmarkResult]:
    """Benchmark different lineshape calculation backends.

    Args:
        n_points: Number of points in the lineshape
        n_iterations: Number of iterations for timing

    Returns:
        Dictionary of backend name to BenchmarkResult
    """
    from peakfit.core.backend import (
        _gaussian_numpy,
        _lorentzian_numpy,
        _pvoigt_numpy,
    )

    x = np.linspace(-50, 50, n_points).astype(np.float64)
    fwhm = 10.0
    eta = 0.5

    results = {}

    # NumPy backend (default)
    results["numpy_gaussian"] = benchmark_function(
        lambda: _gaussian_numpy(x, fwhm),
        "NumPy Gaussian",
        n_iterations,
    )

    results["numpy_lorentzian"] = benchmark_function(
        lambda: _lorentzian_numpy(x, fwhm),
        "NumPy Lorentzian",
        n_iterations,
    )

    results["numpy_pvoigt"] = benchmark_function(
        lambda: _pvoigt_numpy(x, fwhm, eta),
        "NumPy Pseudo-Voigt",
        n_iterations,
    )

    # Numba backend (if available)
    try:
        from peakfit.core.optimized import gaussian_jit, lorentzian_jit, pvoigt_jit

        results["numba_gaussian"] = benchmark_function(
            lambda: gaussian_jit(x, fwhm),
            "Numba Gaussian",
            n_iterations,
        )

        results["numba_lorentzian"] = benchmark_function(
            lambda: lorentzian_jit(x, fwhm),
            "Numba Lorentzian",
            n_iterations,
        )

        results["numba_pvoigt"] = benchmark_function(
            lambda: pvoigt_jit(x, fwhm, eta),
            "Numba Pseudo-Voigt",
            n_iterations,
        )
    except ImportError:
        pass

    # JAX backend (if available)
    try:
        from peakfit.core.jax_backend import (
            gaussian_jax,
            is_jax_available,
            lorentzian_jax,
            pseudo_voigt_jax,
        )

        if is_jax_available():
            results["jax_gaussian"] = benchmark_function(
                lambda: gaussian_jax(x, fwhm),
                "JAX Gaussian",
                n_iterations,
            )

            results["jax_lorentzian"] = benchmark_function(
                lambda: lorentzian_jax(x, fwhm),
                "JAX Lorentzian",
                n_iterations,
            )

            results["jax_pvoigt"] = benchmark_function(
                lambda: pseudo_voigt_jax(x, fwhm, eta),
                "JAX Pseudo-Voigt",
                n_iterations,
            )
    except ImportError:
        pass

    return results


def benchmark_fitting_methods(
    params: "Parameters",
    cluster: "Cluster",
    noise: float,
    n_iterations: int = 10,
) -> dict[str, BenchmarkResult]:
    """Benchmark different fitting methods.

    Args:
        params: Initial parameters
        cluster: Cluster to fit
        noise: Noise level
        n_iterations: Number of fitting iterations

    Returns:
        Dictionary of method name to BenchmarkResult
    """
    from peakfit.core.fitting import fit_cluster

    results = {}

    # Standard least-squares
    results["least_squares"] = benchmark_function(
        lambda: fit_cluster(params.copy(), cluster, noise, max_nfev=500),
        "Least Squares",
        n_iterations,
        warmup=1,
    )

    # Basin-hopping (if practical)
    try:
        from peakfit.core.advanced_optimization import fit_basin_hopping

        results["basin_hopping"] = benchmark_function(
            lambda: fit_basin_hopping(params.copy(), cluster, noise, n_iterations=10),
            "Basin Hopping (10 iter)",
            n_iterations // 2,  # Fewer iterations since it's slower
            warmup=1,
        )
    except ImportError:
        pass

    return results


def compare_backends_report(
    results: dict[str, BenchmarkResult],
) -> str:
    """Generate a formatted report comparing backends.

    Args:
        results: Dictionary of benchmark results

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 70,
        "PERFORMANCE BENCHMARK REPORT",
        "=" * 70,
        "",
    ]

    # Sort by mean time
    sorted_results = sorted(results.items(), key=lambda x: x[1].mean_time)

    # Find baseline (fastest)
    if sorted_results:
        baseline_time = sorted_results[0][1].mean_time
    else:
        baseline_time = 1.0

    for name, result in sorted_results:
        speedup = baseline_time / result.mean_time if result.mean_time > 0 else 0
        lines.append(f"{result.name:30s}")
        lines.append(f"  Mean time:    {result.mean_time*1000:10.3f} ms")
        lines.append(f"  Std dev:      {result.std_time*1000:10.3f} ms")
        lines.append(f"  Min time:     {result.min_time*1000:10.3f} ms")
        lines.append(f"  Max time:     {result.max_time*1000:10.3f} ms")
        if speedup != 1.0:
            lines.append(f"  Speedup:      {speedup:10.2f}x")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def profile_fit_cluster(
    params: "Parameters",
    cluster: "Cluster",
    noise: float,
) -> dict[str, float]:
    """Profile different stages of cluster fitting.

    Args:
        params: Initial parameters
        cluster: Cluster to fit
        noise: Noise level

    Returns:
        Dictionary of stage name to time in seconds
    """
    from peakfit.computing import calculate_shapes, residuals
    from peakfit.core.fitting import fit_cluster

    profile = {}

    # Profile shape calculation
    start = time.perf_counter()
    for _ in range(100):
        calculate_shapes(params, cluster)
    profile["shape_calculation"] = (time.perf_counter() - start) / 100

    # Profile residual calculation
    start = time.perf_counter()
    for _ in range(100):
        residuals(params, cluster, noise)
    profile["residual_calculation"] = (time.perf_counter() - start) / 100

    # Profile full fitting
    start = time.perf_counter()
    fit_cluster(params.copy(), cluster, noise, max_nfev=500)
    profile["full_fit"] = time.perf_counter() - start

    return profile


def create_synthetic_cluster(
    n_peaks: int = 3,
    n_points_per_dim: int = 64,
    n_planes: int = 10,
) -> tuple["Cluster", float]:
    """Create a synthetic cluster for benchmarking.

    Args:
        n_peaks: Number of peaks
        n_points_per_dim: Points per spectral dimension
        n_planes: Number of planes (pseudo-3D)

    Returns:
        Tuple of (Cluster, noise_level)
    """
    from peakfit.clustering import Cluster
    from peakfit.peak import Peak
    from peakfit.shapes import PseudoVoigt

    # Create synthetic peaks
    peaks = []
    for i in range(n_peaks):
        x_pos = 20.0 + i * 15.0
        y_pos = 20.0 + i * 15.0

        positions = np.array([x_pos, y_pos])
        shapes = [
            PseudoVoigt(f"peak{i}_x", fwhm=5.0, eta=0.5),
            PseudoVoigt(f"peak{i}_y", fwhm=5.0, eta=0.5),
        ]

        peak = Peak(f"peak{i}", positions, shapes)
        peaks.append(peak)

    # Create synthetic data
    x = np.arange(n_points_per_dim, dtype=np.float64)
    y = np.arange(n_points_per_dim, dtype=np.float64)
    positions_2d = np.array(np.meshgrid(x, y)).reshape(2, -1)

    # Generate data with peaks
    data = np.zeros((n_planes, n_points_per_dim * n_points_per_dim))
    noise_level = 10.0

    # Add Gaussian noise
    data += np.random.normal(0, noise_level, data.shape)

    # Create cluster
    cluster = Cluster(positions_2d, peaks)
    cluster.data = data

    return cluster, noise_level
