"""Benchmarking utilities for PeakFit performance analysis.

Provides tools for profiling and comparing different optimization backends,
lineshape calculations, and fitting strategies.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from peakfit.core.shared.constants import BENCHMARK_MAX_NFEV

if TYPE_CHECKING:
    from peakfit.core.domain.cluster import Cluster
    from peakfit.core.fitting.parameters import Parameters


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
    extra_info: dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"<BenchmarkResult {self.name}: {self.mean_time * 1000:.3f} ± "
            f"{self.std_time * 1000:.3f} ms ({self.iterations} iterations)>"
        )


def benchmark_function(
    func: Callable[[], object],
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
    from peakfit.core.lineshapes.functions import gaussian, lorentzian, pvoigt

    x = np.linspace(-50, 50, n_points).astype(np.float64)
    fwhm = 10.0
    eta = 0.5

    results = {}

    # NumPy backend (default)
    results["numpy_gaussian"] = benchmark_function(
        lambda: gaussian(x, fwhm),
        "NumPy Gaussian",
        n_iterations,
    )

    results["numpy_lorentzian"] = benchmark_function(
        lambda: lorentzian(x, fwhm),
        "NumPy Lorentzian",
        n_iterations,
    )

    results["numpy_pvoigt"] = benchmark_function(
        lambda: pvoigt(x, fwhm, eta),
        "NumPy Pseudo-Voigt",
        n_iterations,
    )

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
    from peakfit.core.fitting.optimizer import fit_cluster

    results = {}

    # Standard least-squares
    results["least_squares"] = benchmark_function(
        lambda: fit_cluster(params.copy(), cluster, noise, max_nfev=BENCHMARK_MAX_NFEV),
        "Least Squares",
        n_iterations,
        warmup=1,
    )

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
    baseline_time = sorted_results[0][1].mean_time if sorted_results else 1.0

    for _name, result in sorted_results:
        speedup = baseline_time / result.mean_time if result.mean_time > 0 else 0
        lines.append(f"{result.name:30s}")
        lines.append(f"  Mean time:    {result.mean_time * 1000:10.3f} ms")
        lines.append(f"  Std dev:      {result.std_time * 1000:10.3f} ms")
        lines.append(f"  Min time:     {result.min_time * 1000:10.3f} ms")
        lines.append(f"  Max time:     {result.max_time * 1000:10.3f} ms")
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
    from peakfit.core.fitting.computation import calculate_shapes, residuals
    from peakfit.core.fitting.optimizer import fit_cluster

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
    fit_cluster(params.copy(), cluster, noise, max_nfev=BENCHMARK_MAX_NFEV)
    profile["full_fit"] = time.perf_counter() - start

    return profile


@dataclass
class _BenchmarkOptions:
    """Minimal implementation of FittingOptions protocol for benchmarks."""

    jx: bool = False
    phx: bool = False
    phy: bool = False
    noise: float | None = None
    pvoigt: bool = False
    lorentzian: bool = False
    gaussian: bool = False
    path_list: Path = Path("synthetic_peaks.list")


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
    from peakfit.core.domain.cluster import Cluster
    from peakfit.core.domain.peaks import create_params, create_peak
    from peakfit.core.domain.spectrum import Spectra, SpectralParameters

    noise_level = 10.0
    rng = np.random.default_rng(seed=1234)

    data_cube = np.zeros((n_planes, n_points_per_dim, n_points_per_dim), dtype=np.float64)

    spec_params = []
    for dim_index, axis_size in enumerate(data_cube.shape):
        spec_params.append(
            SpectralParameters(
                size=axis_size,
                sw=5000.0,
                obs=500.0,
                car=0.0,
                aq_time=0.1,
                apocode=0.0,
                apodq1=0.0,
                apodq2=0.0,
                apodq3=0.0,
                p180=False,
                direct=dim_index == data_cube.ndim - 1,
                ft=True,
            )
        )

    # Build a minimal NMRPipe-like header (dic) so we can instantiate a real
    # Spectra object rather than casting a SimpleNamespace. This gives us a
    # consistent `Spectra.params` computed by the existing helper functions.
    dic: dict[str, object] = {}
    fddimorder = list(range(1, data_cube.ndim + 1))
    dic["FDDIMORDER"] = fddimorder
    for fdf in fddimorder:
        dic[f"FDF{fdf}SW"] = 5000.0
        dic[f"FDF{fdf}ORIG"] = 0.0
        dic[f"FDF{fdf}OBS"] = 500.0
        dic[f"FDF{fdf}APOD"] = 0.0
        dic[f"FDF{fdf}P1"] = 0.0
        dic[f"FDF{fdf}APODCODE"] = 0.0
        dic[f"FDF{fdf}APODQ1"] = 0.0
        dic[f"FDF{fdf}APODQ2"] = 0.0
        dic[f"FDF{fdf}APODQ3"] = 0.0
    z_values = np.arange(n_planes)

    spectra = Spectra(dic, data_cube, z_values)
    options = _BenchmarkOptions()
    spatial_dims = data_cube.ndim - 1
    shape_names = ["pvoigt"] * spatial_dims

    peaks = []
    for i in range(n_peaks):
        x_pos = 20.0 + i * 15.0
        y_pos = 20.0 + i * 15.0
        positions = [x_pos, y_pos]
        peak = create_peak(f"peak{i}", positions, shape_names, spectra, options)
        peaks.append(peak)

    params = create_params(peaks)

    grid_y, grid_x = np.meshgrid(
        np.arange(n_points_per_dim, dtype=np.int_),
        np.arange(n_points_per_dim, dtype=np.int_),
        indexing="ij",
    )
    segment_positions = [grid_y.ravel(), grid_x.ravel()]

    shape_values = np.array([peak.evaluate(segment_positions, params) for peak in peaks])
    amplitudes = rng.uniform(0.5, 1.5, size=(len(peaks), n_planes))
    data = shape_values.T @ amplitudes
    data += rng.normal(0.0, noise_level, size=data.shape)

    cluster = Cluster(
        cluster_id=0,
        peaks=peaks,
        positions=segment_positions,
        data=data.astype(np.float64),
    )

    return cluster, noise_level
