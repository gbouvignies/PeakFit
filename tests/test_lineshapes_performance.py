"""Performance benchmarks for Numba implementations.

These are not run by default (marked with @pytest.mark.benchmark).
Run with: uv run pytest tests/test_lineshapes_performance.py -v -m benchmark

Expected performance targets:
- Single-peak functions: 50× faster than pure NumPy
- Multi-peak parallel: 10-50× depending on CPU cores
- Linear scaling up to physical cores
"""

import time

import numba
import numpy as np
import pytest


@pytest.mark.benchmark
def test_gaussian_benchmark():
    """Benchmark Gaussian performance."""
    from peakfit.lineshapes import gaussian

    dx = np.linspace(-500, 500, 10000)
    fwhm = 10.0

    # Warm-up (trigger compilation)
    _ = gaussian(dx, fwhm)

    # Benchmark
    n_iterations = 1000
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = gaussian(dx, fwhm)
    elapsed = time.perf_counter() - start

    throughput = (10000 * n_iterations) / elapsed
    print(f"\nGaussian throughput: {throughput:.2e} evals/s")
    print(f"Average time per call: {elapsed / n_iterations * 1000:.3f} ms")

    # Performance target: should be >1M evals/s on modern CPU
    assert throughput > 1e6, f"Performance too slow: {throughput:.2e} evals/s"


@pytest.mark.benchmark
@pytest.mark.parametrize("n_peaks", [10, 50, 100, 200])
def test_multi_peak_scaling(n_peaks):
    """Test parallel scaling of multi-peak computation."""
    from peakfit.lineshapes import compute_all_gaussian_shapes

    n_points = 5000
    positions = np.linspace(0, 1000, n_points)
    centers = np.linspace(100, 900, n_peaks)
    fwhms = np.full(n_peaks, 10.0)

    # Warm-up
    _ = compute_all_gaussian_shapes(positions, centers, fwhms)

    # Benchmark (10 runs)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        shapes = compute_all_gaussian_shapes(positions, centers, fwhms)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nn_peaks={n_peaks:3d}: {mean_time * 1000:.3f} ± {std_time * 1000:.3f} ms")

    # Performance target: should scale linearly with n_peaks
    # (actual test would compare with n_peaks=10 baseline)


@pytest.mark.benchmark
def test_parallel_efficiency():
    """Test parallel efficiency across different thread counts."""
    from peakfit.lineshapes import compute_all_gaussian_shapes

    n_points = 10000
    n_peaks = 100
    positions = np.linspace(0, 1000, n_points)
    centers = np.linspace(100, 900, n_peaks)
    fwhms = np.full(n_peaks, 10.0)

    max_threads = numba.config.NUMBA_NUM_THREADS

    print(f"\nParallel efficiency test (max threads: {max_threads})")
    print("-" * 60)

    baseline_time = None
    for n_threads in range(1, min(max_threads + 1, 9)):
        numba.set_num_threads(n_threads)

        # Warm-up
        _ = compute_all_gaussian_shapes(positions, centers, fwhms)

        # Benchmark
        times = []
        for _ in range(50):
            start = time.perf_counter()
            shapes = compute_all_gaussian_shapes(positions, centers, fwhms)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_time = np.mean(times)

        if baseline_time is None:
            baseline_time = mean_time
            speedup = 1.0
            efficiency = 100.0
        else:
            speedup = baseline_time / mean_time
            efficiency = (speedup / n_threads) * 100

        print(
            f"Threads={n_threads:2d}: {mean_time * 1000:6.3f} ms | "
            f"Speedup={speedup:5.2f}× | Efficiency={efficiency:5.1f}%"
        )

    # Reset to default
    numba.set_num_threads(max_threads)


@pytest.mark.benchmark
def test_lorentzian_benchmark():
    """Benchmark Lorentzian performance."""
    from peakfit.lineshapes import lorentzian

    dx = np.linspace(-500, 500, 10000)
    fwhm = 10.0

    # Warm-up
    _ = lorentzian(dx, fwhm)

    # Benchmark
    n_iterations = 1000
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = lorentzian(dx, fwhm)
    elapsed = time.perf_counter() - start

    throughput = (10000 * n_iterations) / elapsed
    print(f"\nLorentzian throughput: {throughput:.2e} evals/s")
    print(f"Average time per call: {elapsed / n_iterations * 1000:.3f} ms")

    assert throughput > 1e6, f"Performance too slow: {throughput:.2e} evals/s"


@pytest.mark.benchmark
def test_pvoigt_benchmark():
    """Benchmark Pseudo-Voigt performance."""
    from peakfit.lineshapes import pvoigt

    dx = np.linspace(-500, 500, 10000)
    fwhm = 10.0
    eta = 0.5

    # Warm-up
    _ = pvoigt(dx, fwhm, eta)

    # Benchmark
    n_iterations = 1000
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = pvoigt(dx, fwhm, eta)
    elapsed = time.perf_counter() - start

    throughput = (10000 * n_iterations) / elapsed
    print(f"\nPseudo-Voigt throughput: {throughput:.2e} evals/s")
    print(f"Average time per call: {elapsed / n_iterations * 1000:.3f} ms")

    assert throughput > 5e5, f"Performance too slow: {throughput:.2e} evals/s"


@pytest.mark.benchmark
def test_no_apod_benchmark():
    """Benchmark no_apod performance."""
    from peakfit.lineshapes import no_apod

    dx = np.linspace(-500, 500, 10000)
    r2 = 5.0
    aq = 0.05
    phase = 0.0

    # Warm-up
    _ = no_apod(dx, r2, aq, phase)

    # Benchmark
    n_iterations = 100
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = no_apod(dx, r2, aq, phase)
    elapsed = time.perf_counter() - start

    throughput = (10000 * n_iterations) / elapsed
    print(f"\nno_apod throughput: {throughput:.2e} evals/s")
    print(f"Average time per call: {elapsed / n_iterations * 1000:.3f} ms")

    # Lower threshold for complex FID functions
    assert throughput > 1e5, f"Performance too slow: {throughput:.2e} evals/s"


@pytest.mark.benchmark
def test_compilation_time():
    """Measure JIT compilation time."""
    # Force recompilation by clearing cache
    # This test just shows compilation is fast, not a performance assertion

    from peakfit.lineshapes import gaussian

    dx = np.linspace(-100, 100, 1000)
    fwhm = 10.0

    # First call includes compilation
    start = time.perf_counter()
    _ = gaussian(dx, fwhm)
    first_call = time.perf_counter() - start

    # Second call is cached
    start = time.perf_counter()
    _ = gaussian(dx, fwhm)
    cached_call = time.perf_counter() - start

    print(f"\nFirst call (with compilation): {first_call * 1000:.3f} ms")
    print(f"Cached call: {cached_call * 1000:.3f} ms")
    print(f"Compilation overhead: {(first_call - cached_call) * 1000:.3f} ms")


@pytest.mark.benchmark
def test_multi_peak_lorentzian_scaling():
    """Test multi-peak Lorentzian scaling."""
    from peakfit.lineshapes import compute_all_lorentzian_shapes

    n_points = 5000
    positions = np.linspace(0, 1000, n_points)

    for n_peaks in [10, 50, 100, 200]:
        centers = np.linspace(100, 900, n_peaks)
        fwhms = np.full(n_peaks, 10.0)

        # Warm-up
        _ = compute_all_lorentzian_shapes(positions, centers, fwhms)

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            shapes = compute_all_lorentzian_shapes(positions, centers, fwhms)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_time = np.mean(times)
        print(f"n_peaks={n_peaks:3d}: {mean_time * 1000:.3f} ms")


@pytest.mark.benchmark
def test_multi_peak_pvoigt_scaling():
    """Test multi-peak Pseudo-Voigt scaling."""
    from peakfit.lineshapes import compute_all_pvoigt_shapes

    n_points = 5000
    positions = np.linspace(0, 1000, n_points)

    for n_peaks in [10, 50, 100, 200]:
        centers = np.linspace(100, 900, n_peaks)
        fwhms = np.full(n_peaks, 10.0)
        etas = np.full(n_peaks, 0.5)

        # Warm-up
        _ = compute_all_pvoigt_shapes(positions, centers, fwhms, etas)

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            shapes = compute_all_pvoigt_shapes(positions, centers, fwhms, etas)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_time = np.mean(times)
        print(f"n_peaks={n_peaks:3d}: {mean_time * 1000:.3f} ms")
