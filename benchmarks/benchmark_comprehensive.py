"""Comprehensive benchmarking suite for PeakFit Numba optimization.

Usage:
    uv run python benchmarks/benchmark_comprehensive.py

This script will:
1. Benchmark all lineshape functions
2. Test parallel scaling efficiency
3. Generate performance report
4. Save results to CSV
"""

import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from peakfit.lineshapes import (
    compute_all_gaussian_shapes,
    gaussian,
    lorentzian,
    no_apod,
    pvoigt,
    sp1,
    sp2,
)


@contextmanager
def timer(name: str):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.6f} s")


def benchmark_lineshape_function(
    func: Any, name: str, dx: np.ndarray, *args, n_iterations: int = 1000
) -> dict[str, float]:
    """Benchmark a single lineshape function.

    Returns:
        Dictionary with benchmark results
    """
    # Warm-up (trigger compilation)
    _ = func(dx, *args)

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        func(dx, *args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)

    throughput = (len(dx) * n_iterations) / sum(times)

    print(
        f"{name:20s}: {mean_time * 1e6:8.2f} ± {std_time * 1e6:6.2f} µs/call | "
        f"Throughput: {throughput:.2e} evals/s"
    )

    return {
        "function": name,
        "mean_time_us": mean_time * 1e6,
        "std_time_us": std_time * 1e6,
        "min_time_us": min_time * 1e6,
        "throughput_evals_per_sec": throughput,
    }


def benchmark_all_lineshapes():
    """Benchmark all single-peak lineshape functions."""
    print("=" * 80)
    print("Single-Peak Lineshape Functions")
    print("=" * 80)

    n_points = 10000
    dx = np.linspace(-500, 500, n_points)

    results = []

    # Gaussian
    results.append(benchmark_lineshape_function(gaussian, "Gaussian", dx, 10.0))

    # Lorentzian
    results.append(benchmark_lineshape_function(lorentzian, "Lorentzian", dx, 10.0))

    # Pseudo-Voigt
    results.append(benchmark_lineshape_function(pvoigt, "Pseudo-Voigt", dx, 10.0, 0.5))

    # No apod
    results.append(
        benchmark_lineshape_function(no_apod, "No Apod (FID)", dx, 5.0, 0.05, 0.0, n_iterations=100)
    )

    # SP1
    results.append(
        benchmark_lineshape_function(
            sp1, "SP1 (Sine Bell)", dx, 5.0, 0.05, 2.0, 0.5, 0.0, n_iterations=100
        )
    )

    # SP2
    results.append(
        benchmark_lineshape_function(
            sp2, "SP2 (Sine²)", dx, 5.0, 0.05, 2.0, 0.5, 0.0, n_iterations=100
        )
    )

    return pd.DataFrame(results)


def benchmark_multi_peak(n_peaks_list: list[int]):
    """Benchmark multi-peak parallel computation."""
    print("\n" + "=" * 80)
    print("Multi-Peak Parallel Computation")
    print("=" * 80)

    n_points = 5000
    positions = np.linspace(0, 1000, n_points)

    results = []

    for n_peaks in n_peaks_list:
        centers = np.linspace(100, 900, n_peaks)
        fwhms = np.full(n_peaks, 10.0)

        # Warm-up
        _ = compute_all_gaussian_shapes(positions, centers, fwhms)

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            compute_all_gaussian_shapes(positions, centers, fwhms)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_time = np.mean(times)
        std_time = np.std(times)

        total_evals = n_peaks * n_points * 10
        throughput = total_evals / sum(times)

        print(
            f"n_peaks={n_peaks:4d}: {mean_time * 1000:7.3f} ± {std_time * 1000:5.3f} ms | "
            f"Throughput: {throughput:.2e} evals/s"
        )

        results.append(
            {
                "n_peaks": n_peaks,
                "mean_time_ms": mean_time * 1000,
                "std_time_ms": std_time * 1000,
                "throughput_evals_per_sec": throughput,
            }
        )

    return pd.DataFrame(results)


def benchmark_parallel_scaling():
    """Benchmark parallel scaling efficiency."""
    print("\n" + "=" * 80)
    print("Parallel Scaling Efficiency")
    print("=" * 80)

    n_points = 10000
    n_peaks = 100
    positions = np.linspace(0, 1000, n_points)
    centers = np.linspace(100, 900, n_peaks)
    fwhms = np.full(n_peaks, 10.0)

    max_threads = numba.config.NUMBA_NUM_THREADS

    results = []
    baseline_time = None

    for n_threads in range(1, min(max_threads + 1, 17)):
        numba.set_num_threads(n_threads)

        # Warm-up
        _ = compute_all_gaussian_shapes(positions, centers, fwhms)

        # Benchmark
        times = []
        for _ in range(50):
            start = time.perf_counter()
            compute_all_gaussian_shapes(positions, centers, fwhms)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_time = np.mean(times)
        std_time = np.std(times)

        if baseline_time is None:
            baseline_time = mean_time
            speedup = 1.0
            efficiency = 100.0
        else:
            speedup = baseline_time / mean_time
            efficiency = (speedup / n_threads) * 100

        print(
            f"Threads={n_threads:2d}: {mean_time * 1000:6.3f} ± {std_time * 1000:5.3f} ms | "
            f"Speedup={speedup:6.2f}× | Efficiency={efficiency:6.1f}%"
        )

        results.append(
            {
                "n_threads": n_threads,
                "mean_time_ms": mean_time * 1000,
                "std_time_ms": std_time * 1000,
                "speedup": speedup,
                "efficiency_percent": efficiency,
            }
        )

    # Reset to default
    numba.set_num_threads(max_threads)

    return pd.DataFrame(results)


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 80)
    print("PeakFit Numba Comprehensive Benchmark Suite")
    print("=" * 80)

    # System info
    print(f"\nSystem: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"NumPy: {np.__version__}")
    print(f"Numba: {numba.__version__}")
    print(f"Available threads: {numba.config.NUMBA_NUM_THREADS}")

    # Check for Intel SVML
    try:
        from numba.core import config

        if hasattr(config, "USING_SVML"):
            print(f"Intel SVML: {'Enabled' if config.USING_SVML else 'Disabled'}")
    except (ImportError, AttributeError):
        print("Intel SVML: Unknown")

    print()

    # Run benchmarks
    df_lineshapes = benchmark_all_lineshapes()
    df_multi_peak = benchmark_multi_peak([10, 50, 100, 200, 500])
    df_scaling = benchmark_parallel_scaling()

    # Save results
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    df_lineshapes.to_csv(output_dir / f"lineshapes_{timestamp}.csv", index=False)
    df_multi_peak.to_csv(output_dir / f"multi_peak_{timestamp}.csv", index=False)
    df_scaling.to_csv(output_dir / f"scaling_{timestamp}.csv", index=False)

    print("\n" + "=" * 80)
    print("Benchmark Results Summary")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nFastest lineshape function:")
    fastest = df_lineshapes.loc[df_lineshapes["mean_time_us"].idxmin()]
    print(f"  {fastest['function']}: {fastest['mean_time_us']:.2f} µs/call")

    print("\nMulti-peak performance (100 peaks):")
    row_100 = df_multi_peak[df_multi_peak["n_peaks"] == 100].iloc[0]
    print(f"  Time: {row_100['mean_time_ms']:.3f} ms")
    print(f"  Throughput: {row_100['throughput_evals_per_sec']:.2e} evals/s")

    print("\nParallel efficiency (at max threads):")
    max_row = df_scaling.iloc[-1]
    print(f"  Threads: {int(max_row['n_threads'])}")
    print(f"  Speedup: {max_row['speedup']:.2f}×")
    print(f"  Efficiency: {max_row['efficiency_percent']:.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
