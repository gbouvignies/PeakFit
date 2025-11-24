"""Benchmark comparing lstsq vs normal equations for amplitude calculation."""

import time

import numpy as np

from peakfit.lineshapes.functions import compute_ata_symmetric, compute_atb


def benchmark_lstsq(shapes, data, n_iterations=100):
    """Benchmark using np.linalg.lstsq."""
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = np.linalg.lstsq(shapes.T, data, rcond=None)[0]
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)


def benchmark_normal_equations(shapes, data, n_iterations=100):
    """Benchmark using normal equations with optimized Numba functions."""
    # Warm up
    ata = compute_ata_symmetric(shapes)
    atb = compute_atb(shapes, data)
    _ = np.linalg.solve(ata, atb)

    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        ata = compute_ata_symmetric(shapes)
        atb = compute_atb(shapes, data)
        _ = np.linalg.solve(ata, atb)
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)


def main():
    """Run benchmarks for different problem sizes."""
    print("Benchmark: Amplitude Calculation Methods")
    print("=" * 60)

    # Different problem sizes (n_peaks, n_points)
    sizes = [
        (3, 100, "Small cluster (3 peaks)"),
        (10, 200, "Medium cluster (10 peaks)"),
        (20, 500, "Large cluster (20 peaks)"),
    ]

    for n_peaks, n_points, description in sizes:
        print(f"\n{description}")
        print(f"  Problem size: {n_peaks} peaks × {n_points} points")
        print("-" * 60)

        # Generate random data
        shapes = np.random.randn(n_peaks, n_points).astype(np.float64)
        data = np.random.randn(n_points).astype(np.float64)

        # Benchmark lstsq
        mean_lstsq, std_lstsq = benchmark_lstsq(shapes, data)
        print(f"  np.linalg.lstsq:      {mean_lstsq * 1e6:8.2f} ± {std_lstsq * 1e6:6.2f} μs")

        # Benchmark normal equations
        mean_normal, std_normal = benchmark_normal_equations(shapes, data)
        print(f"  Normal equations:     {mean_normal * 1e6:8.2f} ± {std_normal * 1e6:6.2f} μs")

        # Speedup
        speedup = mean_lstsq / mean_normal
        print(f"  Speedup:              {speedup:.2f}×")


if __name__ == "__main__":
    main()
