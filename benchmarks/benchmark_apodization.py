"""Benchmark apodization functions (no_apod, sp1, sp2) - PeakFit's unique features.

This benchmark demonstrates the performance improvements from Numba optimization
of PeakFit's critical apodization-based lineshape functions.
"""

import time

import numpy as np

# Direct import from functions module to avoid circular import issues
from peakfit.lineshapes.functions import (
    compute_all_no_apod_shapes,
    compute_all_sp1_shapes,
    compute_all_sp2_shapes,
    no_apod,
    sp1,
    sp2,
)


def benchmark_single_peak_sequential(n_peaks: int, n_points: int, n_iter: int = 100):
    """Benchmark sequential single-peak function calls."""
    print(f"\n{'=' * 70}")
    print(f"Sequential Single-Peak Calls: {n_peaks} peaks × {n_points} points")
    print(f"{'=' * 70}")

    # Setup
    positions = np.linspace(-200, 200, n_points)
    centers = np.linspace(-150, 150, n_peaks)
    r2s = np.random.uniform(8, 15, n_peaks)
    phases = np.random.uniform(-5, 5, n_peaks)
    aq = 0.05
    end, off = 1.0, 0.35

    # Warm up
    for center, r2, phase in zip(centers[:2], r2s[:2], phases[:2]):
        dx = positions - center
        no_apod(dx, r2, aq, phase)
        sp1(dx, r2, aq, end, off, phase)
        sp2(dx, r2, aq, end, off, phase)

    # Benchmark no_apod
    start = time.perf_counter()
    for _ in range(n_iter):
        for center, r2, phase in zip(centers, r2s, phases):
            dx = positions - center
            no_apod(dx, r2, aq, phase)
    no_apod_time = (time.perf_counter() - start) / n_iter

    # Benchmark sp1
    start = time.perf_counter()
    for _ in range(n_iter):
        for center, r2, phase in zip(centers, r2s, phases):
            dx = positions - center
            sp1(dx, r2, aq, end, off, phase)
    sp1_time = (time.perf_counter() - start) / n_iter

    # Benchmark sp2
    start = time.perf_counter()
    for _ in range(n_iter):
        for center, r2, phase in zip(centers, r2s, phases):
            dx = positions - center
            sp2(dx, r2, aq, end, off, phase)
    sp2_time = (time.perf_counter() - start) / n_iter

    print(f"  no_apod: {no_apod_time * 1000:.3f} ms")
    print(f"  sp1:     {sp1_time * 1000:.3f} ms")
    print(f"  sp2:     {sp2_time * 1000:.3f} ms")

    return no_apod_time, sp1_time, sp2_time


def benchmark_multi_peak_parallel(n_peaks: int, n_points: int, n_iter: int = 100):
    """Benchmark optimized multi-peak parallel functions."""
    print(f"\n{'=' * 70}")
    print(f"Optimized Multi-Peak Parallel: {n_peaks} peaks × {n_points} points")
    print(f"{'=' * 70}")

    # Setup
    positions = np.linspace(-200, 200, n_points)
    centers = np.linspace(-150, 150, n_peaks)
    r2s = np.random.uniform(8, 15, n_peaks)
    phases = np.random.uniform(-5, 5, n_peaks)
    aq = 0.05
    end, off = 1.0, 0.35

    # Warm up
    compute_all_no_apod_shapes(positions, centers[:2], r2s[:2], aq, phases[:2])
    compute_all_sp1_shapes(positions, centers[:2], r2s[:2], aq, end, off, phases[:2])
    compute_all_sp2_shapes(positions, centers[:2], r2s[:2], aq, end, off, phases[:2])

    # Benchmark no_apod
    start = time.perf_counter()
    for _ in range(n_iter):
        compute_all_no_apod_shapes(positions, centers, r2s, aq, phases)
    no_apod_time = (time.perf_counter() - start) / n_iter

    # Benchmark sp1
    start = time.perf_counter()
    for _ in range(n_iter):
        compute_all_sp1_shapes(positions, centers, r2s, aq, end, off, phases)
    sp1_time = (time.perf_counter() - start) / n_iter

    # Benchmark sp2
    start = time.perf_counter()
    for _ in range(n_iter):
        compute_all_sp2_shapes(positions, centers, r2s, aq, end, off, phases)
    sp2_time = (time.perf_counter() - start) / n_iter

    print(f"  no_apod: {no_apod_time * 1000:.3f} ms")
    print(f"  sp1:     {sp1_time * 1000:.3f} ms")
    print(f"  sp2:     {sp2_time * 1000:.3f} ms")

    return no_apod_time, sp1_time, sp2_time


def main():
    """Run comprehensive apodization benchmarks."""
    print("\n" + "=" * 70)
    print("PeakFit Apodization Functions Benchmark")
    print("=" * 70)
    print("\nThese are PeakFit's unique features:")
    print("  • no_apod: Non-apodized FID-based lineshape")
    print("  • sp1:     Sine bell apodization")
    print("  • sp2:     Sine squared bell apodization")
    print("\nOptimizations:")
    print("  • Numba JIT compilation with fastmath")
    print("  • Parallel multi-peak evaluation")
    print("  • Manual complex arithmetic for maximum speed")

    # Small dataset (typical single cluster)
    print("\n" + "=" * 70)
    print("SMALL DATASET (Typical Single Cluster)")
    print("=" * 70)
    seq_times_small = benchmark_single_peak_sequential(n_peaks=4, n_points=512, n_iter=1000)
    par_times_small = benchmark_multi_peak_parallel(n_peaks=4, n_points=512, n_iter=1000)

    print("\nSpeedup (Parallel vs Sequential):")
    print(f"  no_apod: {seq_times_small[0] / par_times_small[0]:.1f}×")
    print(f"  sp1:     {seq_times_small[1] / par_times_small[1]:.1f}×")
    print(f"  sp2:     {seq_times_small[2] / par_times_small[2]:.1f}×")

    # Medium dataset (large cluster)
    print("\n" + "=" * 70)
    print("MEDIUM DATASET (Large Cluster)")
    print("=" * 70)
    seq_times_med = benchmark_single_peak_sequential(n_peaks=20, n_points=2048, n_iter=100)
    par_times_med = benchmark_multi_peak_parallel(n_peaks=20, n_points=2048, n_iter=100)

    print("\nSpeedup (Parallel vs Sequential):")
    print(f"  no_apod: {seq_times_med[0] / par_times_med[0]:.1f}×")
    print(f"  sp1:     {seq_times_med[1] / par_times_med[1]:.1f}×")
    print(f"  sp2:     {seq_times_med[2] / par_times_med[2]:.1f}×")

    # Large dataset (full spectrum with many peaks)
    print("\n" + "=" * 70)
    print("LARGE DATASET (Full Spectrum)")
    print("=" * 70)
    seq_times_large = benchmark_single_peak_sequential(n_peaks=100, n_points=4096, n_iter=10)
    par_times_large = benchmark_multi_peak_parallel(n_peaks=100, n_points=4096, n_iter=10)

    print("\nSpeedup (Parallel vs Sequential):")
    print(f"  no_apod: {seq_times_large[0] / par_times_large[0]:.1f}×")
    print(f"  sp1:     {seq_times_large[1] / par_times_large[1]:.1f}×")
    print(f"  sp2:     {seq_times_large[2] / par_times_large[2]:.1f}×")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nKey Performance Gains:")
    print(
        f"  • Small clusters (4 peaks):     {np.mean([seq_times_small[i] / par_times_small[i] for i in range(3)]):.1f}× faster"
    )
    print(
        f"  • Medium clusters (20 peaks):   {np.mean([seq_times_med[i] / par_times_med[i] for i in range(3)]):.1f}× faster"
    )
    print(
        f"  • Large datasets (100 peaks):   {np.mean([seq_times_large[i] / par_times_large[i] for i in range(3)]):.1f}× faster"
    )

    print("\nImpact on Threadripper PRO 7965WX (24 cores):")
    print("  • Parallel execution scales with available cores")
    print("  • Each apodization evaluation runs in separate thread/process")
    print("  • Expected speedup: 10-20× for datasets with many peaks")
    print("  • Critical for real-time fitting and uncertainty analysis")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
