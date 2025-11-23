"""
Demonstration: Why compute_all_sp2_shapes is MUCH better than calling sp2 repeatedly

This shows the actual performance difference and explains when each is used.
"""

import time

import numpy as np

from peakfit.lineshapes.functions import compute_all_sp2_shapes, sp2

# ============================================================================
# SCENARIO: Fitting a cluster with 20 peaks (typical in real NMR data)
# ============================================================================

# Setup: 20 peaks at different positions
n_peaks = 20
n_points = 2048

positions_hz = np.linspace(-1000, 1000, n_points)  # Frequency axis in Hz
centers_hz = np.linspace(-500, 500, n_peaks)  # 20 peak centers
r2s = np.full(n_peaks, 15.0)  # Relaxation rates
aq = 0.05  # Acquisition time
end = 0.98  # SP2 end parameter
off = 0.35  # SP2 offset parameter
phases = np.zeros(n_peaks)  # Phase corrections

# ============================================================================
# APPROACH 1: Sequential sp2 calls (what PeakFit did BEFORE optimization)
# ============================================================================
print("=" * 70)
print("APPROACH 1: Sequential sp2() calls (OLD way)")
print("=" * 70)
print(f"Scenario: {n_peaks} peaks × {n_points} points")
print()


def sequential_approach(positions_hz, centers_hz, r2s, aq, end, off, phases):
    """OLD: Call sp2() once per peak in a Python loop."""
    n_peaks = len(centers_hz)
    n_points = len(positions_hz)
    shapes = np.empty((n_peaks, n_points))

    for i in range(n_peaks):
        # For each peak, compute offset from its center
        dx = positions_hz - centers_hz[i]
        # Call sp2 for this single peak
        shapes[i] = sp2(dx, r2s[i], aq, end, off, phases[i])

    return shapes


# Warm up Numba JIT
_ = sequential_approach(positions_hz[:10], centers_hz[:2], r2s[:2], aq, end, off, phases[:2])

# Benchmark
start = time.perf_counter()
n_iter = 100
for _ in range(n_iter):
    result_seq = sequential_approach(positions_hz, centers_hz, r2s, aq, end, off, phases)
time_seq = (time.perf_counter() - start) / n_iter

print(f"Time per call: {time_seq * 1000:.3f} ms")
print()
print("Why is this slow?")
print("  1. Python loop overhead (calling sp2() 20 times)")
print("  2. Each sp2() call processes points sequentially")
print("  3. No parallelization across peaks")
print("  4. Function call overhead for each peak")
print()

# ============================================================================
# APPROACH 2: Batch compute_all_sp2_shapes (NEW optimized way)
# ============================================================================
print("=" * 70)
print("APPROACH 2: Batch compute_all_sp2_shapes() call (NEW way)")
print("=" * 70)

# Warm up
_ = compute_all_sp2_shapes(positions_hz[:10], centers_hz[:2], r2s[:2], aq, end, off, phases[:2])

# Benchmark
start = time.perf_counter()
for _ in range(n_iter):
    result_batch = compute_all_sp2_shapes(positions_hz, centers_hz, r2s, aq, end, off, phases)
time_batch = (time.perf_counter() - start) / n_iter

print(f"Time per call: {time_batch * 1000:.3f} ms")
print()
print("Why is this fast?")
print("  1. Single function call (no Python loop)")
print("  2. Numba parallelizes across peaks using prange")
print("  3. All peaks processed simultaneously on multiple CPU cores")
print("  4. Manual complex arithmetic (no function call overhead)")
print("  5. Compiler optimizations (SIMD, register allocation)")
print()

# ============================================================================
# RESULTS COMPARISON
# ============================================================================
print("=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)
speedup = time_seq / time_batch
print(f"Sequential time:  {time_seq * 1000:.3f} ms")
print(f"Batch time:       {time_batch * 1000:.3f} ms")
print(f"Speedup:          {speedup:.1f}x faster!")
print()

# Verify results are identical
print("Correctness check:")
if np.allclose(result_seq, result_batch, rtol=1e-10, atol=1e-10):
    print(
        f"  ✓ Results match perfectly (max diff: {np.max(np.abs(result_seq - result_batch)):.2e})"
    )
else:
    print("  ✗ Results differ!")
print()

# ============================================================================
# WHEN IS EACH USED IN PEAKFIT?
# ============================================================================
print("=" * 70)
print("WHEN IS EACH FUNCTION USED IN PEAKFIT?")
print("=" * 70)
print()
print("sp2() - Single Peak Function:")
print("  • Used for: Single peak evaluation")
print("  • Called by: ApodShape.evaluate() for individual peaks")
print("  • Use case: Testing, validation, single-peak fitting")
print()
print("compute_all_sp2_shapes() - Multi-Peak Batch Function:")
print("  • Used for: Multiple peaks in a cluster")
print("  • Called by: ApodShape.batch_evaluate_apod_shapes()")
print("  • Automatically detected by: calculate_shapes() in fitting pipeline")
print("  • Use case: Production fitting (AUTOMATIC - no code changes needed!)")
print()
print("AUTOMATIC OPTIMIZATION IN ACTION:")
print("  When you run 'peakfit fit', the code detects:")
print("    - Is this cluster all sp2 shapes? → Use compute_all_sp2_shapes()")
print("    - Is this cluster all sp1 shapes? → Use compute_all_sp1_shapes()")
print("    - Is this cluster all no_apod?   → Use compute_all_no_apod_shapes()")
print("    - Mixed shape types?              → Fall back to sequential")
print()

# ============================================================================
# REAL-WORLD IMPACT
# ============================================================================
print("=" * 70)
print("REAL-WORLD IMPACT")
print("=" * 70)
print()
print("Your recent fit (examples/02-advanced-fitting):")
print("  • 166 peaks across 121 clusters")
print("  • All using sp2 shapes")
print("  • Completed in 6.0 seconds")
print("  • CPU usage: 299% (multi-core parallelism working!)")
print()
print("Without this optimization:")
print(f"  • Would take ~{6.0 * speedup:.0f} seconds (estimated)")
print("  • CPU usage: ~100% (single-core sequential)")
print()
print("Critical for:")
print("  ✓ MCMC uncertainty analysis (millions of evaluations)")
print("  ✓ Profile likelihood (thousands of evaluations per parameter)")
print("  ✓ Real-time interactive fitting")
print("  ✓ Large 3D/4D datasets with hundreds of peaks")
print()
print("=" * 70)
