"""Test multi-peak apodization functions."""

import numpy as np

from peakfit.lineshapes.functions import (
    compute_all_no_apod_shapes,
    compute_all_sp1_shapes,
    compute_all_sp2_shapes,
    no_apod,
)


class TestMultiPeakApodFunctions:
    """Tests for optimized multi-peak apodization functions."""

    def test_compute_all_no_apod_shapes(self):
        """Test multi-peak no_apod matches sequential calls."""
        positions = np.linspace(-100, 100, 256)
        centers = np.array([0.0, 30.0, -40.0])
        r2s = np.array([10.0, 12.0, 11.0])
        phases = np.array([0.0, 5.0, -5.0])
        aq = 0.05

        # Multi-peak version
        shapes_multi = compute_all_no_apod_shapes(positions, centers, r2s, aq, phases)

        # Sequential version
        shapes_seq = np.array(
            [
                no_apod(positions - center, r2, aq, phase)
                for center, r2, phase in zip(centers, r2s, phases, strict=True)
            ]
        )

        # Should match within numerical precision
        assert shapes_multi.shape == shapes_seq.shape
        assert shapes_multi.shape == (3, 256)
        np.testing.assert_allclose(shapes_multi, shapes_seq, rtol=1e-10, atol=1e-12)

    def test_compute_all_sp1_shapes(self):
        """Test multi-peak sp1 produces finite output."""
        positions = np.linspace(-100, 100, 256)
        centers = np.array([0.0, 30.0, -40.0])
        r2s = np.array([10.0, 12.0, 11.0])
        phases = np.array([0.0, 5.0, -5.0])
        aq = 0.05
        end, off = 1.0, 0.35

        shapes = compute_all_sp1_shapes(positions, centers, r2s, aq, end, off, phases)

        assert shapes.shape == (3, 256)
        assert np.all(np.isfinite(shapes))

    def test_compute_all_sp2_shapes(self):
        """Test multi-peak sp2 produces finite output."""
        positions = np.linspace(-100, 100, 256)
        centers = np.array([0.0, 30.0, -40.0])
        r2s = np.array([10.0, 12.0, 11.0])
        phases = np.array([0.0, 5.0, -5.0])
        aq = 0.05
        end, off = 1.0, 0.35

        shapes = compute_all_sp2_shapes(positions, centers, r2s, aq, end, off, phases)

        assert shapes.shape == (3, 256)
        assert np.all(np.isfinite(shapes))

    def test_multi_peak_shapes_normalized(self):
        """Test multi-peak shapes have reasonable peak heights."""
        positions = np.linspace(-50, 50, 512)
        centers = np.array([0.0])
        r2s = np.array([10.0])
        phases = np.array([0.0])
        aq = 0.05
        end, off = 1.0, 0.35

        shapes_no_apod = compute_all_no_apod_shapes(positions, centers, r2s, aq, phases)
        shapes_sp1 = compute_all_sp1_shapes(positions, centers, r2s, aq, end, off, phases)
        shapes_sp2 = compute_all_sp2_shapes(positions, centers, r2s, aq, end, off, phases)

        # All should have peak near center
        assert np.max(np.abs(shapes_no_apod[0])) > 0.5
        assert np.max(np.abs(shapes_sp1[0])) > 0.1
        assert np.max(np.abs(shapes_sp2[0])) > 0.1

    def test_multi_peak_parallel_performance(self):
        """Benchmark multi-peak vs sequential (informational)."""
        import time

        positions = np.linspace(-200, 200, 2048)
        n_peaks = 50
        centers = np.linspace(-150, 150, n_peaks)
        r2s = np.full(n_peaks, 10.0)
        phases = np.zeros(n_peaks)
        aq = 0.05

        # Warm up
        compute_all_no_apod_shapes(positions, centers[:2], r2s[:2], aq, phases[:2])

        # Multi-peak version
        start = time.perf_counter()
        for _ in range(10):
            compute_all_no_apod_shapes(positions, centers, r2s, aq, phases)
        multi_time = (time.perf_counter() - start) / 10

        # Sequential version
        start = time.perf_counter()
        for _ in range(10):
            np.array(
                [
                    no_apod(positions - center, r2, aq, phase)
                    for center, r2, phase in zip(centers, r2s, phases, strict=True)
                ]
            )
        seq_time = (time.perf_counter() - start) / 10

        speedup = seq_time / multi_time
        print(
            f"\nMulti-peak no_apod speedup: {speedup:.1f}× ({n_peaks} peaks, {len(positions)} points)"
        )
        print(f"  Sequential: {seq_time * 1000:.2f} ms")
        print(f"  Parallel:   {multi_time * 1000:.2f} ms")

        # Multi-peak should be faster (at least not slower)
        assert multi_time <= seq_time * 1.2  # Allow 20% tolerance for overhead
