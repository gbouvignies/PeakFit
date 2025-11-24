"""Test batch apodization shape evaluation integration into fitting pipeline."""

import numpy as np
import pytest

from peakfit.data.clustering import Cluster
from peakfit.data.peaks import Peak
from peakfit.data.spectrum import SpectralParameters
from peakfit.fitting.computation import calculate_shapes
from peakfit.fitting.parameters import Parameters
from peakfit.lineshapes.models import ApodShape, NoApod


@pytest.fixture
def mock_spectra():
    """Create a mock Spectra object for testing."""
    # Create minimal spectrum parameters
    spec_params = SpectralParameters(
        size=512,
        sw=1200.0,
        obs=600.13,
        car=4.7,
        aq_time=0.05,
        apocode=0.0,
        apodq1=0.35,
        apodq2=0.98,
        apodq3=0.0,
        p180=False,
        direct=True,
        ft=True,
    )

    # Mock Spectra object
    class MockSpectra:
        def __init__(self):
            rng = np.random.default_rng()
            self.params = [spec_params]
            self.data = [rng.standard_normal(512)]

    return MockSpectra()


def test_batch_evaluation_single_no_apod(mock_spectra):
    """Test batch evaluation with single no_apod peak."""
    # Create a single no_apod peak
    shape = NoApod("test", 0, 0.0, mock_spectra, args=None)
    params = shape.create_params()

    # Create test points
    x_pt = np.arange(100, 200, dtype=np.int64)

    # Batch evaluation with single peak should work
    result = ApodShape.batch_evaluate_apod_shapes([shape], x_pt, params)

    assert result.shape == (1, 100)
    assert np.all(np.isfinite(result))


def test_batch_evaluation_multiple_no_apod(mock_spectra):
    """Test batch evaluation with multiple no_apod peaks."""
    # Create multiple no_apod peaks at different positions
    shapes = [NoApod(f"peak{i}", 0, float(i), mock_spectra, args=None) for i in range(5)]

    # Create combined parameters
    params = Parameters()
    for shape in shapes:
        params.update(shape.create_params())

    # Create test points
    x_pt = np.arange(100, 200, dtype=np.int64)

    # Batch evaluation
    result = ApodShape.batch_evaluate_apod_shapes(shapes, x_pt, params)

    assert result.shape == (5, 100)
    assert np.all(np.isfinite(result))

    # Each peak should have different values (different centers)
    for i in range(4):
        assert not np.allclose(result[i], result[i + 1])


def test_batch_evaluation_matches_sequential(mock_spectra):
    """Test that batch evaluation matches sequential evaluation."""
    # Create test peaks
    shapes = [NoApod(f"peak{i}", 0, float(i * 10), mock_spectra, args=None) for i in range(3)]

    # Create combined parameters
    params = Parameters()
    for shape in shapes:
        params.update(shape.create_params())

    # Create test points
    x_pt = np.arange(150, 250, dtype=np.int64)

    # Batch evaluation
    batch_result = ApodShape.batch_evaluate_apod_shapes(shapes, x_pt, params)

    # Sequential evaluation
    sequential_result = np.array([shape.evaluate(x_pt, params) for shape in shapes])

    # Should match (with small numerical tolerance)
    assert batch_result.shape == sequential_result.shape
    assert np.allclose(batch_result, sequential_result, rtol=1e-10, atol=1e-10)


def test_calculate_shapes_uses_batch_optimization(mock_spectra):
    """Test that calculate_shapes automatically uses batch optimization for apodization shapes."""
    # Create cluster with multiple no_apod peaks
    peaks = []
    for i in range(5):
        shape = NoApod(f"peak{i}", 0, float(i * 10), mock_spectra, args=None)
        peak = Peak(f"peak{i}", np.array([float(i * 10)]), [shape])
        peaks.append(peak)

    # Create combined parameters
    params = Parameters()
    for peak in peaks:
        params.update(peak.create_params())

    # Create cluster
    rng = np.random.default_rng()
    x_pt = np.arange(150, 250, dtype=np.int64)
    data = rng.standard_normal(100)
    cluster = Cluster(cluster_id=0, peaks=peaks, positions=[x_pt], data=data)

    # Calculate shapes (should use batch optimization internally)
    result = calculate_shapes(params, cluster)

    assert result.shape == (5, 100)
    assert np.all(np.isfinite(result))


def test_calculate_shapes_performance_improvement(mock_spectra):
    """Verify that batch evaluation is faster than sequential (smoke test)."""
    import time

    # Create cluster with many peaks
    n_peaks = 20
    peaks = []
    for i in range(n_peaks):
        shape = NoApod(f"peak{i}", 0, float(i * 5), mock_spectra, args=None)
        peak = Peak(f"peak{i}", np.array([float(i * 5)]), [shape])
        peaks.append(peak)

    # Create combined parameters
    params = Parameters()
    for peak in peaks:
        params.update(peak.create_params())

    # Create cluster with many points
    rng = np.random.default_rng()
    x_pt = np.arange(100, 400, dtype=np.int64)
    data = rng.standard_normal(300)
    cluster = Cluster(cluster_id=0, peaks=peaks, positions=[x_pt], data=data)

    # Time batch evaluation (via calculate_shapes)
    start = time.perf_counter()
    for _ in range(10):
        result_batch = calculate_shapes(params, cluster)
    time_batch = time.perf_counter() - start

    # Time sequential evaluation (direct call)
    start = time.perf_counter()
    for _ in range(10):
        result_seq = np.array([peak.evaluate(cluster.positions, params) for peak in peaks])
    time_seq = time.perf_counter() - start

    # Verify results match
    assert np.allclose(result_batch, result_seq, rtol=1e-10, atol=1e-10)

    # Batch should be faster (at least 2x for this size)
    speedup = time_seq / time_batch
    print(
        f"\nSpeedup: {speedup:.1f}x (batch: {time_batch * 100:.1f}ms, seq: {time_seq * 100:.1f}ms)"
    )
    assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.1f}x"


def test_empty_shapes_list():
    """Test batch evaluation with empty shapes list."""
    result = ApodShape.batch_evaluate_apod_shapes([], np.arange(10), Parameters())
    assert result.shape == (0, 10)
