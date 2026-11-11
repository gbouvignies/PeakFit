"""Native slices, the asymmetric intensity range and focus framing on real data."""

from pathlib import Path

import numpy as np
import pytest

from peakfit.gui.editor.slices import (
    FOCUS_MAX_SHARE,
    FOCUS_SAMPLES,
    clipped,
    focus_ranges,
    index_range,
    intensity_range,
    native_index,
    plane_extremes,
    residual,
)
from peakfit.gui.spectrum import display_position, read_dataset

DATA = Path(__file__).resolve().parents[1] / "examples/02-advanced-fitting/data"


@pytest.fixture(scope="module")
def dataset():
    return read_dataset(DATA / "pseudo3d.ft2", DATA / "pseudo3d.list", DATA / "b1_offsets.txt")


def test_a_folded_negative_peak_slices_at_its_displayed_row(dataset):
    # 39N-H is listed at 133.42 ppm ¹⁵N but aliased into the window; its displayed
    # row holds its negative signal. Unfolded ppm would give an index outside the data.
    y_axis, x_axis = dataset.spectra.spectral_params
    peak = next(p for p in dataset.peaks if p.name == "39N-H")
    y, x = peak.positions
    assert float(y_axis.ppm2pts(y)) < 0
    row, column = native_index(y_axis, y), native_index(x_axis, x)
    assert 0 <= row < y_axis.size
    assert row == round(float(y_axis.ppm2pts(display_position(y_axis, y))))
    assert dataset.spectra.data[0, row, column] < -50 * dataset.noise


def test_intensity_range_is_asymmetric_and_includes_zero():
    data = np.zeros((3, 4, 4))
    data[1, 1, 1] = 10.0
    assert intensity_range(plane_extremes(data), (0, 3), (0, 3)) == pytest.approx((-0.5, 10.5))
    data[2, 2, 2] = -2.0
    assert intensity_range(plane_extremes(data), (0, 3), (0, 3)) == pytest.approx((-2.6, 10.6))
    negative = -np.abs(data)
    low, high = intensity_range(plane_extremes(negative), (0, 3), (0, 3))
    assert high > 0 > low
    assert intensity_range(plane_extremes(np.zeros((2, 2, 2))), (0, 1), (0, 1)) == (-1.0, 1.0)


def test_intensity_range_covers_every_plane_of_the_region_only():
    data = np.zeros((2, 6, 6))
    data[1, 0, 0] = 100.0  # outside the region
    data[0, 3, 3] = 5.0  # inside, on another plane
    assert intensity_range(plane_extremes(data), (2, 4), (2, 4))[1] == pytest.approx(5.25)


def test_index_range_and_clipping(dataset):
    _y_axis, x_axis = dataset.spectra.spectral_params
    first, last = index_range(x_axis, (8.40, 8.30))
    assert first <= native_index(x_axis, 8.35) <= last
    assert index_range(x_axis, (1e6, -1e6)) == (0, x_axis.size - 1)
    assert list(clipped(np.array([-3.0, 0.0, 2.0, 5.0]), -1.0, 4.0)) == [0, 3]


def test_residual_is_in_sigma_at_native_samples():
    np.testing.assert_allclose(
        residual(np.array([3.0, 1.0]), np.array([1.0, 1.0]), 2.0), [1.0, 0.0]
    )


def test_focus_frames_members_with_a_linewidth_or_sample_margin(dataset):
    y_axis, x_axis = dataset.spectra.spectral_params
    axes = (y_axis, x_axis)
    pair = [p.positions for p in dataset.peaks if p.name in ("53N-H", "147N-H")]
    widths = [(0.2, 0.02)] * 2
    (x_low, _x_high), (_y_low, y_high) = focus_ranges(pair, widths, axes)
    assert x_low == pytest.approx(min(p[1] for p in pair) - 3 * 0.02)
    assert y_high == pytest.approx(max(p[0] for p in pair) + 3 * 0.2)
    # Without a linewidth, the sample margin alone applies.
    spacing = abs(float(x_axis.pts2ppm(1)) - float(x_axis.pts2ppm(0)))
    (x_low, _), _ = focus_ranges(pair, [(np.nan, np.nan)] * 2, axes)
    assert x_low == pytest.approx(min(p[1] for p in pair) - FOCUS_SAMPLES * spacing)


def test_focus_handles_folding_and_extreme_widths(dataset):
    y_axis, x_axis = dataset.spectra.spectral_params
    folded = next(p for p in dataset.peaks if p.name == "39N-H").positions
    _x_range, y_range = focus_ranges([folded], [(np.nan, np.nan)], (y_axis, x_axis))
    shown = display_position(y_axis, folded[0])
    assert y_range[0] < shown < y_range[1]
    edges = sorted(float(y_axis.pts2ppm(p)) for p in (0, y_axis.size - 1))
    assert edges[0] <= y_range[0]
    assert y_range[1] <= edges[1]
    # An absurd width is capped at a quarter of the spectral width.
    (x_low, x_high), _ = focus_ranges([folded], [(np.nan, 1e3)], (y_axis, x_axis))
    x_edges = sorted(float(x_axis.pts2ppm(p)) for p in (0, x_axis.size - 1))
    assert x_high - x_low <= 2 * FOCUS_MAX_SHARE * (x_edges[1] - x_edges[0]) + 1e-9
