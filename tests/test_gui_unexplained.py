"""Unexplained-signal regions are exact connected sets of the whole-spectrum residual."""

import numpy as np

from peakfit.gui.editor.unexplained import unexplained_regions


def _residual():
    residual = np.zeros((40, 60))
    residual[10:13, 20:24] = 6.0  # a 12-sample positive block
    residual[11, 22] = 9.5  # its extreme
    residual[13:15, 20:22] = -7.0  # touching negative block: a separate region
    residual[30, 40] = 5.5  # diagonal pair: one region
    residual[31, 41] = 5.2
    residual[5, 50] = 4.99  # below the threshold
    return residual


def test_regions_are_signed_connected_sets_beyond_threshold():
    regions = unexplained_regions(_residual(), 5.0, [])

    assert [(r.extreme, r.size) for r in regions] == [(9.5, 12), (-7.0, 4), (5.5, 2)]
    first, negative, diagonal = regions
    assert (first.row, first.column, first.rows, first.columns) == (11, 22, (10, 12), (20, 23))
    assert (negative.rows, negative.columns) == ((13, 14), (20, 21))
    assert (diagonal.rows, diagonal.columns) == ((30, 31), (40, 41))
    assert first.nearest is None
    assert np.isnan(first.distance)


def test_nearest_working_peak_is_measured_from_the_extreme_on_the_native_grid():
    peaks = [("a", 11.0, 25.0), ("b", 30.0, 44.0), ("c", 0.0, 0.0)]

    first, negative, diagonal = unexplained_regions(_residual(), 5.0, peaks)

    assert (first.nearest, first.distance) == ("a", 3.0)
    assert negative.nearest == "a"
    assert negative.distance == np.hypot(13 - 11, 20 - 25)
    assert (diagonal.nearest, diagonal.distance) == ("b", 4.0)


def test_threshold_is_strict_and_noise_alone_rarely_crosses_five_sigma():
    residual = np.random.default_rng(3).standard_normal((256, 512))

    assert unexplained_regions(residual, 5.0, []) == []
    assert unexplained_regions(_residual(), 9.5, []) == []
    assert len(unexplained_regions(residual, 3.0, [])) > 0
