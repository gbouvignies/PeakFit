"""Native slices of the displayed plane: indices, intensity range, residual, framing.

Pure numpy over the spectrum's own sampling. Positions on screen are displayed
(aliased) ppm; one fold-aware conversion turns them into native sample indices.
Nothing here evaluates a model off the acquired grid.
"""

from typing import TYPE_CHECKING

import numpy as np

from peakfit.gui.spectrum import display_position

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.engine.domain.spectrum import SpectralParameters

RANGE_PAD = 0.05  # each bound of the intensity range gains this share of the span
FOCUS_WIDTHS = 3  # a focused cluster's margin, in effective linewidths
FOCUS_SAMPLES = 8  # and at least this many native samples
FOCUS_MAX_SHARE = 0.25  # but at most this share of the spectral width


def native_index(axis: SpectralParameters, ppm: float) -> int:
    """The nearest native sample of a position, folding indirect aliases first."""
    point = float(axis.ppm2pts(display_position(axis, ppm)))
    return int(np.clip(round(point), 0, axis.size - 1))


def index_range(axis: SpectralParameters, ppm_range: Sequence[float]) -> tuple[int, int]:
    """Inclusive native sample range covered by a displayed ppm range."""
    points = sorted(float(axis.ppm2pts(value)) for value in ppm_range)
    first = int(np.clip(np.floor(points[0]), 0, axis.size - 1))
    last = int(np.clip(np.ceil(points[1]), 0, axis.size - 1))
    return first, last


def intensity_range(
    data: np.ndarray, rows: tuple[int, int], columns: tuple[int, int]
) -> tuple[float, float]:
    """Asymmetric bounds over a region and every plane, always including zero.

    The lower bound is min(0, min D) and the upper max(0, max D), each padded by
    ``RANGE_PAD`` of the span. ``data`` is (plane, row, column).
    """
    region = data[:, rows[0] : rows[1] + 1, columns[0] : columns[1] + 1]
    finite = region[np.isfinite(region)]
    low = min(0.0, float(finite.min())) if finite.size else 0.0
    high = max(0.0, float(finite.max())) if finite.size else 0.0
    span = high - low
    if span == 0:
        return -1.0, 1.0
    return low - RANGE_PAD * span, high + RANGE_PAD * span


def residual(observed: np.ndarray, model: np.ndarray, noise: float) -> np.ndarray:
    """The whole-spectrum display residual in σ_stat, at native samples."""
    return (np.asarray(observed, dtype=float) - np.asarray(model, dtype=float)) / noise


def clipped(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Indices of samples outside the shown range."""
    values = np.asarray(values, dtype=float)
    return np.flatnonzero((values < low) | (values > high))


def focus_ranges(
    positions: Sequence[tuple[float, float]],
    widths: Sequence[tuple[float, float]],
    axes: Sequence[SpectralParameters],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Frame members' displayed positions, independent of any fitting mask.

    ``positions`` are unfolded (y, x) ppm, ``widths`` effective FWHM (y, x) in ppm,
    NaN when unavailable, and ``axes`` the (y, x) spectral axes. Each axis gains
    max(3 FWHM, 8 samples), at most a quarter of the spectral width, and stays
    within the spectrum. Returns (x_range, y_range), each (low, high).
    """
    ranges = []
    for index, axis in enumerate(axes):
        shown = [display_position(axis, p[index]) for p in positions]
        edges = sorted(float(axis.pts2ppm(point)) for point in (0, axis.size - 1))
        spacing = abs(float(axis.pts2ppm(1)) - float(axis.pts2ppm(0)))
        finite = [abs(w[index]) for w in widths if np.isfinite(w[index])]
        margin = max(FOCUS_WIDTHS * max(finite, default=0.0), FOCUS_SAMPLES * spacing)
        margin = min(margin, FOCUS_MAX_SHARE * (edges[1] - edges[0]))
        low = max(edges[0], min(shown) - margin)
        high = min(edges[1], max(shown) + margin)
        ranges.append((low, high))
    y_range, x_range = ranges
    return x_range, y_range
