"""Signal not explained on one plane: connected whole-spectrum residual regions (§8).

Pure functions over the whole-spectrum residual ``(D − Σ usable M) / σ_stat`` of one
plane. A region is a connected set of samples (including diagonal neighbours) beyond
+k or beyond −k σ_stat; positive and negative excursions are separate regions. The
list is descriptive: it never proposes peaks or judges clusters.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_THRESHOLD = 5.0  # k, in σ_stat; a display preference
_CONNECTIVITY = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class UnexplainedRegion:
    """One connected residual region; indices are native samples (row, column)."""

    row: int  # location: the sample with the extreme value
    column: int
    extreme: float  # signed, in σ_stat
    size: int  # samples in the region
    rows: tuple[int, int]  # inclusive extent
    columns: tuple[int, int]
    nearest: str | None  # working ID of the nearest working peak
    distance: float  # from the location to that peak, in samples on the native grid


def nearest_peak(
    row: int, column: int, peaks: Sequence[tuple[str, float, float]]
) -> tuple[str | None, float]:
    """The working ID nearest a sample and its distance in native samples, if any peak."""
    if not peaks:
        return None, float("nan")
    distances = [np.hypot(r - row, c - column) for _id, r, c in peaks]
    closest = int(np.argmin(distances))
    return peaks[closest][0], float(distances[closest])


def unexplained_regions(
    residual: np.ndarray,
    threshold: float,
    peaks: Sequence[tuple[str, float, float]],
) -> list[UnexplainedRegion]:
    """Regions beyond ±threshold, largest |extreme| first.

    ``peaks`` holds (working ID, row, column) in fractional native samples.
    """
    regions = []
    for sign in (1.0, -1.0):
        labels, _count = ndimage.label(sign * residual > threshold, structure=_CONNECTIVITY)
        for index, extent in enumerate(ndimage.find_objects(labels), start=1):
            inside = labels[extent] == index
            values = np.where(inside, sign * residual[extent], -np.inf)
            local = np.unravel_index(int(np.argmax(values)), values.shape)
            row, column = extent[0].start + int(local[0]), extent[1].start + int(local[1])
            nearest, distance = nearest_peak(row, column, peaks)
            regions.append(
                UnexplainedRegion(
                    row=row,
                    column=column,
                    extreme=float(residual[row, column]),
                    size=int(np.count_nonzero(inside)),
                    rows=(extent[0].start, extent[0].stop - 1),
                    columns=(extent[1].start, extent[1].stop - 1),
                    nearest=nearest,
                    distance=distance,
                )
            )
    regions.sort(key=lambda region: -abs(region.extreme))
    return regions
