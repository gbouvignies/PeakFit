"""Spectrum loading and display geometry for the desktop viewer."""

from typing import TYPE_CHECKING, cast

import contourpy
import numpy as np
from PySide6.QtGui import QPainterPath

from peakfit.engine.domain.config import FitConfig
from peakfit.gui.peaks import WorkingPeak
from peakfit.io.readers.peaks import read_list
from peakfit.io.readers.spectrum import read_spectra

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.domain.spectrum import Spectra, SpectralParameters

_SPECTRAL_DIMENSIONS = 2
_MIN_AXIS_POINTS = 2


def load_spectrum(path: Path, plane_values: Path | None = None) -> Spectra:
    """Load two spectral axes with a leading series axis using PeakFit's reader."""
    spectra = read_spectra(path, plane_values)
    spectra.z_values = np.atleast_1d(np.asarray(spectra.z_values, dtype=float))
    if (
        spectra.data.ndim != _SPECTRAL_DIMENSIONS + 1
        or spectra.n_spectral_dims != _SPECTRAL_DIMENSIONS
    ):
        raise ValueError("The viewer currently supports 2D and pseudo-3D NMRPipe spectra.")
    if any(size < _MIN_AXIS_POINTS for size in spectra.data.shape[1:]):
        raise ValueError("Each spectral axis must contain at least two points.")
    if spectra.data.shape[0] == 0 or not np.all(np.isfinite(spectra.data)):
        raise ValueError("The spectrum must contain planes with finite intensity values.")
    validate_plane_values(spectra.z_values, spectra.data.shape[0])
    return spectra


def validate_plane_values(values: np.ndarray, n_planes: int) -> None:
    """Require one finite experimental coordinate per plane."""
    if values.shape != (n_planes,) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"Plane values must contain exactly {n_planes} finite numbers in one column."
        )


def load_peaks(path: Path, spectra: Spectra) -> list[WorkingPeak]:
    """Reuse peak readers, discarding visualization shapes at the working-state handoff."""
    peaks = read_list(path, spectra, ["gaussian"] * spectra.n_spectral_dims, FitConfig())
    if any(not np.all(np.isfinite(peak.positions)) for peak in peaks):
        raise ValueError("Peak positions must be finite ppm coordinates.")
    return [
        WorkingPeak.new(peak.name, (float(peak.positions[0]), float(peak.positions[1])))
        for peak in peaks
    ]


def display_position(axis: SpectralParameters, ppm: float) -> float:
    """Fold indirect coordinates into the sampled window without rounding the peak."""
    point = float(axis.ppm2pts(ppm))
    if not axis.direct:
        point %= axis.size
    return float(axis.pts2ppm(point))


def contour_paths(
    plane: np.ndarray, axes: list[SpectralParameters], levels: np.ndarray
) -> list[QPainterPath]:
    """Extract full-plane contours once and retain their geometry as Qt paths.

    Coordinates come from the same sample conversions used by the fitting engine.
    Navigation only transforms these paths; it never re-extracts contours.
    """
    # Import PySide6 before PyQtGraph so both use the same Qt binding.
    import pyqtgraph as pg  # noqa: PLC0415

    y_axis, x_axis = axes
    generator = contourpy.contour_generator(
        x=x_axis.pts2ppm(np.arange(x_axis.size, dtype=float)),
        y=y_axis.pts2ppm(np.arange(y_axis.size, dtype=float)),
        z=plane,
        line_type="Separate",
    )
    paths = []
    for level in levels:
        segments = cast("list[np.ndarray]", generator.lines(float(level)))
        if not segments:
            paths.append(QPainterPath())
            continue
        points = np.concatenate(segments)
        connect = np.ones(len(points), dtype=np.int32)
        connect[np.cumsum([len(segment) for segment in segments]) - 1] = 0
        paths.append(pg.arrayToQPath(points[:, 0], points[:, 1], connect=connect))
    return paths
