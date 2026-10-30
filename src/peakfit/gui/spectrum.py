"""Spectrum loading and display geometry for the desktop viewer."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import contourpy
import numpy as np

from peakfit.engine.algorithms.noise import estimate_noise
from peakfit.engine.domain.config import FitConfig
from peakfit.gui.peaks import WorkingPeak
from peakfit.io.readers.peaks import read_list
from peakfit.io.readers.spectrum import read_spectra

if TYPE_CHECKING:
    from pathlib import Path

    from PySide6.QtGui import QPainterPath

    from peakfit.engine.domain.spectrum import Spectra, SpectralParameters

_SPECTRAL_DIMENSIONS = 2
_MIN_AXIS_POINTS = 2


@contextmanager
def verified_source_files(
    paths: dict[str, Path], *, canceled=lambda: False, progress=lambda _: None
):
    """Record import paths/checksums only if files stay identical through reading."""

    def records() -> dict[str, str]:
        result = {}
        for role, path in paths.items():
            before_stat = path.stat()
            digest_state = hashlib.sha256()
            with path.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    if canceled():
                        raise CancelledError("Source verification canceled")
                    digest_state.update(block)
            digest = digest_state.hexdigest()
            after_stat = path.stat()
            if (
                before_stat.st_size,
                before_stat.st_mtime_ns,
                before_stat.st_ctime_ns,
                before_stat.st_ino,
                before_stat.st_dev,
            ) != (
                after_stat.st_size,
                after_stat.st_mtime_ns,
                after_stat.st_ctime_ns,
                after_stat.st_ino,
                after_stat.st_dev,
            ):
                raise ValueError("An input changed while hashing. Load it again.")
            progress(f"Verified {path.name} · {before_stat.st_size / 1024**2:.1f} MiB")
            stat = path.stat()
            result[role] = json.dumps(
                {
                    "path": str(path.resolve()),
                    "checksum_sha256": digest,
                    "size": str(stat.st_size),
                    "mtime_ns": str(stat.st_mtime_ns),
                    "ctime_ns": str(stat.st_ctime_ns),
                    "device": str(stat.st_dev),
                    "inode": str(stat.st_ino),
                    "algorithm": "sha256-v1",
                    "format": "NMRPipe" if role == "spectrum" else role,
                }
            )
        return result

    before = records()
    verified = datetime.now(UTC).isoformat()
    yield tuple(
        (role, json.dumps(dict(json.loads(record), verified_at=verified)))
        for role, record in before.items()
    )
    if before != records():
        raise ValueError("An input file changed while loading. Load the inputs again.")


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


type ContourGeometry = list[tuple[np.ndarray, np.ndarray]]


def contour_geometry(
    plane: np.ndarray,
    axes: list[SpectralParameters],
    levels: np.ndarray,
) -> ContourGeometry:
    """Extract numerical paths off the GUI thread; create no Qt objects."""
    y_axis, x_axis = axes
    generator = contourpy.contour_generator(
        x=x_axis.pts2ppm(np.arange(x_axis.size, dtype=float)),
        y=y_axis.pts2ppm(np.arange(y_axis.size, dtype=float)),
        z=plane,
        line_type="Separate",
    )
    geometry: ContourGeometry = []
    for level in levels:
        segments = cast("list[np.ndarray]", generator.lines(float(level)))
        points = np.concatenate(segments) if segments else np.empty((0, 2))
        connect = np.ones(len(points), dtype=np.int32)
        if segments:
            connect[np.cumsum([len(segment) for segment in segments]) - 1] = 0
        geometry.append((points, connect))
    return geometry


def paths_from_geometry(geometry: ContourGeometry) -> list[QPainterPath]:
    """Construct Qt graphics only on the caller's GUI thread."""
    import pyqtgraph as pg  # noqa: PLC0415

    return [
        pg.arrayToQPath(points[:, 0], points[:, 1], connect=connect) for points, connect in geometry
    ]


def contour_paths(
    plane: np.ndarray, axes: list[SpectralParameters], levels: np.ndarray
) -> list[QPainterPath]:
    """Synchronous geometry convenience for callers already on the GUI thread."""
    return paths_from_geometry(contour_geometry(plane, axes, levels))


@dataclass(frozen=True, slots=True)
class Dataset:
    """One verified source with original plane order and immutable working seeds.

    No rounded labels, selection or viewport participates in scientific identity.
    Arrays are read-only; a later input replacement creates a new Dataset.
    """

    path: Path
    spectra: Spectra
    peaks: tuple[WorkingPeak, ...]
    sources: tuple[tuple[str, str], ...]
    has_plane_values: bool
    trace_limit: float
    noise: float


def read_dataset(
    path: Path,
    peaks: Path | None = None,
    values: Path | None = None,
    *,
    swap_peak_axes: bool = False,
    canceled=lambda: False,
    progress=lambda _: None,
) -> Dataset:
    """Validate everything before handing a complete dataset back to Qt.

    The dialog explicitly confirms peak-column mapping. Readers retain authority
    over supported formats; a mapping swap is applied once to full precision ppm.
    """
    sources = {"spectrum": path}
    if values is not None:
        sources["plane_values"] = values
    if peaks is not None:
        sources["peak_list"] = peaks
    with verified_source_files(sources, canceled=canceled, progress=progress) as fingerprints:
        spectra = load_spectrum(path, values)
        for axis in spectra.spectral_params:
            if not axis.ft:
                raise ValueError(f"{axis.label}: process the time-domain axis before opening it.")
            if not all(np.isfinite(v) for v in (axis.first, axis.delta, axis.obs, axis.sw)):
                raise ValueError(
                    f"{axis.label}: non-finite spectral calibration. Check the header."
                )
            if axis.delta == 0 or axis.obs <= 0 or axis.sw <= 0:
                raise ValueError(f"{axis.label}: invalid spectral calibration. Check the header.")
        imported = read_assignments(peaks, spectra, swap=swap_peak_axes) if peaks else ()
        largest = float(np.max(np.abs(spectra.data)))
        trace_limit = max(largest * 1.05, 1.0)
        noise = float(estimate_noise(spectra.data[0]))
        if not np.isfinite(noise) or noise <= 0:
            noise = max(largest / 100, np.finfo(float).tiny)
        spectra.data.flags.writeable = False
        spectra.z_values.flags.writeable = False
    return Dataset(path, spectra, imported, fingerprints, values is not None, trace_limit, noise)


def read_assignments(
    path: Path, spectra: Spectra, *, swap: bool = False
) -> tuple[WorkingPeak, ...]:
    """Load reader-supported assignments, allocating fresh stable identities."""
    imported = load_peaks(path, spectra)
    return tuple(
        replace(p, positions=(p.positions[1], p.positions[0])) if swap else p for p in imported
    )
