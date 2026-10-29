"""Atomic reader handoff for the inspection workspace, without Qt objects."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.noise import estimate_noise
from peakfit.gui.spectrum import load_peaks, load_spectrum, verified_source_files

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.gui.peaks import WorkingPeak


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
) -> Dataset:
    """Validate everything before handing a complete dataset back to Qt.

    The dialog explicitly confirms peak-column mapping. Readers retain authority
    over supported formats; a mapping swap is applied once to full precision ppm.
    """
    sources = {"spectrum": path}
    if peaks is not None:
        sources["peak_list"] = peaks
    if values is not None:
        sources["plane_values"] = values
    with verified_source_files(sources) as fingerprints:
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
