"""Data loading for fit pipeline.

This module handles loading spectrum data, peak lists, and computing
noise levels for the fitting process.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from peakfit.core.algorithms.noise import prepare_noise_level
from peakfit.core.domain.peaks import Peak
from peakfit.core.domain.peaks_io import read_list
from peakfit.core.domain.spectrum import Spectra, get_shape_names, read_spectra
from peakfit.ui import PeakFitUI

if TYPE_CHECKING:
    from peakfit.services.fit.pipeline import FitArguments

ui = PeakFitUI

__all__ = [
    "LoadedData",
    "load_data",
    "load_peaks",
    "load_spectrum",
    "prepare_noise",
]


class LoadedData:
    """Container for loaded fitting data."""

    def __init__(
        self,
        spectra: Spectra,
        peaks: list[Peak],
        noise: float,
        noise_source: str,
        shape_names: list[str],
        contour_level: float,
    ) -> None:
        self.spectra = spectra
        self.peaks = peaks
        self.noise = noise
        self.noise_source = noise_source
        self.shape_names = shape_names
        self.contour_level = contour_level


def load_spectrum(
    spectrum_path: Path,
    z_values_path: Path | None,
    exclude_planes: list[int],
) -> Spectra:
    """Load and validate spectrum data.

    Args:
        spectrum_path: Path to spectrum file
        z_values_path: Optional path to z-values file
        exclude_planes: List of plane indices to exclude

    Returns:
        Loaded Spectra object
    """
    return read_spectra(spectrum_path, z_values_path, exclude_planes)


def load_peaks(
    spectra: Spectra,
    shape_names: list[str],
    clargs: FitArguments,
) -> list[Peak]:
    """Load and validate peak list.

    Args:
        spectra: Loaded spectrum data
        shape_names: List of lineshape names
        clargs: Command line arguments

    Returns:
        List of Peak objects
    """
    return read_list(spectra, shape_names, clargs)


def prepare_noise(clargs: FitArguments, spectra: Spectra) -> tuple[float, str]:
    """Prepare noise level for fitting.

    Args:
        clargs: Command line arguments with optional noise value
        spectra: Loaded spectrum data

    Returns:
        Tuple of (noise_value, noise_source)
    """
    noise_was_provided = clargs.noise is not None and clargs.noise > 0.0
    noise = prepare_noise_level(clargs, spectra)
    if noise is None:
        raise ValueError("Noise must be set by prepare_noise_level")

    noise_value = float(noise)
    noise_source = "user-provided" if noise_was_provided else "estimated"

    return noise_value, noise_source


def load_data(
    spectrum_path: Path,
    peaklist_path: Path,
    z_values_path: Path | None,
    clargs: FitArguments,
) -> LoadedData:
    """Load all data required for fitting.

    Args:
        spectrum_path: Path to spectrum file
        peaklist_path: Path to peak list file
        z_values_path: Optional path to z-values file
        clargs: Command line arguments

    Returns:
        LoadedData containing all loaded data
    """
    # Load spectrum
    spectra = load_spectrum(spectrum_path, z_values_path, clargs.exclude)

    # Prepare noise level
    noise, noise_source = prepare_noise(clargs, spectra)
    clargs.noise = noise

    # Get lineshape names
    shape_names = get_shape_names(clargs, spectra)

    # Set contour level
    contour_level = clargs.contour_level or 5.0 * noise

    # Load peaks
    peaks = load_peaks(spectra, shape_names, clargs)

    return LoadedData(
        spectra=spectra,
        peaks=peaks,
        noise=noise,
        noise_source=noise_source,
        shape_names=shape_names,
        contour_level=contour_level,
    )
