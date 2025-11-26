"""Output writing for fit results.

This module handles writing fitting results to various output files
including profiles, chemical shifts, simulated spectra, and HTML reports.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak
from peakfit.core.domain.spectrum import Spectra
from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.parameters import Parameters
from peakfit.core.fitting.simulation import simulate_data
from peakfit.infra.state import StateRepository
from peakfit.io.output import write_profiles, write_shifts
from peakfit.ui import PeakFitUI, console

if TYPE_CHECKING:
    from peakfit.services.fit.pipeline import FitArguments

ui = PeakFitUI

__all__ = [
    "save_fitting_state",
    "write_all_outputs",
    "write_html_report",
    "write_simulated_spectra",
]


def write_all_outputs(
    output_dir: Path,
    spectra: Spectra,
    clusters: list[Cluster],
    peaks: list[Peak],
    params: Parameters,
    clargs: FitArguments,
    *,
    save_simulated: bool = False,
    save_html_report: bool = True,
) -> None:
    """Write all output files.

    Args:
        output_dir: Directory to write files to
        spectra: Spectrum data
        clusters: List of clusters
        peaks: List of peaks
        params: Fitted parameters
        clargs: Command line arguments
        save_simulated: Whether to save simulated spectra
        save_html_report: Whether to save HTML report
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write profiles
    with console.status("[cyan]Writing profiles...[/cyan]", spinner="dots"):
        write_profiles(output_dir, spectra.z_values, clusters, params, clargs)
    ui.success(f"Peak profiles: {output_dir.name}/{len(peaks)} *.out files")

    # Write shifts
    with console.status("[cyan]Writing shifts...[/cyan]", spinner="dots"):
        write_shifts(peaks, params, output_dir / "shifts.list")
    ui.success(f"Chemical shifts: {output_dir.name}/shifts.list")

    # Write simulated spectra if requested
    if save_simulated:
        write_simulated_spectra(output_dir, spectra, clusters, params)
        ui.success(f"Simulated spectra: {output_dir.name}/simulated_*.ft*")

    # Write HTML report if requested
    if save_html_report:
        write_html_report(output_dir)
        ui.success(f"HTML report: {output_dir.name}/logs.html")


def write_simulated_spectra(
    output_dir: Path,
    spectra: Spectra,
    clusters: list[Cluster],
    params: Parameters,
) -> None:
    """Write simulated spectra to file.

    Args:
        output_dir: Directory to write files to
        spectra: Spectrum data
        clusters: List of clusters
        params: Fitted parameters
    """
    try:
        ng = import_module("nmrglue")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
        msg = (
            "Writing simulated spectra requires the optional 'nmrglue' dependency. "
            "Install it with `uv pip install nmrglue`."
        )
        raise RuntimeError(msg) from exc

    with console.status("[cyan]Writing simulated spectra...[/cyan]", spinner="dots"):
        data_simulated = simulate_data(params, clusters, spectra.data)

        if spectra.pseudo_dim_added:
            data_simulated = np.squeeze(data_simulated, axis=0)

        ng.pipe.write(
            str(output_dir / f"simulated.ft{data_simulated.ndim}"),
            spectra.dic,
            data_simulated.astype(np.float32),
            overwrite=True,
        )


def write_html_report(output_dir: Path) -> None:
    """Write HTML report with console output.

    Args:
        output_dir: Directory to write files to
    """
    with console.status("[cyan]Generating HTML report...[/cyan]", spinner="dots"):
        ui.export_html(output_dir / "logs.html")


def save_fitting_state(
    output_dir: Path,
    clusters: list[Cluster],
    params: Parameters,
    noise: float,
    peaks: list[Peak],
) -> Path:
    """Save fitting state for later analysis.

    Args:
        output_dir: Directory to save state to
        clusters: List of clusters
        params: Fitted parameters
        noise: Noise level
        peaks: List of peaks

    Returns:
        Path to saved state file
    """
    with console.status("[cyan]Saving fitting state...[/cyan]", spinner="dots"):
        state_file = StateRepository.default_path(output_dir)
        state = FittingState(
            clusters=clusters,
            params=params,
            noise=noise,
            peaks=peaks,
        )
        StateRepository.save(state_file, state)

    ui.success(f"Fitting state: {output_dir.name}/.peakfit_state.pkl")
    return state_file
