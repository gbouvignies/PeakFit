"""Simulation and README writing utilities."""

from __future__ import annotations

from datetime import datetime
from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.engine.fitting.simulation import simulate_data

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.shared.reporter import Reporter


def write_simulated_spectra(
    output_dir: Path,
    spectra: Spectra,
    clusters: list[Cluster],
    params: Parameters,
    reporter: Reporter | None = None,
) -> Path | None:
    """Write simulated spectra to file.

    Simulates the spectra based on the fitted parameters and writes it to an NMRPipe file.

    Args:
        output_dir: Directory where the simulated file should be saved.
        spectra: The original spectra object (used for metadata/header).
        clusters: List of clusters used in the fit.
        params: The fitted parameters.
        reporter: Optional reporter for progress updates.
    """
    try:
        ng = import_module("nmrglue")
    except ModuleNotFoundError:
        return None

    if reporter:
        reporter.action("Writing simulated spectra...")

    data_simulated = simulate_data(params, clusters, spectra.data)

    if spectra.pseudo_dim_added:
        data_simulated = np.squeeze(data_simulated, axis=0)

    output_path = output_dir / f"simulated.ft{data_simulated.ndim}"
    ng.pipe.write(
        str(output_path),
        spectra.dic,
        data_simulated.astype(np.float32),
        overwrite=True,
    )
    return output_path


def write_readme(output_dir: Path, summary: Any) -> Path:
    """Generate a README.md file for the output directory.

    Args:
        output_dir: Path to the output directory.
        summary: RunSummary object containing run metrics.
    """
    readme_path = output_dir / "README.md"

    lines = [
        "# PeakFit Run",
        "",
        "## Summary",
        "",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Peaks**: {summary.n_peaks}",
        f"- **Converged clusters**: {summary.n_converged}/{summary.n_clusters} "
        f"({summary.success_rate:.1%})",
        f"- **Median reduced chi2**: {summary.median_redchi:.4g}",
        "",
        "## Files",
        "",
    ]

    file_descriptions = [
        ("summary/fit.json", "Main machine-readable fit summary."),
        ("summary/report.md", "Concise human-readable report."),
        ("tables/parameters.csv", "Model parameters."),
        ("tables/intensities.csv", "Per-plane fitted intensities and errors."),
        ("tables/shifts.csv", "Chemical shifts by peak."),
        ("metadata/fitting_state.pkl", "Saved state for MCMC and plotting workflows."),
    ]

    for rel_path, description in file_descriptions:
        if (output_dir / rel_path).exists():
            lines.append(f"- `{rel_path}`: {description}")

    readme_path.write_text("\n".join(lines) + "\n")
    return readme_path


__all__ = [
    "write_readme",
    "write_simulated_spectra",
]
