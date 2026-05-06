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

    from .config import WriterConfig


def write_simulated_spectra(
    output_dir: Path,
    spectra: Spectra,
    clusters: list[Cluster],
    params: Parameters,
    reporter: Reporter | None = None,
) -> None:
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
        return

    if reporter:
        reporter.action("Writing simulated spectra...")

    data_simulated = simulate_data(params, clusters, spectra.data)

    if spectra.pseudo_dim_added:
        data_simulated = np.squeeze(data_simulated, axis=0)

    ng.pipe.write(
        str(output_dir / f"simulated.ft{data_simulated.ndim}"),
        spectra.dic,
        data_simulated.astype(np.float32),
        overwrite=True,
    )


def write_readme(output_dir: Path, config: WriterConfig, summary: Any) -> None:
    """Generate a README.md file for the output directory.

    Args:
        output_dir: Path to the output directory.
        config: The configuration object used for the run.
        summary: RunSummary object containing run metrics.
    """
    readme_path = output_dir / "README.md"

    lines = [
        "# PeakFit Run Results",
        "",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Clusters**: {summary.n_clusters}",
        f"- **Success Rate**: {summary.success_rate:.1%}",
        "",
        "## File Guide",
        "",
    ]

    file_descriptions = [
        ("manifest.json", "Run manifest with a full file index and run metrics."),
        ("summary/fit_summary.json", "Canonical structured fit summary for downstream tools."),
        ("summary/report.md", "Concise human-readable report."),
        ("parameters/parameters.csv", "Model parameters (one row per parameter)."),
        ("parameters/intensities.csv", "Per-plane fitted intensities and errors."),
        ("parameters/shifts.csv", "Wide-format chemical shifts by peak."),
        ("statistics/fit_statistics.json", "Per-cluster and global fit quality statistics."),
        ("diagnostics/mcmc_diagnostics.json", "MCMC convergence diagnostics."),
        ("metadata/run_metadata.json", "Run metadata and reproducibility context."),
        ("metadata/fitting_state.pkl", "Serialized fitting state for reuse."),
    ]

    for rel_path, description in file_descriptions:
        if (output_dir / rel_path).exists():
            lines.append(f"- `{rel_path}`: {description}")

    if config.include_legacy and (output_dir / "legacy").exists():
        lines.append("- `legacy/`: Legacy `.out` outputs for backward compatibility.")

    readme_path.write_text("\n".join(lines))


__all__ = [
    "write_readme",
    "write_simulated_spectra",
]
