"""Run-level output file utilities."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.shared.reporter import Reporter
    from peakfit.shared.typing import FloatArray


def write_simulated_spectra(
    output_dir: Path,
    spectra: Spectra,
    data_simulated: FloatArray,
    reporter: Reporter | None = None,
) -> Path | None:
    """Write optional simulated spectra to an NMRPipe file."""
    try:
        ng = import_module("nmrglue")
    except ModuleNotFoundError:
        return None

    if reporter:
        reporter.action("Writing simulated spectra...")

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
    """Generate the output-directory README."""
    readme_path = output_dir / "README.md"
    median_redchi = (
        f"{summary.median_redchi:.4g}"
        if summary.median_redchi is not None
        else "N/A (no usable outcomes)"
    )

    lines = [
        "# PeakFit Run",
        "",
        "## Summary",
        "",
        f"- **Date**: {datetime.now(UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Peaks**: {summary.n_peaks}",
        f"- **Converged clusters**: {summary.n_converged}/{summary.n_clusters} "
        f"({summary.success_rate:.1%})",
        f"- **Usable, not converged**: {summary.n_usable_non_converged}",
        f"- **Unusable clusters**: {summary.n_unusable}",
        f"- **Usable clusters**: {summary.n_usable}/{summary.n_clusters}",
        f"- **Reduced chi2 population**: {summary.redchi_population_size}",
        f"- **Median reduced chi2**: {median_redchi}",
        "",
        "## Files",
        "",
    ]

    file_descriptions = [
        ("summary/fit.json", "Main machine-readable fit summary."),
        ("summary/report.md", "Concise human-readable report."),
        ("tables/parameters.csv", "Model parameters."),
        ("tables/intensities.csv", "Per-plane fitted intensities and errors."),
        ("tables/clusters.csv", "Final cluster classifications and optimizer provenance."),
        ("tables/shifts.csv", "Chemical shifts by peak."),
        ("metadata/fitting_state.pkl", "Saved state for MCMC and plotting workflows."),
    ]

    for rel_path, description in file_descriptions:
        if (output_dir / rel_path).exists():
            lines.append(f"- `{rel_path}`: {description}")

    lines.extend(
        [
            "",
            "## Next Steps",
            "",
            "- Inspect `summary/fit.json` and `tables/parameters.csv` for fitted values.",
            "- Plot fitted amplitudes with `peakfit plot intensity <this-run-directory>`.",
            "- Plot CEST or CPMG profiles from `tables/intensities.csv` when the z-axis "
            "matches that experiment.",
            "- Run MCMC with `peakfit mcmc <this-run-directory> --peaks <peak-name>` "
            "when uncertainty estimates are needed.",
        ]
    )

    readme_path.write_text("\n".join(lines) + "\n")
    return readme_path


__all__ = [
    "write_readme",
    "write_simulated_spectra",
]
