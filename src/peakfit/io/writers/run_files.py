"""Run-level output file utilities."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


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
        ("scientific.npz", "Lossless scientific result, samples, coordinates and provenance."),
        ("simulated.npy", "Optional full-grid float64 model; calibration is in scientific.npz."),
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
]
