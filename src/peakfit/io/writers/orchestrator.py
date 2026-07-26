"""Plan and write fit result artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.csv import (
    has_final_shift_parameters,
    write_final_outcome_clusters,
    write_final_outcome_intensities,
    write_final_outcome_parameters,
    write_final_outcome_shifts,
)
from peakfit.io.writers.json import write_final_outcome_summary
from peakfit.io.writers.markdown import write_final_outcome_report

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.fit.final_outcome import FinalFitOutcome
    from peakfit.fit.output_metadata import RunMetadata
    from peakfit.fit.run_models import RunSummary
    from peakfit.shared.typing import FloatArray


def build_output_plan(
    outcome: FinalFitOutcome,
    output_dir: Path,
    config: WriterConfig,
) -> dict[str, Path]:
    """Resolve output paths from requested formats and available result data."""
    files: dict[str, Path] = {}

    write_json = config.enables("json")
    write_csv = config.enables("csv")
    write_txt = config.enables("txt")

    if write_json:
        files["summary_fit"] = output_dir / "summary" / "fit.json"

    if write_csv:
        files["parameters"] = output_dir / "tables" / "parameters.csv"
        files["intensities"] = output_dir / "tables" / "intensities.csv"
        files["clusters"] = output_dir / "tables" / "clusters.csv"

    if write_txt:
        files["report"] = output_dir / "summary" / "report.md"

    if write_csv and has_final_shift_parameters(outcome):
        files["shifts"] = output_dir / "tables" / "shifts.csv"

    return files


def write_fit_outputs(
    final_outcome: FinalFitOutcome,
    output_dir: Path,
    config: WriterConfig | None = None,
    *,
    metadata: RunMetadata | None = None,
    z_values: FloatArray | None = None,
    summary: RunSummary | None = None,
) -> dict[str, Path]:
    """Write fit result artifacts to output files.

    Run-level files such as README.md and fitting state are written by the fit
    workflow after fit artifacts are known.

    Args:
        final_outcome: Authoritative completed scientific result
        output_dir: Base output directory
        config: Writer configuration (uses defaults if None)
        metadata: Operational run metadata required for JSON 4.0.0
        z_values: Ordered series coordinate values for JSON and intensity tables
        summary: Ticket-06 run summary required for the Markdown report

    Returns:
    -------
        Dictionary mapping output type to written file paths

    Output Structure:
        output_dir/
        ├── summary/
        │   ├── fit.json
        │   └── report.md            # when txt format is enabled
        ├── tables/
        │   ├── clusters.csv          # final status and provenance per cluster
        │   ├── parameters.csv
        │   ├── intensities.csv
        │   └── shifts.csv           # when shift parameters are present
    """
    cfg = config or WriterConfig()
    written_files: dict[str, Path] = {}
    plan = build_output_plan(final_outcome, output_dir, cfg)

    if fit_json := plan.get("summary_fit"):
        if metadata is None:
            raise ValueError("JSON 4.0.0 output requires run metadata.")
        written_files["summary_fit"] = write_final_outcome_summary(
            final_outcome,
            metadata=metadata,
            z_values=z_values,
            path=fit_json,
        )

    if params_csv := plan.get("parameters"):
        write_final_outcome_parameters(final_outcome, params_csv, cfg)
        written_files["parameters"] = params_csv

    if intensities_csv := plan.get("intensities"):
        write_final_outcome_intensities(final_outcome, z_values, intensities_csv, cfg)
        written_files["intensities"] = intensities_csv

    if clusters_csv := plan.get("clusters"):
        write_final_outcome_clusters(final_outcome, clusters_csv, cfg)
        written_files["clusters"] = clusters_csv

    if report_md := plan.get("report"):
        if summary is None:
            raise ValueError("Markdown output requires the final-outcome RunSummary.")
        written_files["report"] = write_final_outcome_report(final_outcome, report_md, summary, cfg)

    if shifts_csv := plan.get("shifts"):
        write_final_outcome_shifts(final_outcome, shifts_csv, cfg)
        written_files["shifts"] = shifts_csv

    return written_files


__all__ = [
    "build_output_plan",
    "write_fit_outputs",
]
