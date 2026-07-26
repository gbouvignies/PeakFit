"""Plan and write fit result artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.csv import (
    has_shift_parameters,
    write_intensities,
    write_parameters,
    write_shifts,
)
from peakfit.io.writers.json import write_final_outcome_summary
from peakfit.io.writers.markdown import write_report

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.fit.final_outcome import FinalFitOutcome
    from peakfit.fit.result_models import FitResults, RunMetadata
    from peakfit.shared.typing import FloatArray


def build_output_plan(
    results: FitResults | None,
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

    if write_txt:
        files["report"] = output_dir / "summary" / "report.md"

    if write_csv and results is None:
        raise ValueError("CSV output requires the deferred FitResults projection.")

    if write_csv and results is not None and has_shift_parameters(results):
        files["shifts"] = output_dir / "tables" / "shifts.csv"

    return files


def write_fit_outputs(
    results: FitResults | None,
    output_dir: Path,
    config: WriterConfig | None = None,
    *,
    final_outcome: FinalFitOutcome | None = None,
    metadata: RunMetadata | None = None,
    z_values: FloatArray | None = None,
) -> dict[str, Path]:
    """Write fit result artifacts to output files.

    Run-level files such as README.md and fitting state are written by the fit
    workflow after fit artifacts are known.

    Args:
        results: FitResults object containing all output data
        output_dir: Base output directory
        config: Writer configuration (uses defaults if None)
        final_outcome: Authoritative completed result required for JSON 4.0.0
        metadata: Operational run metadata required for JSON 4.0.0
        z_values: Ordered series coordinate values for JSON 4.0.0

    Returns:
    -------
        Dictionary mapping output type to written file paths

    Output Structure:
        output_dir/
        ├── summary/
        │   ├── fit.json
        │   └── report.md            # when txt format is enabled
        ├── tables/
        │   ├── parameters.csv
        │   ├── intensities.csv
        │   └── shifts.csv           # when shift parameters are present
    """
    cfg = config or WriterConfig()
    written_files: dict[str, Path] = {}
    plan = build_output_plan(results, output_dir, cfg)

    if fit_json := plan.get("summary_fit"):
        if final_outcome is None:
            raise ValueError("JSON 4.0.0 output requires the authoritative FinalFitOutcome.")
        if metadata is None:
            raise ValueError("JSON 4.0.0 output requires run metadata.")
        written_files["summary_fit"] = write_final_outcome_summary(
            final_outcome,
            metadata=metadata,
            z_values=z_values,
            path=fit_json,
        )

    if params_csv := plan.get("parameters"):
        if results is None:
            raise AssertionError("CSV output plan requires FitResults.")
        write_parameters(results, params_csv, cfg)
        written_files["parameters"] = params_csv

    if intensities_csv := plan.get("intensities"):
        if results is None:
            raise AssertionError("CSV output plan requires FitResults.")
        write_intensities(results, intensities_csv, cfg)
        written_files["intensities"] = intensities_csv

    if report_md := plan.get("report"):
        if results is None:
            raise AssertionError("Markdown output plan requires FitResults.")
        written_files["report"] = write_report(results, report_md, cfg)

    if shifts_csv := plan.get("shifts"):
        if results is None:
            raise AssertionError("CSV output plan requires FitResults.")
        write_shifts(results, shifts_csv, cfg)
        written_files["shifts"] = shifts_csv

    return written_files


__all__ = [
    "build_output_plan",
    "write_fit_outputs",
]
