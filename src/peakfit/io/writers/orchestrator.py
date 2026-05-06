"""Main orchestrator for all output writers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from peakfit.io.writers.config import Verbosity, WriterConfig
from peakfit.io.writers.csv import CSVWriter
from peakfit.io.writers.json import JSONWriter
from peakfit.io.writers.legacy import LegacyWriter
from peakfit.io.writers.markdown import MarkdownReportGenerator
from peakfit.io.writers.simulation import write_readme, write_simulated_spectra

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.engine.results import FitResults
    from peakfit.shared.reporter import Reporter


def _write_manifest(
    output_dir: Path,
    results: FitResults,
    config: WriterConfig,
    written_files: dict[str, Path],
) -> Path:
    """Write a compact index of generated outputs."""
    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def _as_rel(path: Path) -> str:
        return str(path.relative_to(output_dir))

    payload = {
        "schema_version": "2.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "verbosity": config.verbosity.value,
        "formats": list(config.formats),
        "n_clusters": results.n_clusters,
        "n_peaks": results.n_peaks,
        "n_parameters": sum(len(cluster.lineshape_params) for cluster in results.clusters),
        "n_intensities": sum(len(cluster.amplitudes) for cluster in results.clusters),
        "files": {name: _as_rel(path) for name, path in written_files.items()},
    }

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    return manifest_path


def _write_fit_outputs(
    results: FitResults,
    output_dir: Path,
    config: WriterConfig,
) -> dict[str, Path]:
    """Write output files according to configured format and verbosity."""
    written_files: dict[str, Path] = {}

    json_writer = JSONWriter(config)
    csv_writer = CSVWriter(config)
    md_writer = MarkdownReportGenerator(config)

    write_json = config.enables("json")
    write_csv = config.enables("csv")
    write_txt = config.enables("txt")
    write_extended = config.verbosity in (Verbosity.STANDARD, Verbosity.FULL)

    if write_json:
        fit_json = output_dir / "summary" / "fit_summary.json"
        json_writer.write_results(results, fit_json)
        written_files["summary_fit"] = fit_json

        metadata_json = output_dir / "metadata" / "run_metadata.json"
        json_writer.write_metadata(results, metadata_json)
        written_files["metadata_run"] = metadata_json

    if write_csv:
        params_csv = output_dir / "parameters" / "parameters.csv"
        csv_writer.write_parameters(results, params_csv)
        written_files["parameters"] = params_csv

        intensities_csv = output_dir / "parameters" / "intensities.csv"
        csv_writer.write_intensities(results, intensities_csv)
        written_files["intensities"] = intensities_csv

    if write_extended:
        if write_txt:
            report_md = output_dir / "summary" / "report.md"
            md_writer.generate_full_report(results, report_md)
            written_files["report"] = report_md

        if write_json:
            statistics_json = output_dir / "statistics" / "fit_statistics.json"
            json_writer.write_statistics(results, statistics_json)
            written_files["statistics"] = statistics_json

            if results.mcmc_diagnostics:
                diagnostics_json = output_dir / "diagnostics" / "mcmc_diagnostics.json"
                json_writer.write_diagnostics(results, diagnostics_json)
                written_files["diagnostics_mcmc"] = diagnostics_json

        if write_csv and csv_writer.has_shift_parameters(results):
            shifts_csv = output_dir / "parameters" / "shifts.csv"
            csv_writer.write_shifts(results, shifts_csv)
            written_files["shifts"] = shifts_csv

    if config.include_legacy:
        legacy_writer = LegacyWriter(config)
        legacy_writer.write_all(results, output_dir)
        written_files["legacy_dir"] = output_dir / "legacy"

    manifest = _write_manifest(output_dir, results, config, written_files)
    written_files["manifest"] = manifest

    return written_files


def write_fit_outputs(
    results: FitResults,
    output_dir: Path,
    config: WriterConfig | None = None,
) -> dict[str, Path]:
    """Write fit results to output files.

    Main entry point for writing all fit outputs. The verbosity level
    in the config determines which files are written.

    Args:
        results: FitResults object containing all output data
        output_dir: Base output directory
        config: Writer configuration (uses defaults if None)

    Returns:
    -------
        Dictionary mapping output type to written file paths

    Output Structure (standard verbosity):
        output_dir/
        ├── manifest.json
        ├── summary/
        │   ├── fit_summary.json
        │   └── report.md            # when txt format is enabled
        ├── parameters/
        │   ├── parameters.csv
        │   ├── intensities.csv
        │   └── shifts.csv           # when shift parameters are present
        ├── statistics/
        │   └── fit_statistics.json
        ├── diagnostics/
        │   └── mcmc_diagnostics.json # when MCMC diagnostics exist
        ├── metadata/
        │   └── run_metadata.json
        └── legacy/                  # when include_legacy is enabled
    """
    cfg = config or WriterConfig()
    return _write_fit_outputs(results, output_dir, cfg)


def write_simulation_outputs(
    output_dir: Path,
    spectra: Spectra,
    clusters: list[Cluster],
    params: Parameters,
    config: WriterConfig | None = None,
    reporter: Reporter | None = None,
) -> None:
    """Write simulated spectra to file.

    Args:
        output_dir: Directory where the simulated file should be saved.
        spectra: The original spectra object (used for metadata/header).
        clusters: List of clusters used in the fit.
        params: The fitted parameters.
        config: Writer configuration (uses defaults if None)
        reporter: Optional reporter for progress updates.
    """
    cfg = config or WriterConfig()
    if cfg.save_simulated:
        write_simulated_spectra(output_dir, spectra, clusters, params, reporter)


def write_run_readme(
    output_dir: Path,
    summary: Any,
    config: WriterConfig | None = None,
) -> None:
    """Write README.md for the output directory.

    Args:
        output_dir: Path to the output directory.
        summary: RunSummary object containing run metrics.
        config: Writer configuration (uses defaults if None)
    """
    cfg = config or WriterConfig()
    write_readme(output_dir, cfg, summary)


# Keep ResultsWriter for backward compatibility (thin wrapper)
class ResultsWriter:
    """Backward-compatible wrapper for write_fit_outputs.

    Deprecated: Use write_fit_outputs() function directly.
    """

    def __init__(
        self,
        config: WriterConfig | None = None,
        formats: set[str] | None = None,
        include_legacy: bool | None = None,
    ) -> None:
        """Initialize results writer."""
        self.config = config or WriterConfig()
        if formats:
            self.config.formats = tuple(sorted(formats))
        if include_legacy is not None:
            self.config.include_legacy = include_legacy

    def write_for_verbosity(
        self, results: FitResults, output_dir: Path, verbosity: Verbosity
    ) -> dict[str, Path]:
        """Write output files based on verbosity level."""
        cfg = WriterConfig(
            verbosity=verbosity,
            formats=self.config.formats,
            include_legacy=self.config.include_legacy,
            include_amplitudes_in_summary=self.config.include_amplitudes_in_summary,
            save_simulated=self.config.save_simulated,
            precision=self.config.precision,
        )
        return write_fit_outputs(results, output_dir, cfg)

    def write_simulation(
        self,
        output_dir: Path,
        spectra: Spectra,
        clusters: list[Cluster],
        params: Parameters,
        reporter: Reporter | None = None,
    ) -> None:
        """Write simulated spectra."""
        write_simulation_outputs(output_dir, spectra, clusters, params, self.config, reporter)

    def write_readme(self, output_dir: Path, summary: Any) -> None:
        """Write README.md."""
        write_run_readme(output_dir, summary, self.config)


__all__ = [
    "ResultsWriter",
    "write_fit_outputs",
    "write_run_readme",
    "write_simulation_outputs",
]
