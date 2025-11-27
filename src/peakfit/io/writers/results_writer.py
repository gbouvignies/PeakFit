"""Integrated results writer service.

This module provides the main ResultsWriter service that coordinates
all output writers to produce the complete set of output files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from peakfit.io.writers.base import Verbosity, WriterConfig
from peakfit.io.writers.csv_writer import CSVWriter
from peakfit.io.writers.json_writer import JSONWriter
from peakfit.io.writers.markdown_writer import MarkdownReportGenerator

if TYPE_CHECKING:
    from peakfit.core.results.fit_results import FitResults


class ResultsWriter:
    """Coordinated results writer for all output formats.

    This service orchestrates all output writers to produce a complete
    set of output files in a clean, flat directory structure.

    Output Structure (default):
        output_dir/
        ├── fit_results.json      # Complete machine-readable results
        ├── parameters.csv        # All parameters (long format)
        ├── shifts.csv            # Chemical shifts (wide format, easy to use)
        ├── intensities.csv       # Fitted intensities for all peaks
        ├── report.md             # Human-readable summary
        ├── peakfit.log           # Log file (written by pipeline)
        ├── simulated.ft3         # Simulated spectrum (written by pipeline)
        └── cache/
            └── state.pkl         # Internal state for MCMC continuation

    With MCMC diagnostics:
        output_dir/
        ├── fit_results.json
        ├── parameters.csv
        ├── shifts.csv
        ├── intensities.csv
        ├── report.md
        ├── mcmc/
        │   ├── diagnostics.json
        │   └── chains.npz        # If save_chains=True
        └── figures/              # If figures generated
            └── ...

    With legacy files (--legacy):
        output_dir/
        ├── fit_results.json
        ├── parameters.csv
        ├── shifts.csv
        ├── intensities.csv
        ├── report.md
        └── legacy/
            ├── {peak_name}.out
            └── shifts.list
    """

    def __init__(
        self,
        config: WriterConfig | None = None,
        include_legacy: bool = False,
    ) -> None:
        """Initialize results writer.

        Args:
            config: Writer configuration. Defaults to standard settings.
            include_legacy: Whether to write legacy format files.
        """
        self.config = config or WriterConfig()
        self.include_legacy = include_legacy

        # Initialize individual writers
        self.json_writer = JSONWriter(self.config)
        self.csv_writer = CSVWriter(self.config)
        self.markdown_writer = MarkdownReportGenerator(self.config)

    def write_all(self, results: FitResults, output_dir: Path) -> dict[str, Path]:
        """Write all output files.

        Args:
            results: FitResults object containing all output data
            output_dir: Base output directory

        Returns:
            Dictionary mapping output type to written file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        written_files: dict[str, Path] = {}

        # Main files at root level
        fit_json = output_dir / "fit_results.json"
        self.json_writer.write_results(results, fit_json)
        written_files["fit_results"] = fit_json

        report_md = output_dir / "report.md"
        self.markdown_writer.generate_full_report(results, report_md)
        written_files["report"] = report_md

        params_csv = output_dir / "parameters.csv"
        self.csv_writer.write_parameters(results, params_csv)
        written_files["parameters"] = params_csv

        shifts_csv = output_dir / "shifts.csv"
        self.csv_writer.write_shifts(results, shifts_csv)
        written_files["shifts"] = shifts_csv

        intensities_csv = output_dir / "intensities.csv"
        self.csv_writer.write_intensities(results, intensities_csv)
        written_files["intensities"] = intensities_csv

        # MCMC diagnostics in mcmc/ subdirectory (if applicable)
        if results.mcmc_diagnostics:
            mcmc_dir = output_dir / "mcmc"
            mcmc_dir.mkdir(exist_ok=True)

            diag_json = mcmc_dir / "diagnostics.json"
            self.json_writer.write_diagnostics(results, diag_json)
            written_files["mcmc_diagnostics"] = diag_json

        return written_files

    def write_minimal(self, results: FitResults, output_dir: Path) -> dict[str, Path]:
        """Write minimal output files (essential only).

        Args:
            results: FitResults object
            output_dir: Base output directory

        Returns:
            Dictionary mapping output type to written file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        written_files: dict[str, Path] = {}

        # Essential: fit results JSON and parameters CSV
        fit_json = output_dir / "fit_results.json"
        self.json_writer.write_results(results, fit_json)
        written_files["fit_results"] = fit_json

        params_csv = output_dir / "parameters.csv"
        self.csv_writer.write_parameters(results, params_csv)
        written_files["parameters"] = params_csv

        shifts_csv = output_dir / "shifts.csv"
        self.csv_writer.write_shifts(results, shifts_csv)
        written_files["shifts"] = shifts_csv

        intensities_csv = output_dir / "intensities.csv"
        self.csv_writer.write_intensities(results, intensities_csv)
        written_files["intensities"] = intensities_csv

        return written_files

    def write_for_verbosity(
        self, results: FitResults, output_dir: Path, verbosity: Verbosity
    ) -> dict[str, Path]:
        """Write output files based on verbosity level.

        Args:
            results: FitResults object
            output_dir: Base output directory
            verbosity: Verbosity level controlling what to write

        Returns:
            Dictionary mapping output type to written file paths
        """
        if verbosity == Verbosity.MINIMAL:
            return self.write_minimal(results, output_dir)
        if verbosity == Verbosity.FULL:
            return self.write_all(results, output_dir)
        # STANDARD: write main outputs
        return self._write_standard(results, output_dir)

    def _write_standard(self, results: FitResults, output_dir: Path) -> dict[str, Path]:
        """Write standard output files (clean, flat structure)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        written_files: dict[str, Path] = {}

        # Main files at root level
        fit_json = output_dir / "fit_results.json"
        self.json_writer.write_results(results, fit_json)
        written_files["fit_results"] = fit_json

        report_md = output_dir / "report.md"
        self.markdown_writer.generate_summary_report(results, report_md)
        written_files["report"] = report_md

        params_csv = output_dir / "parameters.csv"
        self.csv_writer.write_parameters(results, params_csv)
        written_files["parameters"] = params_csv

        shifts_csv = output_dir / "shifts.csv"
        self.csv_writer.write_shifts(results, shifts_csv)
        written_files["shifts"] = shifts_csv

        intensities_csv = output_dir / "intensities.csv"
        self.csv_writer.write_intensities(results, intensities_csv)
        written_files["intensities"] = intensities_csv

        return written_files
