"""Output directory management for PeakFit results.

This module provides the OutputDirectoryManager class for creating
and managing the standardized output directory structure.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path


class OutputType(str, Enum):
    """Types of output files for directory organization."""

    # Summary files
    SUMMARY = "summary"

    # Parameter outputs
    PARAMETERS = "parameters"

    # Statistical outputs
    STATISTICS = "statistics"

    # Diagnostic outputs
    DIAGNOSTICS = "diagnostics"

    # Figures and plots
    FIGURES = "figures"

    # Metadata and provenance
    METADATA = "metadata"

    # Legacy format (for backward compatibility)
    LEGACY = "legacy"

    # MCMC chain data
    CHAINS = "chains"


# Subdirectory names for each output type
OUTPUT_SUBDIRS: dict[OutputType, str] = {
    OutputType.SUMMARY: "summary",
    OutputType.PARAMETERS: "parameters",
    OutputType.STATISTICS: "statistics",
    OutputType.DIAGNOSTICS: "diagnostics",
    OutputType.FIGURES: "figures",
    OutputType.METADATA: "metadata",
    OutputType.LEGACY: "legacy",
    OutputType.CHAINS: "chains",
}


class OutputDirectoryManager:
    """Manages the standardized output directory structure.

    Directory Structure:
        output_YYYYMMDD_HHMMSS/
        ├── summary/
        │   ├── fit_summary.json      # Main results file
        │   ├── analysis_report.md    # Human-readable report
        │   └── quick_results.csv     # Key results for quick import
        ├── parameters/
        │   ├── parameters.csv        # All parameters (long format)
        │   ├── amplitudes.csv        # Intensities per plane
        │   └── parameters.json       # Parameters with full metadata
        ├── statistics/
        │   ├── fit_statistics.json   # Chi-squared, AIC, BIC
        │   ├── residuals.csv         # Residual values
        │   └── model_comparison.json # If multiple models
        ├── diagnostics/
        │   ├── mcmc_diagnostics.json # R-hat, ESS, convergence
        │   ├── convergence.csv       # Per-parameter convergence
        │   └── warnings.txt          # Collected warnings
        ├── figures/
        │   ├── profiles/             # Fit profiles per peak
        │   ├── diagnostics/          # MCMC diagnostic plots
        │   └── correlations/         # Correlation plots
        ├── metadata/
        │   ├── run_metadata.json     # Reproducibility info
        │   └── configuration.toml    # Copy of input config
        ├── chains/
        │   └── mcmc_chains.h5        # Full MCMC chains (HDF5)
        └── legacy/
            └── *.out                 # Legacy format files

    Attributes
    ----------
        base_dir: Base output directory path
        timestamp: Timestamp string used in directory name
        include_timestamp: Whether directory name includes timestamp
    """

    def __init__(
        self,
        base_dir: Path | str,
        *,
        include_timestamp: bool = True,
        timestamp: str | None = None,
    ) -> None:
        """Initialize output directory manager.

        Args:
            base_dir: Base directory path (will be created if needed)
            include_timestamp: Whether to add timestamp to directory name
            timestamp: Custom timestamp string (auto-generated if None)
        """
        self.include_timestamp = include_timestamp

        if timestamp:
            self.timestamp = timestamp
        else:
            self.timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        base_path = Path(base_dir)
        if include_timestamp:
            self.base_dir = base_path.parent / f"{base_path.name}_{self.timestamp}"
        else:
            self.base_dir = base_path

        self._subdirs_created = False

    @property
    def summary_dir(self) -> Path:
        """Path to summary directory."""
        return self.base_dir / OUTPUT_SUBDIRS[OutputType.SUMMARY]

    @property
    def parameters_dir(self) -> Path:
        """Path to parameters directory."""
        return self.base_dir / OUTPUT_SUBDIRS[OutputType.PARAMETERS]

    @property
    def statistics_dir(self) -> Path:
        """Path to statistics directory."""
        return self.base_dir / OUTPUT_SUBDIRS[OutputType.STATISTICS]

    @property
    def diagnostics_dir(self) -> Path:
        """Path to diagnostics directory."""
        return self.base_dir / OUTPUT_SUBDIRS[OutputType.DIAGNOSTICS]

    @property
    def figures_dir(self) -> Path:
        """Path to figures directory."""
        return self.base_dir / OUTPUT_SUBDIRS[OutputType.FIGURES]

    @property
    def metadata_dir(self) -> Path:
        """Path to metadata directory."""
        return self.base_dir / OUTPUT_SUBDIRS[OutputType.METADATA]

    @property
    def legacy_dir(self) -> Path:
        """Path to legacy format directory."""
        return self.base_dir / OUTPUT_SUBDIRS[OutputType.LEGACY]

    @property
    def chains_dir(self) -> Path:
        """Path to MCMC chains directory."""
        return self.base_dir / OUTPUT_SUBDIRS[OutputType.CHAINS]

    def get_dir(self, output_type: OutputType) -> Path:
        """Get directory path for an output type.

        Args:
            output_type: Type of output

        Returns
        -------
            Path to the appropriate subdirectory
        """
        return self.base_dir / OUTPUT_SUBDIRS[output_type]

    def ensure_structure(self) -> None:
        """Create the base directory structure.

        Only the base directory is created here; subdirectories are created
        lazily when files are written using get_path() with ensure_dir=True.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # Directories are created lazily by get_path() when files are written
        self._subdirs_created = False

    def get_path(
        self,
        output_type: OutputType,
        filename: str,
        *,
        ensure_dir: bool = True,
    ) -> Path:
        """Get full path for an output file.

        Args:
            output_type: Type of output (determines subdirectory)
            filename: Name of the file
            ensure_dir: Create subdirectory if it doesn't exist

        Returns
        -------
            Full path to the file
        """
        subdir = self.get_dir(output_type)
        if ensure_dir:
            subdir.mkdir(parents=True, exist_ok=True)
        return subdir / filename

    def get_figure_path(
        self,
        category: str,
        filename: str,
        *,
        ensure_dir: bool = True,
    ) -> Path:
        """Get path for a figure file.

        Args:
            category: Figure category (profiles, diagnostics, correlations)
            filename: Name of the figure file
            ensure_dir: Create directory if needed

        Returns
        -------
            Full path to the figure file
        """
        fig_subdir = self.figures_dir / category
        if ensure_dir:
            fig_subdir.mkdir(parents=True, exist_ok=True)
        return fig_subdir / filename

    # Standard file paths
    @property
    def fit_summary_path(self) -> Path:
        """Path to main fit_summary.json."""
        return self.get_path(OutputType.SUMMARY, "fit_summary.json")

    @property
    def analysis_report_path(self) -> Path:
        """Path to analysis_report.md."""
        return self.get_path(OutputType.SUMMARY, "analysis_report.md")

    @property
    def parameters_csv_path(self) -> Path:
        """Path to parameters.csv."""
        return self.get_path(OutputType.PARAMETERS, "parameters.csv")

    @property
    def amplitudes_csv_path(self) -> Path:
        """Path to amplitudes.csv."""
        return self.get_path(OutputType.PARAMETERS, "amplitudes.csv")

    @property
    def mcmc_diagnostics_path(self) -> Path:
        """Path to mcmc_diagnostics.json."""
        return self.get_path(OutputType.DIAGNOSTICS, "mcmc_diagnostics.json")

    @property
    def run_metadata_path(self) -> Path:
        """Path to run_metadata.json."""
        return self.get_path(OutputType.METADATA, "run_metadata.json")

    @property
    def mcmc_chains_path(self) -> Path:
        """Path to MCMC chains file."""
        return self.get_path(OutputType.CHAINS, "mcmc_chains.h5")

    @property
    def mcmc_chains_npz_path(self) -> Path:
        """Path to MCMC chains as NumPy archive (fallback)."""
        return self.get_path(OutputType.CHAINS, "mcmc_chains.npz")

    def get_legacy_profile_path(self, peak_name: str) -> Path:
        """Get path for legacy .out profile file.

        Args:
            peak_name: Name of the peak

        Returns
        -------
            Path to the legacy profile file
        """
        return self.get_path(OutputType.LEGACY, f"{peak_name}.out")

    def write_readme(self) -> None:
        """Write a README file explaining the directory structure."""
        readme_content = f"""# PeakFit Analysis Output

Generated: {self.timestamp}

## Directory Structure

- `summary/` - Quick-access results
  - `fit_summary.json` - Complete results in structured format
  - `analysis_report.md` - Human-readable report
  - `quick_results.csv` - Key results for spreadsheet import

- `parameters/` - Fitted parameter values
  - `parameters.csv` - All parameters in long format
  - `amplitudes.csv` - Peak intensities per Z-plane
  - `parameters.json` - Parameters with full metadata

- `statistics/` - Fit quality metrics
  - `fit_statistics.json` - Chi-squared, AIC, BIC, etc.
  - `residuals.csv` - Fit residuals

- `diagnostics/` - Convergence and quality checks
  - `mcmc_diagnostics.json` - R-hat, ESS for MCMC
  - `convergence.csv` - Per-parameter diagnostics
  - `warnings.txt` - Any flagged issues

- `figures/` - Generated plots
  - `profiles/` - Fit profiles for each peak
  - `diagnostics/` - MCMC trace plots, autocorrelation
  - `correlations/` - Parameter correlation plots

- `metadata/` - Reproducibility information
  - `run_metadata.json` - Software version, input checksums
  - `configuration.toml` - Copy of fitting configuration

- `chains/` - Full MCMC data (if applicable)
  - `mcmc_chains.h5` - Complete chains in HDF5 format

- `legacy/` - Legacy format files (for backward compatibility)
  - `*.out` - Traditional profile output files

## File Formats

- JSON files can be loaded with Python's `json` module or pandas
- CSV files use comma separators with header row
- HDF5 files require h5py or similar library

## Quick Start

### Python
```python
import json
import pandas as pd

# Load main results
with open("summary/fit_summary.json") as f:
    results = json.load(f)

# Load parameters as DataFrame
params = pd.read_csv("parameters/parameters.csv")
```

### R
```r
library(jsonlite)
library(readr)

results <- fromJSON("summary/fit_summary.json")
params <- read_csv("parameters/parameters.csv")
```
"""
        readme_path = self.base_dir / "README.md"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        readme_path.write_text(readme_content)


def generate_filename(
    base_name: str,
    *,
    cluster_id: int | None = None,
    peak_names: list[str] | None = None,
    extension: str = "",
    timestamp: str | None = None,
) -> str:
    """Generate a standardized filename.

    Naming convention:
    - Use underscores for word separation
    - Include identifiers when relevant
    - Avoid spaces and special characters

    Args:
        base_name: Base name for the file (e.g., "parameters", "trace_plot")
        cluster_id: Optional cluster identifier
        peak_names: Optional list of peak names
        extension: File extension (with or without leading dot)
        timestamp: Optional timestamp to include

    Returns
    -------
        Generated filename

    Examples
    --------
        >>> generate_filename("parameters", extension=".csv")
        "parameters.csv"
        >>> generate_filename("trace", cluster_id=1, extension=".pdf")
        "trace_cluster_1.pdf"
        >>> generate_filename("profile", peak_names=["G23N", "G23H"])
        "profile_G23N_G23H"
    """
    parts = [base_name]

    if cluster_id is not None:
        parts.append(f"cluster_{cluster_id}")

    if peak_names:
        # Sanitize peak names
        sanitized = [_sanitize_filename_part(name) for name in peak_names]
        parts.append("_".join(sanitized))

    if timestamp:
        parts.append(timestamp)

    filename = "_".join(parts)

    if extension:
        if not extension.startswith("."):
            extension = f".{extension}"
        filename += extension

    return filename


def _sanitize_filename_part(name: str) -> str:
    """Sanitize a string for use in filenames.

    Replaces or removes characters that may cause issues.
    """
    # Replace common problematic characters
    replacements = {
        " ": "_",
        "/": "-",
        "\\": "-",
        ":": "-",
        "*": "",
        "?": "",
        '"': "",
        "<": "",
        ">": "",
        "|": "-",
    }
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result
