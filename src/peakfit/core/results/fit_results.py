"""Top-level fit results model.

This module defines the FitResults dataclass that aggregates all
output from a fitting run, including parameters, statistics, and
diagnostics for all clusters.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peakfit.core.results.diagnostics import MCMCDiagnostics
    from peakfit.core.results.estimates import ClusterEstimates
    from peakfit.core.results.statistics import FitStatistics, ModelComparison
    from peakfit.core.shared.typing import FloatArray


class FitMethod(str, Enum):
    """Fitting method used."""

    LEAST_SQUARES = "least_squares"
    BASIN_HOPPING = "basin_hopping"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    MCMC = "mcmc"
    PROFILE_LIKELIHOOD = "profile_likelihood"


class OutputVerbosity(str, Enum):
    """Output verbosity levels.

    Controls which outputs are generated:
    - MINIMAL: Essential outputs only (parameters CSV, summary JSON)
    - STANDARD: Default outputs (+ diagnostics, figures)
    - FULL: All outputs including posteriors and debug info
    """

    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL = "full"


@dataclass
class RunMetadata:
    """Metadata about the fitting run for reproducibility.

    Attributes:
        timestamp: When the analysis was run (ISO 8601)
        software_version: PeakFit version string
        git_commit: Git commit hash if in repository
        python_version: Python interpreter version
        platform: Operating system platform
        input_files: Dictionary of input file paths and checksums
        configuration: Complete configuration used for fitting
        command_line: Command line arguments if available
        run_duration_seconds: Total run time
    """

    timestamp: str = ""
    software_version: str = ""
    git_commit: str | None = None
    python_version: str = ""
    platform: str = ""
    input_files: dict[str, dict[str, str]] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    command_line: str = ""
    run_duration_seconds: float | None = None

    @classmethod
    def capture(cls, config: dict | None = None) -> RunMetadata:
        """Capture current environment metadata.

        Args:
            config: Configuration dictionary to include

        Returns:
            RunMetadata with populated fields
        """
        import platform
        import sys

        # Try to get git commit
        git_commit = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode == 0:
                git_commit = result.stdout.strip()[:12]
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass

        # Get version
        try:
            from peakfit import __version__

            version = __version__
        except ImportError:
            version = "unknown"

        return cls(
            timestamp=datetime.now(UTC).isoformat(),
            software_version=version,
            git_commit=git_commit,
            python_version=sys.version,
            platform=platform.platform(),
            configuration=config or {},
        )

    def add_input_file(self, name: str, path: Path) -> None:
        """Add an input file with its checksum.

        Args:
            name: Descriptive name for the file
            path: Path to the file
        """
        if path.exists():
            checksum = _compute_file_checksum(path)
            self.input_files[name] = {
                "path": str(path.name),  # Relative, not absolute
                "checksum_sha256": checksum,
            }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "timestamp": self.timestamp,
            "software_version": self.software_version,
            "python_version": self.python_version,
            "platform": self.platform,
        }

        if self.git_commit:
            result["git_commit"] = self.git_commit
        if self.input_files:
            result["input_files"] = self.input_files
        if self.configuration:
            result["configuration"] = self.configuration
        if self.command_line:
            result["command_line"] = self.command_line
        if self.run_duration_seconds is not None:
            result["run_duration_seconds"] = self.run_duration_seconds

        return result


@dataclass
class FitResults:
    """Complete results from a fitting run.

    This is the top-level result object that aggregates all outputs
    from fitting: parameter estimates, statistics, and diagnostics
    for all clusters.

    Attributes:
        metadata: Run metadata for reproducibility
        method: Fitting method used
        clusters: Per-cluster parameter estimates
        statistics: Per-cluster fit statistics
        global_statistics: Overall fit statistics (if applicable)
        mcmc_diagnostics: Per-cluster MCMC diagnostics (if MCMC used)
        model_comparisons: Model comparison results (if multiple models)
        z_values: Z-dimension values (e.g., relaxation delays)
        z_unit: Unit of Z-values (e.g., "s", "Hz")
        experiment_type: Type of experiment (e.g., "CPMG", "CEST", "R1")
    """

    metadata: RunMetadata = field(default_factory=RunMetadata)
    method: FitMethod = FitMethod.LEAST_SQUARES
    clusters: list[ClusterEstimates] = field(default_factory=list)
    statistics: list[FitStatistics] = field(default_factory=list)
    global_statistics: FitStatistics | None = None
    mcmc_diagnostics: list[MCMCDiagnostics] = field(default_factory=list)
    model_comparisons: list[ModelComparison] = field(default_factory=list)
    z_values: FloatArray | None = None
    z_unit: str = ""
    experiment_type: str = ""

    @property
    def n_clusters(self) -> int:
        """Number of fitted clusters."""
        return len(self.clusters)

    @property
    def n_peaks(self) -> int:
        """Total number of peaks across all clusters."""
        return sum(c.n_peaks for c in self.clusters)

    @property
    def all_peak_names(self) -> list[str]:
        """List of all peak names across clusters."""
        names = []
        for cluster in self.clusters:
            names.extend(cluster.peak_names)
        return names

    @property
    def is_mcmc(self) -> bool:
        """Check if MCMC was used."""
        return self.method == FitMethod.MCMC

    @property
    def has_converged(self) -> bool:
        """Check if all MCMC analyses converged."""
        if not self.mcmc_diagnostics:
            return True  # Non-MCMC assumed converged
        return all(d.converged for d in self.mcmc_diagnostics)

    def get_cluster_by_peak(self, peak_name: str) -> ClusterEstimates | None:
        """Find cluster containing a specific peak."""
        for cluster in self.clusters:
            if peak_name in cluster.peak_names:
                return cluster
        return None

    def get_all_problematic_params(self) -> list[tuple[str, str]]:
        """Get all problematic parameters across clusters.

        Returns:
            List of (cluster_label, param_name) tuples
        """
        problems = []
        for cluster in self.clusters:
            label = ", ".join(cluster.peak_names)
            problems.extend((label, param.name) for param in cluster.get_problematic_params())
        return problems

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "metadata": self.metadata.to_dict(),
            "method": self.method.value,
            "n_clusters": self.n_clusters,
            "n_peaks": self.n_peaks,
            "experiment_type": self.experiment_type,
            "clusters": [c.to_dict() for c in self.clusters],
        }

        if self.statistics:
            result["statistics"] = [s.to_dict() for s in self.statistics]

        if self.global_statistics:
            result["global_statistics"] = self.global_statistics.to_dict()

        if self.mcmc_diagnostics:
            result["mcmc_diagnostics"] = [d.to_dict() for d in self.mcmc_diagnostics]

        if self.model_comparisons:
            result["model_comparisons"] = [m.to_dict() for m in self.model_comparisons]

        if self.z_values is not None:
            result["z_values"] = self.z_values.tolist()
            result["z_unit"] = self.z_unit

        return result

    def summary_dict(self) -> dict:
        """Generate a summary dictionary for quick inspection.

        This is a condensed version suitable for the executive summary.
        """
        summary = {
            "timestamp": self.metadata.timestamp,
            "method": self.method.value,
            "n_clusters": self.n_clusters,
            "n_peaks": self.n_peaks,
            "experiment_type": self.experiment_type,
        }

        if self.global_statistics:
            summary["reduced_chi_squared"] = self.global_statistics.reduced_chi_squared
            summary["fit_converged"] = self.global_statistics.fit_converged

        if self.mcmc_diagnostics:
            summary["mcmc_converged"] = self.has_converged
            n_problematic = sum(len(d.get_problematic_parameters()) for d in self.mcmc_diagnostics)
            summary["n_problematic_params"] = n_problematic

        problems = self.get_all_problematic_params()
        summary["n_problematic_total"] = len(problems)
        if problems:
            summary["problematic_params"] = problems[:10]  # First 10 only

        return summary


def _compute_file_checksum(path: Path, algorithm: str = "sha256") -> str:
    """Compute checksum of a file.

    Args:
        path: Path to file
        algorithm: Hash algorithm (default sha256)

    Returns:
        Hex digest of file contents
    """
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
