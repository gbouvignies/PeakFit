"""CSV writer for PeakFit results.

Implements CSV output in long format for easy import into
pandas, R, Excel, and other analysis tools.
"""

from __future__ import annotations

import csv
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.io.writers.base import WriterConfig, format_float

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.core.results.estimates import (
        AmplitudeEstimate,
        ClusterEstimates,
        ParameterEstimate,
    )
    from peakfit.core.results.fit_results import FitResults


class CSVWriter:
    """Writer for CSV format outputs.

    Produces long-format CSV files suitable for data analysis tools.
    Long format has one row per parameter/measurement, making it easy
    to filter, group, and join data.
    """

    def __init__(self, config: WriterConfig | None = None) -> None:
        """Initialize CSV writer.

        Args:
            config: Writer configuration (uses defaults if None)
        """
        self.config = config or WriterConfig()

    def write_results(self, results: FitResults, path: Path) -> None:
        """Write complete results summary to CSV.

        This writes a condensed summary with key results per peak.

        Args:
            results: FitResults object
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="") as f:
            writer = csv.writer(
                f,
                delimiter=self.config.csv_delimiter,
                quoting=csv.QUOTE_MINIMAL if self.config.csv_quoting else csv.QUOTE_NONE,
            )

            # Write header comment if enabled
            if self.config.include_comments:
                f.write("# PeakFit Results Summary\n")
                f.write(f"# Generated: {results.metadata.timestamp}\n")
                f.write(f"# Method: {results.method.value}\n")
                f.write("#\n")

            # Header
            writer.writerow(
                [
                    "cluster_id",
                    "peak_name",
                    "n_params",
                    "n_planes",
                    "reduced_chi_squared",
                    "converged",
                ]
            )

            # Data rows
            for i, cluster in enumerate(results.clusters):
                stats = results.statistics[i] if i < len(results.statistics) else None
                diag = results.mcmc_diagnostics[i] if i < len(results.mcmc_diagnostics) else None

                for peak_name in cluster.peak_names:
                    writer.writerow(
                        [
                            cluster.cluster_id,
                            peak_name,
                            cluster.n_lineshape_params,
                            cluster.n_planes,
                            format_float(stats.reduced_chi_squared, self.config.precision)
                            if stats
                            else "",
                            diag.converged if diag else "",
                        ]
                    )

    def write_parameters(self, results: FitResults, path: Path) -> None:
        """Write parameters in long format.

        Columns:
        - cluster_id, peak_name, parameter, category
        - value, std_error, ci_68_lower, ci_68_upper, ci_95_lower, ci_95_upper
        - unit, min_bound, max_bound, is_fixed, is_global

        Args:
            results: FitResults object
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="") as f:
            # Write header comment
            if self.config.include_comments:
                f.write("# PeakFit Parameter Estimates\n")
                f.write(f"# Generated: {results.metadata.timestamp}\n")
                f.write(f"# Method: {results.method.value}\n")
                f.write("# Format: Long format (one row per parameter)\n")
                f.write("#\n")
                f.write("# Column descriptions:\n")
                f.write("#   cluster_id: Cluster identifier\n")
                f.write("#   peak_name: Peak from input peak list\n")
                f.write("#   parameter: Parameter name\n")
                f.write("#   category: Parameter type (lineshape, amplitude, etc.)\n")
                f.write("#   value: Best-fit or MAP estimate\n")
                f.write("#   std_error: Standard error (symmetric uncertainty)\n")
                f.write("#   ci_68_lower/upper: 68% credible interval bounds\n")
                f.write("#   ci_95_lower/upper: 95% credible interval bounds\n")
                f.write("#   unit: Physical unit\n")
                f.write("#   min_bound/max_bound: Fitting bounds\n")
                f.write("#   is_fixed: Parameter was fixed during fitting\n")
                f.write("#   is_global: Parameter shared across clusters\n")
                f.write("#\n")

            writer = csv.writer(f, delimiter=self.config.csv_delimiter)

            # Header
            writer.writerow(
                [
                    "cluster_id",
                    "peak_name",
                    "parameter",
                    "category",
                    "value",
                    "std_error",
                    "ci_68_lower",
                    "ci_68_upper",
                    "ci_95_lower",
                    "ci_95_upper",
                    "unit",
                    "min_bound",
                    "max_bound",
                    "is_fixed",
                    "is_global",
                ]
            )

            # Data rows
            for cluster in results.clusters:
                # Lineshape parameters
                for param in cluster.lineshape_params:
                    self._write_parameter_row(writer, cluster, param)

    def _write_parameter_row(
        self,
        writer: Any,
        cluster: ClusterEstimates,
        param: ParameterEstimate,
    ) -> None:
        """Write a single parameter row."""
        prec = self.config.precision
        thresh = self.config.scientific_notation_threshold

        # Determine peak name from parameter
        peak_name = self._extract_peak_name_from_param(param, cluster.peak_names)

        # Get user-friendly parameter name
        # Use ParameterEstimate.user_name which uses param_id if available
        friendly_name = param.user_name

        writer.writerow(
            [
                cluster.cluster_id,
                peak_name,
                friendly_name,
                param.category.value,
                format_float(param.value, prec, thresh),
                format_float(param.std_error, prec, thresh),
                format_float(param.ci_68_lower, prec, thresh)
                if param.ci_68_lower is not None
                else "",
                format_float(param.ci_68_upper, prec, thresh)
                if param.ci_68_upper is not None
                else "",
                format_float(param.ci_95_lower, prec, thresh)
                if param.ci_95_lower is not None
                else "",
                format_float(param.ci_95_upper, prec, thresh)
                if param.ci_95_upper is not None
                else "",
                param.unit,
                format_float(param.min_bound, prec, thresh)
                if not np.isinf(param.min_bound)
                else "",
                format_float(param.max_bound, prec, thresh)
                if not np.isinf(param.max_bound)
                else "",
                param.is_fixed,
                param.is_global,
            ]
        )

    def _extract_peak_name_from_param(self, param: ParameterEstimate, peak_names: list[str]) -> str:
        """Extract the original peak name from a parameter.

        Args:
            param: ParameterEstimate object
            peak_names: List of peak names in the cluster

        Returns
        -------
            Original peak name like '2N-H'
        """
        # Use param_id if available (new format)
        if param.param_id is not None:
            return param.param_id.peak_name

        # Legacy fallback: parse from internal name
        param_name = param.name
        for peak_name in peak_names:
            # New format: "peak_name.axis.type"
            if param_name.startswith(peak_name + "."):
                return peak_name
            # Legacy format: sanitized prefix
            safe_prefix = re.sub(r"\W+|^(?=\d)", "_", peak_name)
            if param_name.startswith(safe_prefix + "_"):
                return peak_name

        # Fallback to first peak
        return peak_names[0] if peak_names else ""

    def write_amplitudes(self, results: FitResults, path: Path) -> None:
        """Write amplitude (intensity) values to CSV.

        Columns:
        - cluster_id, peak_name, plane_index, z_value
        - value, std_error, ci_68_lower, ci_68_upper

        Args:
            results: FitResults object
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="") as f:
            if self.config.include_comments:
                f.write("# PeakFit Amplitude (Intensity) Values\n")
                f.write(f"# Generated: {results.metadata.timestamp}\n")
                if results.z_unit:
                    f.write(f"# Z-axis unit: {results.z_unit}\n")
                f.write("#\n")

            writer = csv.writer(f, delimiter=self.config.csv_delimiter)

            # Header
            writer.writerow(
                [
                    "cluster_id",
                    "peak_name",
                    "plane_index",
                    "z_value",
                    "value",
                    "std_error",
                    "ci_68_lower",
                    "ci_68_upper",
                ]
            )

            # Data
            for cluster in results.clusters:
                for amp in cluster.amplitudes:
                    self._write_amplitude_row(writer, cluster, amp)

    def _write_amplitude_row(
        self,
        writer: Any,
        cluster: ClusterEstimates,
        amp: AmplitudeEstimate,
    ) -> None:
        """Write a single amplitude row."""
        prec = self.config.precision
        thresh = self.config.scientific_notation_threshold

        writer.writerow(
            [
                cluster.cluster_id,
                amp.peak_name,
                amp.plane_index,
                format_float(amp.z_value, prec, thresh) if amp.z_value is not None else "",
                format_float(amp.value, prec, thresh),
                format_float(amp.std_error, prec, thresh),
                format_float(amp.ci_68_lower, prec, thresh) if amp.ci_68_lower is not None else "",
                format_float(amp.ci_68_upper, prec, thresh) if amp.ci_68_upper is not None else "",
            ]
        )

    def write_shifts(self, results: FitResults, path: Path) -> None:
        """Write chemical shifts in wide format for easy downstream use.

        Creates a table with one row per peak containing chemical shifts
        for all dimensions. Automatically adapts to the number of dimensions
        using F1/F2/F3/F4 naming convention.

        For 2D: peak_name, cs_F1_ppm, cs_F1_err, cs_F2_ppm, cs_F2_err
        For 3D: peak_name, cs_F1_ppm, cs_F1_err, cs_F2_ppm, cs_F2_err, cs_F3_ppm, cs_F3_err

        Args:
            results: FitResults object
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        prec = self.config.precision
        thresh = self.config.scientific_notation_threshold

        # Detect dimensions from parameter names
        dim_labels = self._detect_dimension_labels(results)

        with path.open("w", newline="") as f:
            writer = csv.writer(f, delimiter=self.config.csv_delimiter)

            # Build header dynamically based on detected dimensions
            header = ["peak_name"]
            for dim in dim_labels:
                header.extend([f"cs_{dim}_ppm", f"cs_{dim}_err"])
            writer.writerow(header)

            # Collect shift data for each peak
            for cluster in results.clusters:
                # Group parameters by peak
                peak_shifts: dict[str, dict[str, float | None]] = {}

                for param in cluster.lineshape_params:
                    peak_name = self._extract_peak_name_from_param(param, cluster.peak_names)
                    if peak_name not in peak_shifts:
                        # Initialize with None for all dimensions
                        peak_shifts[peak_name] = {}
                        for dim in dim_labels:
                            peak_shifts[peak_name][f"cs_{dim}"] = None
                            peak_shifts[peak_name][f"cs_{dim}_err"] = None

                    # Extract dimension from param_id for position parameters
                    if (
                        param.param_id is not None
                        and param.param_id.param_type.value == "position"
                        and param.param_id.axis
                    ):
                        dim_label = param.param_id.axis
                        peak_shifts[peak_name][f"cs_{dim_label}"] = param.value
                        peak_shifts[peak_name][f"cs_{dim_label}_err"] = param.std_error

                # Write rows for each peak
                for peak_name, shifts in peak_shifts.items():
                    row = [peak_name]
                    for dim in dim_labels:
                        cs_val = shifts.get(f"cs_{dim}")
                        cs_err = shifts.get(f"cs_{dim}_err")
                        row.append(format_float(cs_val, prec, thresh) if cs_val is not None else "")
                        row.append(format_float(cs_err, prec, thresh) if cs_err is not None else "")
                    writer.writerow(row)

    def _detect_dimension_labels(self, results: FitResults) -> list[str]:
        """Detect dimension labels from parameter names.

        Returns ordered list like ['F1', 'F2'].
        """
        dim_labels: set[str] = set()

        for cluster in results.clusters:
            for param in cluster.lineshape_params:
                if param.param_id is not None and param.param_id.axis:
                    dim_labels.add(param.param_id.axis)

        # Sort: F1, F2, F3, F4
        return sorted(dim_labels, key=lambda x: int(x[1:]) if x.startswith("F") else 999)

    def write_intensities(self, results: FitResults, path: Path) -> None:
        """Write fitted intensities for CEST/relaxation analysis.

        Creates a simple table with:
        - peak_name, offset, intensity, intensity_err

        This replaces the need for per-peak .out files.

        Args:
            results: FitResults object
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        prec = self.config.precision
        thresh = self.config.scientific_notation_threshold

        with path.open("w", newline="") as f:
            writer = csv.writer(f, delimiter=self.config.csv_delimiter)

            # Header
            writer.writerow(["peak_name", "offset", "intensity", "intensity_err"])

            # Write all amplitude data
            for cluster in results.clusters:
                for amp in cluster.amplitudes:
                    writer.writerow(
                        [
                            amp.peak_name,
                            format_float(amp.z_value, prec, thresh)
                            if amp.z_value is not None
                            else "",
                            format_float(amp.value, prec, thresh),
                            format_float(amp.std_error, prec, thresh),
                        ]
                    )

    def write_statistics(self, results: FitResults, path: Path) -> None:
        """Write fit statistics to CSV.

        Args:
            results: FitResults object
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="") as f:
            if self.config.include_comments:
                f.write("# PeakFit Statistics\n")
                f.write(f"# Generated: {results.metadata.timestamp}\n")
                f.write("#\n")

            writer = csv.writer(f, delimiter=self.config.csv_delimiter)

            writer.writerow(
                [
                    "cluster_id",
                    "peak_names",
                    "chi_squared",
                    "reduced_chi_squared",
                    "n_data",
                    "n_params",
                    "dof",
                    "aic",
                    "bic",
                    "fit_converged",
                ]
            )

            for i, cluster in enumerate(results.clusters):
                if i >= len(results.statistics):
                    continue

                stats = results.statistics[i]
                prec = self.config.precision

                writer.writerow(
                    [
                        cluster.cluster_id,
                        ";".join(cluster.peak_names),
                        format_float(stats.chi_squared, prec),
                        format_float(stats.reduced_chi_squared, prec),
                        stats.n_data,
                        stats.n_params,
                        stats.dof,
                        format_float(stats.aic, prec) if stats.aic is not None else "",
                        format_float(stats.bic, prec) if stats.bic is not None else "",
                        stats.fit_converged,
                    ]
                )

    def write_diagnostics(self, results: FitResults, path: Path) -> None:
        """Write MCMC diagnostics to CSV.

        Args:
            results: FitResults object
            path: Output file path
        """
        if not results.mcmc_diagnostics:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="") as f:
            if self.config.include_comments:
                f.write("# PeakFit MCMC Convergence Diagnostics\n")
                f.write(f"# Generated: {results.metadata.timestamp}\n")
                f.write("#\n")
                f.write("# R-hat: Should be ≤ 1.01 (excellent) or ≤ 1.05 (acceptable)\n")
                f.write("# ESS_bulk: Effective sample size for distribution bulk\n")
                f.write("# ESS_tail: Effective sample size for distribution tails\n")
                f.write("#\n")

            writer = csv.writer(f, delimiter=self.config.csv_delimiter)

            writer.writerow(
                [
                    "cluster_id",
                    "peak_names",
                    "parameter",
                    "rhat",
                    "ess_bulk",
                    "ess_tail",
                    "status",
                ]
            )

            for i, cluster in enumerate(results.clusters):
                if i >= len(results.mcmc_diagnostics):
                    continue

                diag = results.mcmc_diagnostics[i]
                peak_names_str = ";".join(cluster.peak_names)

                for param_diag in diag.parameter_diagnostics:
                    writer.writerow(
                        [
                            cluster.cluster_id,
                            peak_names_str,
                            param_diag.name,
                            format_float(param_diag.rhat, 4) if param_diag.rhat else "",
                            format_float(param_diag.ess_bulk, 0) if param_diag.ess_bulk else "",
                            format_float(param_diag.ess_tail, 0) if param_diag.ess_tail else "",
                            param_diag.status.value,
                        ]
                    )

    def write_correlations(self, results: FitResults, path: Path) -> None:
        """Write parameter correlations to CSV.

        Args:
            results: FitResults object
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="") as f:
            if self.config.include_comments:
                f.write("# PeakFit Parameter Correlations\n")
                f.write(f"# Generated: {results.metadata.timestamp}\n")
                f.write("#\n")

            writer = csv.writer(f, delimiter=self.config.csv_delimiter)

            writer.writerow(
                [
                    "cluster_id",
                    "peak_names",
                    "param_1",
                    "param_2",
                    "correlation",
                ]
            )

            for cluster in results.clusters:
                if cluster.correlation_matrix is None:
                    continue

                peak_names_str = ";".join(cluster.peak_names)
                names = cluster.correlation_param_names
                n = len(names)

                for i in range(n):
                    for j in range(i + 1, n):
                        corr = cluster.correlation_matrix[i, j]
                        writer.writerow(
                            [
                                cluster.cluster_id,
                                peak_names_str,
                                names[i],
                                names[j],
                                format_float(corr, 4),
                            ]
                        )

    def parameters_to_string(self, results: FitResults) -> str:
        """Generate parameters CSV as a string.

        Useful for testing or streaming output.

        Args:
            results: FitResults object

        Returns
        -------
            CSV content as string
        """
        content_lines = []

        # Build content manually
        if self.config.include_comments:
            content_lines.append("# PeakFit Parameter Estimates")
            content_lines.append(f"# Generated: {results.metadata.timestamp}")
            content_lines.append(f"# Method: {results.method.value}")
            content_lines.append("#")

        # Header
        header = [
            "cluster_id",
            "peak_name",
            "parameter",
            "category",
            "value",
            "std_error",
            "ci_68_lower",
            "ci_68_upper",
            "ci_95_lower",
            "ci_95_upper",
            "unit",
            "min_bound",
            "max_bound",
            "is_fixed",
            "is_global",
        ]
        content_lines.append(self.config.csv_delimiter.join(header))

        # Data
        for cluster in results.clusters:
            for param in cluster.lineshape_params:
                prec = self.config.precision
                thresh = self.config.scientific_notation_threshold
                peak_name = cluster.peak_names[0] if cluster.peak_names else ""

                row = [
                    str(cluster.cluster_id),
                    peak_name,
                    param.name,
                    param.category.value,
                    format_float(param.value, prec, thresh),
                    format_float(param.std_error, prec, thresh),
                    format_float(param.ci_68_lower, prec, thresh)
                    if param.ci_68_lower is not None
                    else "",
                    format_float(param.ci_68_upper, prec, thresh)
                    if param.ci_68_upper is not None
                    else "",
                    format_float(param.ci_95_lower, prec, thresh)
                    if param.ci_95_lower is not None
                    else "",
                    format_float(param.ci_95_upper, prec, thresh)
                    if param.ci_95_upper is not None
                    else "",
                    param.unit,
                    format_float(param.min_bound, prec, thresh)
                    if not np.isinf(param.min_bound)
                    else "",
                    format_float(param.max_bound, prec, thresh)
                    if not np.isinf(param.max_bound)
                    else "",
                    str(param.is_fixed),
                    str(param.is_global),
                ]
                content_lines.append(self.config.csv_delimiter.join(row))

        return "\n".join(content_lines)
