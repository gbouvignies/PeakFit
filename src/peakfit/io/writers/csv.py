"""CSV format output writer."""

from __future__ import annotations

import csv
import re
from io import StringIO
from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.utils import flatten_diagnostics, format_float, get_peak_name

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.results import (
        FitResults,
        ParameterEstimate,
    )


_LEGACY_USER_NAME_PATTERN = re.compile(r"^(?P<label>[A-Za-z]+)(?P<index>\d+)?_(?P<axis>F\d+)$")
_LEGACY_PARAM_PATTERN = re.compile(r"^(?P<peak>.+)_(?P<label>[A-Za-z]+\d*)$")
_CANONICAL_NAME_PARTS = 3


class CSVWriter:
    """Writer for CSV outputs."""

    def __init__(self, config: WriterConfig | None = None) -> None:
        self.config = config or WriterConfig()

    def has_shift_parameters(self, results: FitResults) -> bool:
        """Return whether shift parameters are present in results."""
        return bool(self._detect_dimension_labels(results))

    def write_results(self, results: FitResults, path: Path) -> None:
        """Write a compact per-cluster summary table."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(
                f,
                delimiter=self.config.csv_delimiter,
                quoting=csv.QUOTE_MINIMAL if self.config.csv_quoting else csv.QUOTE_NONE,
            )

            self._write_header_comments(f, "PeakFit Results Summary", results)
            writer.writerow(
                [
                    "cluster_id",
                    "peak_name",
                    "n_parameters",
                    "n_series",
                    "reduced_chi_squared",
                    "fit_converged",
                ]
            )

            for i, cluster in enumerate(results.clusters):
                stats = results.statistics[i] if i < len(results.statistics) else None
                for peak_name in cluster.peak_names:
                    writer.writerow(
                        [
                            cluster.cluster_id,
                            peak_name,
                            cluster.n_lineshape_params,
                            cluster.n_series,
                            self._fmt_required(stats.reduced_chi_squared if stats else None),
                            stats.fit_converged if stats else "",
                        ]
                    )

    def write_parameters(self, results: FitResults, path: Path) -> None:
        """Write model parameters with canonical identifiers.

        The clean-break schema uses one parameter per row with:
        - canonical `parameter_name` like `2N-H.F2.lw`
        - optional uncertainty/bounds columns included only when present.
        """
        rows = self._build_parameter_rows(results)

        required_columns = [
            "cluster_id",
            "peak_name",
            "parameter_name",
            "category",
            "value",
            "std_error",
            "is_fixed",
            "is_global",
        ]
        optional_columns = [
            "ci_68_lower",
            "ci_68_upper",
            "ci_95_lower",
            "ci_95_upper",
            "unit",
            "min_bound",
            "max_bound",
        ]

        present_optional = [
            col
            for col in optional_columns
            if any(self._is_present(row.get(col, "")) for row in rows)
        ]
        header = required_columns + present_optional

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            self._write_header_comments(
                f,
                "PeakFit Parameter Estimates",
                results,
                extra_lines=["Format: Long format (one row per model parameter)"],
            )
            writer = csv.writer(f, delimiter=self.config.csv_delimiter)
            writer.writerow(header)
            for row in rows:
                writer.writerow([row.get(col, "") for col in header])

    def _build_parameter_rows(self, results: FitResults) -> list[dict[str, Any]]:
        """Build normalized rows for parameter export."""
        rows: list[dict[str, Any]] = []

        for cluster in results.clusters:
            for param in cluster.lineshape_params:
                peak_name = get_peak_name(param, cluster.peak_names)
                canonical_name = self._canonical_parameter_name(param, peak_name)

                row: dict[str, Any] = {
                    "cluster_id": cluster.cluster_id,
                    "peak_name": peak_name,
                    "parameter_name": canonical_name,
                    "category": param.category.value,
                    "value": self._fmt_required(param.value),
                    "std_error": self._fmt_required(param.std_error),
                    "ci_68_lower": self._fmt_optional(param.ci_68_lower),
                    "ci_68_upper": self._fmt_optional(param.ci_68_upper),
                    "ci_95_lower": self._fmt_optional(param.ci_95_lower),
                    "ci_95_upper": self._fmt_optional(param.ci_95_upper),
                    "unit": param.unit,
                    "min_bound": self._fmt_optional(param.min_bound, allow_infinite=False),
                    "max_bound": self._fmt_optional(param.max_bound, allow_infinite=False),
                    "is_fixed": param.is_fixed,
                    "is_global": param.is_global,
                }
                rows.append(row)

        return rows

    def _canonical_parameter_name(self, param: ParameterEstimate, peak_name: str) -> str:
        """Return canonical dot-notation parameter identifier."""
        if param.param_id is not None:
            if param.param_id.axis:
                return param.param_id.name
            entity = (
                f"cluster_{param.param_id.cluster_id}"
                if param.param_id.cluster_id is not None
                else param.param_id.peak_name
            )
            suffix = (
                f"{param.param_id.label}{param.param_id.index}"
                if param.param_id.index is not None
                else param.param_id.label
            )
            return f"{entity}.F0.{suffix}"

        name = param.name
        parts = name.split(".")
        if len(parts) == _CANONICAL_NAME_PARTS:
            return name

        # Legacy fallback from user_name style (e.g., cs_F2, I0_F1).
        user_name = param.user_name
        match = _LEGACY_USER_NAME_PATTERN.match(user_name)
        if match:
            label = match.group("label")
            index = match.group("index") or ""
            axis = match.group("axis")
            return f"{peak_name}.{axis}.{label}{index}"

        # Last-resort fallback keeps identifier deterministic.
        legacy_match = _LEGACY_PARAM_PATTERN.match(name)
        if legacy_match:
            return f"{peak_name}.F0.{legacy_match.group('label')}"

        return f"{peak_name}.F0.{name.replace('.', '_')}"

    def write_amplitudes(self, results: FitResults, path: Path) -> None:
        """Backward-compatible alias for `write_intensities`."""
        self.write_intensities(results, path)

    def write_intensities(self, results: FitResults, path: Path) -> None:
        """Write per-plane fitted amplitudes for each peak."""
        rows = self._build_intensity_rows(results)

        required_columns = [
            "cluster_id",
            "peak_name",
            "plane_index",
            "intensity",
            "intensity_err",
        ]
        optional_columns = [
            "z_value",
            "ci_68_lower",
            "ci_68_upper",
        ]
        present_optional = [
            col
            for col in optional_columns
            if any(self._is_present(row.get(col, "")) for row in rows)
        ]
        header = required_columns + present_optional

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=self.config.csv_delimiter)
            writer.writerow(header)
            for row in rows:
                writer.writerow([row.get(col, "") for col in header])

    def _build_intensity_rows(self, results: FitResults) -> list[dict[str, Any]]:
        """Build normalized rows for intensity export."""
        rows: list[dict[str, Any]] = []

        for cluster in results.clusters:
            for amp in cluster.amplitudes:
                rows.append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "peak_name": amp.peak_name,
                        "plane_index": amp.plane_index,
                        "z_value": self._fmt_optional(amp.z_value),
                        "intensity": self._fmt_required(amp.value),
                        "intensity_err": self._fmt_required(amp.std_error),
                        "ci_68_lower": self._fmt_optional(amp.ci_68_lower),
                        "ci_68_upper": self._fmt_optional(amp.ci_68_upper),
                    }
                )

        return rows

    def write_shifts(self, results: FitResults, path: Path) -> None:
        """Write chemical shifts in wide format for quick navigation."""
        dim_labels = self._detect_dimension_labels(results)
        rows = self._collect_shift_data(results, dim_labels)

        if not dim_labels or not rows:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=self.config.csv_delimiter)
            header = ["peak_name"]
            for dim in dim_labels:
                header.extend([f"cs_{dim}_ppm", f"cs_{dim}_err"])
            writer.writerow(header)
            writer.writerows(rows)

    def _detect_dimension_labels(self, results: FitResults) -> list[str]:
        """Detect available dimension labels from canonical shift parameters."""
        dim_labels: set[str] = set()

        for cluster in results.clusters:
            for param in cluster.lineshape_params:
                if param.param_id is not None:
                    if param.param_id.label == "cs" and param.param_id.axis:
                        dim_labels.add(param.param_id.axis)
                    continue

                canonical_name = self._canonical_parameter_name(
                    param, get_peak_name(param, cluster.peak_names)
                )
                parts = canonical_name.split(".")
                if len(parts) == _CANONICAL_NAME_PARTS and parts[2] == "cs" and parts[1]:
                    dim_labels.add(parts[1])

        return sorted(dim_labels, key=lambda x: int(x[1:]) if x.startswith("F") else 999)

    def _collect_shift_data(self, results: FitResults, dim_labels: list[str]) -> list[list[str]]:
        """Collect and pivot chemical shift data by peak."""
        if not dim_labels:
            return []

        rows: list[list[str]] = []
        prec = self.config.precision
        thresh = self.config.scientific_notation_threshold

        for cluster in results.clusters:
            peak_shifts: dict[str, dict[str, float | None]] = {}

            for param in cluster.lineshape_params:
                peak_name = get_peak_name(param, cluster.peak_names)
                canonical_name = self._canonical_parameter_name(param, peak_name)
                parts = canonical_name.split(".")
                if len(parts) != _CANONICAL_NAME_PARTS or parts[2] != "cs":
                    continue

                dim_label = parts[1]
                if dim_label not in dim_labels:
                    continue

                peak_shifts.setdefault(peak_name, {})
                peak_shifts[peak_name][f"cs_{dim_label}"] = param.value
                peak_shifts[peak_name][f"cs_{dim_label}_err"] = param.std_error

            for peak_name, shifts in peak_shifts.items():
                row = [peak_name]
                for dim in dim_labels:
                    cs_val = shifts.get(f"cs_{dim}")
                    cs_err = shifts.get(f"cs_{dim}_err")
                    row.append(format_float(cs_val, prec, thresh) if cs_val is not None else "")
                    row.append(format_float(cs_err, prec, thresh) if cs_err is not None else "")
                rows.append(row)

        return rows

    def write_statistics(self, results: FitResults, path: Path) -> None:
        """Write per-cluster fit statistics to CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
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
                writer.writerow(
                    [
                        cluster.cluster_id,
                        ";".join(cluster.peak_names),
                        self._fmt_required(stats.chi_squared),
                        self._fmt_required(stats.reduced_chi_squared),
                        stats.n_data,
                        stats.n_params,
                        stats.dof,
                        self._fmt_optional(stats.aic),
                        self._fmt_optional(stats.bic),
                        stats.fit_converged,
                    ]
                )

    def write_residuals(self, results: FitResults, path: Path) -> None:
        """Write residual arrays when available.

        This method skips file creation when no residual arrays are present.
        """
        rows: list[list[Any]] = []

        for i, cluster in enumerate(results.clusters):
            if i >= len(results.statistics):
                continue

            residuals = results.statistics[i].residuals
            raw = residuals.raw_residuals
            norm = residuals.normalized_residuals
            if raw is None and norm is None:
                continue

            n_points = 0
            if raw is not None:
                n_points = len(raw)
            elif norm is not None:
                n_points = len(norm)

            for idx in range(n_points):
                raw_val = float(raw[idx]) if raw is not None else None
                norm_val = float(norm[idx]) if norm is not None else None
                rows.append(
                    [
                        cluster.cluster_id,
                        idx,
                        self._fmt_optional(raw_val),
                        self._fmt_optional(norm_val),
                    ]
                )

        if not rows:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=self.config.csv_delimiter)
            writer.writerow(["cluster_id", "index", "raw_residual", "normalized_residual"])
            writer.writerows(rows)

    def write_correlations(self, results: FitResults, path: Path) -> None:
        """Write pairwise parameter correlations when available."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            if self.config.include_comments:
                f.write("# PeakFit Parameter Correlations\n")
                f.write(f"# Generated: {results.metadata.timestamp}\n")
                f.write("#\n")

            writer = csv.writer(f, delimiter=self.config.csv_delimiter)
            writer.writerow(["cluster_id", "peak_names", "param_1", "param_2", "correlation"])

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
                                format_float(float(corr), 4),
                            ]
                        )

    def parameters_to_string(self, results: FitResults) -> str:
        """Generate parameters CSV as a string."""
        buffer = StringIO()
        writer = csv.writer(buffer, delimiter=self.config.csv_delimiter)

        rows = self._build_parameter_rows(results)
        required_columns = [
            "cluster_id",
            "peak_name",
            "parameter_name",
            "category",
            "value",
            "std_error",
            "is_fixed",
            "is_global",
        ]
        optional_columns = [
            "ci_68_lower",
            "ci_68_upper",
            "ci_95_lower",
            "ci_95_upper",
            "unit",
            "min_bound",
            "max_bound",
        ]
        present_optional = [
            col
            for col in optional_columns
            if any(self._is_present(row.get(col, "")) for row in rows)
        ]
        header = required_columns + present_optional

        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(col, "") for col in header])

        return buffer.getvalue().rstrip("\n")

    def write_diagnostics(self, results: FitResults, path: Path) -> None:
        """Write flattened MCMC diagnostics to CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            if self.config.include_comments:
                f.write("# PeakFit MCMC Diagnostics\n")
                f.write(f"# Generated: {results.metadata.timestamp}\n")
                f.write(f"# Method: {results.method.value}\n")
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
                    "convergence",
                ]
            )

            for (
                cluster_id,
                peak_names,
                param_name,
                rhat,
                ess_bulk,
                ess_tail,
                status,
            ) in flatten_diagnostics(results):
                writer.writerow(
                    [
                        cluster_id,
                        ";".join(peak_names),
                        param_name,
                        self._fmt_optional(rhat),
                        f"{ess_bulk:.0f}" if ess_bulk is not None else "",
                        f"{ess_tail:.0f}" if ess_tail is not None else "",
                        status,
                    ]
                )

    def _write_header_comments(
        self, f: Any, title: str, results: FitResults, extra_lines: list[str] | None = None
    ) -> None:
        """Write standard header comments."""
        if not self.config.include_comments:
            return

        f.write(f"# {title}\n")
        f.write(f"# Generated: {results.metadata.timestamp}\n")

        if hasattr(results, "method") and results.method:
            f.write(f"# Method: {results.method.value}\n")

        if extra_lines:
            for line in extra_lines:
                if line.startswith("#"):
                    f.write(f"{line}\n")
                else:
                    f.write(f"# {line}\n")

        if not extra_lines or extra_lines[-1].strip() != "#":
            f.write("#\n")

    def _fmt_required(self, value: float | None) -> str:
        """Format required numeric value; never emits empty strings."""
        safe_value = value
        if safe_value is None:
            safe_value = 0.0
        try:
            numeric_value = float(safe_value)
        except (TypeError, ValueError):
            numeric_value = 0.0
        if not np.isfinite(numeric_value):
            numeric_value = 0.0
        return format_float(
            numeric_value,
            self.config.precision,
            self.config.scientific_notation_threshold,
        )

    def _fmt_optional(self, value: float | None, allow_infinite: bool = True) -> str:
        """Format optional numeric value; emits empty strings when absent."""
        if value is None:
            return ""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return ""
        if np.isnan(numeric_value):
            return ""
        if np.isinf(numeric_value):
            return "" if not allow_infinite else str(numeric_value)

        return format_float(
            numeric_value,
            self.config.precision,
            self.config.scientific_notation_threshold,
        )

    @staticmethod
    def _is_present(value: Any) -> bool:
        """Return true if a cell should count as present for column retention."""
        return value not in ("", None)
