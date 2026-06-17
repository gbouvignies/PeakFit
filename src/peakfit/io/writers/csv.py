"""CSV output writers."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.utils import canonical_parameter_name, format_float, get_peak_name

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.fit.results import FitResults

_CANONICAL_NAME_PARTS = 3


def has_shift_parameters(results: FitResults) -> bool:
    """Return whether shift parameters are present in results."""
    return bool(_detect_dimension_labels(results))


def write_parameters(
    results: FitResults,
    path: Path,
    config: WriterConfig | None = None,
) -> None:
    """Write model parameters with canonical identifiers."""
    cfg = config or WriterConfig()
    rows = _build_parameter_rows(results, cfg)

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
        col for col in optional_columns if any(_is_present(row.get(col, "")) for row in rows)
    ]
    header = required_columns + present_optional
    _write_rows(path, header, rows)


def write_intensities(
    results: FitResults,
    path: Path,
    config: WriterConfig | None = None,
) -> None:
    """Write per-plane fitted amplitudes for each peak."""
    cfg = config or WriterConfig()
    rows = _build_intensity_rows(results, cfg)

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
        col for col in optional_columns if any(_is_present(row.get(col, "")) for row in rows)
    ]
    header = required_columns + present_optional
    _write_rows(path, header, rows)


def write_shifts(
    results: FitResults,
    path: Path,
    config: WriterConfig | None = None,
) -> None:
    """Write chemical shifts in wide format for quick navigation."""
    cfg = config or WriterConfig()
    dim_labels = _detect_dimension_labels(results)
    rows = _collect_shift_data(results, dim_labels, cfg)

    if not dim_labels or not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["peak_name"]
        for dim in dim_labels:
            header.extend([f"cs_{dim}_ppm", f"cs_{dim}_err"])
        writer.writerow(header)
        writer.writerows(rows)


def _build_parameter_rows(results: FitResults, config: WriterConfig) -> list[dict[str, Any]]:
    """Build normalized rows for parameter export."""
    rows: list[dict[str, Any]] = []

    for cluster in results.clusters:
        for param in cluster.lineshape_params:
            peak_name = get_peak_name(param, cluster.peak_names)
            parameter_name = canonical_parameter_name(param, peak_name)

            row: dict[str, Any] = {
                "cluster_id": cluster.cluster_id,
                "peak_name": peak_name,
                "parameter_name": parameter_name,
                "category": param.category.value,
                "value": _fmt_required(param.value, config),
                "std_error": _fmt_required(param.std_error, config),
                "ci_68_lower": _fmt_optional(param.ci_68_lower, config),
                "ci_68_upper": _fmt_optional(param.ci_68_upper, config),
                "ci_95_lower": _fmt_optional(param.ci_95_lower, config),
                "ci_95_upper": _fmt_optional(param.ci_95_upper, config),
                "unit": param.unit,
                "min_bound": _fmt_optional(param.min_bound, config, allow_infinite=False),
                "max_bound": _fmt_optional(param.max_bound, config, allow_infinite=False),
                "is_fixed": param.is_fixed,
                "is_global": param.is_global,
            }
            rows.append(row)

    return rows


def _build_intensity_rows(results: FitResults, config: WriterConfig) -> list[dict[str, Any]]:
    """Build normalized rows for intensity export."""
    rows: list[dict[str, Any]] = []

    for cluster in results.clusters:
        for amp in cluster.amplitudes:
            rows.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "peak_name": amp.peak_name,
                    "plane_index": amp.plane_index,
                    "z_value": _fmt_optional(amp.z_value, config),
                    "intensity": _fmt_required(amp.value, config),
                    "intensity_err": _fmt_required(amp.std_error, config),
                    "ci_68_lower": _fmt_optional(amp.ci_68_lower, config),
                    "ci_68_upper": _fmt_optional(amp.ci_68_upper, config),
                }
            )

    return rows


def _detect_dimension_labels(results: FitResults) -> list[str]:
    """Detect available dimension labels from canonical shift parameters."""
    dim_labels: set[str] = set()

    for cluster in results.clusters:
        for param in cluster.lineshape_params:
            if param.param_id is not None:
                if param.param_id.label == "cs" and param.param_id.axis:
                    dim_labels.add(param.param_id.axis)
                continue

            canonical_name = canonical_parameter_name(
                param, get_peak_name(param, cluster.peak_names)
            )
            parts = canonical_name.split(".")
            if len(parts) == _CANONICAL_NAME_PARTS and parts[2] == "cs" and parts[1]:
                dim_labels.add(parts[1])

    return sorted(dim_labels, key=lambda x: int(x[1:]) if x.startswith("F") else 999)


def _collect_shift_data(
    results: FitResults,
    dim_labels: list[str],
    config: WriterConfig,
) -> list[list[str]]:
    """Collect and pivot chemical shift data by peak."""
    if not dim_labels:
        return []

    rows: list[list[str]] = []
    prec = config.precision
    thresh = config.scientific_notation_threshold

    for cluster in results.clusters:
        peak_shifts: dict[str, dict[str, float | None]] = {}

        for param in cluster.lineshape_params:
            peak_name = get_peak_name(param, cluster.peak_names)
            canonical_name = canonical_parameter_name(param, peak_name)
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


def _fmt_required(value: float | None, config: WriterConfig) -> str:
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
        config.precision,
        config.scientific_notation_threshold,
    )


def _fmt_optional(
    value: float | None,
    config: WriterConfig,
    allow_infinite: bool = True,
) -> str:
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
        config.precision,
        config.scientific_notation_threshold,
    )


def _is_present(value: Any) -> bool:
    """Return true if a cell should count as present for column retention."""
    return value not in ("", None)


def _write_rows(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as CSV rows using a fixed header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(col, "") for col in header])


__all__ = [
    "has_shift_parameters",
    "write_intensities",
    "write_parameters",
    "write_shifts",
]
