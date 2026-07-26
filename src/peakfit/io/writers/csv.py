"""CSV output writers."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.utils import canonical_parameter_name, format_float, get_peak_name

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.fit.final_outcome import FinalFitOutcome
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

    base_columns = [
        "peak_name",
        "parameter_name",
        "value",
        "std_error",
        "is_fixed",
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
    header = base_columns + present_optional + ["cluster_id"]
    _write_rows(path, header, rows)


def write_intensities(
    results: FitResults,
    path: Path,
    config: WriterConfig | None = None,
) -> None:
    """Write per-plane fitted amplitudes for each peak."""
    cfg = config or WriterConfig()
    rows = _build_intensity_rows(results, cfg)

    base_columns = [
        "peak_name",
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
    header = ["peak_name"]
    if "z_value" in present_optional:
        header.append("z_value")
        present_optional.remove("z_value")
    header.extend(col for col in base_columns if col != "peak_name")
    header.extend(present_optional)
    header.append("plane_index")
    header.append("cluster_id")
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
        header.append("cluster_id")
        writer.writerow(header)
        writer.writerows(rows)


def has_final_shift_parameters(outcome: FinalFitOutcome) -> bool:
    """Return whether the authoritative outcome contains chemical shifts."""
    return any(
        parameter.name.split(".")[-1] == "cs"
        for cluster in outcome.clusters
        for parameter in cluster.final_nonlinear_parameters
    )


def write_final_outcome_parameters(
    outcome: FinalFitOutcome,
    path: Path,
    config: WriterConfig | None = None,
) -> None:
    """Write final nonlinear parameters without consulting continuation state."""
    cfg = config or WriterConfig()
    rows = [
        {
            **_outcome_row(cluster, cfg),
            "peak_name": _parameter_peak_name(parameter.name, cluster.peak_names),
            "parameter_name": parameter.name,
            "value": _fmt_final_value(parameter.value, cfg, unavailable="unavailable"),
            "std_error": _fmt_final_value(parameter.standard_error, cfg, unavailable="unavailable"),
            "is_fixed": not parameter.vary,
            "unit": parameter.unit,
            "min_bound": _fmt_optional(parameter.min, cfg, allow_infinite=False),
            "max_bound": _fmt_optional(parameter.max, cfg, allow_infinite=False),
        }
        for cluster in outcome.clusters
        if cluster.usable
        for parameter in cluster.final_nonlinear_parameters
    ]
    _write_rows(path, _parameter_header(rows), rows)


def write_final_outcome_intensities(
    outcome: FinalFitOutcome,
    z_values: np.ndarray | None,
    path: Path,
    config: WriterConfig | None = None,
) -> None:
    """Write frozen analytical amplitudes for usable final outcomes only."""
    cfg = config or WriterConfig()
    rows: list[dict[str, Any]] = []
    for cluster in outcome.clusters:
        if not cluster.usable:
            continue
        evaluation = cluster.analytical_evaluation
        if evaluation is None:
            raise ValueError("Usable final outcome is missing its analytical evaluation.")
        for peak_index, peak_name in enumerate(cluster.peak_names):
            for plane_index in range(evaluation.amplitudes.shape[1]):
                z_value = (
                    float(z_values[plane_index])
                    if z_values is not None and plane_index < len(z_values)
                    else float(plane_index)
                )
                rows.append(
                    {
                        **_outcome_row(cluster, cfg),
                        "peak_name": peak_name,
                        "z_value": _fmt_optional(z_value, cfg),
                        "intensity": _fmt_final_value(
                            float(evaluation.amplitudes[peak_index, plane_index]),
                            cfg,
                            unavailable="unavailable",
                        ),
                        "intensity_err": _fmt_final_value(
                            float(evaluation.scaled_amplitude_standard_errors[peak_index]),
                            cfg,
                            unavailable="unavailable",
                        ),
                        "plane_index": plane_index,
                    }
                )
    _write_rows(path, _intensity_header(), rows)


def write_final_outcome_clusters(
    outcome: FinalFitOutcome,
    path: Path,
    config: WriterConfig | None = None,
) -> None:
    """Write one authoritative status row per cluster, including unusable ones."""
    cfg = config or WriterConfig()
    rows = [_outcome_row(cluster, cfg) for cluster in outcome.clusters]
    _write_rows(
        path,
        [
            "cluster_id",
            "classification",
            "converged",
            "usable",
            "unusable_reason",
            "correction_revision",
            "optimizer_kind",
            "optimizer_success",
            "termination_message",
            "function_evaluations",
            "jacobian_evaluations",
            "iterations",
            "optimality",
            "final_cost",
            "chi_squared",
            "reduced_chi_squared",
            "n_observations",
            "n_fitted_parameters",
            "degrees_of_freedom",
        ],
        rows,
    )


def write_final_outcome_shifts(
    outcome: FinalFitOutcome,
    path: Path,
    config: WriterConfig | None = None,
) -> None:
    """Write usable final chemical shifts in the existing wide-table shape."""
    cfg = config or WriterConfig()
    dimensions = sorted(
        {
            parts[1]
            for cluster in outcome.clusters
            for parameter in cluster.final_nonlinear_parameters
            if len(parts := parameter.name.split(".")) == _CANONICAL_NAME_PARTS
            and parts[-1] == "cs"
        },
        key=lambda label: int(label[1:]) if label.startswith("F") else 999,
    )
    rows: list[dict[str, Any]] = []
    for cluster in outcome.clusters:
        if not cluster.usable:
            continue
        by_peak: dict[str, dict[str, str]] = {}
        for parameter in cluster.final_nonlinear_parameters:
            parts = parameter.name.split(".")
            if len(parts) != _CANONICAL_NAME_PARTS or parts[-1] != "cs":
                continue
            peak = _parameter_peak_name(parameter.name, cluster.peak_names)
            by_peak.setdefault(peak, {})[parts[1]] = _fmt_final_value(
                parameter.value, cfg, unavailable="unavailable"
            )
            by_peak[peak][f"{parts[1]}_err"] = _fmt_final_value(
                parameter.standard_error, cfg, unavailable="unavailable"
            )
        for peak, shifts in sorted(by_peak.items()):
            rows.append(
                {
                    **_outcome_row(cluster, cfg),
                    "peak_name": peak,
                    **{
                        column: shifts.get(column.removeprefix("cs_").removesuffix("_ppm"), "")
                        if column.endswith("_ppm")
                        else shifts.get(column.removeprefix("cs_").removesuffix("_err"), "")
                        for dimension in dimensions
                        for column in (f"cs_{dimension}_ppm", f"cs_{dimension}_err")
                    },
                }
            )
    header = ["peak_name"]
    for dimension in dimensions:
        header.extend((f"cs_{dimension}_ppm", f"cs_{dimension}_err"))
    header.extend(_outcome_columns())
    _write_rows(path, header, rows)


def _parameter_header(rows: list[dict[str, Any]]) -> list[str]:
    header = ["peak_name", "parameter_name", "value", "std_error", "is_fixed"]
    header.extend(
        column
        for column in ("unit", "min_bound", "max_bound")
        if any(_is_present(row.get(column, "")) for row in rows)
    )
    return header + _outcome_columns()


def _intensity_header() -> list[str]:
    header = ["peak_name", "z_value", "intensity", "intensity_err", "plane_index"]
    return header + _outcome_columns()


def _outcome_columns() -> list[str]:
    return [
        "cluster_id",
        "classification",
        "converged",
        "usable",
        "unusable_reason",
        "correction_revision",
        "optimizer_kind",
        "optimizer_success",
        "termination_message",
        "function_evaluations",
        "jacobian_evaluations",
        "iterations",
        "optimality",
        "final_cost",
    ]


def _outcome_row(cluster: Any, config: WriterConfig) -> dict[str, Any]:
    """Copy trustworthy final-outcome facts into one tabular row."""
    provenance = cluster.optimizer_provenance
    evaluation = cluster.analytical_evaluation
    statistics = evaluation.statistics if evaluation is not None else None
    return {
        "cluster_id": cluster.cluster_id,
        "classification": cluster.classification.value,
        "converged": cluster.classification.value == "converged",
        "usable": cluster.usable,
        "unusable_reason": cluster.unusable_reason or "",
        "correction_revision": cluster.correction_revision,
        "optimizer_kind": provenance.optimizer_kind or "",
        "optimizer_success": provenance.converged,
        "termination_message": provenance.termination_message or "",
        "function_evaluations": _optional_value(provenance.function_evaluations),
        "jacobian_evaluations": _optional_value(provenance.jacobian_evaluations),
        "iterations": provenance.iterations if provenance.iterations is not None else "",
        "optimality": _fmt_optional(provenance.optimality, config),
        "final_cost": _fmt_final_value(provenance.final_cost, config),
        "chi_squared": _fmt_optional(statistics.chi_squared if statistics else None, config),
        "reduced_chi_squared": _fmt_optional(
            statistics.reduced_chi_squared if statistics else None, config
        ),
        "n_observations": statistics.n_observations if statistics else "",
        "n_fitted_parameters": statistics.n_fitted_parameters if statistics else "",
        "degrees_of_freedom": statistics.degrees_of_freedom if statistics else "",
    }


def _parameter_peak_name(name: str, peak_names: tuple[str, ...]) -> str:
    candidate = name.split(".", maxsplit=1)[0]
    return candidate if candidate in peak_names else (peak_names[0] if peak_names else candidate)


def _optional_value(value: Any) -> Any:
    return value if value is not None else ""


def _fmt_final_value(
    value: float | None,
    config: WriterConfig,
    *,
    unavailable: str = "",
) -> str:
    """Format a final-outcome scalar without inventing unavailable values."""
    if value is None:
        return unavailable
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return unavailable
    if not np.isfinite(numeric_value):
        return unavailable
    return format_float(numeric_value, config.precision, config.scientific_notation_threshold)


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
            }
            rows.append(row)

    return rows


def _build_intensity_rows(results: FitResults, config: WriterConfig) -> list[dict[str, Any]]:
    """Build normalized rows for intensity export."""
    return [
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
        for cluster in results.clusters
        for amp in cluster.amplitudes
    ]


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
            row.append(str(cluster.cluster_id))
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
    "has_final_shift_parameters",
    "has_shift_parameters",
    "write_final_outcome_clusters",
    "write_final_outcome_intensities",
    "write_final_outcome_parameters",
    "write_final_outcome_shifts",
    "write_intensities",
    "write_parameters",
    "write_shifts",
]
