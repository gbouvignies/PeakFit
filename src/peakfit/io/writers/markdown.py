"""Concise Markdown projection of immutable completed-fit outcomes."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.utils import format_float

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.fit.final_outcome import FinalClusterOutcome, FinalFitOutcome
    from peakfit.fit.run_models import RunSummary

_MAX_CLUSTER_ROWS = 40
_MAX_PARAMETER_ROWS = 40


def write_final_outcome_report(
    outcome: FinalFitOutcome,
    path: Path,
    summary: RunSummary,
    config: WriterConfig | None = None,
) -> Path:
    """Write a bounded report directly from immutable final outcomes."""
    cfg = config or WriterConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "# PeakFit Report",
        "",
        f"- Generated: {datetime.now(UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Clusters: {summary.n_clusters}",
        f"- Peaks: {summary.n_peaks}",
        f"- Converged: {summary.n_converged}",
        f"- Usable, not converged: {summary.n_usable_non_converged}",
        f"- Unusable: {summary.n_unusable}",
        f"- Usable clusters: {summary.n_usable}",
        f"- Reduced chi2 population: {summary.redchi_population_size}",
    ]
    if summary.n_usable:
        summary_lines.extend(
            [
                f"- Global reduced chi2: {outcome.statistics.reduced_chi_squared:.4g}",
                f"- Data points: {outcome.statistics.n_observations}",
                f"- Fit parameters: {outcome.statistics.n_fitted_parameters}",
            ]
        )
    else:
        summary_lines.append("- Global reduced chi2: N/A (no usable outcomes)")

    cluster_rows = [_cluster_row(cluster) for cluster in outcome.clusters]
    clusters = [
        "## Clusters",
        "",
        (
            "| Cluster | Peaks | Reduced chi2 | Classification | Correction revision | "
            "Optimizer | Terminal message |"
        ),
        "| --- | --- | ---: | --- | ---: | --- | --- |",
        *cluster_rows[:_MAX_CLUSTER_ROWS],
        *_omitted_note(len(cluster_rows), min(len(cluster_rows), _MAX_CLUSTER_ROWS), "clusters"),
    ]

    parameter_rows = [
        _parameter_row(cluster, parameter.name, parameter.value, parameter.standard_error, cfg)
        for cluster in outcome.clusters
        if cluster.usable
        for parameter in cluster.final_nonlinear_parameters
        if parameter.vary
    ]
    parameters = ["## Key Parameters", ""]
    if parameter_rows:
        parameters.extend(
            [
                "| Cluster | Peak | Parameter | Value | Error |",
                "| --- | --- | --- | ---: | ---: |",
                *parameter_rows[:_MAX_PARAMETER_ROWS],
                *_omitted_note(
                    len(parameter_rows), min(len(parameter_rows), _MAX_PARAMETER_ROWS), "parameters"
                ),
            ]
        )
    else:
        parameters.append("No usable final parameters to display.")

    report = "\n\n".join(("\n".join(summary_lines), "\n".join(clusters), "\n".join(parameters)))
    path.write_text(report + "\n")
    return path


def _cluster_row(cluster: FinalClusterOutcome) -> str:
    evaluation = cluster.analytical_evaluation
    redchi = f"{evaluation.statistics.reduced_chi_squared:.4g}" if evaluation else "N/A"
    return (
        "| "
        f"{cluster.cluster_id} | {', '.join(cluster.peak_names)} | {redchi} | "
        f"{_status(cluster)} | {cluster.correction_revision} | "
        f"{cluster.optimizer_provenance.optimizer_kind or 'N/A'} | "
        f"{cluster.optimizer_provenance.termination_message or 'N/A'} |"
    )


def _parameter_row(
    cluster: FinalClusterOutcome,
    name: str,
    value: float,
    standard_error: float,
    config: WriterConfig,
) -> str:
    peak = name.split(".", maxsplit=1)[0]
    if peak not in cluster.peak_names:
        peak = cluster.peak_names[0] if cluster.peak_names else peak
    return (
        f"| {cluster.cluster_id} | {peak} | {name} | {_format_final_value(value, config)} | "
        f"{_format_final_value(standard_error, config)} |"
    )


def _status(cluster: FinalClusterOutcome) -> str:
    if cluster.classification.value == "converged":
        return "converged"
    if cluster.classification.value == "usable_non_converged":
        return "usable, not converged"
    return f"unusable: {cluster.unusable_reason}"


def _format_final_value(value: float, config: WriterConfig) -> str:
    return (
        format_float(value, config.precision, config.scientific_notation_threshold)
        if math.isfinite(value)
        else "N/A"
    )


def _omitted_note(total: int, shown: int, label: str) -> list[str]:
    if total <= shown:
        return []
    return ["", f"_Showing {shown} of {total} {label}. See JSON/CSV outputs for full detail._"]


__all__ = ["write_final_outcome_report"]
