"""Concise Markdown projection of immutable completed-fit outcomes."""

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from peakfit.engine.domain.spectrum import format_planes
from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.utils import format_float

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from peakfit.engine.diagnostics.couplings import CouplingQualification
    from peakfit.engine.scientific.final_outcome import FinalClusterOutcome, FinalFitOutcome
    from peakfit.engine.scientific.summary import RunSummary

_MAX_CLUSTER_ROWS = 40
_MAX_PARAMETER_ROWS = 40


def write_final_outcome_report(
    outcome: FinalFitOutcome,
    path: Path,
    summary: RunSummary,
    config: WriterConfig | None = None,
    couplings: Mapping[str, CouplingQualification] | None = None,
) -> Path:
    """Write a bounded report directly from immutable final outcomes.

    ``couplings`` (``CompletedFit.couplings``) add a section on the fitted J values.
    """
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
        *(
            [
                "- Excluded planes (zero-based, not fitted): "
                + format_planes(outcome.excluded_planes)
            ]
            if outcome.excluded_planes
            else []
        ),
        f"- {outcome.noise_model.description}",
        *(
            [f"- **Warning.** {outcome.noise_model.power_warning}"]
            if outcome.noise_model.power_warning
            else []
        ),
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

    sections = [summary_lines, clusters, parameters]
    if couplings:
        sections.append(_coupling_section(outcome, couplings, cfg))
    report = "\n\n".join("\n".join(section) for section in sections)
    path.write_text(report + "\n")
    return path


def _coupling_section(
    outcome: FinalFitOutcome,
    couplings: Mapping[str, CouplingQualification],
    config: WriterConfig,
) -> list[str]:
    parameters = {p.name: p for p in outcome.final_nonlinear_parameters}
    rows = []
    for name, qualification in couplings.items():
        coupling = parameters[name]
        rho = (qualification.linewidth_correlation, qualification.phase_correlation)
        rows.append(
            f"| {name} | {_format_final_value(coupling.value, config)} | "
            f"{_format_final_value(coupling.standard_error, config)} | "
            f"{'yes' if qualification.below_linewidth else 'no'} | "
            + " | ".join(
                "N/A" if value is None else _format_final_value(value, config) for value in rho
            )
            + " |"
        )
    return [
        "## J Couplings",
        "",
        "Below LW: J is smaller than the fitted linewidth on its axis, so J and that "
        "linewidth are determined together, not separately. ρ is first-order, from the "
        "recorded lineshape covariance; N/A where it is withheld (with σ(J)) or the partner "
        "is fixed or at a bound.",
        "",
        "| Parameter | J (Hz) | Error | Below LW | ρ(J, LW) | ρ(J, phase) |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
        *rows[:_MAX_PARAMETER_ROWS],
        *_omitted_note(len(rows), min(len(rows), _MAX_PARAMETER_ROWS), "couplings"),
    ]


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
    if cluster.analytical_evaluation and cluster.analytical_evaluation.uncertainty_warning:
        return "unreliable uncertainties (inspect fit)"
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
