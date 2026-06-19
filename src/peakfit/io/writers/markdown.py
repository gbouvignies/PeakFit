"""Concise Markdown report writer."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.utils import format_float

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.fit.results import (
        ClusterEstimates,
        FitResults,
        FitStatistics,
        MCMCDiagnostics,
        ParameterEstimate,
    )


_GOOD_REDCHI_MIN = 0.5
_GOOD_REDCHI_MAX = 2.0
_MAX_CLUSTER_ROWS = 40
_MAX_PARAMETER_ROWS = 40
_MAX_WARNING_ROWS = 20
_MAX_MCMC_ROWS = 40
_MAX_PEAK_NAMES = 3
_POOR_RELATIVE_ERROR_THRESHOLD = 0.5
_MCMC_MARGINAL = "marginal"
_MCMC_POOR = "poor"


def write_report(
    results: FitResults,
    path: Path,
    config: WriterConfig | None = None,
) -> Path:
    """Write a compact human-readable fit report.

    CSV and JSON remain the source for complete per-parameter data. The Markdown
    report is intentionally bounded so large runs stay readable in terminals,
    notebooks, and code review diffs.
    """
    cfg = config or WriterConfig()
    path.parent.mkdir(parents=True, exist_ok=True)

    warnings = _collect_warnings(results)
    sections = [
        _summary_section(results),
        _warning_section(warnings),
        _cluster_section(results),
        _parameter_section(results, cfg),
    ]
    if results.mcmc_diagnostics:
        sections.append(_mcmc_section(results))

    path.write_text("\n\n".join(section for section in sections if section) + "\n")
    return path


def _summary_section(results: FitResults) -> str:
    lines = [
        "# PeakFit Report",
        "",
        f"- Generated: {datetime.now(UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Method: {results.method}",
        f"- Clusters: {results.n_clusters}",
        f"- Peaks: {results.n_peaks}",
    ]

    if results.global_statistics is not None:
        stats = results.global_statistics
        lines.extend(
            [
                f"- Reduced chi2: {stats.reduced_chi_squared:.4g} "
                f"({_fit_status(stats.reduced_chi_squared, stats.fit_converged)})",
                f"- Data points: {stats.n_data}",
                f"- Fit parameters: {stats.n_params}",
            ]
        )

    if results.mcmc_diagnostics:
        status = "ok" if results.has_converged else "check"
        n_problematic = sum(
            len(diag.get_problematic_parameters()) for diag in results.mcmc_diagnostics
        )
        lines.append(f"- MCMC convergence: {status}")
        if n_problematic:
            lines.append(f"- MCMC parameters to check: {n_problematic}")

    return "\n".join(lines)


def _warning_section(warnings: list[str]) -> str:
    if not warnings:
        return "## Checks\n\nNo warnings."

    shown = warnings[:_MAX_WARNING_ROWS]
    lines = [f"## Checks ({len(warnings)})", ""]
    lines.extend(f"- {warning}" for warning in shown)
    lines.extend(_omitted_note(len(warnings), len(shown), "warnings"))
    return "\n".join(lines)


def _cluster_section(results: FitResults) -> str:
    rows = list(_cluster_rows(results))
    shown = rows[:_MAX_CLUSTER_ROWS]

    lines = [
        "## Clusters",
        "",
        "| Cluster | Peaks | Reduced chi2 | Status |",
        "| --- | --- | ---: | --- |",
    ]
    lines.extend(shown)
    lines.extend(_omitted_note(len(rows), len(shown), "clusters"))
    return "\n".join(lines)


def _cluster_rows(results: FitResults) -> list[str]:
    rows: list[tuple[int, int, str]] = []
    for index, cluster in enumerate(results.clusters):
        stats = results.statistics[index] if index < len(results.statistics) else None
        sort_key = _cluster_sort_key(stats)
        redchi = f"{stats.reduced_chi_squared:.4g}" if stats is not None else ""
        status = _cluster_status(stats)
        rows.append(
            (
                sort_key,
                cluster.cluster_id,
                f"| {cluster.cluster_id} | {_peak_label(cluster)} | {redchi} | {status} |",
            )
        )

    rows.sort(key=lambda row: (row[0], row[1]))
    return [row for _, _, row in rows]


def _parameter_section(results: FitResults, config: WriterConfig) -> str:
    rows = list(_parameter_rows(results, config))
    if not rows:
        return "## Parameters To Check\n\nNo parameter warnings."

    shown = rows[:_MAX_PARAMETER_ROWS]
    lines = [
        "## Parameters To Check",
        "",
        "| Cluster | Peak | Parameter | Value | Error | Status |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    lines.extend(shown)
    lines.extend(_omitted_note(len(rows), len(shown), "parameters"))
    return "\n".join(lines)


def _parameter_rows(results: FitResults, config: WriterConfig) -> list[str]:
    rows: list[tuple[int, int, str, str, str]] = []
    for cluster in results.clusters:
        for param in cluster.lineshape_params:
            if not _show_parameter(param):
                continue
            rows.append(
                (
                    _parameter_sort_key(param),
                    cluster.cluster_id,
                    _peak_name(cluster, param),
                    param.name,
                    _parameter_row(cluster, param, config),
                )
            )

    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    return [row for *_, row in rows]


def _parameter_row(
    cluster: ClusterEstimates,
    param: ParameterEstimate,
    config: WriterConfig,
) -> str:
    precision = config.precision
    threshold = config.scientific_notation_threshold
    value = format_float(param.value, precision, threshold)
    error = _format_error(param, precision, threshold)
    return (
        f"| {cluster.cluster_id} | {_peak_name(cluster, param)} | {param.name} | "
        f"{value} | {error} | {_parameter_status(param)} |"
    )


def _mcmc_section(results: FitResults) -> str:
    rows = list(_mcmc_rows(results))
    if not rows:
        return ""

    shown = rows[:_MAX_MCMC_ROWS]
    lines = [
        "## MCMC Diagnostics",
        "",
        "| Cluster | Parameter | R-hat | ESS bulk | ESS tail | Status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    lines.extend(shown)
    lines.extend(_omitted_note(len(rows), len(shown), "MCMC diagnostics"))
    return "\n".join(lines)


def _mcmc_rows(results: FitResults) -> list[str]:
    rows: list[tuple[int, int, str]] = []
    for index, diag in enumerate(results.mcmc_diagnostics):
        cluster_id = results.clusters[index].cluster_id if index < len(results.clusters) else index
        for param in diag.parameter_diagnostics:
            status = _status_value(param.status)
            if status not in (_MCMC_MARGINAL, _MCMC_POOR):
                continue
            rhat = f"{param.rhat:.4g}" if param.rhat is not None else ""
            ess_bulk = f"{param.ess_bulk:.0f}" if param.ess_bulk is not None else ""
            ess_tail = f"{param.ess_tail:.0f}" if param.ess_tail is not None else ""
            rows.append(
                (
                    _mcmc_sort_key(diag),
                    cluster_id,
                    (
                        f"| {cluster_id} | {param.name} | {rhat} | {ess_bulk} | {ess_tail} | "
                        f"{status} |"
                    ),
                )
            )

    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return [row for _, _, row in rows]


def _collect_warnings(results: FitResults) -> list[str]:
    warnings: list[str] = []

    for index, cluster in enumerate(results.clusters):
        stats = results.statistics[index] if index < len(results.statistics) else None
        if stats is not None and not stats.fit_converged:
            warnings.append(f"Cluster {cluster.cluster_id} did not converge.")
        if stats is not None and not _is_good_redchi(stats.reduced_chi_squared):
            warnings.append(
                f"Cluster {cluster.cluster_id} has reduced chi2 {stats.reduced_chi_squared:.4g}."
            )

        for param in cluster.lineshape_params:
            if param.is_fixed:
                continue
            if param.is_at_boundary():
                warnings.append(f"{param.name} is at a fitting boundary.")
            rel_err = param.relative_error
            if rel_err is not None and rel_err > _POOR_RELATIVE_ERROR_THRESHOLD:
                warnings.append(f"{param.name} has relative uncertainty {rel_err:.0%}.")
            elif param.std_error <= 0:
                warnings.append(f"{param.name} has no positive uncertainty estimate.")

    for diag in results.mcmc_diagnostics:
        warnings.extend(diag.all_warnings)

    return list(dict.fromkeys(warnings))


def _show_parameter(param: ParameterEstimate) -> bool:
    if param.is_fixed:
        return False
    if param.is_problematic:
        return True
    return param.name.endswith(".cs") or param.name.endswith(".lw")


def _parameter_sort_key(param: ParameterEstimate) -> int:
    if param.is_at_boundary():
        return 0
    if param.is_problematic:
        return 1
    return 2


def _cluster_sort_key(stats: FitStatistics | None) -> int:
    if stats is None:
        return 2
    if not stats.fit_converged:
        return 0
    if not _is_good_redchi(stats.reduced_chi_squared):
        return 1
    return 2


def _mcmc_sort_key(diag: MCMCDiagnostics) -> int:
    status = _status_value(diag.overall_status)
    if status == _MCMC_POOR:
        return 0
    if status == _MCMC_MARGINAL:
        return 1
    return 2


def _status_value(status: object) -> str:
    return str(getattr(status, "value", status))


def _cluster_status(stats: FitStatistics | None) -> str:
    if stats is None:
        return "missing statistics"
    return _fit_status(stats.reduced_chi_squared, stats.fit_converged)


def _fit_status(redchi: float, converged: bool) -> str:
    if not converged:
        return "failed"
    if _is_good_redchi(redchi):
        return "ok"
    return "check"


def _parameter_status(param: ParameterEstimate) -> str:
    if param.is_at_boundary():
        return "at boundary"
    if param.is_problematic:
        return "check"
    return "ok"


def _peak_label(cluster: ClusterEstimates) -> str:
    names = cluster.peak_names[:_MAX_PEAK_NAMES]
    suffix = (
        f" +{len(cluster.peak_names) - len(names)}" if len(cluster.peak_names) > len(names) else ""
    )
    return ", ".join(names) + suffix


def _peak_name(cluster: ClusterEstimates, param: ParameterEstimate) -> str:
    if param.param_id is not None and param.param_id.peak_name:
        return param.param_id.peak_name
    if "." in param.name:
        return param.name.split(".", 1)[0]
    if cluster.peak_names:
        return cluster.peak_names[0]
    return f"cluster_{cluster.cluster_id}"


def _format_error(param: ParameterEstimate, precision: int, threshold: int) -> str:
    if (
        param.has_asymmetric_error
        and param.ci_68_lower is not None
        and param.ci_68_upper is not None
    ):
        upper = param.ci_68_upper - param.value
        lower = param.value - param.ci_68_lower
        upper_text = format_float(upper, precision, threshold)
        lower_text = format_float(lower, precision, threshold)
        return f"+{upper_text}/-{lower_text}"
    return format_float(param.std_error, precision, threshold)


def _is_good_redchi(redchi: float) -> bool:
    return _GOOD_REDCHI_MIN <= redchi <= _GOOD_REDCHI_MAX


def _omitted_note(total: int, shown: int, label: str) -> list[str]:
    omitted = total - shown
    if omitted <= 0:
        return []
    return ["", f"_Showing {shown} of {total} {label}. See JSON/CSV outputs for full detail._"]


__all__ = [
    "write_report",
]
