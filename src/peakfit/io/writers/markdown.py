"""Markdown report generator for human-readable output."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from peakfit.engine.results import ConvergenceStatus
from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.utils import format_float

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.results import (
        ClusterEstimates,
        FitResults,
        FitStatistics,
        MCMCDiagnostics,
        ParameterEstimate,
    )

# Constants for markdown formatting
_REDUCED_CHI2_GOOD_MIN = 0.5
_REDUCED_CHI2_GOOD_MAX = 2.0
_MAX_PEAK_NAMES_IN_SUMMARY = 3
_MAX_WARNINGS_IN_BRIEF = 5
_MAX_PARAMETERS_IN_REPORT = 80
_SECONDS_PER_MINUTE = 60
_POOR_RELATIVE_ERROR_THRESHOLD = 0.5


class MarkdownReportGenerator:
    """Generate human-readable Markdown reports from fit results.

    Creates structured reports including:
    - Executive summary with key metrics
    - Per-cluster parameter tables
    - Convergence diagnostics (for MCMC)
    - Warnings and recommendations
    """

    def __init__(self, config: WriterConfig | None = None) -> None:
        """Initialize report generator.

        Args:
            config: Writer configuration for formatting.
        """
        self.config = config or WriterConfig()

    def generate_full_report(self, results: FitResults, path: Path) -> None:
        """Generate complete analysis report.

        Args:
            results: FitResults object
            path: Output file path (e.g., results/reports/analysis_report.md)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        sections = [
            self._generate_header(results),
            self._generate_executive_summary(results),
            self._generate_cluster_summary(results),
            self._generate_key_parameters(results),
        ]

        # Add MCMC diagnostics section if applicable
        if results.mcmc_diagnostics:
            sections.append(self._generate_diagnostics_section(results))

        # Add warnings section if any
        warnings = self._collect_all_warnings(results)
        if warnings:
            sections.append(self._generate_warnings_section(warnings))

        sections.append(self._generate_footer(results))

        content = "\n\n".join(sections)
        path.write_text(content)

    def generate_summary_report(self, results: FitResults, path: Path) -> None:
        """Generate brief summary report.

        Suitable for quick review with essential information only.

        Args:
            results: FitResults object
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        sections = [
            self._generate_header(results),
            self._generate_executive_summary(results),
            self._generate_cluster_summary(results),
        ]

        # Brief warnings if any
        warnings = self._collect_all_warnings(results)
        if warnings:
            sections.append(self._generate_brief_warnings(warnings))

        sections.append(self._generate_footer(results))

        content = "\n\n".join(sections)
        path.write_text(content)

    def generate_cluster_report(
        self, cluster: ClusterEstimates, statistics: FitStatistics | None, path: Path
    ) -> None:
        """Generate report for a single cluster.

        Args:
            cluster: ClusterEstimates object
            statistics: FitStatistics for this cluster (optional)
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        sections = [
            f"# Cluster {cluster.cluster_id}: {', '.join(cluster.peak_names)}",
            "",
            self._generate_cluster_table(cluster),
        ]

        if statistics:
            sections.append(self._generate_statistics_summary(statistics))

        content = "\n\n".join(sections)
        path.write_text(content)

    # ----------------------------------------------------------------
    # Section generators
    # ----------------------------------------------------------------

    def _generate_header(self, results: FitResults) -> str:
        """Generate report header."""
        lines = [
            "# PeakFit Analysis Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Software Version:** {results.metadata.software_version}",
        ]

        if results.metadata.git_commit:
            lines.append(f"**Git Commit:** {results.metadata.git_commit}")

        lines.append(f"**Fitting Method:** {results.method.value}")

        return "\n".join(lines)

    def _generate_executive_summary(self, results: FitResults) -> str:
        """Generate executive summary section."""
        lines = [
            "## Executive Summary",
            "",
            f"- **Clusters analyzed:** {results.n_clusters}",
            f"- **Total peaks:** {results.n_peaks}",
        ]

        # Global fit quality
        if results.global_statistics:
            red_chi2 = results.global_statistics.reduced_chi_squared
            status = (
                "✓ Good"
                if _REDUCED_CHI2_GOOD_MIN <= red_chi2 <= _REDUCED_CHI2_GOOD_MAX
                else "⚠ Check"
            )
            lines.append(f"- **Reduced χ²:** {red_chi2:.4f} ({status})")

        # MCMC convergence summary
        if results.mcmc_diagnostics:
            converged = results.has_converged
            status = "✓ Converged" if converged else "⚠ Issues"
            lines.append(f"- **MCMC Convergence:** {status}")

            # Count problematic parameters
            n_problems = sum(
                len(
                    [
                        p
                        for p in d.parameter_diagnostics
                        if p.status in (ConvergenceStatus.MARGINAL, ConvergenceStatus.POOR)
                    ]
                )
                for d in results.mcmc_diagnostics
            )
            if n_problems > 0:
                lines.append(f"- **Parameters with issues:** {n_problems}")

        return "\n".join(lines)

    def _generate_cluster_summary(self, results: FitResults) -> str:
        """Generate summary table of cluster statistics."""
        lines = [
            "## Cluster Summary",
            "",
            "| Cluster | Peaks | χ² | Reduced χ² | Status |",
            "|---------|-------|-----|------------|--------|",
        ]

        for i, cluster in enumerate(results.clusters):
            peak_names = ", ".join(cluster.peak_names[:_MAX_PEAK_NAMES_IN_SUMMARY])
            if len(cluster.peak_names) > _MAX_PEAK_NAMES_IN_SUMMARY:
                peak_names += f" +{len(cluster.peak_names) - _MAX_PEAK_NAMES_IN_SUMMARY}"

            if i < len(results.statistics):
                stats = results.statistics[i]
                chi2_str = f"{stats.chi_squared:.1f}"
                red_chi2 = stats.reduced_chi_squared
                red_chi2_str = f"{red_chi2:.2f}"
                status = (
                    "✓" if _REDUCED_CHI2_GOOD_MIN <= red_chi2 <= _REDUCED_CHI2_GOOD_MAX else "⚠"
                )
            else:
                chi2_str = "—"
                red_chi2_str = "—"
                status = "?"

            lines.append(
                f"| {cluster.cluster_id} | {peak_names} | {chi2_str} | {red_chi2_str} | {status} |"
            )

        return "\n".join(lines)

    def _generate_parameter_summary(self, results: FitResults) -> str:
        """Generate parameter summary tables."""
        lines = ["## Parameter Estimates", ""]

        for i, cluster in enumerate(results.clusters):
            lines.append(f"### Cluster {cluster.cluster_id}: {', '.join(cluster.peak_names)}")
            lines.append("")
            lines.append(self._generate_cluster_table(cluster))
            lines.append("")

            # Add statistics if available
            if i < len(results.statistics):
                lines.append(self._generate_statistics_summary(results.statistics[i]))
                lines.append("")

        return "\n".join(lines)

    def _generate_key_parameters(self, results: FitResults) -> str:
        """Generate a compact parameter table focused on relevant entries."""
        candidates: list[tuple[int, int, str, ParameterEstimate]] = []

        for cluster in results.clusters:
            for param in cluster.lineshape_params:
                priority = 2
                if param.is_problematic:
                    priority = 0
                elif param.name.endswith(".cs") or param.name.endswith(".lw"):
                    priority = 1

                peak_name = (
                    param.param_id.peak_name
                    if param.param_id is not None and param.param_id.peak_name
                    else (
                        param.name.split(".", 1)[0]
                        if "." in param.name
                        else (
                            cluster.peak_names[0]
                            if cluster.peak_names
                            else f"cluster_{cluster.cluster_id}"
                        )
                    )
                )
                candidates.append((priority, cluster.cluster_id, peak_name, param))

        if not candidates:
            return "## Key Parameters\n\nNo parameters were exported."

        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3].name))
        selected = candidates[:_MAX_PARAMETERS_IN_REPORT]

        lines = [
            "## Key Parameters",
            "",
            "| Cluster | Peak | Parameter | Value | Uncertainty | Status |",
            "|---------|------|-----------|-------|-------------|--------|",
        ]

        prec = self.config.precision
        thresh = self.config.scientific_notation_threshold
        for _, cluster_id, peak_name, param in selected:
            value_str = format_float(param.value, prec, thresh)
            unc_str = self._format_uncertainty(param, prec, thresh)
            status = self._get_param_status_indicator(param)
            param_name = param.param_id.name if param.param_id is not None else param.name
            row = (
                f"| {cluster_id} | {peak_name} | {param_name} | "
                f"{value_str} | {unc_str} | {status} |"
            )
            lines.append(row)

        hidden = len(candidates) - len(selected)
        if hidden > 0:
            lines.append("")
            lines.append(
                f"_Showing top {_MAX_PARAMETERS_IN_REPORT} parameters ({hidden} omitted)._"
            )

        return "\n".join(lines)

    def _generate_cluster_table(self, cluster: ClusterEstimates) -> str:
        """Generate parameter table for a cluster."""
        lines = [
            "| Parameter | Value | Uncertainty | Unit | Status |",
            "|-----------|-------|-------------|------|--------|",
        ]

        prec = self.config.precision
        thresh = self.config.scientific_notation_threshold

        lines.extend(
            self._format_parameter_row(param, prec, thresh) for param in cluster.lineshape_params
        )

        return "\n".join(lines)

    def _generate_statistics_summary(self, stats: FitStatistics) -> str:
        """Generate statistics summary."""
        prec = self.config.precision

        lines = [
            "**Fit Statistics:**",
            f"- χ² = {stats.chi_squared:.{prec}f}",
            f"- Reduced χ² = {stats.reduced_chi_squared:.{prec}f}",
            f"- DOF = {stats.dof}",
        ]

        if stats.aic is not None:
            lines.append(f"- AIC = {stats.aic:.{prec}f}")
        if stats.bic is not None:
            lines.append(f"- BIC = {stats.bic:.{prec}f}")

        return "\n".join(lines)

    def _generate_diagnostics_section(self, results: FitResults) -> str:
        """Generate MCMC diagnostics section."""
        lines = ["## MCMC Convergence Diagnostics", ""]

        for i, diag in enumerate(results.mcmc_diagnostics):
            cluster_label = (
                ", ".join(results.clusters[i].peak_names)
                if i < len(results.clusters)
                else f"Cluster {i}"
            )

            lines.append(f"### {cluster_label}")
            lines.append("")
            lines.append(f"**Overall Status:** {self._status_badge(diag.overall_status)}")
            lines.append(f"- Chains: {diag.n_chains}")
            lines.append(f"- Samples per chain: {diag.n_samples}")
            lines.append(f"- Burn-in: {diag.burn_in}")
            lines.append("")

            # Parameter diagnostics table
            if diag.parameter_diagnostics:
                lines.append(self._generate_diagnostics_table(diag))
                lines.append("")

        return "\n".join(lines)

    def _generate_diagnostics_table(self, diag: MCMCDiagnostics) -> str:
        """Generate diagnostics table for a cluster."""
        lines = [
            "| Parameter | R-hat | ESS (bulk) | ESS (tail) | Status |",
            "|-----------|-------|------------|------------|--------|",
        ]

        for pd in diag.parameter_diagnostics:
            rhat_str = f"{pd.rhat:.4f}" if pd.rhat is not None else "—"
            ess_bulk_str = f"{pd.ess_bulk:.0f}" if pd.ess_bulk is not None else "—"
            ess_tail_str = f"{pd.ess_tail:.0f}" if pd.ess_tail is not None else "—"
            status = self._status_badge(pd.status)

            lines.append(f"| {pd.name} | {rhat_str} | {ess_bulk_str} | {ess_tail_str} | {status} |")

        return "\n".join(lines)

    def _generate_warnings_section(self, warnings: list[str]) -> str:
        """Generate warnings section."""
        lines = ["## ⚠️ Warnings", ""]
        lines.extend(f"- {warning}" for warning in warnings)
        return "\n".join(lines)

    def _generate_brief_warnings(self, warnings: list[str]) -> str:
        """Generate brief warnings for summary report."""
        n_warnings = len(warnings)
        if n_warnings == 0:
            return ""

        lines = [
            f"## ⚠️ Warnings ({n_warnings})",
            "",
        ]

        # Show first 5 only
        lines.extend(f"- {warning}" for warning in warnings[:_MAX_WARNINGS_IN_BRIEF])

        if n_warnings > _MAX_WARNINGS_IN_BRIEF:
            lines.append(f"- ... and {n_warnings - _MAX_WARNINGS_IN_BRIEF} more")

        return "\n".join(lines)

    def _generate_footer(self, results: FitResults) -> str:
        """Generate report footer."""
        lines = [
            "---",
            "",
            "*This report was automatically generated by PeakFit.*",
        ]

        if results.metadata.run_duration_seconds:
            duration = results.metadata.run_duration_seconds
            if duration < _SECONDS_PER_MINUTE:
                time_str = f"{duration:.1f} seconds"
            else:
                minutes = int(duration // _SECONDS_PER_MINUTE)
                seconds = duration % _SECONDS_PER_MINUTE
                time_str = f"{minutes} min {seconds:.0f} sec"
            lines.append(f"*Analysis completed in {time_str}.*")

        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------

    def _format_parameter_row(self, param: ParameterEstimate, prec: int, thresh: int) -> str:
        """Format a single parameter row."""
        value_str = format_float(param.value, prec, thresh)
        unc_str = self._format_uncertainty(param, prec, thresh)
        status = self._get_param_status_indicator(param)
        return f"| {param.name} | {value_str} | {unc_str} | {param.unit} | {status} |"

    def _format_uncertainty(self, param: ParameterEstimate, prec: int, thresh: int) -> str:
        """Format uncertainty string."""
        if param.has_asymmetric_error and param.ci_68_lower is not None:
            upper_diff = param.ci_68_upper - param.value if param.ci_68_upper else 0
            lower_diff = param.value - param.ci_68_lower
            upper_str = format_float(upper_diff, prec, thresh)
            lower_str = format_float(lower_diff, prec, thresh)
            return f"+{upper_str}/−{lower_str}"
        return format_float(param.std_error, prec, thresh)

    def _get_param_status_indicator(self, param: ParameterEstimate) -> str:
        """Get status indicator for a parameter."""
        if param.is_fixed:
            return "🔒 Fixed"
        if param.is_problematic:
            return "⚠️ Check"
        if param.is_at_boundary():
            return "⚠️ At bound"
        return "✓"

    def _status_badge(self, status: ConvergenceStatus) -> str:
        """Get badge for convergence status."""
        badges = {
            ConvergenceStatus.EXCELLENT: "✓ Excellent",
            ConvergenceStatus.GOOD: "✓ Good",
            ConvergenceStatus.ACCEPTABLE: "○ OK",
            ConvergenceStatus.MARGINAL: "⚠ Marginal",
            ConvergenceStatus.POOR: "[BAD] Poor",
            ConvergenceStatus.UNKNOWN: "? Unknown",
        }
        return badges.get(status, "?")

    def _collect_all_warnings(self, results: FitResults) -> list[str]:
        """Collect all warnings from results."""
        warnings = []
        warnings.extend(self._collect_parameter_warnings(results))
        warnings.extend(self._collect_mcmc_warnings(results))
        return warnings

    def _collect_parameter_warnings(self, results: FitResults) -> list[str]:
        """Collect warnings from parameters."""
        warnings = []
        for cluster in results.clusters:
            for param in cluster.lineshape_params:
                if param.is_at_boundary():
                    warnings.append(f"Parameter {param.name} is at a fitting boundary")
                rel_err = param.relative_error
                if (
                    rel_err is not None
                    and rel_err > _POOR_RELATIVE_ERROR_THRESHOLD
                    and not param.is_fixed
                ):
                    warnings.append(
                        f"Parameter {param.name} is poorly determined "
                        f"(uncertainty > {rel_err * 100:.0f}%)"
                    )
        return warnings

    def _collect_mcmc_warnings(self, results: FitResults) -> list[str]:
        """Collect warnings from MCMC diagnostics."""
        warnings = []
        for diag in results.mcmc_diagnostics:
            warnings.extend(diag.all_warnings)
        return warnings


__all__ = [
    "MarkdownReportGenerator",
]
