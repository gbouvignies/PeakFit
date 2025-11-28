"""Markdown report generator for human-readable PeakFit results.

Generates structured Markdown reports with tables, summaries, and
diagnostic information for review by users.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from peakfit.core.results.diagnostics import ConvergenceStatus
from peakfit.io.writers.base import WriterConfig, format_float

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.core.results.diagnostics import MCMCDiagnostics
    from peakfit.core.results.estimates import ClusterEstimates, ParameterEstimate
    from peakfit.core.results.fit_results import FitResults
    from peakfit.core.results.statistics import FitStatistics


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
            self._generate_parameter_summary(results),
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
            f"# Cluster {cluster.cluster_id}: {", ".join(cluster.peak_names)}",
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
            f"**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}",
            f"**Software Version:** {results.metadata.software_version}",
        ]

        if results.metadata.git_commit:
            lines.append(f"**Git Commit:** {results.metadata.git_commit}")

        lines.append(f"**Fitting Method:** {results.method.value}")

        if results.experiment_type:
            lines.append(f"**Experiment Type:** {results.experiment_type}")

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
            status = "✓ Good" if 0.5 <= red_chi2 <= 2.0 else "⚠ Check"
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
            peak_names = ", ".join(cluster.peak_names[:3])
            if len(cluster.peak_names) > 3:
                peak_names += f" +{len(cluster.peak_names) - 3}"

            if i < len(results.statistics):
                stats = results.statistics[i]
                chi2_str = f"{stats.chi_squared:.1f}"
                red_chi2 = stats.reduced_chi_squared
                red_chi2_str = f"{red_chi2:.2f}"
                status = "✓" if 0.5 <= red_chi2 <= 2.0 else "⚠"
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
            lines.append(f"### Cluster {cluster.cluster_id}: {", ".join(cluster.peak_names)}")
            lines.append("")
            lines.append(self._generate_cluster_table(cluster))
            lines.append("")

            # Add statistics if available
            if i < len(results.statistics):
                lines.append(self._generate_statistics_summary(results.statistics[i]))
                lines.append("")

        return "\n".join(lines)

    def _generate_cluster_table(self, cluster: ClusterEstimates) -> str:
        """Generate parameter table for a cluster."""
        lines = [
            "| Parameter | Value | Uncertainty | Unit | Status |",
            "|-----------|-------|-------------|------|--------|",
        ]

        prec = self.config.precision
        thresh = self.config.scientific_notation_threshold

        for param in cluster.lineshape_params:
            value_str = format_float(param.value, prec, thresh)

            # Format uncertainty
            if param.has_asymmetric_error and param.ci_68_lower is not None:
                upper_diff = param.ci_68_upper - param.value if param.ci_68_upper else 0
                lower_diff = param.value - param.ci_68_lower
                unc_str = f"+{format_float(upper_diff, prec, thresh)}/−{format_float(lower_diff, prec, thresh)}"
            else:
                unc_str = format_float(param.std_error, prec, thresh)

            # Status indicator
            status = self._get_param_status_indicator(param)

            lines.append(f"| {param.name} | {value_str} | {unc_str} | {param.unit} | {status} |")

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
        lines.extend(f"- {warning}" for warning in warnings[:5])

        if n_warnings > 5:
            lines.append(f"- ... and {n_warnings - 5} more")

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
            if duration < 60:
                time_str = f"{duration:.1f} seconds"
            else:
                minutes = int(duration // 60)
                seconds = duration % 60
                time_str = f"{minutes} min {seconds:.0f} sec"
            lines.append(f"*Analysis completed in {time_str}.*")

        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------

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

        # Parameter warnings
        for cluster in results.clusters:
            for param in cluster.lineshape_params:
                if param.is_at_boundary():
                    warnings.append(f"Parameter {param.name} is at a fitting boundary")
                rel_err = param.relative_error
                if rel_err is not None and rel_err > 0.5 and not param.is_fixed:
                    warnings.append(
                        f"Parameter {param.name} is poorly determined "
                        f"(uncertainty > {rel_err * 100:.0f}%)"
                    )

        # MCMC warnings
        for diag in results.mcmc_diagnostics:
            warnings.extend(diag.all_warnings)

        return warnings
