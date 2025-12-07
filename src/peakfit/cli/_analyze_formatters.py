"""Rich table formatters for analyze command output.

This module contains Rich-specific formatting functions for displaying
analyze command results. Keeps console/Rich dependencies out of services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.table import Table

from peakfit.ui import console

if TYPE_CHECKING:
    from peakfit.services.analyze.formatters import MCMCClusterSummary, MCMCParameterSummary


def format_rhat(rhat: float | None) -> str:
    """Format R-hat value with color coding."""
    if rhat is None:
        return "[neutral]N/A[/neutral]"
    if rhat <= 1.01:
        return f"[success]{rhat:.4f}[/success]"
    if rhat <= 1.05:
        return f"[info]{rhat:.4f}[/info]"
    return f"[error]{rhat:.4f}[/error]"


def format_ess(ess: float | None, target: int = 10000) -> str:
    """Format ESS value with percentage toward target."""
    if ess is None:
        return "[neutral]N/A[/neutral]"

    pct = min(100, (ess / target) * 100)
    if ess >= target:
        return f"[success]{ess:.0f} (100%)[/success]"
    if ess >= 100:
        return f"[success]{ess:.0f} ({pct:.0f}%)[/success]"
    if ess >= 10:
        return f"[warning]{ess:.0f} ({pct:.0f}%)[/warning]"
    return f"[error]{ess:.0f} ({pct:.0f}%)[/error]"


def format_status(summary: MCMCParameterSummary) -> str:
    """Format convergence status with icon and color."""
    status = summary.convergence_status
    status_map = {
        "excellent": "[success]✓ Excellent[/success]",
        "good": "[success]✓ Good[/success]",
        "acceptable": "[info]○ Acceptable[/info]",
        "marginal": "[warning]⚠ Marginal[/warning]",
        "poor": "[error]✗ Poor[/error]",
        "unknown": "[neutral]? Unknown[/neutral]",
    }
    return status_map.get(status, "[neutral]?[/neutral]")


def print_mcmc_diagnostics_table(summary: MCMCClusterSummary) -> None:
    """Print convergence diagnostics table for a cluster."""
    console.print(f"[header]Convergence Diagnostics - {summary.cluster_label}[/header]")
    console.print(f"  Chains: {summary.n_chains}, Samples per chain: {summary.n_samples}")
    console.print(
        "  [dim]BARG Guidelines: R-hat ≤ 1.01 (excellent), "
        "ESS ≥ 10,000 for stable CIs (Kruschke 2021)[/dim]"
    )
    console.print("")

    table = Table(show_header=True, header_style="header")
    table.add_column("Parameter", style="key", width=20)
    table.add_column("R-hat", justify="right", width=10)
    table.add_column("ESS_bulk", justify="right", width=14)
    table.add_column("ESS_tail", justify="right", width=14)
    table.add_column("Status", width=15)

    for param in summary.parameter_summaries:
        table.add_row(
            param.name,
            format_rhat(param.rhat),
            format_ess(param.ess_bulk),
            format_ess(param.ess_tail),
            format_status(param),
        )

    console.print(table)
    console.print("")


def print_mcmc_results_table(summary: MCMCClusterSummary) -> None:
    """Print MCMC results table for a cluster."""
    table = Table(title=f"MCMC Results - {summary.cluster_label}")
    table.add_column("Parameter", style="key")
    table.add_column("Value", justify="right", style="value")
    table.add_column("Std Error", justify="right")
    table.add_column("68% CI", justify="right")
    table.add_column("95% CI", justify="right")

    for param in summary.parameter_summaries:
        table.add_row(
            param.name,
            f"{param.value:.6f}",
            f"{param.std_error:.6f}",
            f"[{param.ci_68_lower:.6f}, {param.ci_68_upper:.6f}]",
            f"[{param.ci_95_lower:.6f}, {param.ci_95_upper:.6f}]",
        )

    console.print(table)
    console.print("")


def print_correlation_matrix(summary: MCMCClusterSummary) -> None:
    """Print correlation matrix table for a cluster."""
    if summary.correlation_matrix is None or len(summary.parameter_summaries) < 2:
        return

    console.print(f"[header]Correlation Matrix - {summary.cluster_label}[/header]")
    console.print("  (Strong correlations: |r| > 0.7)")

    table = Table(show_header=True, header_style="header")
    table.add_column("", style="key", width=15)

    param_names = [p.name for p in summary.parameter_summaries]

    # Use multiline headers to save horizontal space
    # Format: PeakName\nParamName (e.g. "41N-H\nF2.cs")
    for name in param_names:
        if "." in name:
            parts = name.split(".")
            # Heuristic: First part is usually peak name
            header = f"{parts[0]}\n{'.'.join(parts[1:])}"
            table.add_column(header, justify="right")
        else:
            table.add_column(name, justify="right")

    for i, param in enumerate(summary.parameter_summaries):
        # Use simple name for row label to save space
        # e.g. "41N-H.F2.cs" -> "F2.cs" (if unambiguous) or keep full
        # For now, keep full name but truncate if too long to be safe
        row_label = param.name
        row = [row_label[:20]]

        for j in range(len(summary.parameter_summaries)):
            val = summary.correlation_matrix[i, j]
            if i == j:
                row.append("[neutral]1.0000[/neutral]")
            elif abs(val) > 0.7:
                row.append(f"[warning]{val:7.4f}[/warning]")
            elif abs(val) > 0.3:
                row.append(f"[string]{val:7.4f}[/string]")
            else:
                row.append(f"{val:7.4f}")

        table.add_row(*row)

    console.print(table)
    console.print("")


def print_profile_likelihood_table(
    cluster_name: str,
    results: list[tuple[str, float, float, float]],
) -> None:
    """Print profile likelihood results table.

    Args:
        cluster_name: Name of the cluster
        results: List of (param_name, best_fit, ci_lower, ci_upper)
    """
    table = Table(title=f"Profile Likelihood Results - {cluster_name}")
    table.add_column("Parameter", style="key")
    table.add_column("Best Fit", justify="right", style="value")
    table.add_column("Lower 68% CI", justify="right")
    table.add_column("Upper 68% CI", justify="right")
    table.add_column("Uncertainty", justify="right")

    for name, best_fit, ci_lower, ci_upper in results:
        uncertainty = (ci_upper - ci_lower) / 2
        table.add_row(
            name,
            f"{best_fit:.6f}",
            f"{ci_lower:.6f}",
            f"{ci_upper:.6f}",
            f"± {uncertainty:.6f}",
        )

    console.print(table)
    console.print("")


def print_uncertainty_table(
    cluster_name: str,
    results: list[tuple[str, float, float, float, float]],
) -> None:
    """Print uncertainty estimation results table.

    Args:
        cluster_name: Name of the cluster
        results: List of (param_name, value, error, lower, upper)
    """
    table = Table(title=f"Parameter Uncertainties - {cluster_name}")
    table.add_column("Parameter", style="key")
    table.add_column("Value", justify="right", style="value")
    table.add_column("Error", justify="right")
    table.add_column("Lower Bound", justify="right")
    table.add_column("Upper Bound", justify="right")

    for name, value, error, lower, upper in results:
        table.add_row(
            name,
            f"{value:.6f}",
            f"± {error:.6f}",
            f"{lower:.6f}",
            f"{upper:.6f}",
        )

    console.print(table)
    console.print("")


def print_correlation_analysis_table(
    results: list[tuple[str, str, float]],
) -> None:
    """Print correlation analysis results.

    Args:
        results: List of (param1, param2, correlation)
    """
    if not results:
        console.print("[dim]No significant correlations found.[/dim]")
        return

    table = Table(title="Parameter Correlations")
    table.add_column("Parameter 1", style="key")
    table.add_column("Parameter 2", style="key")
    table.add_column("Correlation", justify="right")
    table.add_column("Strength", justify="center")

    for param1, param2, corr in sorted(results, key=lambda x: abs(x[2]), reverse=True):
        if abs(corr) > 0.9:
            strength = "[error]Very Strong[/error]"
        elif abs(corr) > 0.7:
            strength = "[warning]Strong[/warning]"
        elif abs(corr) > 0.5:
            strength = "[info]Moderate[/info]"
        else:
            strength = "[neutral]Weak[/neutral]"

        corr_str = f"[bold]{corr:+.4f}[/bold]" if abs(corr) > 0.7 else f"{corr:+.4f}"
        table.add_row(param1, param2, corr_str, strength)

    console.print(table)
    console.print("")


def print_mcmc_amplitude_table(summary: MCMCClusterSummary, max_rows: int = 10) -> None:
    """Print MCMC amplitude (intensity) results table for a cluster.

    Args:
        summary: MCMCClusterSummary containing amplitude summaries
        max_rows: Maximum number of rows to display per peak before summarizing
    """
    if not summary.amplitude_summaries:
        return

    console.print(f"[header]Intensity Results - {summary.cluster_label}[/header]")
    console.print("  [dim]Intensities computed from linear least-squares at each MCMC sample[/dim]")
    console.print("")

    # Group amplitudes by peak
    by_peak = summary.get_amplitudes_by_peak()

    for peak_name, amplitudes in by_peak.items():
        # Sort by plane index
        amplitudes = sorted(amplitudes, key=lambda x: x.plane_index)
        n_planes = len(amplitudes)

        table = Table(title=f"Intensities for {peak_name}")
        table.add_column(
            "Z-value" if amplitudes[0].z_value is not None else "Plane", justify="right"
        )
        table.add_column("Intensity", justify="right", style="value")
        table.add_column("Std Error", justify="right")
        table.add_column("68% CI", justify="right")
        table.add_column("Rel. Error (%)", justify="right")

        # Determine if we need to truncate
        show_all = n_planes <= max_rows
        display_amps = (
            amplitudes if show_all else amplitudes[: max_rows // 2] + amplitudes[-max_rows // 2 :]
        )

        for i, amp in enumerate(display_amps):
            # Add ellipsis row if truncated
            if not show_all and i == max_rows // 2:
                table.add_row(
                    "[dim]...[/dim]",
                    "[dim]...[/dim]",
                    "[dim]...[/dim]",
                    "[dim]...[/dim]",
                    "[dim]...[/dim]",
                )

            z_str = f"{amp.z_value:.1f}" if amp.z_value is not None else str(amp.plane_index)

            # Calculate relative error
            rel_err = abs(amp.std_error / amp.value * 100) if amp.value != 0 else 0.0
            rel_err_str = f"{rel_err:.1f}%"
            if rel_err > 10:
                rel_err_str = f"[warning]{rel_err:.1f}%[/warning]"

            table.add_row(
                z_str,
                f"{amp.value:.4e}",
                f"{amp.std_error:.4e}",
                f"[{amp.ci_68_lower:.4e}, {amp.ci_68_upper:.4e}]",
                rel_err_str,
            )

        console.print(table)

        if not show_all:
            console.print(
                f"  [dim]Showing {max_rows} of {n_planes} planes. "
                f"Full results saved to output file.[/dim]"
            )

    console.print("")
