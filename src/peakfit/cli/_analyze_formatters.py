"""Rich table formatters for analyze command output.

This module contains Rich-specific formatting functions for displaying
analyze command results. Keeps console/Rich dependencies out of services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.table import Table

from peakfit.ui import console

if TYPE_CHECKING:
    from peakfit.services.analyze.formatters import (
        MCMCClusterSummary,
        MCMCParameterSummary,
    )


def format_rhat(rhat: float | None) -> str:
    """Format R-hat value with color coding."""
    if rhat is None:
        return "[dim]N/A[/dim]"
    if rhat <= 1.01:
        return f"[green]{rhat:.4f}[/green]"
    if rhat <= 1.05:
        return f"[cyan]{rhat:.4f}[/cyan]"
    return f"[red]{rhat:.4f}[/red]"


def format_ess(ess: float | None, target: int = 10000) -> str:
    """Format ESS value with percentage toward target."""
    if ess is None:
        return "[dim]N/A[/dim]"

    pct = min(100, (ess / target) * 100)
    if ess >= target:
        return f"[green]{ess:.0f} (100%)[/green]"
    if ess >= 100:
        return f"[green]{ess:.0f} ({pct:.0f}%)[/green]"
    if ess >= 10:
        return f"[yellow]{ess:.0f} ({pct:.0f}%)[/yellow]"
    return f"[red]{ess:.0f} ({pct:.0f}%)[/red]"


def format_status(summary: MCMCParameterSummary) -> str:
    """Format convergence status with icon and color."""
    status = summary.convergence_status
    status_map = {
        "excellent": "[green]✓ Excellent[/green]",
        "good": "[green]✓ Good[/green]",
        "acceptable": "[cyan]○ Acceptable[/cyan]",
        "marginal": "[yellow]⚠ Marginal[/yellow]",
        "poor": "[red]✗ Poor[/red]",
        "unknown": "[dim]? Unknown[/dim]",
    }
    return status_map.get(status, "[dim]?[/dim]")


def print_mcmc_diagnostics_table(summary: MCMCClusterSummary) -> None:
    """Print convergence diagnostics table for a cluster."""
    console.print(f"[bold cyan]Convergence Diagnostics - {summary.cluster_label}[/bold cyan]")
    console.print(f"  Chains: {summary.n_chains}, Samples per chain: {summary.n_samples}")
    console.print(
        "  [dim]BARG Guidelines: R-hat ≤ 1.01 (excellent), "
        "ESS ≥ 10,000 for stable CIs (Kruschke 2021)[/dim]"
    )
    console.print("")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="cyan", width=20)
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
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")
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

    console.print(f"[bold cyan]Correlation Matrix - {summary.cluster_label}[/bold cyan]")
    console.print("  (Strong correlations: |r| > 0.7)")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("", style="cyan", width=15)

    param_names = [p.name for p in summary.parameter_summaries]
    for name in param_names:
        short_name = name.split("_")[-1] if "_" in name else name
        table.add_column(short_name[:8], justify="right", width=9)

    for i, param in enumerate(summary.parameter_summaries):
        short_name = param.name.split("_")[-1] if "_" in param.name else param.name
        row = [short_name[:15]]

        for j in range(len(summary.parameter_summaries)):
            val = summary.correlation_matrix[i, j]
            if i == j:
                row.append("[dim]1.0000[/dim]")
            elif abs(val) > 0.7:
                row.append(f"[bold yellow]{val:7.4f}[/bold yellow]")
            elif abs(val) > 0.3:
                row.append(f"[yellow]{val:7.4f}[/yellow]")
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
    table.add_column("Parameter", style="cyan")
    table.add_column("Best Fit", justify="right")
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
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")
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
    table.add_column("Parameter 1", style="cyan")
    table.add_column("Parameter 2", style="cyan")
    table.add_column("Correlation", justify="right")
    table.add_column("Strength", justify="center")

    for param1, param2, corr in sorted(results, key=lambda x: abs(x[2]), reverse=True):
        if abs(corr) > 0.9:
            strength = "[red]Very Strong[/red]"
        elif abs(corr) > 0.7:
            strength = "[yellow]Strong[/yellow]"
        elif abs(corr) > 0.5:
            strength = "[cyan]Moderate[/cyan]"
        else:
            strength = "[dim]Weak[/dim]"

        corr_str = f"[bold]{corr:+.4f}[/bold]" if abs(corr) > 0.7 else f"{corr:+.4f}"
        table.add_row(param1, param2, corr_str, strength)

    console.print(table)
    console.print("")
