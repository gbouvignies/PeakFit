"""Implementation of Uncertainty analysis command."""

from pathlib import Path

from rich.table import Table

from peakfit.cli.analysis.shared import load_fitting_state
from peakfit.services.analyze import NoVaryingParametersFoundError, ParameterUncertaintyService
from peakfit.ui import Verbosity, console, set_verbosity, show_standard_header, success, warning


def run_uncertainty(
    results_dir: Path, output_file: Path | None = None, verbose: bool = False
) -> None:
    """Display parameter uncertainties from fitting results.

    Shows the covariance-based uncertainties computed during fitting.

    Args:
        results_dir: Path to results directory
        output_file: Optional output file for uncertainty summary
        verbose: Show banner and verbose output
    """
    # Set verbosity and show header
    set_verbosity(Verbosity.VERBOSE if verbose else Verbosity.NORMAL)
    show_standard_header("Parameter Uncertainties")

    state = load_fitting_state(results_dir)
    try:
        analysis = ParameterUncertaintyService.analyze(state)
    except NoVaryingParametersFoundError:
        warning("No varying parameters found")
        return

    console.print("  Source: Covariance matrix from least-squares fit")
    console.print(f"  Parameters: {len(analysis.parameters)}")
    console.print("")

    # Create uncertainty table
    table = Table(title="Fitted Parameters with Uncertainties")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Std Error", justify="right")
    table.add_column("Relative Error (%)", justify="right")
    table.add_column("At Boundary?", justify="center")

    large_uncertainty_names = {entry.name for entry in analysis.large_uncertainty_parameters}

    for entry in analysis.parameters:
        at_boundary = "⚠️" if entry.at_boundary else ""

        rel_error_str = f"{entry.rel_error_pct:.2f}%" if entry.rel_error_pct is not None else "N/A"

        if entry.stderr <= 0:
            stderr_str = "[red]Not computed[/red]"
        elif entry.name in large_uncertainty_names:
            stderr_str = f"[yellow]{entry.stderr:.6f}[/yellow]"
        else:
            stderr_str = f"{entry.stderr:.6f}"

        table.add_row(
            entry.name,
            f"{entry.value:.6f}",
            stderr_str,
            rel_error_str,
            at_boundary,
        )

    console.print(table)

    # Report boundary warnings
    if analysis.boundary_parameters:
        console.print()
        warning("Parameters at boundaries:")
        for entry in analysis.boundary_parameters:
            console.print(
                f"  {entry.name}: {entry.value:.6f} (bounds: "
                f"[{entry.min_bound:.6f}, {entry.max_bound:.6f}])"
            )
        console.print("  [dim]Consider adjusting bounds or using global optimization[/dim]")

    # Check for large uncertainties
    if analysis.large_uncertainty_parameters:
        console.print()
        warning("Parameters with large relative uncertainties (>10%):")
        for entry in analysis.large_uncertainty_parameters:
            rel_err = entry.rel_error_pct if entry.rel_error_pct is not None else 0.0
            console.print(f"  {entry.name}: {rel_err:.1f}%")
        console.print("  [dim]Consider MCMC analysis for better uncertainty estimates[/dim]")

    # Suggest next steps
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  • Run MCMC for full posterior distributions and correlations:")
    console.print(f"    [cyan]peakfit analyze mcmc {results_dir}/[/cyan]")
    console.print("  • After MCMC, visualize diagnostics and correlations:")
    console.print(f"    [cyan]peakfit plot diagnostics {results_dir}/[/cyan]")

    if output_file is not None:
        with output_file.open("w") as f:
            f.write("# Parameter Uncertainty Summary\n")
            f.write("# Name  Value  Stderr  RelError(%)  Min  Max\n")
            for entry in analysis.parameters:
                rel_error = entry.rel_error_pct if entry.rel_error_pct is not None else 0.0
                f.write(
                    f"{entry.name}  {entry.value:.6f}  {entry.stderr:.6f}  {rel_error:.2f}  "
                    f"{entry.min_bound:.6f}  {entry.max_bound:.6f}\n"
                )
        success(f"Saved uncertainty summary to: [path]{output_file}[/path]")
