"""Contain IO messages."""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(record=True)

LOGO = r"""
   ___           _     _____ _ _
  / _ \ ___  __ _| | __|  ___(_) |_
 / /_)/ _ \/ _` | |/ /| |_  | | __|
/ ___/  __/ (_| |   < |  _| | | |_
\/    \___|\__,_|_|\_\|_|   |_|\__|
"""


def print_fit_report(result: object) -> None:
    """Print the fitting report.

    Args:
        result: scipy.optimize.OptimizeResult or similar object
    """
    # Create a table for fit statistics
    table = Table(title="Fit Statistics", show_header=False, box=None)
    table.add_column("Property", style="cyan", width=20)
    table.add_column("Value", style="green")

    if hasattr(result, "success"):
        status_style = "green" if result.success else "red"
        status_text = "✓ success" if result.success else "✗ failure"
        table.add_row("Status", f"[{status_style}]{status_text}[/]")

    if hasattr(result, "message"):
        table.add_row("Message", str(result.message))

    if hasattr(result, "nfev"):
        table.add_row("Function evals", str(result.nfev))

    if hasattr(result, "njev") and result.njev:
        table.add_row("Jacobian evals", str(result.njev))

    if hasattr(result, "cost"):
        table.add_row("Final cost", f"{result.cost:.6e}")

    if hasattr(result, "optimality"):
        table.add_row("Optimality", f"{result.optimality:.6e}")

    if hasattr(result, "x"):
        table.add_row("Variables", str(len(result.x)))

    console.print()
    console.print(table)
    console.print()


def print_fit_summary(
    total_clusters: int,
    total_peaks: int,
    total_time: float,
    success_count: int,
) -> None:
    """Print a summary of the fitting results."""
    console.print()
    panel_content = Text()
    panel_content.append("Fitting Complete\n\n", style="bold green")
    panel_content.append(f"  Clusters fitted:  {total_clusters}\n")
    panel_content.append(f"  Total peaks:      {total_peaks}\n")
    panel_content.append(f"  Successful:       {success_count}/{total_clusters}\n")
    panel_content.append(f"  Total time:       {total_time:.2f}s\n")

    if total_clusters > 0:
        avg_time = total_time / total_clusters
        panel_content.append(f"  Avg per cluster:  {avg_time:.3f}s")

    console.print(Panel(panel_content, border_style="green"))


def print_optimization_settings(ftol: float, xtol: float, max_nfev: int) -> None:
    """Print optimization settings."""
    console.print(f"[dim]Optimization: ftol={ftol:.0e}, xtol={xtol:.0e}, max_nfev={max_nfev}[/]")


def print_data_summary(
    spectrum_shape: tuple,
    n_planes: int,
    n_peaks: int,
    n_clusters: int,
    noise_level: float,
    contour_level: float,
) -> None:
    """Print a summary of loaded data.

    Args:
        spectrum_shape: Shape of spectrum data
        n_planes: Number of planes (z-values)
        n_peaks: Number of peaks
        n_clusters: Number of clusters
        noise_level: Estimated noise level
        contour_level: Contour level for clustering
    """
    table = Table(title="Data Summary", show_header=False, box=None)
    table.add_column("Property", style="cyan", width=20)
    table.add_column("Value", style="green")

    table.add_row("Spectrum shape", str(spectrum_shape))
    table.add_row("Number of planes", str(n_planes))
    table.add_row("Number of peaks", str(n_peaks))
    table.add_row("Number of clusters", str(n_clusters))
    table.add_row("Noise level", f"{noise_level:.4f}")
    table.add_row("Contour level", f"{contour_level:.4f}")

    console.print()
    console.print(table)
    console.print()


def export_html(filehtml: Path) -> None:
    """Export console output to an HTML file."""
    filehtml.write_text(console.export_html())


def print_reading_files() -> None:
    """Print the message for reading files."""
    console.print("\n — Reading files...", style="bold yellow")


def print_plotting(out: str) -> None:
    """Print the message for plotting."""
    filename = f"[bold green]{out}[/]"
    message = f"\n[bold yellow] — Plotting to[/] {filename}[bold yellow]...[/]"
    console.print(Text.from_markup(message))


def print_filename(filename: Path) -> None:
    """Print the filename."""
    message = f"    ‣ [green]{filename}[/]"
    console.print(Text.from_markup(message))


# File validation and error messages for plotting commands


def print_no_files_specified() -> None:
    """Print error message when no files are specified."""
    console.print("Error: No files specified.", style="bold red")


def print_files_not_found_warning() -> None:
    """Print warning message when some files are not found."""
    console.print("\nWarning: Some files were not found:", style="bold yellow")


def print_missing_file(filename: str | Path) -> None:
    """Print a specific missing file."""
    console.print(f"  - {filename}", style="yellow")


def print_no_valid_files_error() -> None:
    """Print error message when no valid files are found."""
    console.print("Error: No valid files found.", style="bold red")


def print_all_files_missing_error() -> None:
    """Print error message when all specified files are missing."""
    console.print("All specified files are missing or inaccessible.", style="red")


def print_check_file_patterns_help() -> None:
    """Print help message to check file patterns."""
    console.print("Please check your file patterns and ensure files exist.", style="red")


def print_processing_files_count(count: int) -> None:
    """Print the number of files being processed."""
    console.print(f"\nProcessing {count} valid file(s).", style="green")


def print_experimental_file_not_found(filename: str) -> None:
    """Print error message when experimental data file is not found."""
    console.print(f"Error: Experimental data file not found: {filename}", style="bold red")


def print_simulated_file_not_found(filename: str) -> None:
    """Print error message when simulated data file is not found."""
    console.print(f"Error: Simulated data file not found: {filename}", style="bold red")


def print_peak_list_file_not_found(filename: str) -> None:
    """Print error message when peak list file is not found."""
    console.print(f"Error: Peak list file not found: {filename}", style="bold red")


def print_data_loading_error(error: Exception) -> None:
    """Print error message when data files cannot be loaded."""
    console.print(f"Error loading data files: {error}", style="bold red")


def print_data_shape_mismatch_error() -> None:
    """Print error message when data shapes do not match."""
    console.print(
        "Error: Data shapes do not match between experimental and simulated data",
        style="bold red",
    )


def print_performance_summary(
    total_time: float,
    n_items: int,
    item_name: str = "items",
    successful: int | None = None,
) -> None:
    """Print a performance summary table.

    Args:
        total_time: Total elapsed time in seconds
        n_items: Number of items processed
        item_name: Name of items being processed
        successful: Optional number of successful items
    """
    from datetime import timedelta

    table = Table(title="⏱️  Performance Summary", show_header=False, box=None)
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="green")

    table.add_row(f"Total {item_name}", str(n_items))
    if successful is not None:
        success_rate = (successful / n_items * 100) if n_items > 0 else 0
        table.add_row("Successful", f"{successful} ({success_rate:.1f}%)")
        if successful < n_items:
            failed = n_items - successful
            table.add_row("Failed", f"[red]{failed}[/]")

    # Format time nicely
    td = timedelta(seconds=total_time)
    if total_time < 60:
        time_str = f"{total_time:.2f}s"
    elif total_time < 3600:
        time_str = f"{td.seconds // 60}m {td.seconds % 60}s"
    else:
        time_str = str(td)

    table.add_row("Total time", time_str)

    if n_items > 0:
        avg_time = total_time / n_items
        if avg_time < 1:
            avg_str = f"{avg_time * 1000:.0f}ms"
        else:
            avg_str = f"{avg_time:.3f}s"
        table.add_row(f"Average per {item_name.rstrip('s')}", avg_str)

    console.print()
    console.print(table)
    console.print()


def print_next_steps(steps: list[str]) -> None:
    """Print suggested next steps for the user.

    Args:
        steps: List of suggested commands or actions
    """
    console.print("\n[bold cyan]📋 Next steps:[/]")
    for i, step in enumerate(steps, 1):
        console.print(f"  {i}. {step}")
    console.print()
