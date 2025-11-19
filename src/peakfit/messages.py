"""Contain IO messages."""

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from peakfit import __version__

if TYPE_CHECKING:
    from peakfit.peak import Peak

console = Console(record=True)

LOGO = r"""
   ___           _     _____ _ _
  / _ \ ___  __ _| | __|  ___(_) |_
 / /_)/ _ \/ _` | |/ /| |_  | | __|
/ ___/  __/ (_| |   < |  _| | | |_
\/    \___|\__,_|_|\_\|_|   |_|\__|
"""


def print_logo() -> None:
    """Display the logo in the terminal."""
    logo_text = Text(LOGO, style="bold cyan")
    description_text = Text(
        "Modern NMR Peak Fitting for Pseudo-3D Spectra\n"
        "https://github.com/gbouvignies/PeakFit\n\n",
        style="dim"
    )
    version_text = Text("Version: ", style="dim")
    version_number_text = Text(f"{__version__}", style="bold green")
    all_text = Text.assemble(logo_text, description_text, version_text, version_number_text)
    panel = Panel.fit(all_text, border_style="cyan", title="🎯 PeakFit")
    console.print(panel)


def print_message(message: str, style: str) -> None:
    """Print a styled message to the console."""
    console.print(message, style=style)


def print_fitting() -> None:
    """Print the fitting message."""
    print_message("\n — Fitting peaks...", "bold yellow")


def print_peaks(peaks: list["Peak"]) -> None:
    """Print the peak names that are being fitted."""
    peak_list = ", ".join(peak.name for peak in peaks)
    message = f"Peak(s): {peak_list}"
    panel = Panel.fit(message, style="green")
    console.print(panel)


def print_segmenting() -> None:
    """Print the segmenting message."""
    print_message("\n — Segmenting the spectra and clustering the peaks...", "bold yellow")


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


def print_cluster_summary(cluster_index: int, total_clusters: int, peak_names: list[str]) -> None:
    """Print a summary of the cluster being fitted."""
    peaks_str = ", ".join(peak_names)
    console.print(
        f"\n[bold cyan]Cluster {cluster_index}/{total_clusters}[/] │ Peaks: [green]{peaks_str}[/]"
    )


def print_fitting_progress(current: int, total: int) -> None:
    """Print fitting progress."""
    percentage = (current / total) * 100 if total > 0 else 0
    bar_width = 30
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    console.print(
        f"\r[cyan]Progress[/] │[{bar}] {percentage:.0f}% ({current}/{total})",
        end="",
    )
    if current == total:
        console.print()  # New line when complete


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


def print_parameter_table(params: object) -> None:
    """Print a formatted table of fitted parameters.

    Args:
        params: Parameters object with fitted values
    """
    table = Table(title="Fitted Parameters", show_lines=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_column("Min", style="dim", justify="right")
    table.add_column("Max", style="dim", justify="right")
    table.add_column("Vary", style="yellow", justify="center")

    for name in params:
        param = params[name]
        vary_str = "✓" if param.vary else "✗"
        min_str = f"{param.min:.2f}" if param.min > -1e10 else "-∞"
        max_str = f"{param.max:.2f}" if param.max < 1e10 else "∞"
        table.add_row(name, f"{param.value:.6f}", min_str, max_str, vary_str)

    console.print(table)


def print_optimization_settings(ftol: float, xtol: float, max_nfev: int) -> None:
    """Print optimization settings."""
    console.print(f"[dim]Optimization: ftol={ftol:.0e}, xtol={xtol:.0e}, max_nfev={max_nfev}[/]")


def print_boundary_warning(param_names: list[str]) -> None:
    """Print a warning about parameters at their boundaries.

    Args:
        param_names: List of parameter names at boundaries
    """
    if param_names:
        console.print(f"\n[bold yellow]⚠ Warning:[/] {len(param_names)} parameter(s) at boundary:")
        for name in param_names[:10]:  # Show max 10
            console.print(f"  • [yellow]{name}[/]")
        if len(param_names) > 10:
            console.print(f"  ... and {len(param_names) - 10} more")


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


def print_success_message(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]✓[/] {message}")


def print_warning_message(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]⚠[/] {message}")


def print_error_message(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]✗[/] {message}")


def export_html(filehtml: Path) -> None:
    """Export console output to an HTML file."""
    filehtml.write_text(console.export_html())


def print_reading_files() -> None:
    """Print the message for reading files."""
    print_message("\n — Reading files...", "bold yellow")


def print_plotting(out: str) -> None:
    """Print the message for plotting."""
    filename = f"[bold green]{out}[/]"
    message = f"\n[bold yellow] — Plotting to[/] {filename}[bold yellow]...[/]"
    console.print(Text.from_markup(message))


def print_filename(filename: Path) -> None:
    """Print the filename."""
    message = f"    ‣ [green]{filename}[/]"
    console.print(Text.from_markup(message))


def print_estimated_noise(noise: float) -> None:
    """Print the estimated noise."""
    message = f"\n [bold yellow]— Estimated noise:[/] [bold green]{noise:.2f}[/]"
    console.print(Text.from_markup(message))


def print_writing_spectra() -> None:
    """Print the message for writing the spectra."""
    print_message("\n — Writing the simulated spectra...", "bold yellow")


def print_writing_profiles() -> None:
    """Print the message for writing the profiles."""
    print_message("\n — Writing the profiles...", "bold yellow")


def print_writing_shifts() -> None:
    """Print the message for writing the shifts."""
    print_message("\n — Writing the shifts...", "bold yellow")


def print_refining(index: int, refine_nb: int) -> None:
    """Print the message for refining the peaks."""
    print_message(f"\n — Refining the peak parameters ({index}/{refine_nb})...", "bold yellow")


# File validation and error messages for plotting commands


def print_no_files_specified() -> None:
    """Print error message when no files are specified."""
    print_message("Error: No files specified.", "bold red")


def print_files_not_found_warning() -> None:
    """Print warning message when some files are not found."""
    print_message("\nWarning: Some files were not found:", "bold yellow")


def print_missing_file(filename: str | Path) -> None:
    """Print a specific missing file."""
    print_message(f"  - {filename}", "yellow")


def print_no_valid_files_error() -> None:
    """Print error message when no valid files are found."""
    print_message("Error: No valid files found.", "bold red")


def print_all_files_missing_error() -> None:
    """Print error message when all specified files are missing."""
    print_message("All specified files are missing or inaccessible.", "red")


def print_check_file_patterns_help() -> None:
    """Print help message to check file patterns."""
    print_message("Please check your file patterns and ensure files exist.", "red")


def print_processing_files_count(count: int) -> None:
    """Print the number of files being processed."""
    print_message(f"\nProcessing {count} valid file(s).", "green")


def print_experimental_file_not_found(filename: str) -> None:
    """Print error message when experimental data file is not found."""
    print_message(f"Error: Experimental data file not found: {filename}", "bold red")


def print_simulated_file_not_found(filename: str) -> None:
    """Print error message when simulated data file is not found."""
    print_message(f"Error: Simulated data file not found: {filename}", "bold red")


def print_peak_list_file_not_found(filename: str) -> None:
    """Print error message when peak list file is not found."""
    print_message(f"Error: Peak list file not found: {filename}", "bold red")


def print_data_loading_error(error: Exception) -> None:
    """Print error message when data files cannot be loaded."""
    print_message(f"Error loading data files: {error}", "bold red")


def print_data_shape_mismatch_error() -> None:
    """Print error message when data shapes do not match."""
    print_message(
        "Error: Data shapes do not match between experimental and simulated data",
        "bold red",
    )


def print_file_not_found_with_suggestions(filepath: Path, similar_files: list[Path] | None = None) -> None:
    """Print file not found error with helpful suggestions.

    Args:
        filepath: Path that was not found
        similar_files: Optional list of similar files to suggest
    """
    console.print(f"\n[bold red]✗ Error:[/] File not found: [yellow]{filepath}[/]")

    if similar_files:
        console.print("\n[cyan]Did you mean one of these?[/]")
        for file in similar_files[:5]:  # Show max 5 suggestions
            console.print(f"  • [green]{file}[/]")

    # Show files in current directory
    parent = filepath.parent if filepath.parent.exists() else Path(".")
    if parent.is_dir():
        pattern = f"*{filepath.suffix}" if filepath.suffix else "*"
        matching_files = list(parent.glob(pattern))
        if matching_files and not similar_files:
            console.print(f"\n[cyan]Available {pattern} files in {parent}:[/]")
            for file in matching_files[:10]:
                console.print(f"  • [green]{file.name}[/]")
            if len(matching_files) > 10:
                console.print(f"  [dim]... and {len(matching_files) - 10} more[/]")


def print_auto_detection(parameter: str, value: str, source: str = "spectrum") -> None:
    """Print auto-detected parameter information.

    Args:
        parameter: Name of the parameter
        value: Detected value
        source: Source of detection (e.g., "spectrum", "data")
    """
    console.print(f"[dim]ℹ Auto-detected {parameter} from {source}:[/] [cyan]{value}[/]")


def print_smart_default(option: str, value: str, reason: str) -> None:
    """Print smart default selection message.

    Args:
        option: Option name
        value: Default value being used
        reason: Reason for this default
    """
    console.print(f"[dim]→ Using {option}:[/] [cyan]{value}[/] [dim]({reason})[/]")


def print_confirmation_prompt(message: str) -> bool:
    """Show confirmation prompt and return user's choice.

    Args:
        message: Message to display

    Returns:
        True if user confirms, False otherwise
    """
    from rich.prompt import Confirm
    return Confirm.ask(f"[yellow]{message}[/]")


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
            avg_str = f"{avg_time*1000:.0f}ms"
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
