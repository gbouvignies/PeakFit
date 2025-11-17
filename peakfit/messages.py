"""Contain IO messages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from peakfit import __version__
from peakfit.peak import Peak

console = Console(record=True)

LOGO = r"""
   ___           _      ___ _ _
  / _ \___  __ _| | __ / __(_) |_
 / /_)/ _ \/ _` | |/ // _\ | | __|
/ ___/  __/ (_| |   </ /   | | |_
\/    \___|\__,_|_|\_\/    |_|\__|
"""


def print_logo() -> None:
    """Display the logo in the terminal."""
    logo_text = Text(LOGO, style="blue")
    description_text = Text("Perform peak integration in  \npseudo-3D spectra\n\n")
    version_text = Text("Version: ")
    version_number_text = Text(f"{__version__}", style="red")
    all_text = Text.assemble(
        logo_text, description_text, version_text, version_number_text
    )
    panel = Panel.fit(all_text)
    console.print(panel)


def print_message(message: str, style: str) -> None:
    """Print a styled message to the console."""
    console.print(message, style=style)


def print_fitting() -> None:
    """Print the fitting message."""
    print_message("\n — Fitting peaks...", "bold yellow")


def print_peaks(peaks: list[Peak]) -> None:
    """Print the peak names that are being fitted."""
    peak_list = ", ".join(peak.name for peak in peaks)
    message = f"Peak(s): {peak_list}"
    panel = Panel.fit(message, style="green")
    console.print(panel)


def print_segmenting() -> None:
    """Print the segmenting message."""
    print_message(
        "\n — Segmenting the spectra and clustering the peaks...", "bold yellow"
    )


def print_fit_report(result: Any) -> None:
    """Print the fitting report.

    Args:
        result: scipy.optimize.OptimizeResult or similar object
    """
    # Format scipy.optimize.OptimizeResult
    report_lines = ["[[Fit Statistics]]"]

    if hasattr(result, "success"):
        status = "success" if result.success else "failure"
        report_lines.append(f"    fit status:        {status}")

    if hasattr(result, "message"):
        report_lines.append(f"    message:           {result.message}")

    if hasattr(result, "nfev"):
        report_lines.append(f"    # function evals:  {result.nfev}")

    if hasattr(result, "njev"):
        report_lines.append(f"    # jacobian evals:  {result.njev}")

    if hasattr(result, "cost"):
        report_lines.append(f"    final cost:        {result.cost:.6e}")

    if hasattr(result, "optimality"):
        report_lines.append(f"    optimality:        {result.optimality:.6e}")

    if hasattr(result, "x"):
        report_lines.append(f"    # variables:       {len(result.x)}")

    report_text = "\n".join(report_lines)
    console.print("\n", Text(report_text), "\n")


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
    print_message(
        f"\n — Refining the peak parameters ({index}/{refine_nb})...", "bold yellow"
    )


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
