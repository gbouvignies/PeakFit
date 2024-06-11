"""Contain IO messages."""

from __future__ import annotations

from pathlib import Path

from lmfit.minimizer import MinimizerResult
from lmfit.printfuncs import fit_report
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


def print_fit_report(minimizer_result: MinimizerResult) -> None:
    """Print the fitting report."""
    console.print("\n", Text(fit_report(minimizer_result, min_correl=0.5)), "\n")


def export_html(filehtml: Path) -> None:
    """Export console output to an HTML file."""
    filehtml.write_text(console.export_html())


def print_reading_files() -> None:
    """Print the message for reading files."""
    print_message("\n — Reading files...", "bold yellow")


def print_plotting() -> None:
    """Print the message for plotting."""
    filename = "[bold green]profiles.pdf[/]"
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
