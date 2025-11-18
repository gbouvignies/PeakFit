"""Implementation of the plot command."""

from pathlib import Path

from rich.console import Console

console = Console()


def run_plot(
    results: Path,
    spectrum: Path | None,
    output: Path | None,
    show: bool,
    plot_type: str,
) -> None:
    """Run the plotting command.

    Args:
        results: Path to results directory or file.
        spectrum: Optional path to spectrum for overlay.
        output: Optional output file path.
        show: Whether to display plots interactively.
        plot_type: Type of plot to generate.
    """
    console.print(f"[bold]Generating {plot_type} plots...[/bold]\n")

    if plot_type == "intensity":
        _plot_intensity(results, output, show)
    elif plot_type == "cest":
        _plot_cest(results, output, show)
    elif plot_type == "cpmg":
        _plot_cpmg(results, output, show)
    elif plot_type == "spectra":
        if spectrum is None:
            console.print("[red]Error:[/red] --spectrum is required for spectra plots")
            raise SystemExit(1)
        _plot_spectra(results, spectrum, show)
    else:
        console.print(f"[red]Error:[/red] Unknown plot type: {plot_type}")
        raise SystemExit(1)


def _plot_intensity(results: Path, output: Path | None, show: bool) -> None:
    """Generate intensity profile plots."""
    from peakfit.plotting.plots.intensity import print_plotting

    if results.is_dir():
        files = list(results.glob("*.out"))
    else:
        files = [results]

    if not files:
        console.print("[yellow]Warning:[/yellow] No result files found")
        return

    console.print(f"Found {len(files)} result files")

    # Use existing plotting function
    import matplotlib.pyplot as plt

    for file in files:
        print_plotting(str(file), output_file=output)

    if show:
        plt.show()


def _plot_cest(results: Path, output: Path | None, show: bool) -> None:
    """Generate CEST plots."""
    console.print("[yellow]CEST plotting - use peakfit-plot cest for full functionality[/yellow]")
    # Placeholder - integrate with existing cest.py module


def _plot_cpmg(results: Path, output: Path | None, show: bool) -> None:
    """Generate CPMG plots."""
    console.print("[yellow]CPMG plotting - use peakfit-plot cpmg for full functionality[/yellow]")
    # Placeholder - integrate with existing cpmg.py module


def _plot_spectra(results: Path, spectrum: Path, show: bool) -> None:
    """Launch interactive spectra viewer."""
    console.print("[yellow]Launching interactive spectra viewer...[/yellow]")

    # Import and launch the PyQt5 viewer
    from peakfit.plotting.plots.spectra import main as spectra_main

    import sys

    sys.argv = ["peakfit", str(spectrum)]
    if results.is_dir():
        sim_path = results / f"simulated.ft{2}"
        if sim_path.exists():
            sys.argv.extend(["--simulated", str(sim_path)])

    spectra_main()
