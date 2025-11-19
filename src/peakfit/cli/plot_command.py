"""Implementation of the plot command - fully integrated plotting for all types."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from rich.console import Console

console = Console()

# Maximum number of plots to display interactively (to avoid opening hundreds of windows)
MAX_DISPLAY_PLOTS = 10


def _get_result_files(results: Path, extension: str = "*.out") -> list[Path]:
    """Get result files from path."""
    if results.is_dir():
        files = sorted(results.glob(extension))
    elif results.is_file():
        files = [results]
    else:
        files = []

    if not files:
        console.print(f"[yellow]Warning:[/yellow] No {extension} files found in {results}")
        return []

    console.print(f"[green]Found {len(files)} result files[/green]")
    return files


def _save_figure_to_pdf(pdf: PdfPages, fig: Figure) -> None:
    """Save a single figure to PDF and close it."""
    pdf.savefig(fig)
    plt.close(fig)


# ==================== INTENSITY PLOTTING ====================


def _make_intensity_figure(name: str, data: np.ndarray) -> Figure:
    """Create intensity profile plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(data["xlabel"], data["intensity"], yerr=data["error"], fmt=".", markersize=8)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Intensity", fontsize=11)
    ax.set_xlabel("Index", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_intensity_profiles(results: Path, output: Path | None, show: bool) -> None:
    """Generate intensity profile plots."""
    console.print("[bold]Generating intensity profile plots...[/bold]\n")

    files = _get_result_files(results, "*.out")
    if not files:
        return

    output_path = output or Path("intensity_profiles.pdf")
    console.print(f"[green]Saving plots to:[/green] {output_path}")

    # Limit interactive display to avoid opening hundreds of windows
    if show and len(files) > MAX_DISPLAY_PLOTS:
        console.print(f"[yellow]Note:[/yellow] Displaying only first {MAX_DISPLAY_PLOTS} of {len(files)} plots")
        console.print(f"       All plots are saved to {output_path}")

    plot_data_for_display = [] if show else None

    with PdfPages(output_path) as pdf:
        for idx, file in enumerate(files):
            try:
                data = np.genfromtxt(file, dtype=None, names=("xlabel", "intensity", "error"))
                fig = _make_intensity_figure(file.stem, data)
                _save_figure_to_pdf(pdf, fig)

                if show and idx < MAX_DISPLAY_PLOTS:
                    # Store data for recreating figure later
                    plot_data_for_display.append((file.stem, data))

            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Failed to plot {file.name}: {e}")

    if show and plot_data_for_display:
        for name, data in plot_data_for_display:
            fig = _make_intensity_figure(name, data)
            fig.show()
        plt.show()


# ==================== CEST PLOTTING ====================


def _make_cest_figure(name: str, offset: np.ndarray, intensity: np.ndarray, error: np.ndarray) -> Figure:
    """Create CEST profile plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(offset, intensity, yerr=error, fmt=".", markersize=8, capsize=3)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$B_1$ offset (Hz)", fontsize=11)
    ax.set_ylabel(r"$I/I_0$", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def plot_cest_profiles(results: Path, output: Path | None, show: bool, ref_points: list[int]) -> None:
    """Generate CEST plots."""
    console.print("[bold]Generating CEST profile plots...[/bold]\n")

    files = _get_result_files(results, "*.out")
    if not files:
        return

    THRESHOLD = 1e4  # Threshold for automatic reference selection
    output_path = output or Path("cest_profiles.pdf")
    console.print(f"[green]Saving plots to:[/green] {output_path}")

    # Limit interactive display to avoid opening hundreds of windows
    if show and len(files) > MAX_DISPLAY_PLOTS:
        console.print(f"[yellow]Note:[/yellow] Displaying only first {MAX_DISPLAY_PLOTS} of {len(files)} plots")
        console.print(f"       All plots are saved to {output_path}")

    plot_data_for_display = [] if show else None
    plots_saved = 0

    with PdfPages(output_path) as pdf:
        for file in files:
            try:
                offset, intensity, error = np.loadtxt(file, unpack=True)

                # Determine reference points
                if ref_points == [-1]:
                    # Automatic: use points with |offset| >= threshold
                    ref = np.abs(offset) >= THRESHOLD
                else:
                    # Manual: use specified indices
                    ref = np.zeros_like(offset, dtype=bool)
                    for idx in ref_points:
                        if 0 <= idx < len(offset):
                            ref[idx] = True

                if not np.any(ref):
                    console.print(f"[yellow]Warning:[/yellow] No reference points found for {file.name}")
                    continue

                # Normalize by reference intensity
                intensity_ref = np.mean(intensity[ref])
                offset_norm = offset[~ref]
                intensity_norm = intensity[~ref] / intensity_ref
                error_norm = error[~ref] / np.abs(intensity_ref)

                fig = _make_cest_figure(file.stem, offset_norm, intensity_norm, error_norm)
                _save_figure_to_pdf(pdf, fig)

                if show and plots_saved < MAX_DISPLAY_PLOTS:
                    # Store data for recreating figure later
                    plot_data_for_display.append((file.stem, offset_norm, intensity_norm, error_norm))

                plots_saved += 1

            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Failed to plot {file.name}: {e}")

    if show and plot_data_for_display:
        for name, offset_norm, intensity_norm, error_norm in plot_data_for_display:
            fig = _make_cest_figure(name, offset_norm, intensity_norm, error_norm)
            fig.show()
        plt.show()


# ==================== CPMG PLOTTING ====================


def _ncyc_to_nu_cpmg(ncyc: np.ndarray, time_t2: float) -> np.ndarray:
    """Convert ncyc values to nu_CPMG values."""
    return np.where(ncyc > 0, ncyc / time_t2, 0.5 / time_t2)


def _intensity_to_r2eff(
    intensity: np.ndarray,
    intensity_ref: np.ndarray | float,
    time_t2: float
) -> np.ndarray:
    """Convert intensity values to R2 effective values."""
    return -np.log(intensity / intensity_ref) / time_t2


def _make_ensemble(data: np.ndarray, size: int = 1000) -> np.ndarray:
    """Generate ensemble of intensity values for error estimation."""
    rng = np.random.default_rng()
    return data["intensity"] + data["error"] * rng.standard_normal((size, len(data["intensity"])))


def _make_cpmg_figure(
    name: str,
    nu_cpmg: np.ndarray,
    r2_exp: np.ndarray,
    r2_err_down: np.ndarray,
    r2_err_up: np.ndarray,
) -> Figure:
    """Create CPMG relaxation dispersion plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(nu_cpmg, r2_exp, yerr=(r2_err_down, r2_err_up), fmt="o", markersize=8, capsize=3)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$\nu_{CPMG}$ (Hz)", fontsize=11)
    ax.set_ylabel(r"$R_{2,\mathrm{eff}}$ (s$^{-1}$)", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_cpmg_profiles(results: Path, output: Path | None, show: bool, time_t2: float) -> None:
    """Generate CPMG relaxation dispersion plots."""
    console.print("[bold]Generating CPMG relaxation dispersion plots...[/bold]\n")

    files = _get_result_files(results, "*.out")
    if not files:
        return

    output_path = output or Path("cpmg_profiles.pdf")
    console.print(f"[green]Saving plots to:[/green] {output_path}")

    # Limit interactive display to avoid opening hundreds of windows
    if show and len(files) > MAX_DISPLAY_PLOTS:
        console.print(f"[yellow]Note:[/yellow] Displaying only first {MAX_DISPLAY_PLOTS} of {len(files)} plots")
        console.print(f"       All plots are saved to {output_path}")

    plot_data_for_display = [] if show else None
    plots_saved = 0

    with PdfPages(output_path) as pdf:
        for file in files:
            try:
                data = np.loadtxt(
                    file,
                    dtype={"names": ("ncyc", "intensity", "error"), "formats": ("i4", "f8", "f8")},
                )

                # Separate reference (ncyc=0) and CPMG data
                data_ref = data[data["ncyc"] == 0]
                data_cpmg = data[data["ncyc"] != 0]

                if len(data_ref) == 0:
                    console.print(f"[yellow]Warning:[/yellow] No reference point (ncyc=0) in {file.name}")
                    continue

                # Calculate reference intensity
                intensity_ref = float(np.mean(data_ref["intensity"]))
                error_ref = np.mean(data_ref["error"]) / np.sqrt(len(data_ref))

                # Convert to CPMG frequency and R2eff
                nu_cpmg = _ncyc_to_nu_cpmg(data_cpmg["ncyc"], time_t2)
                r2_exp = _intensity_to_r2eff(data_cpmg["intensity"], intensity_ref, time_t2)

                # Bootstrap error estimation
                data_ref_ens = np.array(
                    [(intensity_ref, error_ref)],
                    dtype=[("intensity", float), ("error", float)]
                )
                r2_ensemble = _intensity_to_r2eff(
                    _make_ensemble(data_cpmg),
                    _make_ensemble(data_ref_ens),
                    time_t2
                )
                r2_err_down, r2_err_up = np.abs(
                    np.percentile(r2_ensemble, [15.9, 84.1], axis=0) - r2_exp
                )

                fig = _make_cpmg_figure(file.stem, nu_cpmg, r2_exp, r2_err_down, r2_err_up)
                _save_figure_to_pdf(pdf, fig)

                if show and plots_saved < MAX_DISPLAY_PLOTS:
                    # Store data for recreating figure later
                    plot_data_for_display.append((file.stem, nu_cpmg, r2_exp, r2_err_down, r2_err_up))

                plots_saved += 1

            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Failed to plot {file.name}: {e}")

    if show and plot_data_for_display:
        for name, nu_cpmg, r2_exp, r2_err_down, r2_err_up in plot_data_for_display:
            fig = _make_cpmg_figure(name, nu_cpmg, r2_exp, r2_err_down, r2_err_up)
            fig.show()
        plt.show()


# ==================== SPECTRA VIEWER ====================


def plot_spectra_viewer(results: Path, spectrum: Path) -> None:
    """Launch interactive spectra viewer (PyQt5)."""
    console.print("[green]Launching interactive spectra viewer...[/green]")

    try:
        import sys

        from peakfit.plotting.plots.spectra import main as spectra_main

        # Build arguments for the viewer
        sys.argv = ["peakfit", str(spectrum)]

        # Add simulated spectrum if available
        if results.is_dir():
            # Try both ft2 and ft3
            for dim in [2, 3]:
                sim_path = results / f"simulated.ft{dim}"
                if sim_path.exists():
                    sys.argv.extend(["--sim", str(sim_path)])
                    console.print(f"[green]Loading simulated spectrum:[/green] {sim_path.name}")
                    break

        # Launch the viewer
        spectra_main()

    except ImportError as e:
        console.print(f"[red]Error:[/red] PyQt5 not available: {e}")
        console.print("[yellow]Install with:[/yellow] pip install 'peakfit[gui]'")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to launch spectra viewer: {e}")
        raise SystemExit(1)
