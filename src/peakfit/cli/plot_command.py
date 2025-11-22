"""Implementation of the plot command - fully integrated plotting for all types."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from peakfit.ui import PeakFitUI as ui, console

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
        ui.warning(f"No {extension} files found in {results}")
        return []

    ui.success(f"Found {len(files)} result files")
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


def plot_intensity_profiles(
    results: Path, output: Path | None, show: bool, verbose: bool = False
) -> None:
    """Generate intensity profile plots."""
    import time

    # Show banner based on verbosity
    ui.show_banner(verbose)

    ui.show_header("Generating Intensity Profile Plots")

    files = _get_result_files(results, "*.out")
    if not files:
        ui.warning(f"No .out files found in {results}")
        return

    output_path = output or Path("intensity_profiles.pdf")
    ui.spacer()
    ui.success(f"Saving plots to: [path]{output_path}[/path]")

    # Limit interactive display to avoid opening hundreds of windows
    if show and len(files) > MAX_DISPLAY_PLOTS:
        ui.info(f"Displaying only first {MAX_DISPLAY_PLOTS} of {len(files)} plots")
        console.print(f"       [dim]All plots are saved to {output_path}[/dim]")

    plot_data_for_display = [] if show else None

    start_time = time.time()

    with ui.create_progress() as progress:
        task = progress.add_task("[cyan]Generating plots...", total=len(files))

        with PdfPages(output_path) as pdf:
            for idx, file in enumerate(files):
                try:
                    data = np.genfromtxt(file, dtype=None, names=("xlabel", "intensity", "error"))
                    fig = _make_intensity_figure(file.stem, data)
                    _save_figure_to_pdf(pdf, fig)

                    if show and idx < MAX_DISPLAY_PLOTS:
                        # Store data for recreating figure later
                        plot_data_for_display.append((file.stem, data))

                    progress.update(task, advance=1)

                except Exception as e:
                    ui.warning(f"Failed to plot {file.name}: {e}")
                    progress.update(task, advance=1)

    plot_time = time.time() - start_time

    # Summary
    file_size = output_path.stat().st_size / 1024 / 1024  # MB
    ui.spacer()
    summary_table = ui.create_table("Plot Summary")
    summary_table.add_column("Item", style="cyan")
    summary_table.add_column("Value", style="green", justify="right")

    summary_table.add_row("PDF file", str(output_path.name))
    summary_table.add_row("Total plots", str(len(files)))
    summary_table.add_row("File size", f"{file_size:.1f} MB")
    summary_table.add_row("Generation time", f"{plot_time:.1f}s")

    console.print(summary_table)
    ui.spacer()
    ui.success("Plots saved successfully!")

    # Next steps
    ui.print_next_steps(
        [
            f"Open PDF: [cyan]open {output_path}[/cyan]",
            f"Plot CEST profiles: [cyan]peakfit plot cest {results}/[/cyan]",
            f"Interactive viewer: [cyan]peakfit plot spectra {results}/ --spectrum SPECTRUM.ft2[/cyan]",
        ]
    )

    if show and plot_data_for_display:
        for name, data in plot_data_for_display:
            fig = _make_intensity_figure(name, data)
            fig.show()
        plt.show()


# ==================== CEST PLOTTING ====================


def _make_cest_figure(
    name: str, offset: np.ndarray, intensity: np.ndarray, error: np.ndarray
) -> Figure:
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


def plot_cest_profiles(
    results: Path, output: Path | None, show: bool, ref_points: list[int], verbose: bool = False
) -> None:
    """Generate CEST plots."""
    # Show banner based on verbosity
    ui.show_banner(verbose)

    ui.show_header("Generating CEST Profile Plots")

    files = _get_result_files(results, "*.out")
    if not files:
        return

    threshold = 1e4  # Threshold for automatic reference selection
    output_path = output or Path("cest_profiles.pdf")
    ui.success(f"Saving plots to: [path]{output_path}[/path]")

    # Limit interactive display to avoid opening hundreds of windows
    if show and len(files) > MAX_DISPLAY_PLOTS:
        ui.info(f"Displaying only first {MAX_DISPLAY_PLOTS} of {len(files)} plots")
        console.print(f"       [dim]All plots are saved to {output_path}[/dim]")

    plot_data_for_display = [] if show else None
    plots_saved = 0

    with PdfPages(output_path) as pdf:
        for file in files:
            try:
                offset, intensity, error = np.loadtxt(file, unpack=True)

                # Determine reference points
                if ref_points == [-1]:
                    # Automatic: use points with |offset| >= threshold
                    ref = np.abs(offset) >= threshold
                else:
                    # Manual: use specified indices
                    ref = np.zeros_like(offset, dtype=bool)
                    for idx in ref_points:
                        if 0 <= idx < len(offset):
                            ref[idx] = True

                if not np.any(ref):
                    ui.warning(f"No reference points found for {file.name}")
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
                    plot_data_for_display.append(
                        (file.stem, offset_norm, intensity_norm, error_norm)
                    )

                plots_saved += 1

            except Exception as e:
                ui.warning(f"Failed to plot {file.name}: {e}")

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
    intensity: np.ndarray, intensity_ref: np.ndarray | float, time_t2: float
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


def plot_cpmg_profiles(
    results: Path, output: Path | None, show: bool, time_t2: float, verbose: bool = False
) -> None:
    """Generate CPMG relaxation dispersion plots."""
    # Show banner based on verbosity
    ui.show_banner(verbose)

    ui.show_header("Generating CPMG Relaxation Dispersion Plots")

    files = _get_result_files(results, "*.out")
    if not files:
        return

    output_path = output or Path("cpmg_profiles.pdf")
    ui.success(f"Saving plots to: [path]{output_path}[/path]")

    # Limit interactive display to avoid opening hundreds of windows
    if show and len(files) > MAX_DISPLAY_PLOTS:
        ui.info(f"Displaying only first {MAX_DISPLAY_PLOTS} of {len(files)} plots")
        console.print(f"       [dim]All plots are saved to {output_path}[/dim]")

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
                    ui.warning(f"No reference point (ncyc=0) in {file.name}")
                    continue

                # Calculate reference intensity
                intensity_ref = float(np.mean(data_ref["intensity"]))
                error_ref = np.mean(data_ref["error"]) / np.sqrt(len(data_ref))

                # Convert to CPMG frequency and R2eff
                nu_cpmg = _ncyc_to_nu_cpmg(data_cpmg["ncyc"], time_t2)
                r2_exp = _intensity_to_r2eff(data_cpmg["intensity"], intensity_ref, time_t2)

                # Bootstrap error estimation
                data_ref_ens = np.array(
                    [(intensity_ref, error_ref)], dtype=[("intensity", float), ("error", float)]
                )
                r2_ensemble = _intensity_to_r2eff(
                    _make_ensemble(data_cpmg), _make_ensemble(data_ref_ens), time_t2
                )
                r2_err_down, r2_err_up = np.abs(
                    np.percentile(r2_ensemble, [15.9, 84.1], axis=0) - r2_exp
                )

                fig = _make_cpmg_figure(file.stem, nu_cpmg, r2_exp, r2_err_down, r2_err_up)
                _save_figure_to_pdf(pdf, fig)

                if show and plots_saved < MAX_DISPLAY_PLOTS:
                    # Store data for recreating figure later
                    plot_data_for_display.append(
                        (file.stem, nu_cpmg, r2_exp, r2_err_down, r2_err_up)
                    )

                plots_saved += 1

            except Exception as e:
                ui.warning(f"Failed to plot {file.name}: {e}")

    if show and plot_data_for_display:
        for name, nu_cpmg, r2_exp, r2_err_down, r2_err_up in plot_data_for_display:
            fig = _make_cpmg_figure(name, nu_cpmg, r2_exp, r2_err_down, r2_err_up)
            fig.show()
        plt.show()


# ==================== SPECTRA VIEWER ====================


def plot_spectra_viewer(results: Path, spectrum: Path, verbose: bool = False) -> None:
    """Launch interactive spectra viewer (PyQt5)."""
    # Show banner based on verbosity
    ui.show_banner(verbose)

    ui.info("Launching interactive spectra viewer...")

    try:
        import sys

        from peakfit.plotting.plots.spectra import main as spectra_main

        # Build arguments for the viewer
        sys.argv = ["peakfit", str(spectrum)]

        # Add simulated spectrum if available
        sim_found = False
        if results.is_dir():
            # Try both ft2 and ft3
            for dim in [2, 3]:
                sim_path = results / f"simulated.ft{dim}"
                if sim_path.exists():
                    sys.argv.extend(["--sim", str(sim_path)])
                    ui.success(f"Loading simulated spectrum: {sim_path.name}")
                    sim_found = True
                    break

        if not sim_found:
            ui.warning("No simulated spectrum found in results directory")
            ui.info("Viewer requires both experimental and simulated spectra")
            raise SystemExit(1)

        # Add peak list if available
        if results.is_dir():
            peak_list_path = results / "shifts.list"
            if peak_list_path.exists():
                sys.argv.extend(["--peak-list", str(peak_list_path)])
                ui.success(f"Loading peak list: {peak_list_path.name}")

        # Launch the viewer
        spectra_main()

    except ImportError as e:
        ui.error(f"PyQt5 not available: {e}")
        ui.info("Install with: [code]pip install 'peakfit[gui]'[/code]")
        raise SystemExit(1) from e
    except Exception as e:
        ui.error(f"Failed to launch spectra viewer: {e}")
        raise SystemExit(1) from e


# ==================== MCMC DIAGNOSTICS PLOTTING ====================


def plot_mcmc_diagnostics(
    results: Path,
    output: Path | None,
    peaks: list[str] | None,
    verbose: bool = False,
) -> None:
    """Generate MCMC diagnostic plots from saved chain data."""
    import pickle

    # Show banner based on verbosity
    ui.show_banner(verbose)

    ui.show_header("Generating MCMC Diagnostic Plots")

    # Load MCMC chain data
    mcmc_file = results / ".mcmc_chains.pkl"
    if not mcmc_file.exists():
        ui.error(f"No MCMC chain data found in {results}")
        ui.info("Run 'peakfit analyze mcmc' first to generate MCMC samples")
        raise SystemExit(1)

    ui.success(f"Loading MCMC data from: [path]{mcmc_file}[/path]")

    # Note: pickle.load is safe here as we control the file creation
    with mcmc_file.open("rb") as f:
        mcmc_data = pickle.load(f)

    # Filter peaks if specified
    if peaks is not None:
        peak_set = set(peaks)
        mcmc_data = [d for d in mcmc_data if any(p in peak_set for p in d["peak_names"])]
        if not mcmc_data:
            ui.error(f"No MCMC data found for peaks: {peaks}")
            raise SystemExit(1)

    ui.success(f"Found MCMC data for {len(mcmc_data)} cluster(s)")

    output_path = output or Path("mcmc_diagnostics.pdf")
    ui.success(f"Saving diagnostic plots to: [path]{output_path}[/path]")

    # Import plotting functions
    from peakfit.diagnostics import save_diagnostic_plots

    # Generate plots for each cluster
    with PdfPages(output_path) as pdf:
        for i, data in enumerate(mcmc_data):
            peak_names = data["peak_names"]
            chains = data["chains"]  # Shape: (n_walkers, n_steps, n_params)
            parameter_names = data["parameter_names"]
            burn_in = data.get("burn_in", 0)
            diagnostics = data.get("diagnostics", None)
            best_fit_values = data.get("best_fit_values", None)

            ui.info(f"[cyan]Cluster {i + 1}/{len(mcmc_data)}:[/cyan] {', '.join(peak_names)}")

            # Generate all diagnostic plots
            from peakfit.diagnostics import (
                plot_autocorrelation,
                plot_corner,
                plot_trace,
            )

            # Flatten chains for corner plot
            samples_flat = chains.reshape(-1, chains.shape[2])

            # Page 1: Trace plots
            fig_trace = plot_trace(chains, parameter_names, burn_in, diagnostics)
            pdf.savefig(fig_trace, bbox_inches="tight")
            plt.close(fig_trace)

            # Page 2: Corner plot
            fig_corner = plot_corner(samples_flat, parameter_names, best_fit_values)
            pdf.savefig(fig_corner, bbox_inches="tight")
            plt.close(fig_corner)

            # Page 3: Autocorrelation plots
            fig_autocorr = plot_autocorrelation(chains, parameter_names)
            pdf.savefig(fig_autocorr, bbox_inches="tight")
            plt.close(fig_autocorr)

    ui.success(f"Diagnostic plots saved to: [path]{output_path}[/path]")

    # Summary
    file_size = output_path.stat().st_size / 1024 / 1024  # MB
    console.print()
    console.print(f"[bold]Summary:[/bold]")
    console.print(f"  • Clusters plotted: {len(mcmc_data)}")
    console.print(f"  • File size: {file_size:.1f} MB")
    console.print(f"  • Output: [path]{output_path}[/path]")

    ui.print_next_steps(
        [
            f"Open plots: [cyan]open {output_path}[/cyan]",
            "Review convergence: Check R-hat ≤ 1.01 in trace plots",
            "Inspect correlations: Look for patterns in corner plots",
        ]
    )
