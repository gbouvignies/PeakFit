"""Plot subcommands for PeakFit CLI.

This module contains all plot-related commands extracted from the main app.py.
It creates a Typer sub-application with commands for:
- Intensity profiles
- CEST profiles
- CPMG profiles
- Interactive spectra viewer
- MCMC diagnostics
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from peakfit.ui import (
    console,
    create_progress,
    create_table,
    error,
    info,
    print_next_steps,
    show_banner,
    show_header,
    spacer,
    success,
    warning,
)

# Maximum number of plots to display interactively
MAX_DISPLAY_PLOTS = 10

# Create plot sub-application
plot_app = typer.Typer(
    help="Visualization commands for PeakFit results",
    no_args_is_help=True,
)


# ==================== HELPER FUNCTIONS ====================


def _get_result_files(results: Path, extension: str = "*.out") -> list[Path]:
    """Get result files from path."""
    if results.is_dir():
        files = sorted(results.glob(extension))
    elif results.is_file():
        files = [results]
    else:
        files = []

    if not files:
        warning(f"No {extension} files found in {results}")
        return []

    success(f"Found {len(files)} result files")
    return files


def _save_figure_to_pdf(pdf: PdfPages, fig: Figure) -> None:
    """Save a single figure to PDF and close it."""
    pdf.savefig(fig)
    plt.close(fig)


# ==================== INTENSITY COMMAND ====================


@plot_app.command("intensity")
def plot_intensity(
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to results directory or result file",
            exists=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output PDF file (default: intensity_profiles.pdf)",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    show: Annotated[
        bool,
        typer.Option(
            "--show/--no-show",
            help="Display plots interactively",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show banner and verbose output",
        ),
    ] = False,
) -> None:
    """Plot intensity profiles vs. plane index.

    Creates plots showing peak intensity decay/buildup across all planes in
    pseudo-3D spectra. Useful for visualizing CEST, CPMG, or T1/T2 relaxation data.

    Examples
    --------
      Save all plots to PDF:
        $ peakfit plot intensity Fits/ --output intensity.pdf

      Interactive display (first 10 plots only):
        $ peakfit plot intensity Fits/ --show

      Plot single result file:
        $ peakfit plot intensity Fits/A45N-HN.out --show
    """
    from peakfit.plotting.profiles import make_intensity_figure

    # Show banner based on verbosity
    show_banner(verbose)
    show_header("Generating Intensity Profile Plots")

    files = _get_result_files(results, "*.out")
    if not files:
        warning(f"No .out files found in {results}")
        return

    output_path = output or Path("intensity_profiles.pdf")
    spacer()
    success(f"Saving plots to: [path]{output_path}[/path]")

    # Limit interactive display
    if show and len(files) > MAX_DISPLAY_PLOTS:
        info(f"Displaying only first {MAX_DISPLAY_PLOTS} of {len(files)} plots")
        console.print(f"       [dim]All plots are saved to {output_path}[/dim]")

    plot_data_for_display: list[tuple[str, np.ndarray]] | None = [] if show else None
    start_time = time.time()

    with create_progress() as progress:
        task = progress.add_task("[cyan]Generating plots...", total=len(files))

        with PdfPages(output_path) as pdf:
            for idx, file in enumerate(files):
                try:
                    data = np.genfromtxt(file, dtype=None, names=("xlabel", "intensity", "error"))
                    fig = make_intensity_figure(file.stem, data)
                    _save_figure_to_pdf(pdf, fig)

                    if show and idx < MAX_DISPLAY_PLOTS and plot_data_for_display is not None:
                        plot_data_for_display.append((file.stem, data))

                    progress.update(task, advance=1)
                except (OSError, ValueError, TypeError, RuntimeError) as e:
                    # Narrowed exception handling: file read errors, invalid data, or plotting errors
                    warning(f"Failed to plot {file.name}: {e}")
                    progress.update(task, advance=1)

    plot_time = time.time() - start_time

    # Summary
    file_size = output_path.stat().st_size / 1024 / 1024
    spacer()
    summary_table = create_table("Plot Summary")
    summary_table.add_column("Item", style="cyan")
    summary_table.add_column("Value", style="green", justify="right")
    summary_table.add_row("PDF file", str(output_path.name))
    summary_table.add_row("Total plots", str(len(files)))
    summary_table.add_row("File size", f"{file_size:.1f} MB")
    summary_table.add_row("Generation time", f"{plot_time:.1f}s")
    console.print(summary_table)

    spacer()
    success("Plots saved successfully!")
    print_next_steps([
        f"Open PDF: [cyan]open {output_path}[/cyan]",
        f"Plot CEST profiles: [cyan]peakfit plot cest {results}/[/cyan]",
        f"Interactive viewer: [cyan]peakfit plot spectra {results}/ --spectrum SPECTRUM.ft2[/cyan]",
    ])

    if show and plot_data_for_display:
        for name, data in plot_data_for_display:
            fig = make_intensity_figure(name, data)
            fig.show()
        plt.show()


# ==================== CEST COMMAND ====================


@plot_app.command("cest")
def plot_cest(
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to results directory or result file",
            exists=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output PDF file (default: cest_profiles.pdf)",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    show: Annotated[
        bool,
        typer.Option(
            "--show/--no-show",
            help="Display plots interactively",
        ),
    ] = False,
    ref: Annotated[
        list[int] | None,
        typer.Option(
            "--ref",
            "-r",
            help="Reference point indices (default: auto-detect using |offset| >= 10 kHz)",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show banner and verbose output",
        ),
    ] = False,
) -> None:
    """Plot CEST profiles (normalized intensity vs. B1 offset).

    Chemical Exchange Saturation Transfer (CEST) profiles show normalized peak
    intensities as a function of B1 offset frequency. Reference points (off-resonance)
    are used for normalization.

    By default, reference points are auto-detected as |offset| >= 10 kHz.
    Use --ref to manually specify reference point indices.

    Examples
    --------
      Auto-detect reference points:
        $ peakfit plot cest Fits/ --output cest.pdf

      Manual reference selection (indices 0, 1, 2):
        $ peakfit plot cest Fits/ --ref 0 1 2

      Interactive display (first 10 plots):
        $ peakfit plot cest Fits/ --show

      Combine save and display:
        $ peakfit plot cest Fits/ --ref 0 1 --output my_cest.pdf --show
    """
    from peakfit.plotting.profiles import make_cest_figure

    ref_points = ref or [-1]

    # Show banner based on verbosity
    show_banner(verbose)
    show_header("Generating CEST Profile Plots")

    files = _get_result_files(results, "*.out")
    if not files:
        return

    threshold = 1e4  # Threshold for automatic reference selection
    output_path = output or Path("cest_profiles.pdf")
    success(f"Saving plots to: [path]{output_path}[/path]")

    # Limit interactive display
    if show and len(files) > MAX_DISPLAY_PLOTS:
        info(f"Displaying only first {MAX_DISPLAY_PLOTS} of {len(files)} plots")
        console.print(f"       [dim]All plots are saved to {output_path}[/dim]")

    plot_data_for_display: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] | None = (
        [] if show else None
    )
    plots_saved = 0

    with PdfPages(output_path) as pdf:
        for file in files:
            try:
                offset, intensity, error = np.loadtxt(file, unpack=True)

                # Determine reference points
                if ref_points == [-1]:
                    ref_mask = np.abs(offset) >= threshold
                else:
                    ref_mask = np.zeros_like(offset, dtype=bool)
                    for idx in ref_points:
                        if 0 <= idx < len(offset):
                            ref_mask[idx] = True

                if not np.any(ref_mask):
                    warning(f"No reference points found for {file.name}")
                    continue

                # Normalize by reference intensity
                intensity_ref = np.mean(intensity[ref_mask])
                offset_norm = offset[~ref_mask]
                intensity_norm = intensity[~ref_mask] / intensity_ref
                error_norm = error[~ref_mask] / np.abs(intensity_ref)

                fig = make_cest_figure(file.stem, offset_norm, intensity_norm, error_norm)
                _save_figure_to_pdf(pdf, fig)

                if show and plots_saved < MAX_DISPLAY_PLOTS and plot_data_for_display is not None:
                    plot_data_for_display.append((
                        file.stem,
                        offset_norm,
                        intensity_norm,
                        error_norm,
                    ))

                plots_saved += 1
            except (OSError, ValueError, TypeError, RuntimeError) as e:
                # Narrowed exception handling: file read issues or invalid data
                warning(f"Failed to plot {file.name}: {e}")

    if show and plot_data_for_display:
        for name, offset_norm, intensity_norm, error_norm in plot_data_for_display:
            fig = make_cest_figure(name, offset_norm, intensity_norm, error_norm)
            fig.show()
        plt.show()


# ==================== CPMG COMMAND ====================


@plot_app.command("cpmg")
def plot_cpmg(
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to results directory or result file",
            exists=True,
            resolve_path=True,
        ),
    ],
    time_t2: Annotated[
        float,
        typer.Option(
            "--time-t2",
            "-t",
            help="T2 relaxation time in seconds (required)",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output PDF file (default: cpmg_profiles.pdf)",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    show: Annotated[
        bool,
        typer.Option(
            "--show/--no-show",
            help="Display plots interactively",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show banner and verbose output",
        ),
    ] = False,
) -> None:
    """Plot CPMG relaxation dispersion (R2eff vs. νCPMG).

    Carr-Purcell-Meiboom-Gill (CPMG) relaxation dispersion experiments probe
    microsecond-millisecond dynamics. This command converts cycle counts to
    CPMG frequencies (νCPMG) and intensities to effective relaxation rates (R2eff).

    The --time-t2 parameter is the constant time delay in the CPMG block (in seconds).
    Common values: 0.02-0.06s for backbone amides.

    Examples
    --------
      Standard CPMG with T2 = 40ms:
        $ peakfit plot cpmg Fits/ --time-t2 0.04

      Save to custom file:
        $ peakfit plot cpmg Fits/ --time-t2 0.04 --output my_cpmg.pdf

      With interactive display (first 10):
        $ peakfit plot cpmg Fits/ --time-t2 0.04 --show

      Different T2 time (60ms):
        $ peakfit plot cpmg Fits/ --time-t2 0.06 --output cpmg_60ms.pdf
    """
    from peakfit.plotting.profiles import (
        intensity_to_r2eff,
        make_cpmg_figure,
        make_intensity_ensemble,
        ncyc_to_nu_cpmg,
    )

    # Show banner based on verbosity
    show_banner(verbose)
    show_header("Generating CPMG Relaxation Dispersion Plots")

    files = _get_result_files(results, "*.out")
    if not files:
        return

    output_path = output or Path("cpmg_profiles.pdf")
    success(f"Saving plots to: [path]{output_path}[/path]")

    # Limit interactive display
    if show and len(files) > MAX_DISPLAY_PLOTS:
        info(f"Displaying only first {MAX_DISPLAY_PLOTS} of {len(files)} plots")
        console.print(f"       [dim]All plots are saved to {output_path}[/dim]")

    plot_data_for_display: (
        list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None
    ) = [] if show else None
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
                    warning(f"No reference point (ncyc=0) in {file.name}")
                    continue

                # Calculate reference intensity
                intensity_ref = float(np.mean(data_ref["intensity"]))
                error_ref = np.mean(data_ref["error"]) / np.sqrt(len(data_ref))

                # Convert to CPMG frequency and R2eff
                nu_cpmg = ncyc_to_nu_cpmg(data_cpmg["ncyc"], time_t2)
                r2_exp = intensity_to_r2eff(data_cpmg["intensity"], intensity_ref, time_t2)

                # Bootstrap error estimation
                data_ref_ens = np.array(
                    [(intensity_ref, error_ref)], dtype=[("intensity", float), ("error", float)]
                )
                r2_ensemble = intensity_to_r2eff(
                    make_intensity_ensemble(data_cpmg),
                    make_intensity_ensemble(data_ref_ens),
                    time_t2,
                )
                r2_err_down, r2_err_up = np.abs(
                    np.percentile(r2_ensemble, [15.9, 84.1], axis=0) - r2_exp
                )

                fig = make_cpmg_figure(file.stem, nu_cpmg, r2_exp, r2_err_down, r2_err_up)
                _save_figure_to_pdf(pdf, fig)

                if show and plots_saved < MAX_DISPLAY_PLOTS and plot_data_for_display is not None:
                    plot_data_for_display.append((
                        file.stem,
                        nu_cpmg,
                        r2_exp,
                        r2_err_down,
                        r2_err_up,
                    ))

                plots_saved += 1
            except (OSError, ValueError, TypeError, RuntimeError) as e:
                # Narrowed exception handling to cover data, file, and computational issues.
                warning(f"Failed to plot {file.name}: {e}")

    if show and plot_data_for_display:
        for name, nu_cpmg, r2_exp, r2_err_down, r2_err_up in plot_data_for_display:
            fig = make_cpmg_figure(name, nu_cpmg, r2_exp, r2_err_down, r2_err_up)
            fig.show()
        plt.show()


# ==================== SPECTRA COMMAND ====================


@plot_app.command("spectra")
def plot_spectra(
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to results directory",
            exists=True,
            resolve_path=True,
        ),
    ],
    spectrum: Annotated[
        Path,
        typer.Option(
            "--spectrum",
            "-s",
            help="Path to experimental spectrum for overlay (required)",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show banner and verbose output",
        ),
    ] = False,
) -> None:
    """Launch interactive spectra viewer (PyQt5).

    Opens a graphical interface showing experimental and simulated spectra
    side-by-side. Allows interactive plane selection, zooming, and comparison
    of fit quality across all planes.

    Requires PyQt5 to be installed. Install with: pip install PyQt5

    Examples
    --------
      Basic usage:
        $ peakfit plot spectra Fits/ --spectrum data.ft2

      Using relative paths:
        $ peakfit plot spectra ./results --spectrum ../data/spectrum.ft2

      Full path specification:
        $ peakfit plot spectra /path/to/Fits --spectrum /path/to/spectrum.ft2
    """
    import sys

    # Show banner based on verbosity
    show_banner(verbose)
    info("Launching interactive spectra viewer...")

    try:
        from peakfit.plotting.spectra import main as spectra_main

        # Build arguments for the viewer
        sys.argv = ["peakfit", str(spectrum)]

        # Add simulated spectrum if available
        sim_found = False
        if results.is_dir():
            for dim in [2, 3]:
                sim_path = results / f"simulated.ft{dim}"
                if sim_path.exists():
                    sys.argv.extend(["--sim", str(sim_path)])
                    success(f"Loading simulated spectrum: {sim_path.name}")
                    sim_found = True
                    break

        if not sim_found:
            warning("No simulated spectrum found in results directory")
            info("Viewer requires both experimental and simulated spectra")
            raise SystemExit(1)

        # Add peak list if available
        if results.is_dir():
            peak_list_path = results / "shifts.list"
            if peak_list_path.exists():
                sys.argv.extend(["--peak-list", str(peak_list_path)])
                success(f"Loading peak list: {peak_list_path.name}")

        # Launch the viewer
        spectra_main()

    except ImportError as e:
        error(f"PyQt5 not available: {e}")
        info("Install with: [code]pip install 'peakfit[gui]'[/code]")
        raise SystemExit(1) from e
    except (OSError, RuntimeError, ValueError) as e:
        # Narrowed failure modes for launching the viewer: system-level errors or runtime failures
        error(f"Failed to launch spectra viewer: {e}")
        raise SystemExit(1) from e


# ==================== DIAGNOSTICS COMMAND ====================


@plot_app.command("diagnostics")
def plot_diagnostics(
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to results directory from 'peakfit analyze mcmc'",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output PDF file (default: mcmc_diagnostics.pdf)",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    peaks: Annotated[
        list[str] | None,
        typer.Option(
            "--peaks",
            help="Peak names to plot (default: all)",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show banner and verbose output",
        ),
    ] = False,
) -> None:
    """Generate MCMC diagnostic plots (trace, corner, autocorrelation).

    Creates comprehensive diagnostic plots from MCMC sampling results to assess:
    - Chain convergence (trace plots)
    - Parameter correlations (corner plots)
    - Mixing efficiency (autocorrelation plots)

    This command requires MCMC results from 'peakfit analyze mcmc' with saved chain data.

    Examples
    --------
      Generate diagnostics for all peaks:
        $ peakfit plot diagnostics Fits/ --output diagnostics.pdf

      Plot specific peaks only:
        $ peakfit plot diagnostics Fits/ --peaks 2N-H 5L-H

      Quick diagnostics with default output:
        $ peakfit plot diagnostics Fits/
    """
    import numpy as np

    from peakfit.plotting.diagnostics import (
        plot_autocorrelation,
        plot_correlation_pairs,
        plot_marginal_distributions,
        plot_trace,
    )

    # Show banner based on verbosity
    show_banner(verbose)
    show_header("Generating MCMC Diagnostic Plots")

    # Load MCMC chain data
    mcmc_file = results / ".mcmc_chains.pkl"
    if not mcmc_file.exists():
        error(f"No MCMC chain data found in {results}")
        info("Run 'peakfit analyze mcmc' first to generate MCMC samples")
        raise SystemExit(1)

    success(f"Loading MCMC data from: [path]{mcmc_file}[/path]")

    with mcmc_file.open("rb") as f:
        mcmc_data = pickle.load(f)

    # Filter peaks if specified
    if peaks is not None:
        peak_set = set(peaks)
        mcmc_data = [d for d in mcmc_data if any(p in peak_set for p in d["peak_names"])]
        if not mcmc_data:
            error(f"No MCMC data found for peaks: {peaks}")
            raise SystemExit(1)

    success(f"Found MCMC data for {len(mcmc_data)} cluster(s)")

    # Generate separate PDF for each cluster
    output_files = []
    for i, data in enumerate(mcmc_data):
        peak_names = data["peak_names"]
        chains = data["chains"]  # Unified chains: (n_walkers, n_steps, n_all_params)
        parameter_names = list(data["parameter_names"])
        burn_in = data.get("burn_in", 0)
        best_fit_values = data.get("best_fit_values", None)
        if best_fit_values is not None:
            best_fit_values = np.array(best_fit_values)

        # Get metadata for parameter type distinction
        n_lineshape = data.get("n_lineshape_params", len(parameter_names))
        n_planes = data.get("n_planes", 1)
        amplitude_peak_names = data.get("amplitude_names", [])
        n_peaks = len(amplitude_peak_names) if amplitude_peak_names else 0

        # Subsample amplitude parameters for plotting (first, middle, last plane)
        # to avoid memory issues with many planes
        if n_peaks > 0 and n_planes > 0 and n_lineshape < len(parameter_names):
            # Determine which planes to show
            if n_planes > 3:
                plane_indices = [0, n_planes // 2, n_planes - 1]
            else:
                plane_indices = list(range(n_planes))

            # Build indices for subsampled amplitude parameters
            # Amplitudes are ordered as: peak0_plane0, peak0_plane1, ..., peak1_plane0, ...
            amp_subsample_indices = []
            for i_peak in range(n_peaks):
                for i_plane in plane_indices:
                    idx = n_lineshape + i_peak * n_planes + i_plane
                    amp_subsample_indices.append(idx)

            # Extract lineshape params (all) + subsampled amplitude params
            lineshape_indices = list(range(n_lineshape))
            plot_indices = lineshape_indices + amp_subsample_indices
            plot_chains = chains[:, :, plot_indices]
            plot_names = [parameter_names[i] for i in plot_indices]
            plot_best_fit = best_fit_values[plot_indices] if best_fit_values is not None else None

            n_amp_total = len(parameter_names) - n_lineshape
            info(
                f"  Subsampled {n_amp_total} amplitude params to {len(amp_subsample_indices)} "
                f"(planes: {plane_indices})"
            )
        else:
            # No amplitudes or subsampling not needed
            plot_chains = chains
            plot_names = parameter_names
            plot_best_fit = best_fit_values

        # Create output filename for this cluster
        if len(mcmc_data) == 1:
            cluster_output = output or Path("mcmc_diagnostics.pdf")
        else:
            peak_label = "_".join(peak_names)
            if output:
                base = output.stem
                suffix = output.suffix
                cluster_output = output.parent / f"{base}_{peak_label}{suffix}"
            else:
                cluster_output = Path(f"mcmc_diagnostics_{peak_label}.pdf")

        info(f"[cyan]Cluster {i + 1}/{len(mcmc_data)}:[/cyan] {", ".join(peak_names)}")
        info(f"  Saving to: [path]{cluster_output}[/path]")

        # Generate plots for this cluster
        with PdfPages(cluster_output) as pdf:
            # Remove burn-in before flattening for marginal/correlation plots
            chains_post_burnin = plot_chains[:, burn_in:, :] if burn_in > 0 else plot_chains
            samples_flat = chains_post_burnin.reshape(-1, chains_post_burnin.shape[2])

            # Page 1: Trace plots (all parameters)
            fig_trace = plot_trace(plot_chains, plot_names, burn_in, diagnostics=None)
            pdf.savefig(fig_trace, bbox_inches="tight")
            plt.close(fig_trace)

            # Pages 2+: Marginal distributions (all parameters)
            figs_marginal = plot_marginal_distributions(
                samples_flat, plot_names, plot_best_fit, diagnostics=None
            )
            for fig in figs_marginal:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # Pages N+: Correlation pairs - ONLY lineshape parameters
            # Amplitudes are computed via linear least-squares and are conditionally
            # independent given lineshape parameters, so their correlations aren't meaningful
            lineshape_chains = chains[:, :, :n_lineshape]
            lineshape_chains_post_burnin = (
                lineshape_chains[:, burn_in:, :] if burn_in > 0 else lineshape_chains
            )
            lineshape_samples_flat = lineshape_chains_post_burnin.reshape(
                -1, lineshape_chains_post_burnin.shape[2]
            )
            lineshape_names = parameter_names[:n_lineshape]
            lineshape_best_fit = (
                best_fit_values[:n_lineshape] if best_fit_values is not None else None
            )
            figs_corr = plot_correlation_pairs(
                lineshape_samples_flat, lineshape_names, lineshape_best_fit, min_correlation=0.5
            )
            if figs_corr:
                for fig in figs_corr:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
            else:
                info(f"  No strong correlations (|r| ≥ 0.5) found for {", ".join(peak_names)}")

            # Autocorrelation plots (all parameters)
            fig_autocorr = plot_autocorrelation(plot_chains, plot_names)
            pdf.savefig(fig_autocorr, bbox_inches="tight")
            plt.close(fig_autocorr)

        output_files.append(cluster_output)
        success(f"  Saved: [path]{cluster_output}[/path]")
        console.print()

    # Summary
    console.print("[bold]Summary:[/bold]")
    console.print(f"  • Clusters plotted: {len(mcmc_data)}")
    console.print(f"  • PDFs generated: {len(output_files)}")
    for out_file in output_files:
        file_size = out_file.stat().st_size / 1024 / 1024
        console.print(f"    - [path]{out_file}[/path] ({file_size:.1f} MB)")

    # Next steps
    if len(output_files) == 1:
        open_cmd = f"open {output_files[0]}"
    else:
        open_cmd = f"open {" ".join(str(f) for f in output_files)}"

    print_next_steps([
        f"Open plots: [cyan]{open_cmd}[/cyan]",
        "Review trace plots: Check R-hat ≤ 1.01 and chain convergence",
        "Inspect marginal distributions: Review parameter posteriors with full names",
        "Check correlations: Look for strongly correlated parameter pairs (|r| ≥ 0.5)",
    ])
