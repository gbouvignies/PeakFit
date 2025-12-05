from __future__ import annotations

import pickle
import warnings as py_warnings
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table

from peakfit.cli._analyze_formatters import (
    print_correlation_matrix,
    print_mcmc_amplitude_table,
    print_mcmc_diagnostics_table,
    print_mcmc_results_table,
)
from peakfit.cli.analysis.shared import _update_output_files, load_fitting_state
from peakfit.core.diagnostics.burnin import format_burnin_report
from peakfit.services.analyze import MCMCAnalysisService, PeaksNotFoundError
from peakfit.services.analyze.formatters import format_mcmc_cluster_result
from peakfit.ui import (
    Verbosity,
    console,
    create_progress,
    error,
    print_next_steps,
    set_verbosity,
    show_standard_header,
    spacer,
    success,
    warning,
)

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.core.domain.cluster import Cluster
    from peakfit.core.domain.peaks import Peak
    from peakfit.core.fitting.parameters import Parameters


def run_mcmc(
    results_dir: Path,
    n_walkers: int = 32,
    n_steps: int = 1000,
    burn_in: int | None = None,
    auto_burnin: bool = True,
    peaks: list[str] | None = None,
    output_file: Path | None = None,
    workers: int = -1,
    verbose: bool = False,
) -> None:
    """Run MCMC uncertainty estimation on fitted results.

    Args:
        results_dir: Path to results directory
        n_walkers: Number of MCMC walkers
        n_steps: Number of MCMC steps
        burn_in: Number of burn-in steps to discard (manual override)
        auto_burnin: Automatically determine burn-in using R-hat monitoring
        peaks: Optional list of peak names to analyze (default: all)
        output_file: Optional output file for results
        workers: Number of parallel workers (-1 = all CPUs, 1 = sequential)
        verbose: Show banner and verbose output
    """
    # Set verbosity and show header
    set_verbosity(Verbosity.VERBOSE if verbose else Verbosity.NORMAL)
    show_standard_header("MCMC Uncertainty Analysis")

    state = load_fitting_state(results_dir, verbose=False)

    # Handle auto vs manual burn-in
    if not auto_burnin and burn_in is None:
        burn_in = 200
        warning("Both auto-burnin and manual burn-in disabled. Using default: 200 steps")

    # Display Consolidated Configuration Panel
    _display_mcmc_config_panel(results_dir, n_walkers, n_steps, burn_in, auto_burnin, peaks, state)

    try:
        with create_progress(transient=True) as progress_bar:
            # Create a task for MCMC sampling
            mcmc_task = progress_bar.add_task("Initializing...", total=n_steps, visible=False)

            def progress_callback(step: int, total: int, description: str = "Sampling...") -> None:
                """Update progress bar."""
                progress_bar.update(
                    mcmc_task, completed=step, total=total, description=description, visible=True
                )

            with py_warnings.catch_warnings(record=True) as caught_warnings:
                py_warnings.simplefilter("always")
                analysis = MCMCAnalysisService.run(
                    state,
                    peaks=peaks,
                    n_walkers=n_walkers,
                    n_steps=n_steps,
                    burn_in=burn_in,
                    auto_burnin=auto_burnin,
                    workers=workers,
                    progress_callback=progress_callback,
                )

            # Process warnings
            for w in caught_warnings:
                if issubclass(w.category, UserWarning) and "R-hat did not converge" in str(
                    w.message
                ):
                    console.print()
                    warning(str(w.message))
                else:
                    # Re-emit other warnings
                    py_warnings.warn_explicit(
                        message=w.message,
                        category=w.category,
                        filename=w.filename,
                        lineno=w.lineno,
                        source=w.source,
                    )
    except PeaksNotFoundError as exc:
        error(str(exc))
        raise SystemExit(1) from exc

    clusters: list[Cluster] = analysis.clusters
    params: Parameters = analysis.params
    all_peaks: list[Peak] = analysis.peaks
    cluster_results = analysis.cluster_results

    console.print()  # Add spacing after progress bar

    all_results = [cr.result for cr in cluster_results]

    for i, cluster_result in enumerate(cluster_results):
        cluster = cluster_result.cluster
        result = cluster_result.result
        peak_names = [p.name for p in cluster.peaks]
        console.print(f"[cyan]Cluster {i + 1}/{len(clusters)}:[/cyan] {', '.join(peak_names)}")

        # Display burn-in determination report
        if result.burn_in_info is not None:
            _display_burnin_report(result.burn_in_info, peak_names, n_steps, n_walkers)

        # Format and display using formatters
        summary = format_mcmc_cluster_result(cluster_result)

        # Display convergence diagnostics
        if result.mcmc_diagnostics is not None:
            print_mcmc_diagnostics_table(summary)

            # Show warnings if any
            warnings = result.mcmc_diagnostics.get_warnings()
            if warnings:
                console.print()
                warning("Convergence issues detected:")
                for warning_msg in warnings[:5]:  # Limit to first 5 warnings
                    console.print(f"  [dim]• {warning_msg}[/dim]")
                if len(warnings) > 5:
                    console.print(f"  [dim]... and {len(warnings) - 5} more warnings[/dim]")
                console.print("")

        # Display results table
        print_mcmc_results_table(summary)

        # Display correlation matrix if there are multiple parameters
        print_correlation_matrix(summary)

        # Display amplitude (intensity) results
        print_mcmc_amplitude_table(summary)

        # Update global parameters with MCMC uncertainties

    # Save MCMC chain data for diagnostic plotting
    _save_mcmc_chains(results_dir, all_results, clusters)
    success("Saved MCMC chain data for diagnostic plotting")

    # Save updated parameters to output files
    if output_file is not None:
        _save_mcmc_results(output_file, all_results, clusters)
        success(f"Saved MCMC results to: [path]{output_file}[/path]")

    # Update .out files with new uncertainties
    _update_output_files(results_dir, params, all_peaks)
    success("Updated output files with MCMC uncertainties")

    # Provide next steps
    spacer()
    print_next_steps(
        [
            f"Generate diagnostic plots: [cyan]peakfit plot diagnostics {results_dir}/[/cyan]",
            "Review convergence: Check R-hat ≤ 1.01 and ESS values above",
            "Inspect correlations: Check correlation matrices for parameter dependencies",
        ]
    )


def _save_mcmc_results(output_file: Path, results: list, clusters: list[Cluster]) -> None:
    """Save MCMC results to file.

    All parameters (lineshape and amplitudes) are saved uniformly.
    """
    with output_file.open("w") as f:
        f.write("# MCMC Uncertainty Analysis Results\n")
        f.write("# Parameter  Value  StdErr  CI_68_Low  CI_68_High  CI_95_Low  CI_95_High\n")

        for result, cluster in zip(results, clusters, strict=False):
            peak_names = [p.name for p in cluster.peaks]
            f.write(f"\n# Cluster: {', '.join(peak_names)}\n")

            n_lineshape = getattr(result, "n_lineshape_params", len(result.parameter_names))

            # Write all parameters uniformly
            for i, name in enumerate(result.parameter_names):
                ci_68 = result.confidence_intervals_68[i]
                ci_95 = result.confidence_intervals_95[i]
                val = result.values[i]
                err = result.std_errors[i]

                # Format based on parameter type (lineshape vs amplitude)
                if i < n_lineshape:
                    # Lineshape parameters - fixed point notation
                    f.write(
                        f"{name}  {val:.6f}  {err:.6f}  "
                        f"{ci_68[0]:.6f}  {ci_68[1]:.6f}  {ci_95[0]:.6f}  {ci_95[1]:.6f}\n"
                    )
                else:
                    # Amplitude parameters - scientific notation for large dynamic range
                    f.write(
                        f"{name}  {val:.6e}  {err:.6e}  "
                        f"{ci_68[0]:.6e}  {ci_68[1]:.6e}  {ci_95[0]:.6e}  {ci_95[1]:.6e}\n"
                    )

            # Add correlation matrix (lineshape parameters only)
            if result.correlation_matrix is not None and n_lineshape > 1:
                lineshape_names = result.parameter_names[:n_lineshape]
                f.write(f"\n# Correlation Matrix for Cluster: {', '.join(peak_names)}\n")
                f.write("# (lineshape parameters only)\n")
                f.write("# Rows/Columns: " + "  ".join(lineshape_names) + "\n")
                for i, row in enumerate(result.correlation_matrix):
                    f.write(f"# {lineshape_names[i]:<20s}")
                    for val in row:
                        f.write(f"  {val:7.4f}")
                    f.write("\n")
                f.write("\n")


def _save_mcmc_chains(
    results_dir: Path,
    all_results: list,
    clusters: list[Cluster],
) -> None:
    """Save MCMC chain data for diagnostic plotting.

    Args:
        results_dir: Directory to save chain data
        all_results: List of UncertaintyResult objects
        clusters: List of clusters
    """
    mcmc_data = []

    for result, cluster in zip(all_results, clusters, strict=False):
        if result.mcmc_chains is not None:
            # Get best-fit values
            best_fit_values = result.values

            # Get peak names
            peak_names = [p.name for p in cluster.peaks]

            # Extract burn-in from result (may be adaptive or manual)
            burn_in = result.burn_in_info["burn_in"] if result.burn_in_info else 0

            # Store data for this cluster - unified chains include all parameters
            mcmc_data.append(
                {
                    "peak_names": peak_names,
                    "chains": result.mcmc_chains,  # Unified chains (lineshape + amplitudes)
                    "parameter_names": result.parameter_names,  # All parameter names
                    "burn_in": burn_in,
                    "burn_in_info": result.burn_in_info,
                    "diagnostics": result.mcmc_diagnostics,
                    "best_fit_values": best_fit_values,
                    # Metadata for distinguishing parameter types
                    "n_lineshape_params": result.n_lineshape_params,
                    "amplitude_names": result.amplitude_names,
                    "n_planes": result.n_planes,
                }
            )

    # Save to pickle file
    mcmc_file = results_dir / ".mcmc_chains.pkl"
    # Note: pickle.dump is safe here as we control the data being saved
    with mcmc_file.open("wb") as f:
        pickle.dump(mcmc_data, f)


def _display_burnin_report(
    burn_in_info: dict,
    peak_names: list[str],
    n_steps: int,
    n_walkers: int,
) -> None:
    """Display burn-in determination report."""
    burn_in_used = burn_in_info["burn_in"]
    console.print(f"[bold cyan]Burn-in Determination - {', '.join(peak_names)}[/bold cyan]")

    # Format and display the report
    report = format_burnin_report(
        burn_in_used,
        n_steps,
        n_walkers,
        burn_in_info.get("diagnostics", {}),
    )
    console.print(report)

    # Show validation warning if present
    if burn_in_info.get("validation_warning"):
        console.print()
        warning(burn_in_info["validation_warning"])

    console.print()


def _display_mcmc_config_panel(
    results_dir: Path,
    n_walkers: int,
    n_steps: int,
    burn_in: int | None,
    auto_burnin: bool,
    peaks: list[str] | None,
    state: object,
) -> None:
    """Display the MCMC configuration and state panel."""
    # Settings Table
    settings_grid = Table.grid(padding=(0, 2))
    settings_grid.add_column(style="cyan", justify="right")
    settings_grid.add_column(style="white")

    settings_grid.add_row("Walkers:", str(n_walkers))
    settings_grid.add_row("Steps:", str(n_steps))

    burn_in_str = "[cyan]Auto (R-hat)[/cyan]" if auto_burnin else f"{burn_in} (manual)"
    settings_grid.add_row("Burn-in:", burn_in_str)

    target_str = f"{len(peaks)} specific peaks" if peaks else "All peaks"
    settings_grid.add_row("Target:", target_str)

    # State Table
    state_grid = Table.grid(padding=(0, 2))
    state_grid.add_column(style="green", justify="right")
    state_grid.add_column(style="white")

    state_grid.add_row("Source:", f"[dim]{results_dir.name}[/dim]")
    state_grid.add_row("Clusters:", str(len(state.clusters)))
    state_grid.add_row("Peaks:", str(len(state.peaks)))
    state_grid.add_row("Noise:", f"{state.noise:.2f}")

    # Main Grid
    main_grid = Table.grid(expand=True)
    main_grid.add_column(ratio=1)
    main_grid.add_column(ratio=1)

    main_grid.add_row(
        Panel(settings_grid, title="[bold]Analysis Settings[/bold]", border_style="cyan"),
        Panel(state_grid, title="[bold]Fitting State[/bold]", border_style="green"),
    )

    console.print(main_grid)
    console.print()
