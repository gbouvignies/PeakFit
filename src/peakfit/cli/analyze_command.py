"""Implementation of the analyze command for uncertainty estimation."""

from pathlib import Path

from rich.table import Table

import numpy as np

from peakfit.cli._analyze_formatters import (
    print_correlation_matrix,
    print_mcmc_amplitude_table,
    print_mcmc_diagnostics_table,
    print_mcmc_results_table,
)
from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak
from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.parameters import Parameters
from peakfit.services.analyze import (
    FittingStateService,
    MCMCAnalysisService,
    NoVaryingParametersError,
    NoVaryingParametersFoundError,
    ParameterMatchError,
    ParameterUncertaintyService,
    PeaksNotFoundError,
    ProfileLikelihoodService,
    ProfileParameterResult,
    StateFileMissingError,
    StateLoadError,
)
from peakfit.services.analyze.formatters import format_mcmc_cluster_result
from peakfit.ui import (
    console,
    error,
    info,
    print_next_steps,
    show_banner,
    show_header,
    spacer,
    success,
    warning,
)


def load_fitting_state(results_dir: Path) -> FittingState:
    """Load fitting state from results directory.

    Args:
        results_dir: Path to results directory containing .peakfit_state.pkl

    Returns
    -------
        FittingState with clusters, params, noise, and peaks
    """
    try:
        loaded_state = FittingStateService.load(results_dir)
    except StateFileMissingError as exc:
        error(f"No fitting state found in {results_dir}")
        info("Run 'peakfit fit' with --save-state (enabled by default)")
        raise SystemExit(1) from exc
    except StateLoadError as exc:  # pragma: no cover - safety guard
        error(str(exc))
        raise SystemExit(1) from exc

    state = loaded_state.state
    state_file = loaded_state.path

    success(f"Loaded fitting state: [path]{state_file}[/path]")
    console.print(f"  Clusters: {len(state.clusters)}")
    console.print(f"  Peaks: {len(state.peaks)}")
    console.print(f"  Parameters: {len(state.params)}")

    return state


def run_mcmc(
    results_dir: Path,
    n_walkers: int = 32,
    n_steps: int = 1000,
    burn_in: int | None = None,
    auto_burnin: bool = True,
    peaks: list[str] | None = None,
    output_file: Path | None = None,
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
        verbose: Show banner and verbose output
    """
    # Show banner based on verbosity
    show_banner(verbose)

    state = load_fitting_state(results_dir)

    # Handle auto vs manual burn-in
    if not auto_burnin and burn_in is None:
        # Neither auto nor manual specified, use default
        burn_in = 200
        warning("Both auto-burnin and manual burn-in disabled. Using default: 200 steps")

    try:
        analysis = MCMCAnalysisService.run(
            state,
            peaks=peaks,
            n_walkers=n_walkers,
            n_steps=n_steps,
            burn_in=burn_in,
            auto_burnin=auto_burnin,
        )
    except PeaksNotFoundError as exc:
        error(str(exc))
        raise SystemExit(1) from exc

    clusters: list[Cluster] = analysis.clusters
    params: Parameters = analysis.params
    all_peaks: list[Peak] = analysis.peaks
    cluster_results = analysis.cluster_results

    if peaks is not None:
        info(f"Analyzing {len(clusters)} cluster(s) for peaks: {peaks}")

    show_header("Running MCMC Uncertainty Estimation")
    console.print(f"  Walkers: {n_walkers}")
    console.print(f"  Steps: {n_steps}")
    if auto_burnin:
        console.print("  Burn-in: [cyan]Auto-determined using R-hat convergence[/cyan]")
    else:
        console.print(f"  Burn-in: {burn_in} (manual)")
    console.print("")

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


def run_profile_likelihood(
    results_dir: Path,
    param_name: str | None = None,
    n_points: int = 20,
    confidence_level: float = 0.95,
    plot: bool = False,
    output_file: Path | None = None,
    verbose: bool = False,
) -> None:
    """Compute profile likelihood confidence interval for parameter(s).

    Args:
        results_dir: Path to results directory
        param_name: Parameter name to profile (or partial match). If None, profiles all parameters.
        n_points: Number of profile points
        confidence_level: Confidence level (0.95 for 95% CI)
        plot: Whether to plot the profile
        output_file: Optional output file for results
        verbose: Show banner and verbose output
    """
    # Show banner based on verbosity
    show_banner(verbose)

    state = load_fitting_state(results_dir)
    try:
        analysis = ProfileLikelihoodService.run(
            state,
            param_name=param_name,
            n_points=n_points,
            confidence_level=confidence_level,
        )
    except NoVaryingParametersError as exc:
        error("No varying parameters found")
        raise SystemExit(1) from exc
    except ParameterMatchError as exc:
        error(str(exc))
        info("Available parameters:")
        for name in exc.available[:20]:
            console.print(f"  {name}")
        if len(exc.available) > 20:
            console.print(f"  ... and {len(exc.available) - 20} more")
        raise SystemExit(1) from exc

    target_params = analysis.target_parameters

    if param_name is None:
        show_header("Computing Profile Likelihood for All Parameters")
        console.print(f"  Parameters to profile: {len(target_params)}")
        console.print(f"  Confidence level: {confidence_level * 100:.0f}%")
        console.print(f"  Points per parameter: {n_points}")
        console.print("  [yellow]This may take a while...[/yellow]")
    else:
        if len(target_params) > 1:
            info(f"Found {len(target_params)} matching parameters:")
            for name in target_params:
                console.print(f"  {name}")
            console.print("")

        show_header(f"Computing Profile Likelihood for {len(target_params)} Parameter(s)")

    console.print(f"  Δχ² threshold: {analysis.delta_chi2:.4f}")
    console.print("")

    missing_set = set(analysis.missing_parameters)
    results_by_name = {result.parameter_name: result for result in analysis.results}
    ordered_results: list[ProfileParameterResult] = []

    for idx, target_param in enumerate(target_params, 1):
        if len(target_params) > 1:
            console.print(f"[cyan]Parameter {idx}/{len(target_params)}: {target_param}[/cyan]")

        if target_param in missing_set:
            warning(f"Parameter '{target_param}' not found in any cluster")
            continue

        result = results_by_name[target_param]
        ordered_results.append(result)

        best_value = result.best_value
        covar_stderr = result.covariance_stderr

        result_table = Table(show_header=False)
        result_table.add_column("", style="dim")
        result_table.add_column("", style="")

        result_table.add_row("Parameter:", target_param)
        result_table.add_row("Best-fit value:", f"{best_value:.6f}")
        result_table.add_row("Covariance stderr:", f"{covar_stderr:.6f}")
        result_table.add_row(
            f"Profile {confidence_level * 100:.0f}% CI:",
            f"[{result.ci_low:.6f}, {result.ci_high:.6f}]",
        )

        if covar_stderr > 0:
            from scipy.stats import norm

            z = norm.ppf((1 + confidence_level) / 2)
            covar_ci_low = best_value - z * covar_stderr
            covar_ci_high = best_value + z * covar_stderr
            result_table.add_row(
                f"Covariance {confidence_level * 100:.0f}% CI:",
                f"[{covar_ci_low:.6f}, {covar_ci_high:.6f}]",
            )

            profile_lower = best_value - result.ci_low
            profile_upper = result.ci_high - best_value
            asymmetry = abs(profile_upper - profile_lower) / (profile_upper + profile_lower) * 200
            if asymmetry > 20:
                result_table.add_row(
                    "Asymmetry:",
                    f"[yellow]{asymmetry:.1f}% (non-linear parameter)[/yellow]",
                )

        console.print(result_table)
        console.print("")

    if plot and ordered_results:
        for result in ordered_results[:10]:
            _plot_profile_likelihood(
                result.parameter_name,
                result.parameter_values,
                result.chi2_values,
                result.ci_low,
                result.ci_high,
                analysis.delta_chi2,
            )

    if output_file is not None:
        _save_all_profile_results(output_file, ordered_results, confidence_level)
        success(f"Saved profile data to: [path]{output_file}[/path]")


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
    # Show banner based on verbosity
    show_banner(verbose)

    state = load_fitting_state(results_dir)
    try:
        analysis = ParameterUncertaintyService.analyze(state)
    except NoVaryingParametersFoundError:
        warning("No varying parameters found")
        return

    show_header("Parameter Uncertainties")
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


def _update_output_files(results_dir: Path, params: Parameters, peaks: list[Peak]) -> None:
    """Update .out files with new uncertainty estimates."""
    for peak in peaks:
        out_file = results_dir / f"{peak.name}.out"
        if out_file.exists():
            # Read existing file
            lines = out_file.read_text().splitlines()

            # Update parameter lines with new stderr
            new_lines = []
            for line in lines:
                if line.startswith("# ") and ":" in line and "±" in line:
                    # Parse parameter line
                    parts = line.split(":")
                    if len(parts) >= 2:
                        param_part = parts[0].strip("# ").strip()
                        # Find matching parameter
                        for shape in peak.shapes:
                            for param_name in shape.param_names:  # type: ignore[attr-defined]
                                if (
                                    param_name.endswith(param_part) or param_part in param_name
                                ) and param_name in params:
                                    value = params[param_name].value
                                    stderr = params[param_name].stderr
                                    shortname = param_part
                                    updated_line = (
                                        f"# {shortname:<10s}: {value:10.5f} ± {stderr:10.5f}"
                                    )
                                    line = updated_line
                                    break
                new_lines.append(line)

            out_file.write_text("\n".join(new_lines))


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


def _save_all_profile_results(
    output_file: Path, results: list[ProfileParameterResult], confidence_level: float
) -> None:
    """Save profile likelihood results for multiple parameters."""
    with output_file.open("w") as f:
        f.write("# Profile Likelihood Results\n")
        f.write(f"# Confidence Level: {confidence_level * 100:.0f}%\n")
        f.write("#\n")

        for result in results:
            f.write(f"\n# Parameter: {result.parameter_name}\n")
            f.write(f"# Best-fit: {result.best_value:.6f}\n")
            f.write(
                f"# {confidence_level * 100:.0f}% CI: [{result.ci_low:.6f}, {result.ci_high:.6f}]\n"
            )
            f.write("# Parameter_Value  Chi_Squared\n")
            for val, chi2 in zip(result.parameter_values, result.chi2_values, strict=True):
                f.write(f"{val:.6f}  {chi2:.6f}\n")
            f.write("\n")


# NOTE: _save_profile_results removed as it was unused; keep per-parameter file generation centralized


def _plot_profile_likelihood(
    param_name: str,
    param_vals: np.ndarray,
    chi2_vals: np.ndarray,
    ci_low: float,
    ci_high: float,
    delta_chi2: float,
) -> None:
    """Plot profile likelihood curve."""
    try:
        import matplotlib.pyplot as plt

        _fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Plot chi-squared curve
        ax.plot(param_vals, chi2_vals, "b-", linewidth=2, label="Profile")

        # Mark best fit
        best_idx = np.argmin(chi2_vals)
        best_val = float(param_vals[best_idx])
        best_chi2 = float(chi2_vals[best_idx])
        ax.axhline(best_chi2, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(best_val, color="gray", linestyle="--", alpha=0.5)

        # Mark confidence interval
        threshold = best_chi2 + float(delta_chi2)
        ax.axhline(threshold, color="r", linestyle="--", alpha=0.7, label=f"Δχ²={delta_chi2:.2f}")
        ax.axvline(ci_low, color="g", linestyle="--", alpha=0.7)
        ax.axvline(ci_high, color="g", linestyle="--", alpha=0.7)
        ax.axvspan(ci_low, ci_high, alpha=0.2, color="green", label="95% CI")

        ax.set_xlabel(param_name)
        ax.set_ylabel("χ²")
        ax.set_title(f"Profile Likelihood for {param_name}")
        ax.legend()
        ax.grid(visible=True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except ImportError:
        warning("matplotlib not available for plotting")


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
    import pickle

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
    from peakfit.core.diagnostics.burnin import format_burnin_report

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
