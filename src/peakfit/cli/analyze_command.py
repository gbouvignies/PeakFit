"""Implementation of the analyze command for uncertainty estimation."""

from pathlib import Path

import numpy as np
from rich.table import Table

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak
from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.parameters import Parameters
from peakfit.services.analyze import (
    FittingStateService,
    MCMCAnalysisService,
    NotEnoughVaryingParametersError,
    NoVaryingParametersError,
    NoVaryingParametersFoundError,
    ParameterCorrelationService,
    ParameterMatchError,
    ParameterUncertaintyService,
    PeaksNotFoundError,
    ProfileLikelihoodService,
    ProfileParameterResult,
    StateFileMissingError,
    StateLoadError,
)
from peakfit.ui import PeakFitUI as ui, console


def load_fitting_state(results_dir: Path) -> FittingState:
    """Load fitting state from results directory.

    Args:
        results_dir: Path to results directory containing .peakfit_state.pkl

    Returns:
        FittingState with clusters, params, noise, and peaks
    """
    try:
        loaded_state = FittingStateService.load(results_dir)
    except StateFileMissingError as exc:
        ui.error(f"No fitting state found in {results_dir}")
        ui.info("Run 'peakfit fit' with --save-state (enabled by default)")
        raise SystemExit(1) from exc
    except StateLoadError as exc:  # pragma: no cover - safety guard
        ui.error(str(exc))
        raise SystemExit(1) from exc

    state = loaded_state.state
    state_file = loaded_state.path

    ui.success(f"Loaded fitting state: [path]{state_file}[/path]")
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
    ui.show_banner(verbose)

    state = load_fitting_state(results_dir)

    # Handle auto vs manual burn-in
    if not auto_burnin and burn_in is None:
        # Neither auto nor manual specified, use default
        burn_in = 200
        ui.warning("Both auto-burnin and manual burn-in disabled. Using default: 200 steps")

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
        ui.error(str(exc))
        raise SystemExit(1) from exc

    clusters: list[Cluster] = analysis.clusters
    params: Parameters = analysis.params
    all_peaks: list[Peak] = analysis.peaks
    cluster_results = analysis.cluster_results

    if peaks is not None:
        ui.info(f"Analyzing {len(clusters)} cluster(s) for peaks: {peaks}")

    ui.show_header("Running MCMC Uncertainty Estimation")
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
            from peakfit.core.diagnostics.burnin import format_burnin_report

            burn_in_used = result.burn_in_info["burn_in"]
            console.print(f"[bold cyan]Burn-in Determination - {', '.join(peak_names)}[/bold cyan]")

            # Format and display the report
            report = format_burnin_report(
                burn_in_used,
                n_steps,
                n_walkers,
                result.burn_in_info.get("diagnostics", {}),
            )
            console.print(report)

            # Show validation warning if present
            if result.burn_in_info.get("validation_warning"):
                console.print()
                ui.warning(result.burn_in_info["validation_warning"])

            console.print()

        # Display convergence diagnostics
        if result.mcmc_diagnostics is not None:
            diag = result.mcmc_diagnostics
            console.print(
                f"[bold cyan]Convergence Diagnostics - {', '.join(peak_names)}[/bold cyan]"
            )
            console.print(f"  Chains: {diag.n_chains}, Samples per chain: {diag.n_samples}")
            console.print(
                "  [dim]BARG Guidelines: R-hat ≤ 1.01 (excellent), "
                "ESS ≥ 10,000 for stable CIs (Kruschke 2021)[/dim]"
            )
            console.print("")

            # Create diagnostics table
            diag_table = Table(show_header=True, header_style="bold cyan")
            diag_table.add_column("Parameter", style="cyan", width=20)
            diag_table.add_column("R-hat", justify="right", width=10)
            diag_table.add_column("ESS_bulk", justify="right", width=14)
            diag_table.add_column("ESS_tail", justify="right", width=14)
            diag_table.add_column("Status", width=15)

            for j, name in enumerate(result.parameter_names):
                rhat = diag.rhat[j]
                ess_bulk = diag.ess_bulk[j]
                ess_tail = diag.ess_tail[j]

                # Determine overall status based on BARG criteria
                # R-hat ≤ 1.01 is excellent, ≤ 1.05 acceptable
                # ESS ≥ 10,000 is BARG-recommended for publication
                if rhat <= 1.01 and ess_bulk >= 10000:
                    status = "[green]✓ Excellent[/green]"
                elif rhat <= 1.01 and ess_bulk >= 100 * diag.n_chains:
                    status = "[green]✓ Good[/green]"
                elif rhat <= 1.05 and ess_bulk >= 100 * diag.n_chains:
                    status = "[cyan]○ Acceptable[/cyan]"
                elif rhat <= 1.05 and ess_bulk >= 10 * diag.n_chains:
                    status = "[yellow]⚠ Marginal[/yellow]"
                else:
                    status = "[red]✗ Poor[/red]"

                # Format R-hat with color coding (stricter is better)
                if rhat <= 1.01:
                    rhat_str = f"[green]{rhat:.4f}[/green]"
                elif rhat <= 1.05:
                    rhat_str = f"[cyan]{rhat:.4f}[/cyan]"
                else:
                    rhat_str = f"[red]{rhat:.4f}[/red]"

                # Format ESS_bulk with percentage toward BARG target (10,000)
                pct_bulk = min(100, (ess_bulk / 10000) * 100)
                if ess_bulk >= 10000:
                    ess_bulk_str = f"[green]{ess_bulk:.0f} (100%)[/green]"
                elif ess_bulk >= 100 * diag.n_chains:
                    ess_bulk_str = f"[green]{ess_bulk:.0f} ({pct_bulk:.0f}%)[/green]"
                elif ess_bulk >= 10 * diag.n_chains:
                    ess_bulk_str = f"[yellow]{ess_bulk:.0f} ({pct_bulk:.0f}%)[/yellow]"
                else:
                    ess_bulk_str = f"[red]{ess_bulk:.0f} ({pct_bulk:.0f}%)[/red]"

                # Format ESS_tail similarly
                pct_tail = min(100, (ess_tail / 10000) * 100)
                if ess_tail >= 10000:
                    ess_tail_str = f"[green]{ess_tail:.0f} (100%)[/green]"
                elif ess_tail >= 100 * diag.n_chains:
                    ess_tail_str = f"[green]{ess_tail:.0f} ({pct_tail:.0f}%)[/green]"
                elif ess_tail >= 10 * diag.n_chains:
                    ess_tail_str = f"[yellow]{ess_tail:.0f} ({pct_tail:.0f}%)[/yellow]"
                else:
                    ess_tail_str = f"[red]{ess_tail:.0f} ({pct_tail:.0f}%)[/red]"

                diag_table.add_row(name, rhat_str, ess_bulk_str, ess_tail_str, status)

            console.print(diag_table)

            # Show warnings if any
            warnings = diag.get_warnings()
            if warnings:
                console.print()
                ui.warning("Convergence issues detected:")
                for warning in warnings[:5]:  # Limit to first 5 warnings
                    console.print(f"  [dim]• {warning}[/dim]")
                if len(warnings) > 5:
                    console.print(f"  [dim]... and {len(warnings) - 5} more warnings[/dim]")

            console.print("")

        # Display results
        table = Table(title=f"MCMC Results - {', '.join(peak_names)}")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Std Error", justify="right")
        table.add_column("68% CI", justify="right")
        table.add_column("95% CI", justify="right")

        for j, name in enumerate(result.parameter_names):
            ci_68 = result.confidence_intervals_68[j]
            ci_95 = result.confidence_intervals_95[j]
            table.add_row(
                name,
                f"{result.values[j]:.6f}",
                f"{result.std_errors[j]:.6f}",
                f"[{ci_68[0]:.6f}, {ci_68[1]:.6f}]",
                f"[{ci_95[0]:.6f}, {ci_95[1]:.6f}]",
            )

        console.print(table)
        console.print("")

        # Display correlation matrix if there are multiple parameters
        if result.correlation_matrix is not None and len(result.parameter_names) > 1:
            console.print(f"[bold cyan]Correlation Matrix - {', '.join(peak_names)}[/bold cyan]")
            console.print("  (Strong correlations: |r| > 0.7)")

            # Create correlation table
            corr_table = Table(show_header=True, header_style="bold cyan")
            corr_table.add_column("", style="cyan", width=15)
            for name in result.parameter_names:
                # Shorten parameter names for display
                short_name = name.split("_")[-1] if "_" in name else name
                corr_table.add_column(short_name[:8], justify="right", width=9)

            for i, name in enumerate(result.parameter_names):
                short_name = name.split("_")[-1] if "_" in name else name
                row = [short_name[:15]]
                for j, val in enumerate(result.correlation_matrix[i]):
                    if i == j:
                        row.append("[dim]1.0000[/dim]")
                    elif abs(val) > 0.7:
                        # Highlight strong correlations
                        row.append(f"[bold yellow]{val:7.4f}[/bold yellow]")
                    elif abs(val) > 0.3:
                        row.append(f"[yellow]{val:7.4f}[/yellow]")
                    else:
                        row.append(f"{val:7.4f}")
                corr_table.add_row(*row)

            console.print(corr_table)
            console.print("")

        # Update global parameters with MCMC uncertainties

    # Save MCMC chain data for diagnostic plotting
    _save_mcmc_chains(results_dir, all_results, clusters)
    ui.success("Saved MCMC chain data for diagnostic plotting")

    # Save updated parameters to output files
    if output_file is not None:
        _save_mcmc_results(output_file, all_results, clusters)
        ui.success(f"Saved MCMC results to: [path]{output_file}[/path]")

    # Update .out files with new uncertainties
    _update_output_files(results_dir, params, all_peaks)
    ui.success("Updated output files with MCMC uncertainties")

    # Provide next steps
    ui.spacer()
    ui.print_next_steps(
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
    ui.show_banner(verbose)

    state = load_fitting_state(results_dir)
    try:
        analysis = ProfileLikelihoodService.run(
            state,
            param_name=param_name,
            n_points=n_points,
            confidence_level=confidence_level,
        )
    except NoVaryingParametersError as exc:
        ui.error("No varying parameters found")
        raise SystemExit(1) from exc
    except ParameterMatchError as exc:
        ui.error(str(exc))
        ui.info("Available parameters:")
        for name in exc.available[:20]:
            console.print(f"  {name}")
        if len(exc.available) > 20:
            console.print(f"  ... and {len(exc.available) - 20} more")
        raise SystemExit(1) from exc

    target_params = analysis.target_parameters

    if param_name is None:
        ui.show_header("Computing Profile Likelihood for All Parameters")
        console.print(f"  Parameters to profile: {len(target_params)}")
        console.print(f"  Confidence level: {confidence_level * 100:.0f}%")
        console.print(f"  Points per parameter: {n_points}")
        console.print("  [yellow]This may take a while...[/yellow]")
    else:
        if len(target_params) > 1:
            ui.info(f"Found {len(target_params)} matching parameters:")
            for name in target_params:
                console.print(f"  {name}")
            console.print("")

        ui.show_header(f"Computing Profile Likelihood for {len(target_params)} Parameter(s)")

    console.print(f"  Δχ² threshold: {analysis.delta_chi2:.4f}")
    console.print("")

    missing_set = set(analysis.missing_parameters)
    results_by_name = {result.parameter_name: result for result in analysis.results}
    ordered_results: list[ProfileParameterResult] = []

    for idx, target_param in enumerate(target_params, 1):
        if len(target_params) > 1:
            console.print(f"[cyan]Parameter {idx}/{len(target_params)}: {target_param}[/cyan]")

        if target_param in missing_set:
            ui.warning(f"Parameter '{target_param}' not found in any cluster")
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
            from scipy.stats import norm  # type: ignore[import-not-found]

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
        ui.success(f"Saved profile data to: [path]{output_file}[/path]")


def run_correlation(
    results_dir: Path, output_file: Path | None = None, verbose: bool = False
) -> None:
    """Analyze parameter correlations from fitting results.

    Args:
        results_dir: Path to results directory
        output_file: Optional output file for correlation matrix
        verbose: Show banner and verbose output
    """
    # Show banner based on verbosity
    ui.show_banner(verbose)

    state = load_fitting_state(results_dir)
    try:
        correlation = ParameterCorrelationService.analyze(state)
    except NotEnoughVaryingParametersError as exc:
        ui.warning("Not enough varying parameters for correlation analysis")
        if exc.vary_names:
            console.print(f"  Found {len(exc.vary_names)} varying parameter(s): {exc.vary_names}")
        return

    ui.show_header("Parameter Correlation Analysis")
    console.print(f"  Parameters: {len(correlation.parameters)}")

    # For now, just show parameter summary
    # Full correlation requires MCMC samples or covariance matrix
    table = Table(title="Fitted Parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Std Error", justify="right")
    table.add_column("At Boundary?", justify="center")

    for entry in correlation.parameters:
        at_boundary = "⚠️" if entry.at_boundary else ""
        table.add_row(
            entry.name,
            f"{entry.value:.6f}",
            f"{entry.stderr:.6f}",
            at_boundary,
        )

    console.print(table)

    # Report boundary warnings
    if correlation.boundary_parameters:
        console.print()
        ui.warning("Parameters at boundaries:")
        for entry in correlation.boundary_parameters:
            console.print(
                f"  {entry.name}: {entry.value:.6f} (bounds: "
                f"[{entry.min_bound:.6f}, {entry.max_bound:.6f}])"
            )
        console.print("  [dim]Consider adjusting bounds or using global optimization[/dim]")

    if output_file is not None:
        with output_file.open("w") as f:
            f.write("# Parameter summary\n")
            f.write("# Name  Value  Stderr  Min  Max\n")
            for entry in correlation.parameters:
                f.write(
                    f"{entry.name}  {entry.value:.6f}  {entry.stderr:.6f}  "
                    f"{entry.min_bound:.6f}  {entry.max_bound:.6f}\n"
                )
        ui.success(f"Saved parameter summary to: [path]{output_file}[/path]")


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
    ui.show_banner(verbose)

    state = load_fitting_state(results_dir)
    try:
        analysis = ParameterUncertaintyService.analyze(state)
    except NoVaryingParametersFoundError:
        ui.warning("No varying parameters found")
        return

    ui.show_header("Parameter Uncertainties")
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
        ui.warning("Parameters at boundaries:")
        for entry in analysis.boundary_parameters:
            console.print(
                f"  {entry.name}: {entry.value:.6f} (bounds: "
                f"[{entry.min_bound:.6f}, {entry.max_bound:.6f}])"
            )
        console.print("  [dim]Consider adjusting bounds or using global optimization[/dim]")

    # Check for large uncertainties
    if analysis.large_uncertainty_parameters:
        console.print()
        ui.warning("Parameters with large relative uncertainties (>10%):")
        for entry in analysis.large_uncertainty_parameters:
            rel_err = entry.rel_error_pct if entry.rel_error_pct is not None else 0.0
            console.print(f"  {entry.name}: {rel_err:.1f}%")
        console.print("  [dim]Consider MCMC analysis for better uncertainty estimates[/dim]")

    # Suggest next steps
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  • Run MCMC for more accurate uncertainties:")
    console.print(f"    [cyan]peakfit analyze mcmc {results_dir}/[/cyan]")
    console.print("  • Check parameter correlations:")
    console.print(f"    [cyan]peakfit analyze correlation {results_dir}/[/cyan]")

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
        ui.success(f"Saved uncertainty summary to: [path]{output_file}[/path]")


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
    """Save MCMC results to file."""
    with output_file.open("w") as f:
        f.write("# MCMC Uncertainty Analysis Results\n")
        f.write("# Parameter  Value  StdErr  CI_68_Low  CI_68_High  CI_95_Low  CI_95_High\n")

        for result, cluster in zip(results, clusters, strict=False):
            peak_names = [p.name for p in cluster.peaks]
            f.write(f"# Cluster: {', '.join(peak_names)}\n")

            for i, name in enumerate(result.parameter_names):
                ci_68 = result.confidence_intervals_68[i]
                ci_95 = result.confidence_intervals_95[i]
                f.write(
                    f"{name}  {result.values[i]:.6f}  {result.std_errors[i]:.6f}  "
                    f"{ci_68[0]:.6f}  {ci_68[1]:.6f}  {ci_95[0]:.6f}  {ci_95[1]:.6f}\n"
                )

            # Add correlation matrix
            if result.correlation_matrix is not None and len(result.parameter_names) > 1:
                f.write(f"\n# Correlation Matrix for Cluster: {', '.join(peak_names)}\n")
                f.write("# Rows/Columns: " + "  ".join(result.parameter_names) + "\n")
                for i, row in enumerate(result.correlation_matrix):
                    f.write(f"# {result.parameter_names[i]:<20s}")
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


def _save_profile_results(
    output_file: Path,
    param_name: str,
    param_vals: np.ndarray,
    chi2_vals: np.ndarray,
    ci_low: float,
    ci_high: float,
) -> None:
    """Save profile likelihood results to file."""
    with output_file.open("w") as f:
        f.write(f"# Profile Likelihood for {param_name}\n")
        f.write(f"# CI: [{ci_low:.6f}, {ci_high:.6f}]\n")
        f.write("# Parameter_Value  Chi_Squared\n")
        for val, chi2 in zip(param_vals, chi2_vals, strict=True):
            f.write(f"{val:.6f}  {chi2:.6f}\n")


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
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]

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
        ui.warning("matplotlib not available for plotting")


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

            # Store data for this cluster
            mcmc_data.append(
                {
                    "peak_names": peak_names,
                    "chains": result.mcmc_chains,
                    "parameter_names": result.parameter_names,
                    "burn_in": burn_in,
                    "burn_in_info": result.burn_in_info,  # Save full burn-in info
                    "diagnostics": result.mcmc_diagnostics,
                    "best_fit_values": best_fit_values,
                }
            )

    # Save to pickle file
    mcmc_file = results_dir / ".mcmc_chains.pkl"
    # Note: pickle.dump is safe here as we control the data being saved
    with mcmc_file.open("wb") as f:
        pickle.dump(mcmc_data, f)
