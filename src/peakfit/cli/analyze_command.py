"""Implementation of the analyze command for uncertainty estimation."""

from pathlib import Path

import numpy as np
from rich.table import Table

from peakfit.data.clustering import Cluster
from peakfit.data.peaks import Peak
from peakfit.fitting.advanced import compute_profile_likelihood, estimate_uncertainties_mcmc
from peakfit.fitting.parameters import Parameters
from peakfit.ui import PeakFitUI as ui, console


def load_fitting_state(results_dir: Path) -> dict:
    """Load fitting state from results directory.

    Args:
        results_dir: Path to results directory containing .peakfit_state.pkl

    Returns:
        Dictionary with clusters, params, noise, and peaks
    """
    import pickle

    state_file = results_dir / ".peakfit_state.pkl"
    if not state_file.exists():
        ui.error(f"No fitting state found in {results_dir}")
        ui.info("Run 'peakfit fit' with --save-state (enabled by default)")
        raise SystemExit(1)

    # Note: pickle.load is safe here as we control the state file creation
    with state_file.open("rb") as f:
        state = pickle.load(f)

    ui.success(f"Loaded fitting state: [path]{state_file}[/path]")
    console.print(f"  Clusters: {len(state['clusters'])}")
    console.print(f"  Peaks: {len(state['peaks'])}")
    console.print(f"  Parameters: {len(state['params'])}")

    return state


def run_mcmc(
    results_dir: Path,
    n_walkers: int = 32,
    n_steps: int = 1000,
    burn_in: int = 200,
    peaks: list[str] | None = None,
    output_file: Path | None = None,
    verbose: bool = False,
) -> None:
    """Run MCMC uncertainty estimation on fitted results.

    Args:
        results_dir: Path to results directory
        n_walkers: Number of MCMC walkers
        n_steps: Number of MCMC steps
        burn_in: Number of burn-in steps to discard
        peaks: Optional list of peak names to analyze (default: all)
        output_file: Optional output file for results
        verbose: Show banner and verbose output
    """
    # Show banner based on verbosity
    ui.show_banner(verbose)

    state = load_fitting_state(results_dir)

    clusters: list[Cluster] = state["clusters"]
    params: Parameters = state["params"]
    noise: float = state["noise"]
    all_peaks: list[Peak] = state["peaks"]

    # Filter clusters if specific peaks requested
    if peaks is not None:
        peak_set = set(peaks)
        clusters = [c for c in clusters if any(p.name in peak_set for p in c.peaks)]
        if not clusters:
            ui.error(f"No clusters found for peaks: {peaks}")
            raise SystemExit(1)
        ui.info(f"Analyzing {len(clusters)} cluster(s) for peaks: {peaks}")

    ui.show_header("Running MCMC Uncertainty Estimation")
    console.print(f"  Walkers: {n_walkers}")
    console.print(f"  Steps: {n_steps}")
    console.print(f"  Burn-in: {burn_in}")
    console.print("")

    all_results = []

    for i, cluster in enumerate(clusters):
        peak_names = [p.name for p in cluster.peaks]
        console.print(f"[cyan]Cluster {i + 1}/{len(clusters)}:[/cyan] {', '.join(peak_names)}")

        # Get parameters for this cluster
        from peakfit.data.peaks import create_params

        cluster_params = create_params(cluster.peaks)

        # Copy values from fitted parameters
        for key in cluster_params:
            if key in params:
                cluster_params[key].value = params[key].value
                cluster_params[key].stderr = params[key].stderr

        # Run MCMC
        with console.status("  [yellow]Sampling posterior distribution...[/yellow]"):
            result = estimate_uncertainties_mcmc(
                cluster_params,
                cluster,
                noise,
                n_walkers=n_walkers,
                n_steps=n_steps,
                burn_in=burn_in,
            )

        # Display convergence diagnostics
        if result.mcmc_diagnostics is not None:
            diag = result.mcmc_diagnostics
            console.print(f"[bold cyan]Convergence Diagnostics - {', '.join(peak_names)}[/bold cyan]")
            console.print(f"  Chains: {diag.n_chains}, Samples per chain: {diag.n_samples}")

            # Create diagnostics table
            diag_table = Table(show_header=True, header_style="bold cyan")
            diag_table.add_column("Parameter", style="cyan", width=20)
            diag_table.add_column("R-hat", justify="right", width=10)
            diag_table.add_column("ESS_bulk", justify="right", width=12)
            diag_table.add_column("Status", width=12)

            for j, name in enumerate(result.parameter_names):
                rhat = diag.rhat[j]
                ess_bulk = diag.ess_bulk[j]

                # Determine status
                if rhat <= 1.01 and ess_bulk >= 100 * diag.n_chains:
                    status = "[green]✓ Good[/green]"
                elif rhat <= 1.05 and ess_bulk >= 10 * diag.n_chains:
                    status = "[yellow]⚠ Marginal[/yellow]"
                else:
                    status = "[red]✗ Poor[/red]"

                # Format R-hat with color coding
                if rhat <= 1.01:
                    rhat_str = f"[green]{rhat:.4f}[/green]"
                elif rhat <= 1.05:
                    rhat_str = f"[yellow]{rhat:.4f}[/yellow]"
                else:
                    rhat_str = f"[red]{rhat:.4f}[/red]"

                # Format ESS with color coding
                recommended_ess = 100 * diag.n_chains
                if ess_bulk >= recommended_ess:
                    ess_str = f"[green]{ess_bulk:.0f}[/green]"
                elif ess_bulk >= 10 * diag.n_chains:
                    ess_str = f"[yellow]{ess_bulk:.0f}[/yellow]"
                else:
                    ess_str = f"[red]{ess_bulk:.0f}[/red]"

                diag_table.add_row(name, rhat_str, ess_str, status)

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
        for j, name in enumerate(result.parameter_names):
            if name in params:
                params[name].stderr = result.std_errors[j]

        all_results.append(result)

    # Save MCMC chain data for diagnostic plotting
    _save_mcmc_chains(results_dir, all_results, clusters, burn_in)
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
            f"Review convergence: Check R-hat ≤ 1.01 and ESS values above",
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

    clusters: list[Cluster] = state["clusters"]
    params: Parameters = state["params"]
    noise: float = state["noise"]

    # Get all varying parameter names
    all_param_names = params.get_vary_names()

    if not all_param_names:
        ui.error("No varying parameters found")
        raise SystemExit(1)

    # Determine which parameters to profile
    if param_name is None:
        # Profile all parameters
        target_params = all_param_names
        ui.show_header("Computing Profile Likelihood for All Parameters")
        console.print(f"  Parameters to profile: {len(target_params)}")
        console.print(f"  Confidence level: {confidence_level * 100:.0f}%")
        console.print(f"  Points per parameter: {n_points}")
        console.print("  [yellow]This may take a while...[/yellow]")
    else:
        # Find matching parameters
        target_params = _find_matching_parameters(param_name, all_param_names)

        if not target_params:
            ui.error(f"No parameters matching '{param_name}' found")
            ui.info("Available parameters:")
            for name in all_param_names[:20]:  # Show first 20
                console.print(f"  {name}")
            if len(all_param_names) > 20:
                console.print(f"  ... and {len(all_param_names) - 20} more")
            raise SystemExit(1)

        if len(target_params) > 1:
            ui.info(f"Found {len(target_params)} matching parameters:")
            for name in target_params:
                console.print(f"  {name}")
            console.print("")

        ui.show_header(
            f"Computing Profile Likelihood for {len(target_params)} Parameter(s)"
        )

    from scipy.stats import chi2

    delta_chi2 = chi2.ppf(confidence_level, df=1)

    console.print(f"  Δχ² threshold: {delta_chi2:.4f}")
    console.print("")

    # Store results for all parameters
    all_results = []

    for idx, target_param in enumerate(target_params, 1):
        if len(target_params) > 1:
            console.print(
                f"[cyan]Parameter {idx}/{len(target_params)}: {target_param}[/cyan]"
            )

        # Find which cluster contains the parameter
        target_cluster = None
        for cluster in clusters:
            from peakfit.data.peaks import create_params

            cluster_params = create_params(cluster.peaks)
            if target_param in cluster_params:
                target_cluster = cluster
                break

        if target_cluster is None:
            ui.warning(f"Parameter '{target_param}' not found in any cluster")
            continue

        # Get cluster parameters
        from peakfit.data.peaks import create_params

        cluster_params = create_params(target_cluster.peaks)
        for key in cluster_params:
            if key in params:
                cluster_params[key].value = params[key].value
                cluster_params[key].stderr = params[key].stderr

        with console.status(f"[yellow]Computing profile for {target_param}...[/yellow]"):
            param_vals, chi2_vals, (ci_low, ci_high) = compute_profile_likelihood(
                cluster_params,
                target_cluster,
                noise,
                param_name=target_param,
                n_points=n_points,
                delta_chi2=delta_chi2,
            )

        best_value = cluster_params[target_param].value
        covar_stderr = cluster_params[target_param].stderr

        # Display results
        result_table = Table(show_header=False)
        result_table.add_column("", style="dim")
        result_table.add_column("", style="")

        result_table.add_row("Parameter:", target_param)
        result_table.add_row("Best-fit value:", f"{best_value:.6f}")
        result_table.add_row("Covariance stderr:", f"{covar_stderr:.6f}")
        result_table.add_row(
            f"Profile {confidence_level * 100:.0f}% CI:",
            f"[{ci_low:.6f}, {ci_high:.6f}]",
        )

        # Compare with covariance-based CI
        if covar_stderr > 0:
            from scipy.stats import norm

            z = norm.ppf((1 + confidence_level) / 2)
            covar_ci_low = best_value - z * covar_stderr
            covar_ci_high = best_value + z * covar_stderr
            result_table.add_row(
                f"Covariance {confidence_level * 100:.0f}% CI:",
                f"[{covar_ci_low:.6f}, {covar_ci_high:.6f}]",
            )

            # Check for asymmetry
            profile_lower = best_value - ci_low
            profile_upper = ci_high - best_value
            asymmetry = (
                abs(profile_upper - profile_lower) / (profile_upper + profile_lower) * 200
            )
            if asymmetry > 20:
                result_table.add_row(
                    "Asymmetry:",
                    f"[yellow]{asymmetry:.1f}% (non-linear parameter)[/yellow]",
                )

        console.print(result_table)
        console.print("")

        # Store results
        all_results.append(
            {
                "name": target_param,
                "param_vals": param_vals,
                "chi2_vals": chi2_vals,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "best_value": best_value,
            }
        )

    if plot and len(all_results) > 0:
        for result in all_results[:10]:  # Limit to 10 plots
            _plot_profile_likelihood(
                result["name"],
                result["param_vals"],
                result["chi2_vals"],
                result["ci_low"],
                result["ci_high"],
                delta_chi2,
            )

    if output_file is not None:
        _save_all_profile_results(output_file, all_results, confidence_level)
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
    params: Parameters = state["params"]

    # Get varying parameters
    vary_names = params.get_vary_names()

    if len(vary_names) < 2:
        ui.warning("Not enough varying parameters for correlation analysis")
        return

    ui.show_header("Parameter Correlation Analysis")
    console.print(f"  Parameters: {len(vary_names)}")

    # For now, just show parameter summary
    # Full correlation requires MCMC samples or covariance matrix
    table = Table(title="Fitted Parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Std Error", justify="right")
    table.add_column("At Boundary?", justify="center")

    for name in vary_names:
        param = params[name]
        at_boundary = "⚠️" if param.is_at_boundary() else ""
        table.add_row(
            name,
            f"{param.value:.6f}",
            f"{param.stderr:.6f}",
            at_boundary,
        )

    console.print(table)

    # Report boundary warnings
    boundary_params = params.get_boundary_params()
    if boundary_params:
        console.print()
        ui.warning("Parameters at boundaries:")
        for name in boundary_params:
            param = params[name]
            console.print(
                f"  {name}: {param.value:.6f} (bounds: [{param.min:.6f}, {param.max:.6f}])"
            )
        console.print("  [dim]Consider adjusting bounds or using global optimization[/dim]")

    if output_file is not None:
        with output_file.open("w") as f:
            f.write("# Parameter summary\n")
            f.write("# Name  Value  Stderr  Min  Max\n")
            for name in vary_names:
                param = params[name]
                f.write(
                    f"{name}  {param.value:.6f}  {param.stderr:.6f}  "
                    f"{param.min:.6f}  {param.max:.6f}\n"
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
    params: Parameters = state["params"]

    # Get varying parameters
    vary_names = params.get_vary_names()

    if len(vary_names) == 0:
        ui.warning("No varying parameters found")
        return

    ui.show_header("Parameter Uncertainties")
    console.print("  Source: Covariance matrix from least-squares fit")
    console.print(f"  Parameters: {len(vary_names)}")
    console.print("")

    # Create uncertainty table
    table = Table(title="Fitted Parameters with Uncertainties")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Std Error", justify="right")
    table.add_column("Relative Error (%)", justify="right")
    table.add_column("At Boundary?", justify="center")

    for name in vary_names:
        param = params[name]
        at_boundary = "⚠️" if param.is_at_boundary() else ""

        # Calculate relative error
        if param.value != 0 and param.stderr > 0:
            rel_error = abs(param.stderr / param.value) * 100
            rel_error_str = f"{rel_error:.2f}%"
        else:
            rel_error_str = "N/A"

        # Color code based on relative error
        if param.stderr <= 0:
            stderr_str = "[red]Not computed[/red]"
        elif param.value != 0 and abs(param.stderr / param.value) > 0.1:
            # > 10% relative error - warning
            stderr_str = f"[yellow]{param.stderr:.6f}[/yellow]"
        else:
            stderr_str = f"{param.stderr:.6f}"

        table.add_row(
            name,
            f"{param.value:.6f}",
            stderr_str,
            rel_error_str,
            at_boundary,
        )

    console.print(table)

    # Report boundary warnings
    boundary_params = params.get_boundary_params()
    if boundary_params:
        console.print()
        ui.warning("Parameters at boundaries:")
        for name in boundary_params:
            param = params[name]
            console.print(
                f"  {name}: {param.value:.6f} (bounds: [{param.min:.6f}, {param.max:.6f}])"
            )
        console.print("  [dim]Consider adjusting bounds or using global optimization[/dim]")

    # Check for large uncertainties
    large_uncert = [
        name
        for name in vary_names
        if params[name].value != 0
        and params[name].stderr > 0
        and abs(params[name].stderr / params[name].value) > 0.1
    ]
    if large_uncert:
        console.print()
        ui.warning("Parameters with large relative uncertainties (>10%):")
        for name in large_uncert:
            param = params[name]
            rel_err = abs(param.stderr / param.value) * 100
            console.print(f"  {name}: {rel_err:.1f}%")
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
            for name in vary_names:
                param = params[name]
                rel_error = (
                    abs(param.stderr / param.value) * 100
                    if param.value != 0 and param.stderr > 0
                    else 0.0
                )
                f.write(
                    f"{name}  {param.value:.6f}  {param.stderr:.6f}  {rel_error:.2f}  "
                    f"{param.min:.6f}  {param.max:.6f}\n"
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
                            for param_name in shape.param_names:
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


def _find_matching_parameters(pattern: str, all_params: list[str]) -> list[str]:
    """Find parameters matching a pattern.

    Supports:
    - Exact match: "2N-H_x0"
    - Peak name: "2N-H" matches all parameters for that peak
    - Parameter type: "x0" matches all x0 parameters
    - Partial match: "fwhm" matches all fwhm parameters

    Args:
        pattern: Search pattern
        all_params: List of all parameter names

    Returns:
        List of matching parameter names
    """
    matches = []

    # Try exact match first
    if pattern in all_params:
        return [pattern]

    # Try partial matching
    pattern_lower = pattern.lower()
    for param in all_params:
        param_lower = param.lower()

        # Check if pattern matches peak name (before underscore)
        if "_" in param:
            peak_name, param_type = param.rsplit("_", 1)
            if pattern_lower == peak_name.lower() or pattern_lower == param_type.lower() or pattern_lower in param_lower:
                matches.append(param)
        else:
            # No underscore, just check if pattern is in parameter name
            if pattern_lower in param_lower:
                matches.append(param)

    return matches


def _save_all_profile_results(
    output_file: Path, results: list[dict], confidence_level: float
) -> None:
    """Save profile likelihood results for multiple parameters."""
    with output_file.open("w") as f:
        f.write("# Profile Likelihood Results\n")
        f.write(f"# Confidence Level: {confidence_level * 100:.0f}%\n")
        f.write("#\n")

        for result in results:
            param_name = result["name"]
            param_vals = result["param_vals"]
            chi2_vals = result["chi2_vals"]
            ci_low = result["ci_low"]
            ci_high = result["ci_high"]
            best_value = result["best_value"]

            f.write(f"\n# Parameter: {param_name}\n")
            f.write(f"# Best-fit: {best_value:.6f}\n")
            f.write(f"# {confidence_level * 100:.0f}% CI: [{ci_low:.6f}, {ci_high:.6f}]\n")
            f.write("# Parameter_Value  Chi_Squared\n")
            for val, chi2 in zip(param_vals, chi2_vals, strict=True):
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
        import matplotlib.pyplot as plt

        _fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Plot chi-squared curve
        ax.plot(param_vals, chi2_vals, "b-", linewidth=2, label="Profile")

        # Mark best fit
        best_idx = np.argmin(chi2_vals)
        best_val = param_vals[best_idx]
        best_chi2 = chi2_vals[best_idx]
        ax.axhline(best_chi2, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(best_val, color="gray", linestyle="--", alpha=0.5)

        # Mark confidence interval
        threshold = best_chi2 + delta_chi2
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
    burn_in: int,
) -> None:
    """Save MCMC chain data for diagnostic plotting.

    Args:
        results_dir: Directory to save chain data
        all_results: List of UncertaintyResult objects
        clusters: List of clusters
        burn_in: Burn-in steps used
    """
    import pickle

    mcmc_data = []

    for result, cluster in zip(all_results, clusters, strict=False):
        if result.mcmc_chains is not None:
            # Get best-fit values
            best_fit_values = result.values

            # Get peak names
            peak_names = [p.name for p in cluster.peaks]

            # Store data for this cluster
            mcmc_data.append(
                {
                    "peak_names": peak_names,
                    "chains": result.mcmc_chains,
                    "parameter_names": result.parameter_names,
                    "burn_in": burn_in,
                    "diagnostics": result.mcmc_diagnostics,
                    "best_fit_values": best_fit_values,
                }
            )

    # Save to pickle file
    mcmc_file = results_dir / ".mcmc_chains.pkl"
    # Note: pickle.dump is safe here as we control the data being saved
    with mcmc_file.open("wb") as f:
        pickle.dump(mcmc_data, f)
