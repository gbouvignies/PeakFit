"""Implementation of the analyze command for uncertainty estimation."""

from pathlib import Path

import numpy as np
from rich.table import Table

from peakfit.data.clustering import Cluster
from peakfit.fitting.advanced import (
    compute_profile_likelihood,
    estimate_uncertainties_mcmc,
)
from peakfit.fitting.parameters import Parameters
from peakfit.data.peaks import Peak
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
                f"{result.to_numpy()[j]:.6f}",
                f"{result.std_errors[j]:.6f}",
                f"[{ci_68[0]:.6f}, {ci_68[1]:.6f}]",
                f"[{ci_95[0]:.6f}, {ci_95[1]:.6f}]",
            )

        console.print(table)
        console.print("")

        # Update global parameters with MCMC uncertainties
        for j, name in enumerate(result.parameter_names):
            if name in params:
                params[name].stderr = result.std_errors[j]

        all_results.append(result)

    # Save updated parameters to output files
    if output_file is not None:
        _save_mcmc_results(output_file, all_results, clusters)
        ui.success(f"Saved MCMC results to: [path]{output_file}[/path]")

    # Update .out files with new uncertainties
    _update_output_files(results_dir, params, all_peaks)
    ui.success("Updated output files with MCMC uncertainties")


def run_profile_likelihood(
    results_dir: Path,
    param_name: str,
    n_points: int = 20,
    confidence_level: float = 0.95,
    plot: bool = False,
    output_file: Path | None = None,
    verbose: bool = False,
) -> None:
    """Compute profile likelihood confidence interval for a parameter.

    Args:
        results_dir: Path to results directory
        param_name: Parameter name to profile
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

    # Find which cluster contains the parameter
    target_cluster = None
    for cluster in clusters:
        from peakfit.data.peaks import create_params

        cluster_params = create_params(cluster.peaks)
        if param_name in cluster_params:
            target_cluster = cluster
            break

    if target_cluster is None:
        ui.error(f"Parameter '{param_name}' not found")
        ui.info("Available parameters:")
        for name in params:
            console.print(f"  {name}")
        raise SystemExit(1)

    # Get cluster parameters
    from peakfit.data.peaks import create_params

    cluster_params = create_params(target_cluster.peaks)
    for key in cluster_params:
        if key in params:
            cluster_params[key].value = params[key].value
            cluster_params[key].stderr = params[key].stderr

    # Compute delta chi-squared for desired confidence level
    # For 1 parameter: chi2_inv(0.95, df=1) = 3.84
    # For 1 parameter: chi2_inv(0.68, df=1) = 1.0
    from scipy.stats import chi2

    delta_chi2 = chi2.ppf(confidence_level, df=1)

    ui.show_header(f"Computing Profile Likelihood for {param_name}")
    console.print(f"  Confidence level: {confidence_level * 100:.0f}%")
    console.print(f"  Δχ² threshold: {delta_chi2:.4f}")
    console.print(f"  Profile points: {n_points}")

    with console.status("[yellow]Computing profile...[/yellow]"):
        param_vals, chi2_vals, (ci_low, ci_high) = compute_profile_likelihood(
            cluster_params,
            target_cluster,
            noise,
            param_name=param_name,
            n_points=n_points,
            delta_chi2=delta_chi2,
        )

    best_value = cluster_params[param_name].value
    covar_stderr = cluster_params[param_name].stderr

    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Best-fit value: {best_value:.6f}")
    console.print(f"  Covariance stderr: {covar_stderr:.6f}")
    console.print(f"  Profile {confidence_level * 100:.0f}% CI: [{ci_low:.6f}, {ci_high:.6f}]")

    # Compare with covariance-based CI
    if covar_stderr > 0:
        from scipy.stats import norm

        z = norm.ppf((1 + confidence_level) / 2)
        covar_ci_low = best_value - z * covar_stderr
        covar_ci_high = best_value + z * covar_stderr
        console.print(
            f"  Covariance {confidence_level * 100:.0f}% CI: "
            f"[{covar_ci_low:.6f}, {covar_ci_high:.6f}]"
        )

        # Check for asymmetry
        profile_lower = best_value - ci_low
        profile_upper = ci_high - best_value
        asymmetry = abs(profile_upper - profile_lower) / (profile_upper + profile_lower) * 200
        if asymmetry > 20:
            console.print(
                f"  [yellow]Warning: Profile CI is asymmetric ({asymmetry:.1f}%)[/yellow]"
            )
            console.print("  [dim]This indicates non-linear parameter behavior[/dim]")

    if plot:
        _plot_profile_likelihood(param_name, param_vals, chi2_vals, ci_low, ci_high, delta_chi2)

    if output_file is not None:
        _save_profile_results(output_file, param_name, param_vals, chi2_vals, ci_low, ci_high)
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
                    f"{name}  {result.to_numpy()[i]:.6f}  {result.std_errors[i]:.6f}  "
                    f"{ci_68[0]:.6f}  {ci_68[1]:.6f}  {ci_95[0]:.6f}  {ci_95[1]:.6f}\n"
                )


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
