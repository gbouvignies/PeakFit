from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich.table import Table
from scipy.stats import norm

if TYPE_CHECKING:
    from pathlib import Path

from peakfit.cli.analysis.shared import load_fitting_state
from peakfit.services.analyze import (
    NoVaryingParametersError,
    ParameterMatchError,
    ProfileLikelihoodService,
    ProfileParameterResult,
)
from peakfit.ui import (
    Verbosity,
    console,
    create_progress,
    error,
    info,
    set_verbosity,
    show_header,
    show_standard_header,
    success,
    warning,
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
    # Set verbosity and show header
    set_verbosity(Verbosity.VERBOSE if verbose else Verbosity.NORMAL)
    show_standard_header("Profile Likelihood Analysis")

    state = load_fitting_state(results_dir)
    try:
        with create_progress(transient=True) as progress_bar:
            profile_task = progress_bar.add_task("Initializing...", total=100, visible=False)

            def progress_callback(step: int, total: int, description: str) -> None:
                progress_bar.update(
                    profile_task, completed=step, total=total, description=description, visible=True
                )

            analysis = ProfileLikelihoodService.run(
                state,
                param_name=param_name,
                n_points=n_points,
                confidence_level=confidence_level,
                progress_callback=progress_callback,
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

        _display_profile_result(target_param, result, confidence_level)

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


def _display_profile_result(
    target_param: str,
    result: ProfileParameterResult,
    confidence_level: float,
) -> None:
    """Display individual profile likelihood result table."""
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
        z = norm.ppf((1 + confidence_level) / 2)
        covar_ci_low = best_value - z * covar_stderr
        covar_ci_high = best_value + z * covar_stderr
        result_table.add_row(
            f"Covariance {confidence_level * 100:.0f}% CI:",
            f"[{covar_ci_low:.6f}, {covar_ci_high:.6f}]",
        )

        profile_lower = best_value - result.ci_low
        profile_upper = result.ci_high - best_value
        denominator = profile_upper + profile_lower
        if denominator > 0:
            asymmetry = abs(profile_upper - profile_lower) / denominator * 200
            if asymmetry > 20:
                result_table.add_row(
                    "Asymmetry:",
                    f"[yellow]{asymmetry:.1f}% (non-linear parameter)[/yellow]",
                )

    console.print(result_table)
    console.print("")
