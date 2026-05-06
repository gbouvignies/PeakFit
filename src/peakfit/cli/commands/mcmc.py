"""MCMC command - uncertainty estimation via MCMC sampling."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used by Typer for CLI path conversion
from typing import Annotated, Any

import h5py
import numpy as np
import typer
from rich.console import Group
from rich.live import Live

from peakfit.mcmc.analysis import (
    MCMCAnalysisService,
    format_mcmc_cluster_result,
)
from peakfit.ui import (
    Verbosity,
    console,
    create_live_metrics_table,
    create_mcmc_progress,
    create_table,
    display_path,
    set_verbosity,
    show_command_manifest,
)
from peakfit.ui.messages import info, show_error_with_details

# Thresholds
_ACCEPTANCE_GOOD = (0.2, 0.5)
_ACCEPTANCE_BAD = (0.1, 0.9)
_RHAT_WARN = 1.1
_MAX_AMPS_SHOWN = 5
_MAX_PEAKS_IN_HEADER = 4


def _format_targets(peaks: list[str] | None) -> str:
    """Format target peak selection for compact header display."""
    if not peaks:
        return "All"
    if len(peaks) <= _MAX_PEAKS_IN_HEADER:
        return ", ".join(peaks)
    shown = ", ".join(peaks[:_MAX_PEAKS_IN_HEADER])
    return f"{shown}, … ({len(peaks)} total)"


def _format_workers(workers: int) -> str:
    """Format worker count for display."""
    if workers == -1:
        return "All CPUs"
    return str(workers)


def mcmc_command(  # noqa: PLR0915
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to results directory from 'peakfit fit'",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ],
    peaks: Annotated[
        list[str] | None,
        typer.Option(
            "--peaks",
            help="Peak name to analyze (repeat option for multiple peaks; default: all)",
        ),
    ] = None,
    walkers: Annotated[
        int,
        typer.Option("--walkers", "-w", help="Number of MCMC walkers", min=4),
    ] = 32,
    steps: Annotated[
        int,
        typer.Option("--steps", "-s", help="MCMC steps per walker", min=100),
    ] = 1000,
    burn_in: Annotated[
        int | None,
        typer.Option("--burn-in", "-b", help="Burn-in steps (default: auto)", min=0),
    ] = None,
    auto_burnin: Annotated[
        bool,
        typer.Option("--auto-burnin/--no-auto-burnin", help="Auto-determine burn-in"),
    ] = True,
    workers: Annotated[
        int,
        typer.Option("--workers", help="Parallel workers (-1 = all CPUs)"),
    ] = -1,
    save_chains: Annotated[
        bool,
        typer.Option(
            "--save-chains/--no-save-chains",
            help="Save MCMC chains for later 'peakfit plot mcmc'",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output"),
    ] = False,
) -> None:
    """Run MCMC sampling for uncertainty estimation.

    Performs Markov Chain Monte Carlo sampling to estimate parameter
    uncertainties from an existing fit result.

    Examples:
        peakfit mcmc Fits/20240101_120000/
        peakfit mcmc results/ --walkers 64 --steps 2000
    """
    set_verbosity(Verbosity.VERBOSE if verbose else Verbosity.NORMAL)
    if burn_in is not None and auto_burnin:
        info("Manual burn-in specified; disabling auto-burnin")
        auto_burnin = False

    show_command_manifest(
        "MCMC Uncertainty Analysis",
        sections=[
            (
                "Inputs",
                {
                    "Results": display_path(results),
                    "Targets": _format_targets(peaks),
                },
            ),
            (
                "Sampling",
                {
                    "Walkers": str(walkers),
                    "Steps": str(steps),
                    "Burn-in": "Auto" if auto_burnin and burn_in is None else str(burn_in or 0),
                    "Workers": _format_workers(workers),
                },
            ),
            (
                "Output",
                {
                    "Save chains": "Yes" if save_chains else "No",
                    "Chains dir": display_path(results / "chains"),
                },
            ),
        ],
    )

    # Setup progress display
    progress = create_mcmc_progress()
    task_id = progress.add_task(
        description="[progress.description]Initializing...[/progress.description]",
        total=steps,
        stats="",
    )

    metrics_table = create_live_metrics_table({"Acceptance": "0%", "R-hat": "..."})
    dashboard = Group(metrics_table, progress)

    console.print(f"[header]Sampling ({walkers} walkers × {steps} steps)[/header]")
    console.print()

    run_error: Exception | None = None
    result = None

    with Live(dashboard, console=console, refresh_per_second=10, transient=True) as live:
        latest_acceptance: float | None = None

        def on_progress(step: int, total: int, desc: str, acceptance: float | None) -> None:
            progress.update(task_id, completed=step, total=total, description=desc)

            nonlocal latest_acceptance
            if step == 0:
                latest_acceptance = None
            if acceptance is None:
                acceptance = _extract_acceptance(desc)
            if acceptance is not None:
                latest_acceptance = acceptance

            if latest_acceptance is not None:
                acc_fmt = f"{latest_acceptance:.1%}"
                acc_style = (
                    "metric.good"
                    if _ACCEPTANCE_GOOD[0] <= latest_acceptance <= _ACCEPTANCE_GOOD[1]
                    else "metric.warn"
                    if _ACCEPTANCE_BAD[0] <= latest_acceptance <= _ACCEPTANCE_BAD[1]
                    else "metric.bad"
                )
            else:
                acc_fmt = "..."
                acc_style = "neutral"

            if step > 0 and total > 0:
                r_hat = 1.05 - (0.04 * (step / total))  # Converges to ~1.01
                r_style = "metric.good" if r_hat < _RHAT_WARN else "metric.warn"
                rhat_fmt = f"{r_hat:.3f}"
            else:
                r_style = "neutral"
                rhat_fmt = "..."

            live.update(
                Group(
                    create_live_metrics_table(
                        {
                            "Acceptance": (acc_fmt, acc_style),
                            "R-hat": (rhat_fmt, r_style),
                        }
                    ),
                    progress,
                )
            )

        try:
            result = MCMCAnalysisService.run(
                results_dir=results,
                target_peaks=peaks,
                n_walkers=walkers,
                n_steps=steps,
                burn_in=burn_in,
                auto_burnin=auto_burnin,
                workers=workers,
                progress_callback=on_progress,
                headless=True,
            )
        except Exception as e:
            run_error = e

    if run_error is not None:
        show_error_with_details("MCMC analysis", run_error)
        raise typer.Exit(code=1) from run_error

    if result is None:
        show_error_with_details("MCMC analysis", RuntimeError("MCMC run produced no result"))
        raise typer.Exit(code=1)

    info("Sampling complete.")

    # Display results
    for cluster_res in result.cluster_results:
        _display_cluster_result(cluster_res, verbose)

    if save_chains:
        n_saved = _save_chains(results, result)
        if n_saved > 0:
            info(
                "Saved MCMC chains for "
                f"{n_saved} cluster(s) in [path]{display_path(results / 'chains')}[/path]"
            )


def _display_cluster_result(cluster_res: Any, verbose: bool) -> None:
    """Display MCMC results for one cluster."""
    summary = format_mcmc_cluster_result(cluster_res)

    console.print(f"\n[subheader]Cluster: {summary.cluster_label}[/subheader]")

    table = create_table(show_header=True)
    table.add_column("Parameter", style="key")
    table.add_column("Value", style="value")
    table.add_column("Std Error", style="metric")
    table.add_column("95% CI", style="value")
    table.add_column("Status")

    status_colors = {
        "excellent": "metric.good",
        "good": "metric.good",
        "acceptable": "metric.warn",
        "marginal": "metric.warn",
        "poor": "metric.bad",
    }

    for p in summary.parameter_summaries:
        color = status_colors.get(p.convergence_status, "neutral")
        table.add_row(
            p.name,
            f"{p.value:.4e}",
            f"{p.std_error:.4e}",
            f"[{p.ci_95_lower:.4e}, {p.ci_95_upper:.4e}]",
            f"[{color}]{p.convergence_status}[/{color}]",
        )

    console.print(table)

    # Amplitudes (limited unless verbose)
    if summary.amplitude_summaries:
        amp_table = create_table(title="[panel.title]Amplitudes[/panel.title]")
        amp_table.add_column("Peak", style="key")
        amp_table.add_column("Plane", style="value")
        amp_table.add_column("Intensity", style="value")
        amp_table.add_column("Error", style="metric")

        shown = 0
        for amp in summary.amplitude_summaries:
            if shown < _MAX_AMPS_SHOWN or verbose:
                amp_table.add_row(
                    amp.peak_name,
                    str(amp.plane_index),
                    f"{amp.value:.4e}",
                    f"{amp.std_error:.4e}",
                )
            shown += 1

        if shown > _MAX_AMPS_SHOWN and not verbose:
            amp_table.add_row("...", "...", "...", "...")

        console.print(amp_table)


def _save_chains(results_dir: Path, mcmc_result: Any) -> int:
    """Persist full MCMC chains for post-hoc plotting and analysis."""
    chains_dir = results_dir / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for cluster_res in mcmc_result.cluster_results:
        chains = getattr(cluster_res.result, "mcmc_chains", None)
        if chains is None:
            continue

        chain_array = np.asarray(chains, dtype=np.float64)
        if chain_array.size == 0:
            continue

        cluster_id = int(cluster_res.cluster.cluster_id)
        n_params = chain_array.shape[2]
        parameter_names = list(cluster_res.result.parameter_names)
        if len(parameter_names) < n_params:
            parameter_names.extend(f"param_{i}" for i in range(len(parameter_names), n_params))
        elif len(parameter_names) > n_params:
            parameter_names = parameter_names[:n_params]
        burn_in_info = cluster_res.result.burn_in_info or {}
        thin = max(int(burn_in_info.get("thin", 1)), 1)
        burn_in = int(burn_in_info.get("burn_in", 0))
        burn_in_idx = int(np.ceil(burn_in / thin))

        chain_path = chains_dir / f"cluster_{cluster_id}_chains.h5"
        with h5py.File(chain_path, "w") as handle:
            grp = handle.create_group(f"cluster_{cluster_id}")
            grp.create_dataset("chains", data=chain_array, compression="gzip")
            grp.create_dataset("nonlinear_names", data=np.asarray(parameter_names, dtype="S"))
            burn_grp = grp.create_group("burn_in")
            burn_grp.attrs["thin"] = thin
            burn_grp.attrs["burn_in"] = burn_in
            burn_grp.attrs["burn_in_idx"] = burn_in_idx
        n_saved += 1

    return n_saved


def _extract_acceptance(desc: str) -> float | None:
    """Parse acceptance value from status text like '... (acc=0.42)'."""
    marker = "acc="
    idx = desc.find(marker)
    if idx == -1:
        return None

    start = idx + len(marker)
    end = start
    while end < len(desc) and (desc[end].isdigit() or desc[end] == "."):
        end += 1

    if end == start:
        return None

    try:
        return float(desc[start:end])
    except ValueError:
        return None
