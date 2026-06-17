"""MCMC command - uncertainty estimation via MCMC sampling."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used by Typer for CLI path conversion
from typing import Annotated

import typer
from rich.console import Group
from rich.live import Live

from peakfit.cli.commands.mcmc_output import (
    display_cluster_result,
    extract_acceptance,
)
from peakfit.cli.commands.mcmc_output import (
    save_chains as save_mcmc_chains,
)
from peakfit.mcmc.analysis import run_mcmc_analysis
from peakfit.ui.branding import show_command_summary
from peakfit.ui.console import (
    Verbosity,
    console,
    display_path,
    set_verbosity,
)
from peakfit.ui.messages import info, show_error_with_details
from peakfit.ui.progress import create_mcmc_progress
from peakfit.ui.tables import create_live_metrics_table

# Thresholds
_ACCEPTANCE_GOOD = (0.2, 0.5)
_ACCEPTANCE_BAD = (0.1, 0.9)
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


def mcmc_command(
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

    show_command_summary(
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

    metrics_table = create_live_metrics_table({"Step": f"0/{steps}", "Acceptance": "..."})
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
                acceptance = extract_acceptance(desc)
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

            live.update(
                Group(
                    create_live_metrics_table(
                        {
                            "Step": (f"{step}/{total}", "neutral"),
                            "Acceptance": (acc_fmt, acc_style),
                        }
                    ),
                    progress,
                )
            )

        try:
            result = run_mcmc_analysis(
                results_dir=results,
                target_peaks=peaks,
                n_walkers=walkers,
                n_steps=steps,
                burn_in=burn_in,
                auto_burnin=auto_burnin,
                workers=workers,
                progress_callback=on_progress,
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
        display_cluster_result(cluster_res, verbose)

    if save_chains:
        n_saved = save_mcmc_chains(results, result)
        if n_saved > 0:
            info(
                "Saved MCMC chains for "
                f"{n_saved} cluster(s) in [path]{display_path(results / 'chains')}[/path]"
            )
