"""The replicates command: check a run's uncertainties against its replicate planes."""

from collections import Counter, defaultdict
from pathlib import Path  # noqa: TC003 - needed at runtime for Typer
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import typer

from peakfit.engine.scientific.profiles import CEST_REFERENCE_HZ
from peakfit.engine.scientific.replicates import (
    BASELINE_HZ,
    BaselineScatter,
    ExclusionReason,
    duplicate_groups,
    run_baseline_scatter,
    run_duplicate_check,
)
from peakfit.io.scientific import read_results
from peakfit.ui.branding import show_command_summary
from peakfit.ui.console import Verbosity, console, display_path, set_verbosity
from peakfit.ui.messages import bullet, info, show_error_with_details, success, warning
from peakfit.ui.tables import create_table

if TYPE_CHECKING:
    from collections.abc import Mapping

    from peakfit.engine.scientific.final_outcome import FinalFitOutcome
    from peakfit.engine.scientific.replicates import DuplicatePool

_DEFAULT_REFERENCE = f"|offset| ≥ {CEST_REFERENCE_HZ / 1000:g} kHz"
_DUPLICATE_METHOD = "duplicate planes"
_BASELINE_METHOD = "CEST baseline planes"

# What the duplicate check does not show (spec #172, "Stated limits").
_DUPLICATE_LIMITS = (
    "It measures repeatability, not accuracy.",
    "It tests the conditional σ only, the noise part: it cannot detect a "
    "miscalibration of the lineshape-parameter part of the marginal or profile σ.",
    "It checks the uncertainty of contrasts between copies, not marginal σ.",
    "It checks raw amplitudes, not normalized profiles.",
    "Equal plane values do not prove independent, equivalent acquisitions.",
    "It is not calibrated for copies reconstructed jointly.",
    "No error is changed.",
)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def replicates_command(
    results: Annotated[
        Path,
        typer.Argument(help="Results directory", exists=True, file_okay=False, resolve_path=True),
    ],
    baseline: Annotated[
        bool,
        typer.Option(
            "--baseline",
            help="Use the CEST baseline method, even when the run has duplicate planes",
        ),
    ] = False,
    ref: Annotated[
        list[int] | None,
        typer.Option(
            "--ref",
            "-r",
            help=(
                "Baseline method: reference point indices, in offset order; "
                f"omit for {_DEFAULT_REFERENCE}"
            ),
        ),
    ] = None,
    window: Annotated[
        float | None,
        typer.Option(
            "--window",
            help=(
                "Baseline method: planes closer than this to a peak's own offset (Hz) are "
                f"left out [default: {BASELINE_HZ:g}]"
            ),
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="List every peak (baseline method)"),
    ] = False,
) -> None:
    """Check a run's uncertainties against its replicate planes.

    Two methods read the run from scientific.npz, never refit it and change no error.

    Duplicate planes, used when the run has planes with exactly equal plane values: the
    contrasts between copies are standardized by each peak's conditional amplitude
    variance, the noise part of its uncertainty; the lineshape-parameter part cancels
    between copies. The pooled RMS is 1 when the conditional σ is calibrated, reported
    apart for peaks with the correlated-profile flag.

    CEST baseline planes, used otherwise or with --baseline: far from a peak's own
    offset, CEST planes are approximate replicates. The robust scatter of their second
    differences across offsets is compared with the published profile σ: observed /
    stated is 1 when the errors are calibrated. Plane values must be offsets in Hz from
    the carrier of the first spectral axis.

    Both measure repeatability, not accuracy: an error common to every copy, such as a
    lineshape misfit, is invisible.

    Examples:
        peakfit replicates Fits/20240101_120000/
        peakfit replicates Fits/20240101_120000/ --baseline
    """
    set_verbosity(Verbosity.VERBOSE if verbose else Verbosity.NORMAL)
    try:
        outcome, _samples, context = read_results(results, "the check reads")
        if not context:
            archive = results / "scientific.npz"
            raise ValueError(f"{archive} records no plane values or spectral axes.")
    except Exception as e:
        show_error_with_details("checking replicates", e)
        raise typer.Exit(1) from e

    groups = duplicate_groups(np.asarray(context["plane_values"], dtype=float))
    if groups and not baseline:
        if ref is not None or window is not None:
            raise typer.BadParameter(
                "--ref and --window belong to the CEST baseline method; "
                "add --baseline to run it on a run with duplicate planes."
            )
        _check_duplicates(results, outcome, context, groups)
    else:
        _check_baseline(
            results,
            outcome,
            context,
            ref=ref,
            window=BASELINE_HZ if window is None else window,
            verbose=verbose,
            duplicates=bool(groups),
        )


def _check_duplicates(
    results: Path,
    outcome: FinalFitOutcome,
    context: Mapping[str, Any],
    groups: tuple[tuple[int, ...], ...],
) -> None:
    show_command_summary(
        "Check Uncertainties Against Duplicate Planes",
        [("Inputs", {"Results": display_path(results), "Method": _DUPLICATE_METHOD})],
    )
    contrasts = sum(len(group) - 1 for group in groups)
    info(
        f"{_plural(len(groups), 'group')} of equal plane values, "
        f"{_plural(contrasts, 'contrast')} per peak"
    )
    try:
        check = run_duplicate_check(outcome, context)
    except Exception as e:
        show_error_with_details("checking duplicate planes", e)
        raise typer.Exit(1) from e

    by_reason: defaultdict[ExclusionReason, list[str]] = defaultdict(list)
    for name, reason in check.excluded.items():
        by_reason[reason].append(name)
    for reason in ExclusionReason:  # in their fixed precedence
        if not (names := by_reason[reason]):
            continue
        warning(
            f"Excluded {_plural(len(names), 'peak')}, {reason.value}: {', '.join(sorted(names))}"
        )
    if not check.unflagged.peaks and not check.flagged.peaks:
        warning("No peak has duplicate planes it can be checked on.")
        raise typer.Exit(1)
    _report_pool("Peaks without the correlated-profile flag", check.unflagged)
    _report_pool("Peaks with the correlated-profile flag", check.flagged)
    info(
        "Contrasts from peaks of one cluster are correlated: the effective sample is "
        "smaller than the count of contrasts."
    )
    info("Limits of this check:")
    for limit in _DUPLICATE_LIMITS:
        bullet(limit)


def _report_pool(label: str, pool: DuplicatePool) -> None:
    if not pool.peaks:
        info(f"{label}: none")
        return
    success(
        f"{label}: RMS of standardized contrasts {pool.rms:.2f} "
        f"({_plural(pool.contrasts, 'contrast')}, {_plural(pool.peaks, 'peak')}, "
        f"{_plural(pool.clusters, 'cluster')})"
    )
    if pool.low_sample:
        warning(
            "Few contrasts: the sampling uncertainty of the RMS is at least "
            f"1/√(2n) = {pool.standard_error_bound:.2f}, a lower bound",
            indent=1,
        )


def _check_baseline(
    results: Path,
    outcome: FinalFitOutcome,
    context: Mapping[str, Any],
    *,
    ref: list[int] | None,
    window: float,
    verbose: bool,
    duplicates: bool,
) -> None:
    show_command_summary(
        "Check Profile σ Against CEST Baseline Replicates",
        [
            (
                "Inputs",
                {
                    "Results": display_path(results),
                    "Method": _BASELINE_METHOD,
                    "Reference points": ", ".join(str(i) for i in ref)
                    if ref
                    else _DEFAULT_REFERENCE,
                    "Peak window": f"±{window:g} Hz",
                },
            )
        ],
    )
    if not duplicates:
        info("No duplicate planes: the run has no repeated plane values.")
    try:
        peaks = run_baseline_scatter(outcome, context, reference=ref, window=window)
    except Exception as e:
        show_error_with_details("checking baseline replicates", e)
        raise typer.Exit(1) from e

    checked = {name: r for name, r in peaks.items() if isinstance(r, BaselineScatter)}
    if verbose and checked:
        table = create_table("Baseline replicates")
        for column in ("Peak", "Planes", "Relative scatter", "Observed / stated"):
            table.add_column(column, justify="left" if column == "Peak" else "right")
        for name, r in sorted(checked.items()):
            table.add_row(name, str(r.planes), f"{r.observed / r.level:.4f}", f"{r.ratio:.2f}")
        console.print(table)
    for reason, count in sorted(Counter(r for r in peaks.values() if isinstance(r, str)).items()):
        warning(f"Skipped {count} peak{'s' if count > 1 else ''}: {reason}")
    if not checked:
        warning("No peak has enough baseline planes to check.")
        raise typer.Exit(1)
    ratios = np.array([r.ratio for r in checked.values()])
    low, high = np.percentile(ratios, [10, 90])
    scatter = np.median([r.observed / r.level for r in checked.values()])
    info(f"Relative baseline scatter: median {scatter:.4f}")
    success(
        f"Observed / stated profile σ: median {np.median(ratios):.2f} "
        f"(10–90%: {low:.2f}–{high:.2f}), {ratios.size} peaks"
    )
