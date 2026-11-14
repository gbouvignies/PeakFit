"""Output helpers for the MCMC CLI command."""

from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.diagnostics.ensemble import MIN_TAUS
from peakfit.ui.console import console
from peakfit.ui.tables import create_table

if TYPE_CHECKING:
    from peakfit.engine.scientific.sampler import PosteriorSample


_PLANES_SHOWN = 5


def _decimals(sd: float) -> int:
    """Enough decimals to show the posterior standard deviation to two figures."""
    if not np.isfinite(sd) or sd <= 0:
        return 6
    return max(0, 1 - int(np.floor(np.log10(sd))))


def _interval(low: float, high: float, decimals: int) -> str:
    return f"[{low:.{decimals}f}, {high:.{decimals}f}]"


def display_posterior(posterior: PosteriorSample, verbose: bool = False) -> None:
    """Show one cluster's status, diagnostics, parameter and amplitude intervals."""
    diagnosis = posterior.diagnosis
    console.print(f"\n[subheader]Cluster: {', '.join(posterior.peak_names)}[/subheader]")
    if posterior.provisional:
        console.print("[metric.bad]Provisional: no final intervals[/metric.bad]")
        for reason in posterior.reasons:
            console.print(f"  - {reason}")
    else:
        console.print(
            "[metric.good]Passed the convergence checks[/metric.good] "
            "(walkers start near the fit; a distant mode would not be found)"
        )
    kept = posterior.run.chain.shape[0] - diagnosis.burn_in
    tau_max = float(np.nanmax(diagnosis.tau)) if np.isfinite(diagnosis.tau).any() else np.nan
    console.print(
        f"Burn-in {diagnosis.burn_in} steps · kept {kept} steps "
        f"({MIN_TAUS} τ_max = {MIN_TAUS * tau_max:.0f}) · R̂ not assessed (coupled walkers)"
    )
    console.print(posterior.noise_convention)

    table = create_table(show_header=True)
    table.add_column("Parameter", style="key")
    table.add_column("Median", style="value")
    table.add_column("68% interval", style="value")
    table.add_column("95% interval", style="value")
    table.add_column("τ", style="metric")
    table.add_column("ESS", style="metric")
    table.add_column("At bound", style="metric")
    for p in diagnosis.parameters:
        low95, low68, median, high68, high95 = p.quantiles
        decimals = _decimals(p.sd)
        unreported = posterior.unreported(p.name)
        caution = posterior.caution(p.name.rsplit(".", 2)[0])
        if unreported is not None:
            intervals = (unreported,) * 2
        else:
            intervals = (
                _interval(low68, high68, decimals),
                _interval(low95, high95, decimals),
            )
            if caution is not None:
                intervals = tuple(f"{interval} ‡" for interval in intervals)
        pileup = max(p.at_lower, p.at_upper)
        table.add_row(
            p.name,
            f"{median:.{decimals}f}",
            *intervals,
            f"{p.tau:.1f}",
            f"{p.ess:.0f}",
            f"{pileup:.1%}" if pileup else "",
        )
    console.print(table)
    exchanged = {peak for e in posterior.exchanges for peak in (e.first, e.second)}
    for peak in posterior.peak_names:
        caution = posterior.caution(peak)
        if caution is not None and peak not in exchanged:
            console.print(f"‡ {peak}: {caution}; read its intervals with care")
    _display_amplitudes(posterior, verbose)


def _display_amplitudes(posterior: PosteriorSample, verbose: bool) -> None:
    amplitudes = posterior.amplitudes
    exchanged = {peak for e in posterior.exchanges for peak in (e.first, e.second)}
    for peak, fraction in amplitudes.undetermined:
        console.print(f"Amplitude of {peak} withheld: not determined in {fraction:.1%} of samples")
    pairs = sorted({(c.first, c.second) for c in amplitudes.correlated})
    for first, second in pairs:
        planes = [c.plane for c in amplitudes.correlated if (c.first, c.second) == (first, second)]
        console.print(
            f"Amplitudes of {first} and {second} are not separately determined "
            f"(|ρ| ≥ 0.9 in {len(planes)} plane(s)); read their sum"
        )
    rows = [a for a in amplitudes.amplitudes if a.peaks[0] not in exchanged]
    rows += list(amplitudes.sums)
    if not rows:
        return
    table = create_table(
        title=(
            f"[panel.title]Amplitudes · {amplitudes.samples} samples, Monte Carlo errors "
            f"from {amplitudes.effective:.0f} effective[/panel.title]"
        ),
        show_header=True,
    )
    table.add_column("Peak", style="key")
    table.add_column("Plane", style="value")
    table.add_column("Mean", style="value")
    table.add_column("68% interval", style="value")
    table.add_column("95% interval", style="value")
    shown: dict[tuple[str, ...], int] = {}
    for a in rows:
        shown[a.peaks] = shown.get(a.peaks, 0) + 1
        if shown[a.peaks] > _PLANES_SHOWN and not verbose:
            continue
        decimals = _decimals(a.sd)
        low95, low68, _, high68, high95 = a.quantiles
        intervals = (
            ("provisional",) * 2
            if posterior.provisional
            else (_interval(low68, high68, decimals), _interval(low95, high95, decimals))
        )
        if len(a.peaks) == 1 and posterior.caution(a.peaks[0]) and not posterior.provisional:
            intervals = tuple(f"{interval} ‡" for interval in intervals)
        table.add_row(" + ".join(a.peaks), str(a.plane), f"{a.mean:.{decimals}f}", *intervals)
    console.print(table)
    if amplitudes.sums:
        console.print(
            f"Sums are checked for identifiability at the {amplitudes.samples} evaluated "
            "samples; single peaks at every kept sample."
        )
    if not verbose and any(count > _PLANES_SHOWN for count in shown.values()):
        console.print(f"First {_PLANES_SHOWN} planes per amplitude shown; use --verbose for all.")
