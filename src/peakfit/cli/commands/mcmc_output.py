"""Output helpers for the MCMC CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np

from peakfit.engine.diagnostics.ensemble import MIN_TAUS
from peakfit.ui.console import console
from peakfit.ui.tables import create_table

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.scientific.sampler import PosteriorSample
    from peakfit.mcmc.analysis import MCMCAnalysisResult


def _decimals(sd: float) -> int:
    """Enough decimals to show the posterior standard deviation to two figures."""
    if not np.isfinite(sd) or sd <= 0:
        return 6
    return max(0, 1 - int(np.floor(np.log10(sd))))


def _interval(low: float, high: float, decimals: int) -> str:
    return f"[{low:.{decimals}f}, {high:.{decimals}f}]"


def display_posterior(posterior: PosteriorSample) -> None:
    """Show one cluster's status, diagnostics and parameter intervals."""
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
        withheld = posterior.withheld(p.name)
        if withheld is not None:
            intervals = (f"withheld: {withheld}",) * 2
        elif posterior.provisional:
            intervals = ("provisional",) * 2
        else:
            intervals = (
                _interval(low68, high68, decimals),
                _interval(low95, high95, decimals),
            )
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


def save_chains(results_dir: Path, analysis: MCMCAnalysisResult) -> int:
    """Write each cluster's whole chain, burn-in included, for ``peakfit plot mcmc``."""
    chains_dir = results_dir / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)
    for posterior in analysis.posteriors:
        cluster_id = posterior.cluster_id
        # (walkers, steps, parameters): the layout the plots read.
        chain = np.swapaxes(posterior.run.chain, 0, 1)
        burn_in = posterior.diagnosis.burn_in
        with h5py.File(chains_dir / f"cluster_{cluster_id}_chains.h5", "w") as handle:
            group = handle.create_group(f"cluster_{cluster_id}")
            group.create_dataset("chains", data=chain, compression="gzip")
            group.create_dataset("nonlinear_names", data=np.asarray(posterior.names, dtype="S"))
            group.attrs["entropy"] = str(posterior.run.entropy)
            group.attrs["spawn_key"] = np.asarray(posterior.run.spawn_key, dtype=np.int64)
            burn = group.create_group("burn_in")
            burn.attrs["thin"] = 1
            burn.attrs["burn_in"] = burn_in
            burn.attrs["burn_in_idx"] = burn_in
    return len(analysis.posteriors)
