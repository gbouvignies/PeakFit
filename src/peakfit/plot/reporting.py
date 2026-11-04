"""MCMC summary report page generation (Matplotlib)."""

from __future__ import annotations

from textwrap import wrap
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from peakfit.engine.diagnostics.ensemble import MIN_TAUS

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from peakfit.engine.scientific.sampler import PosteriorSample


def generate_mcmc_report_page(posterior: PosteriorSample) -> Figure:
    """One page with a cluster's recorded status, diagnostics and reasons (§7).

    Everything shown is read from the record, exactly as ``peakfit mcmc`` decided it;
    nothing is recomputed from the chain.
    """
    diagnosis = posterior.diagnosis
    steps, walkers = posterior.run.chain.shape[:2]
    kept = steps - diagnosis.burn_in
    tau_max = float(np.nanmax(diagnosis.tau)) if np.isfinite(diagnosis.tau).any() else np.nan
    ess = [p.ess for p in diagnosis.parameters if np.isfinite(p.ess)]
    if posterior.provisional:
        status = ["  Status: PROVISIONAL - NO FINAL INTERVALS"]
    else:
        status = [
            "  Status: PASSED THE CONVERGENCE CHECKS",
            "    Not proof of convergence: walkers start near the fit, so a distant",
            "    mode would not be found.",
        ]
    cautions = [
        f"- {peak}: {caution}"
        for peak in posterior.peak_names
        if (caution := posterior.caution(peak)) is not None
    ]
    lines = [
        f"MCMC Report: Cluster {posterior.cluster_id} ({', '.join(posterior.peak_names)})",
        "PeakFit Diagnostics",
        "",
        "Run Configuration",
        f"  Walkers: {walkers}",
        f"  Steps: {steps:,}",
        f"  Burn-in: {diagnosis.burn_in:,} steps (discarded)",
        f"  Kept: {kept:,} steps per walker",
        f"  Seed: {posterior.run.entropy}, spawn key {posterior.run.spawn_key}",
        "",
        "Convergence Summary",
        *status,
        f"  Max τ: {tau_max:.1f} steps ({MIN_TAUS} τ_max = {MIN_TAUS * tau_max:.0f} steps)",
        f"  Min ESS: {min(ess):.0f} (approximate)" if ess else "  Min ESS: undefined",
        "  R-hat: not assessed (ensemble walkers are coupled)",
        "",
        "Noise Convention",
        *wrap(posterior.noise_convention, 72, initial_indent="  ", subsequent_indent="  "),
        "",
        "Reasons",
        *(f"- {reason}" for reason in posterior.reasons or ["None of the checks failed."]),
        *(["", "Cautions", *cautions] if cautions else []),
    ]

    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(
        0.06,
        0.96,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        wrap=True,
    )
    return fig
