"""MCMC summary report page generation (Matplotlib)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from peakfit.engine.diagnostics.ensemble import MIN_TAUS

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from peakfit.engine.diagnostics.metrics import TraceMetrics


def generate_mcmc_report_page(
    *,
    n_chains: int,
    n_samples: int,
    burn_in: int,
    thin: int,
    metrics: list[TraceMetrics],
    cluster_id: str,
) -> Figure:
    """Generate a single-page MCMC summary figure for inclusion in a PDF.

    The checks are those of ``docs/mcmc-formulation.md`` §7 that the chain alone
    supports. Stuck walkers and label switching need the run itself and are reported
    by ``peakfit mcmc``.
    """
    taus = [m.tau for m in metrics]
    finite = [t for t in taus if np.isfinite(t)]
    tau_max = max(finite) if finite else float("nan")
    needed = MIN_TAUS * tau_max
    ess = [m.ess for m in metrics if np.isfinite(m.ess)]
    drifting = [str(i + 1) for i, m in enumerate(metrics) if not m.stationary]
    reasons = []
    if len(finite) < len(taus):
        reasons.append("τ is undefined for at least one parameter.")
    elif not n_samples >= needed:
        reasons.append(
            f"The kept chain ({n_samples} steps) is shorter than {MIN_TAUS} τ_max "
            f"({needed:.0f} steps): run longer."
        )
    if drifting:
        reasons.append(
            f"The two halves of the kept chain disagree for parameter(s) {', '.join(drifting)}."
        )
    status_text = "PROVISIONAL - NO FINAL INTERVALS" if reasons else "PASSED THE CHAIN CHECKS"
    status_desc = (
        "Not proof of convergence: walkers start near the fit, so a distant mode "
        "would not be found."
    )
    total_iterations = (n_samples + burn_in) * thin

    lines = [
        f"MCMC Report: Cluster {cluster_id}",
        "PeakFit Diagnostics",
        "",
        "Run Configuration",
        f"  Chains: {n_chains}",
        f"  Total Iterations: {total_iterations:,}",
        f"  Stored Samples: {n_samples:,} per chain",
        f"  Thinning Factor: {thin}",
        f"  Burn-in: {burn_in:,} steps (discarded)",
        "",
        "Convergence Summary",
        f"  Status: {status_text}",
        f"  {status_desc}",
        f"  Max τ: {tau_max:.1f} steps ({MIN_TAUS} τ_max = {needed:.0f} steps)",
        f"  Min ESS: {min(ess):.0f} (approximate)" if ess else "  Min ESS: undefined",
        "  R-hat: not assessed (ensemble walkers are coupled)",
        "",
        "Findings",
        *(f"- {reason}" for reason in reasons or ["None of the chain checks failed."]),
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
    )
    return fig
