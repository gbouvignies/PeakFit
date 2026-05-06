"""MCMC summary report page generation (Matplotlib)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from peakfit.engine.diagnostics.metrics import TraceMetrics

_RHAT_EXCELLENT_THRESHOLD = 1.01
_RHAT_ACCEPTABLE_THRESHOLD = 1.05


def generate_mcmc_report_page(
    *,
    n_chains: int,
    n_samples: int,
    burn_in: int,
    thin: int,
    metrics: list[TraceMetrics],
    cluster_id: str,
) -> Figure:
    """Generate a single-page MCMC summary figure for inclusion in a PDF."""
    rhats = [m.rhat for m in metrics]
    ess_bulks = [m.ess for m in metrics]

    if rhats:
        max_rhat = max(rhats)
        max_rhat_idx = rhats.index(max_rhat) + 1
    else:
        max_rhat = float("nan")
        max_rhat_idx = 0

    if ess_bulks:
        min_ess = min(ess_bulks)
        min_ess_idx = ess_bulks.index(min_ess) + 1
    else:
        min_ess = float("nan")
        min_ess_idx = 0

    total_iterations = n_samples * thin

    if max_rhat <= _RHAT_EXCELLENT_THRESHOLD and min_ess >= 100 * n_chains:
        status_text = "EXCELLENT CONVERGENCE"
        status_desc = "All parameters have fully converged with high sample size."
    elif max_rhat <= _RHAT_ACCEPTABLE_THRESHOLD and min_ess >= 10 * n_chains:
        status_text = "MARGINAL / ACCEPTABLE"
        status_desc = "Most parameters converged, but some metrics are borderline."
    else:
        status_text = "POOR CONVERGENCE - CAUTION"
        status_desc = "Significant issues detected. Results may be unreliable."

    recs: list[str] = []
    n_very_high_rhat = sum(1 for r in rhats if r > _RHAT_ACCEPTABLE_THRESHOLD)
    n_low_ess = sum(1 for e in ess_bulks if e < 100 * n_chains)

    if n_very_high_rhat > 0:
        recs.append(
            f"- CRITICAL: {n_very_high_rhat} parameters have R-hat > 1.05. "
            "Increase iterations (e.g., double --steps)."
        )
        recs.append("- Check trace plots for stuck chains (flat lines).")

    if n_low_ess > 0:
        recs.append(
            f"- WARNING: {n_low_ess} parameters have low Effective Sample Size. "
            "Increase iterations to improve precision."
        )

    if not recs:
        recs.append("- No specific actions required. Convergence looks good.")

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
        f"  Max R-hat: {max_rhat:.4f} (Param #{max_rhat_idx})",
        f"  Min ESS: {min_ess:.0f} (Param #{min_ess_idx})",
        "",
        "Recommendations",
        *recs,
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
