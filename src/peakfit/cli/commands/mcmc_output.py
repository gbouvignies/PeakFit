"""Output helpers for the MCMC CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from peakfit.mcmc.analysis import format_mcmc_cluster_result
from peakfit.ui.console import console
from peakfit.ui.tables import create_table

if TYPE_CHECKING:
    from pathlib import Path

_MAX_AMPS_SHOWN = 5


def display_cluster_result(cluster_res: Any, verbose: bool) -> None:
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


def save_chains(results_dir: Path, mcmc_result: Any) -> int:
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


def extract_acceptance(desc: str) -> float | None:
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
