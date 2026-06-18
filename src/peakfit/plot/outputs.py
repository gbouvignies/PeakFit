"""Generate plot output files from PeakFit results."""

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from peakfit.engine.diagnostics.metrics import compute_all_trace_metrics
from peakfit.plot.diagnostics.autocorr import plot_autocorrelation
from peakfit.plot.diagnostics.corner import plot_correlation_pairs
from peakfit.plot.diagnostics.summary import plot_marginal_distributions
from peakfit.plot.diagnostics.trace import plot_trace
from peakfit.plot.profile_data import prepare_cest_data, prepare_cpmg_data, prepare_intensity_data
from peakfit.plot.profiles import (
    make_cest_figure,
    make_cpmg_figure,
    make_intensity_figure,
)
from peakfit.plot.reporting import generate_mcmc_report_page
from peakfit.shared.reporter import NullReporter, Reporter

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure


_MAX_PLOTS_TO_SHOW = 10


@dataclass(frozen=True)
class PlotOutput:
    """Result of plot generation."""

    path: Path
    plot_type: str
    n_plots: int


def generate_intensity_plots(
    results_dir: Path,
    output_path: Path | None = None,
    show: bool = False,
    reporter: Reporter | None = None,
) -> PlotOutput:
    """Generate intensity profile plots from fit results."""
    if output_path is None:
        output_path = results_dir / "intensity_profiles.pdf"

    def _make_figure(peak: str, data: Any) -> Figure:
        return make_intensity_figure(peak, data)

    return _generate_paginated_plots(
        results_dir=results_dir,
        output_path=output_path,
        plot_type="intensity",
        prepare_data_fn=prepare_intensity_data,
        make_figure_fn=_make_figure,
        show=show,
        reporter=reporter,
    )


def generate_cest_plots(
    results_dir: Path,
    output_path: Path | None = None,
    reference_indices: list[int] | None = None,
    show: bool = False,
    reporter: Reporter | None = None,
) -> PlotOutput:
    """Generate CEST profiles from fit intensities.

    Intensities are normalized to reference points. With no explicit references,
    points at |offset| >= 10000 Hz are used; if none exist, the farthest points
    from the profile center are used.
    """
    if output_path is None:
        output_path = results_dir / "cest_profiles.pdf"

    ref_points = reference_indices or [-1]

    def _prepare_data(points: list[tuple[float, float, float]]) -> Any | None:
        return prepare_cest_data(points, ref_points)

    return _generate_paginated_plots(
        results_dir=results_dir,
        output_path=output_path,
        plot_type="cest",
        prepare_data_fn=_prepare_data,
        make_figure_fn=make_cest_figure,
        show=show,
        reporter=reporter,
    )


def generate_cpmg_plots(
    results_dir: Path,
    time_t2: float,
    output_path: Path | None = None,
    show: bool = False,
    reporter: Reporter | None = None,
) -> PlotOutput:
    """Generate CPMG R2eff profiles from fit intensities."""
    if time_t2 <= 0:
        raise ValueError("time_t2 must be greater than zero")

    if output_path is None:
        output_path = results_dir / "cpmg_profiles.pdf"

    def _prepare_data(points: list[tuple[float, float, float]]) -> Any | None:
        return prepare_cpmg_data(points, time_t2)

    return _generate_paginated_plots(
        results_dir=results_dir,
        output_path=output_path,
        plot_type="cpmg",
        prepare_data_fn=_prepare_data,
        make_figure_fn=make_cpmg_figure,
        show=show,
        reporter=reporter,
    )


def _generate_paginated_plots(
    results_dir: Path,
    output_path: Path,
    plot_type: str,
    prepare_data_fn: Callable[[list[tuple[float, float, float]]], Any | None],
    make_figure_fn: Callable[[str, Any], Figure],
    show: bool = False,
    reporter: Reporter | None = None,
) -> PlotOutput:
    """Generate paginated plots from tabular fit results."""
    reporter = reporter or NullReporter()
    peak_data = _load_peak_data_from_intensities_csv(results_dir)

    n_plots = 0
    sorted_peaks = sorted(peak_data.keys())

    with PdfPages(output_path) as pdf:
        for peak in sorted_peaks:
            try:
                points = peak_data[peak]
                # Sort by X/time/freq
                points.sort(key=lambda p: p[0])

                data = prepare_data_fn(points)
                if data is None:
                    continue

                fig = make_figure_fn(peak, data)
                pdf.savefig(fig)
                plt.close(fig)
                n_plots += 1

                if show and n_plots <= _MAX_PLOTS_TO_SHOW:
                    fig = make_figure_fn(peak, data)
                    fig.show()

            except Exception as e:
                reporter.warning(f"Failed to plot {peak}: {e}")

    if show and n_plots > 0:
        plt.show()

    return PlotOutput(output_path, plot_type, n_plots)


def _load_peak_data_from_intensities_csv(
    results_dir: Path,
) -> defaultdict[str, list[tuple[float, float, float]]]:
    """Load per-peak series from tables/intensities.csv."""
    peak_data: defaultdict[str, list[tuple[float, float, float]]] = defaultdict(list)
    intensities_path = _resolve_intensities_csv(results_dir)
    if intensities_path is None:
        return peak_data

    with intensities_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            peak_name = str(row.get("peak_name", "")).strip()
            if not peak_name:
                continue

            intensity = _parse_float(row.get("intensity"), default=np.nan)
            if not np.isfinite(intensity):
                continue

            plane_index = _parse_float(row.get("plane_index"), default=0.0)
            z_value = _parse_float(row.get("z_value"), default=np.nan)
            x_val = z_value if np.isfinite(z_value) else plane_index

            std_error = _parse_float(row.get("intensity_err"), default=0.0)
            if not np.isfinite(std_error):
                std_error = 0.0

            peak_data[peak_name].append((float(x_val), float(intensity), float(std_error)))

    return peak_data


def _parse_float(value: Any, *, default: float) -> float:
    """Parse float from CSV value, returning default on failure."""
    if value is None:
        return float(default)
    text = str(value).strip()
    if not text:
        return float(default)
    try:
        return float(text)
    except (TypeError, ValueError):
        return float(default)


def _resolve_intensities_csv(results_dir: Path) -> Path | None:
    """Resolve path to the canonical intensities.csv output."""
    path = results_dir / "tables" / "intensities.csv"
    return path if path.exists() else None


def generate_mcmc_diagnostics(
    chains_data: list[tuple[Any, list[str], int, int, int]],
    output_path: Path | None = None,
    burn_in: int = 0,
) -> PlotOutput:
    """Generate MCMC diagnostic plots for all clusters.

    Args:
        chains_data: List of (chains, param_names, cluster_id, burn_in, thin) tuples
        output_path: Optional path for output PDF
        burn_in: Global burn-in override (if > 0)

    Returns:
        PlotOutput object with stats
    """
    if output_path is None:
        output_path = Path("mcmc_diagnostics.pdf")

    total_plots = 0

    with PdfPages(output_path) as pdf:
        for (
            raw_chains,
            raw_parameter_names,
            cluster_id,
            stored_burn_in,
            stored_thin,
        ) in chains_data:
            chains, parameter_names = _select_mcmc_plot_parameters(raw_chains, raw_parameter_names)

            # Use stored burn-in if available and no override provided
            effective_burn_in = burn_in if burn_in > 0 else stored_burn_in

            # Chains: (n_walkers, n_steps, n_params)
            chains_post_burnin = (
                chains[:, effective_burn_in:, :] if effective_burn_in > 0 else chains
            )
            samples_flat = chains_post_burnin.reshape(-1, chains_post_burnin.shape[2])

            # Compute Metrics once for all plots
            metrics = compute_all_trace_metrics(chains_post_burnin)

            fig_report = generate_mcmc_report_page(
                n_chains=chains.shape[0],
                n_samples=chains_post_burnin.shape[1],
                burn_in=effective_burn_in,
                thin=stored_thin,
                metrics=metrics,
                cluster_id=str(cluster_id),
            )
            _save_pdf_figure(pdf, fig_report)

            fig_trace = plot_trace(
                chains,
                parameter_names,
                effective_burn_in,
                metrics=metrics,
                thin=stored_thin,
            )
            _save_pdf_figure(
                pdf,
                fig_trace,
                title=f"Cluster {cluster_id}: Trace Plots",
            )
            total_plots += 1

            figs_marginal = plot_marginal_distributions(
                samples_flat,
                parameter_names,
                n_chains=chains.shape[0],
                n_samples=chains_post_burnin.shape[1],
                thin=stored_thin,
                diagnostics=None,
            )
            total_plots += _save_pdf_figure_pages(
                pdf,
                figs_marginal,
                title_template=f"Cluster {cluster_id}: Marginal Distributions (Page {{page}})",
            )

            figs_corr = plot_correlation_pairs(
                samples_flat,
                parameter_names,
                n_chains=chains.shape[0],
                n_samples=chains_post_burnin.shape[1],
                thin=stored_thin,
            )
            total_plots += _save_pdf_figure_pages(
                pdf,
                figs_corr,
                title_template=f"Cluster {cluster_id}: Strong Correlations (Page {{page}})",
            )

            fig_autocorr = plot_autocorrelation(
                chains_post_burnin,
                parameter_names,
                thin=stored_thin,
            )
            n_samples_stored = chains_post_burnin.shape[1]
            total_steps = n_samples_stored * stored_thin
            _save_pdf_figure(
                pdf,
                fig_autocorr,
                title=(
                    f"Cluster {cluster_id}: Autocorrelation ({chains.shape[0]} chains × "
                    f"{total_steps} iters)"
                ),
            )
            total_plots += 1

    return PlotOutput(output_path, "mcmc_diagnostics", total_plots)


def _save_pdf_figure(pdf: PdfPages, fig: Figure, title: str | None = None) -> None:
    """Save and close one PDF figure."""
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _save_pdf_figure_pages(
    pdf: PdfPages,
    figures: list[Figure],
    *,
    title_template: str,
) -> int:
    """Save titled multi-page figure lists."""
    for index, fig in enumerate(figures, start=1):
        _save_pdf_figure(pdf, fig, title_template.format(page=index))
    return len(figures)


def _is_amplitude_parameter_name(name: str) -> bool:
    """Return True for linear amplitude parameter names."""
    return re.search(r"\.F1\.I\d+$", name) is not None


def _select_mcmc_plot_parameters(chains: Any, parameter_names: list[str]) -> tuple[Any, list[str]]:
    """Select parameters for diagnostics: all nonlinear + first amplitude.

    VARPRO amplitudes are linear parameters and can dominate correlation plots.
    Keeping at most one amplitude preserves a quick visual check without clutter.
    """
    n_params = int(chains.shape[2])

    names = list(parameter_names)
    if len(names) < n_params:
        names.extend(f"param_{i}" for i in range(len(names), n_params))
    elif len(names) > n_params:
        names = names[:n_params]

    amp_indices = [i for i, name in enumerate(names) if _is_amplitude_parameter_name(name)]
    nonlinear_indices = [i for i in range(n_params) if i not in amp_indices]
    selected_indices = nonlinear_indices + (amp_indices[:1] if amp_indices else [])

    if not selected_indices:
        return chains, names

    selected_chains = chains[:, :, selected_indices]
    selected_names = [names[i] for i in selected_indices]
    return selected_chains, selected_names
