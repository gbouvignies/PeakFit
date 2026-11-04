"""Generate plot output files from PeakFit results."""

import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

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
    from pathlib import Path

    from matplotlib.figure import Figure

    from peakfit.io.mcmc_record import MCMCRecord


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


def generate_mcmc_diagnostics(record: MCMCRecord, output_path: Path) -> PlotOutput:
    """Plot every cluster of an MCMC record: report, traces, marginals, correlations.

    The report page and the trace titles show the recorded diagnosis. Marginals,
    correlations and autocorrelations are drawn from the kept chain.
    """
    total_plots = 0
    with PdfPages(output_path) as pdf:
        for posterior in record.posteriors:
            cluster_id = posterior.cluster_id
            names = list(posterior.names)
            burn_in = posterior.diagnosis.burn_in
            chains = np.swapaxes(posterior.run.chain, 0, 1)  # (walkers, steps, parameters)
            kept = chains[:, burn_in:, :]
            walkers, steps = kept.shape[:2]
            samples = kept.reshape(-1, kept.shape[2])

            _save_pdf_figure(pdf, generate_mcmc_report_page(posterior))
            _save_pdf_figure(
                pdf,
                plot_trace(chains, names, posterior.diagnosis.parameters, burn_in),
                title=f"Cluster {cluster_id}: Trace Plots",
            )
            total_plots += 1
            total_plots += _save_pdf_figure_pages(
                pdf,
                plot_marginal_distributions(samples, names, walkers, steps),
                title_template=f"Cluster {cluster_id}: Marginal Distributions (Page {{page}})",
            )
            total_plots += _save_pdf_figure_pages(
                pdf,
                plot_correlation_pairs(samples, names, walkers, steps),
                title_template=f"Cluster {cluster_id}: Strong Correlations (Page {{page}})",
            )
            _save_pdf_figure(
                pdf,
                plot_autocorrelation(kept, names),
                title=f"Cluster {cluster_id}: Autocorrelation ({walkers} walkers × {steps} steps)",
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
