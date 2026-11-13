"""Generate plot output files from PeakFit results."""

import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from peakfit.engine.algorithms.covariance_factors import peak_covariances_between_planes
from peakfit.io.scientific import read_results
from peakfit.plot.diagnostics.autocorr import plot_autocorrelation
from peakfit.plot.diagnostics.corner import plot_correlation_pairs
from peakfit.plot.diagnostics.summary import plot_marginal_distributions
from peakfit.plot.diagnostics.trace import plot_trace
from peakfit.plot.profile_data import (
    RecordedProfile,
    prepare_cest_data,
    prepare_cpmg_data,
    prepare_intensity_data,
)
from peakfit.plot.profiles import (
    make_cest_figure,
    make_cpmg_figure,
    make_intensity_figure,
)
from peakfit.plot.reporting import generate_mcmc_report_page
from peakfit.shared.reporter import NullReporter, Reporter

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
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

    peak_data = _load_peak_data_from_intensities_csv(results_dir)
    for points in peak_data.values():
        points.sort(key=lambda p: p[0])
    return _generate_paginated_plots(
        peak_data=peak_data,
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
    """Generate CEST profiles from the recorded amplitudes in ``scientific.npz``.

    Intensities are normalized to reference points, with errors from each peak's
    covariance between planes. Reference indices count points in order of offset. With
    no explicit references, points at |offset| >= 10000 Hz are used; if none exist, the
    farthest points from the profile center are used.
    """
    if output_path is None:
        output_path = results_dir / "cest_profiles.pdf"

    ref_points = reference_indices or [-1]

    def _prepare_data(profile: RecordedProfile) -> Any | str:
        return prepare_cest_data(profile, ref_points)

    return _generate_paginated_plots(
        peak_data=recorded_profiles(results_dir),
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
    """Generate CPMG R2eff profiles from the recorded amplitudes in ``scientific.npz``."""
    if time_t2 <= 0:
        raise ValueError("time_t2 must be greater than zero")

    if output_path is None:
        output_path = results_dir / "cpmg_profiles.pdf"

    def _prepare_data(profile: RecordedProfile) -> Any | str:
        return prepare_cpmg_data(profile, time_t2)

    return _generate_paginated_plots(
        peak_data=recorded_profiles(results_dir),
        output_path=output_path,
        plot_type="cpmg",
        prepare_data_fn=_prepare_data,
        make_figure_fn=make_cpmg_figure,
        show=show,
        reporter=reporter,
    )


def _generate_paginated_plots[T](
    peak_data: Mapping[str, T],
    output_path: Path,
    plot_type: str,
    prepare_data_fn: Callable[[T], Any | str],
    make_figure_fn: Callable[[str, Any], Figure],
    show: bool = False,
    reporter: Reporter | None = None,
) -> PlotOutput:
    """Generate paginated plots, one per peak in name order."""
    reporter = reporter or NullReporter()

    n_plots = 0
    sorted_peaks = sorted(peak_data.keys())

    with PdfPages(output_path) as pdf:
        for peak in sorted_peaks:
            try:
                data = prepare_data_fn(peak_data[peak])
                if isinstance(data, str):
                    reporter.warning(f"Skipped {peak}: {data}")
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


def recorded_profiles(results_dir: Path) -> dict[str, RecordedProfile]:
    """Each usable peak's amplitudes and covariance between planes from ``scientific.npz``.

    Planes are ordered by plane value (stable), or by plane when the archive records
    none. Values are exactly those recorded; nothing is refitted.
    """
    outcome, _samples, context = read_results(results_dir, "CEST and CPMG plots need")
    profiles: dict[str, RecordedProfile] = {}
    for cluster in outcome.clusters:
        evaluation = cluster.analytical_evaluation
        if not cluster.usable or evaluation is None:
            continue
        amplitudes = evaluation.amplitudes
        planes = (
            np.asarray(context["plane_values"], dtype=float)
            if context
            else np.arange(amplitudes.shape[1], dtype=float)
        )
        order = np.argsort(planes, kind="stable")
        blocks = peak_covariances_between_planes(evaluation.covariance_factors, amplitudes)
        for peak, name in enumerate(cluster.peak_names):
            profiles[name] = RecordedProfile(
                planes[order], amplitudes[peak, order], blocks[peak][np.ix_(order, order)]
            )
    return profiles


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

            # An unavailable error stays NaN: no error bar, and no ratio is derived.
            std_error = _parse_float(row.get("intensity_err"), default=np.nan)

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
    correlations and autocorrelations are drawn from the kept chain. Every page of a
    provisional cluster says so, and the marginals mark an interval only where the
    terminal reports it.
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
            cluster = f"Cluster {cluster_id}"
            if posterior.provisional:
                cluster += " (PROVISIONAL: no final intervals)"

            _save_pdf_figure(pdf, generate_mcmc_report_page(posterior))
            _save_pdf_figure(
                pdf,
                plot_trace(chains, names, posterior.diagnosis.parameters, burn_in),
                title=f"{cluster}: Trace Plots",
            )
            total_plots += 1
            total_plots += _save_pdf_figure_pages(
                pdf,
                plot_marginal_distributions(
                    samples,
                    names,
                    walkers,
                    steps,
                    unreported=[posterior.unreported(name) for name in names],
                    cautions=[posterior.caution(name.rsplit(".", 2)[0]) for name in names],
                ),
                title_template=f"{cluster}: Marginal Distributions (Page {{page}})",
            )
            total_plots += _save_pdf_figure_pages(
                pdf,
                plot_correlation_pairs(samples, names, walkers, steps),
                title_template=f"{cluster}: Strong Correlations (Page {{page}})",
            )
            _save_pdf_figure(
                pdf,
                plot_autocorrelation(kept, names),
                title=f"{cluster}: Autocorrelation ({walkers} walkers × {steps} steps)",
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
