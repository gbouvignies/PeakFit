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
from peakfit.plot.profiles import (
    intensity_to_r2eff,
    make_cest_figure,
    make_cpmg_figure,
    make_intensity_ensemble,
    make_intensity_figure,
    ncyc_to_nu_cpmg,
)
from peakfit.plot.reporting import generate_mcmc_report_page
from peakfit.shared.reporter import NullReporter, Reporter

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure


_MAX_PLOTS_TO_SHOW = 10
_CEST_AUTO_REF_OFFSET_THRESHOLD = 10000.0
_CEST_AUTO_REF_FALLBACK_POINTS = 2


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

    def _prepare_intensity_data(points: list[tuple[float, float, float]]) -> Any:
        # Create structured array for plotting functions
        dtype = [("xlabel", "f8"), ("intensity", "f8"), ("error", "f8")]
        return np.array(points, dtype=dtype)

    def _make_figure(peak: str, data: Any) -> Figure:
        return make_intensity_figure(peak, data)

    return _generate_paginated_plots(
        results_dir=results_dir,
        output_path=output_path,
        plot_type="intensity",
        prepare_data_fn=_prepare_intensity_data,
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
    """Generate CEST profile plots."""
    if output_path is None:
        output_path = results_dir / "cest_profiles.pdf"

    ref_points = reference_indices or [-1]

    def _prepare_data(points: list[tuple[float, float, float]]) -> Any | None:
        return _prepare_cest_data(points, ref_points)

    def _make_figure(peak: str, data: Any) -> Figure:
        offset_norm, intensity_norm, error_norm = data
        return make_cest_figure(peak, offset_norm, intensity_norm, error_norm)

    return _generate_paginated_plots(
        results_dir=results_dir,
        output_path=output_path,
        plot_type="cest",
        prepare_data_fn=_prepare_data,
        make_figure_fn=_make_figure,
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
    """Generate CPMG profile plots."""
    if output_path is None:
        output_path = results_dir / "cpmg_profiles.pdf"

    def _prepare_data(points: list[tuple[float, float, float]]) -> Any:
        return _prepare_cpmg_data(points, time_t2)

    def _make_figure(peak: str, data: Any) -> Figure:
        nu_cpmg, r2_exp, r2_err = data
        return make_cpmg_figure(peak, nu_cpmg, r2_exp, r2_err, r2_err)

    return _generate_paginated_plots(
        results_dir=results_dir,
        output_path=output_path,
        plot_type="cpmg",
        prepare_data_fn=_prepare_data,
        make_figure_fn=_make_figure,
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
    """Resolve path to intensities.csv from either results root or summary/."""
    candidates = (
        results_dir / "tables" / "intensities.csv",
        results_dir.parent / "tables" / "intensities.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _prepare_cest_data(
    points: list[tuple[float, float, float]], ref_points: list[int]
) -> tuple[Any, Any, Any] | None:
    """Process raw points into normalized CEST data."""
    offset = np.array([p[0] for p in points])
    intensity = np.array([p[1] for p in points])
    error = np.array([p[2] for p in points])

    # Reference Logic
    if ref_points == [-1]:
        ref_mask = np.abs(offset) >= _CEST_AUTO_REF_OFFSET_THRESHOLD
        if not np.any(ref_mask):
            n_points = len(offset)
            if n_points <= 1:
                return None
            n_fallback = min(_CEST_AUTO_REF_FALLBACK_POINTS, n_points - 1)
            distance_to_center = np.abs(offset - np.median(offset))
            fallback_indices = np.argsort(distance_to_center)[-n_fallback:]
            ref_mask = np.zeros_like(offset, dtype=bool)
            ref_mask[fallback_indices] = True
    else:
        ref_mask = np.zeros_like(offset, dtype=bool)
        for idx in ref_points:
            if 0 <= idx < len(offset):
                ref_mask[idx] = True

    if not np.any(ref_mask):
        return None

    intensity_ref = np.mean(intensity[ref_mask])

    # Avoid division by zero
    if intensity_ref == 0:
        return None

    offset_norm = offset[~ref_mask]
    intensity_norm = intensity[~ref_mask] / intensity_ref
    error_norm = error[~ref_mask] / np.abs(intensity_ref)

    return offset_norm, intensity_norm, error_norm


def _prepare_cpmg_data(
    points: list[tuple[float, float, float]], time_t2: float
) -> tuple[Any, Any, Any]:
    """Process raw points into CPMG R2eff data with errors."""
    ncyc = np.array([p[0] for p in points])
    intensity = np.array([p[1] for p in points])
    error = np.array([p[2] for p in points])

    # Reference Logic (ncyc=0)
    ref_mask = ncyc == 0
    if not np.any(ref_mask):
        # Fallback: assume first point
        ref_mask[0] = True

    intensity_ref = np.mean(intensity[ref_mask])

    # Filter
    ncyc_cpmg = ncyc[~ref_mask]
    intensity_cpmg = intensity[~ref_mask]
    error_cpmg = error[~ref_mask]

    nu_cpmg = ncyc_to_nu_cpmg(ncyc_cpmg, time_t2)
    r2_exp = intensity_to_r2eff(intensity_cpmg, intensity_ref, time_t2)

    # Bootstrap Error
    # Build structured array for helper compatibility
    dt = [("intensity", float), ("error", float)]
    cpmg_data = np.zeros(len(intensity_cpmg), dtype=dt)
    cpmg_data["intensity"] = intensity_cpmg
    cpmg_data["error"] = error_cpmg

    ens = make_intensity_ensemble(cpmg_data, size=1000)
    r2_ens = intensity_to_r2eff(ens, intensity_ref, time_t2)
    r2_err = np.std(r2_ens, axis=0)

    return nu_cpmg, r2_exp, r2_err


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

            # Summary Page
            fig_report = generate_mcmc_report_page(
                n_chains=chains.shape[0],
                n_samples=chains_post_burnin.shape[1],
                burn_in=effective_burn_in,
                thin=stored_thin,
                metrics=metrics,
                cluster_id=str(cluster_id),
            )
            pdf.savefig(fig_report, bbox_inches="tight")
            plt.close(fig_report)

            # Page 1: Trace plots
            fig_trace = plot_trace(
                chains,
                parameter_names,
                effective_burn_in,
                metrics=metrics,
                thin=stored_thin,
            )
            fig_trace.suptitle(f"Cluster {cluster_id}: Trace Plots", fontsize=14, fontweight="bold")
            pdf.savefig(fig_trace, bbox_inches="tight")
            plt.close(fig_trace)
            total_plots += 1

            # Page 2: Marginal Distributions
            figs_marginal = plot_marginal_distributions(
                samples_flat,
                parameter_names,
                n_chains=chains.shape[0],
                n_samples=chains_post_burnin.shape[1],
                thin=stored_thin,
                diagnostics=None,
            )
            for i, fig in enumerate(figs_marginal):
                fig.suptitle(
                    f"Cluster {cluster_id}: Marginal Distributions (Page {i + 1})",
                    fontsize=14,
                    fontweight="bold",
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                total_plots += 1

            # Page 3: Correlation Plots
            figs_corr = plot_correlation_pairs(
                samples_flat,
                parameter_names,
                n_chains=chains.shape[0],
                n_samples=chains_post_burnin.shape[1],
                thin=stored_thin,
            )
            for i, fig in enumerate(figs_corr):
                fig.suptitle(
                    f"Cluster {cluster_id}: Strong Correlations (Page {i + 1})",
                    fontsize=14,
                    fontweight="bold",
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                total_plots += 1

            # Page 4: Autocorrelation Plots
            fig_autocorr = plot_autocorrelation(
                chains_post_burnin,
                parameter_names,
                thin=stored_thin,
            )
            # Force overwrite title to match cluster specificity
            n_samples_stored = chains_post_burnin.shape[1]
            total_steps = n_samples_stored * stored_thin
            fig_autocorr.suptitle(
                f"Cluster {cluster_id}: Autocorrelation ({chains.shape[0]} chains × "
                f"{total_steps} iters)",
                fontsize=14,
                fontweight="bold",
            )
            pdf.savefig(fig_autocorr, bbox_inches="tight")
            plt.close(fig_autocorr)
            total_plots += 1

    return PlotOutput(output_path, "mcmc_diagnostics", total_plots)


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
