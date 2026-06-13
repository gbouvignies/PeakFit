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
    """Generate CEST profiles from fit intensities.

    Intensities are normalized to reference points. With no explicit references,
    points at |offset| >= 10000 Hz are used; if none exist, the farthest points
    from the profile center are used.
    """
    if output_path is None:
        output_path = results_dir / "cest_profiles.pdf"

    ref_points = reference_indices or [-1]

    def _prepare_data(points: list[tuple[float, float, float]]) -> Any | None:
        return _prepare_cest_data(points, ref_points)

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
        return _prepare_cpmg_data(points, time_t2)

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
) -> Any | None:
    """Normalize CEST intensities against explicit or inferred references."""
    offset = np.array([p[0] for p in points], dtype=float)
    intensity = np.array([p[1] for p in points], dtype=float)
    error = np.array([p[2] for p in points], dtype=float)

    ref_mask = _cest_reference_mask(offset, ref_points)
    if not np.any(ref_mask) or np.all(ref_mask):
        return None

    intensity_ref = float(np.mean(intensity[ref_mask]))
    if not np.isfinite(intensity_ref) or intensity_ref == 0:
        return None

    ref_error = _mean_error(error[ref_mask])
    data_mask = ~ref_mask
    normalized = intensity[data_mask] / intensity_ref
    normalized_error = _ratio_error(
        numerator=intensity[data_mask],
        numerator_error=error[data_mask],
        denominator=intensity_ref,
        denominator_error=ref_error,
    )

    dtype = [("offset", "f8"), ("intensity", "f8"), ("error", "f8")]
    return np.array(
        list(zip(offset[data_mask], normalized, normalized_error, strict=True)), dtype=dtype
    )


def _cest_reference_mask(offset: np.ndarray, ref_points: list[int]) -> np.ndarray:
    """Return the points used as CEST references."""
    if ref_points == [-1]:
        ref_mask = np.abs(offset) >= _CEST_AUTO_REF_OFFSET_THRESHOLD
        if np.any(ref_mask):
            return ref_mask

        n_points = len(offset)
        if n_points <= 1:
            return np.zeros_like(offset, dtype=bool)

        n_fallback = min(_CEST_AUTO_REF_FALLBACK_POINTS, n_points - 1)
        distance_to_center = np.abs(offset - np.median(offset))
        fallback_indices = np.argsort(distance_to_center)[-n_fallback:]
        ref_mask = np.zeros_like(offset, dtype=bool)
        ref_mask[fallback_indices] = True
        return ref_mask

    ref_mask = np.zeros_like(offset, dtype=bool)
    for idx in ref_points:
        if 0 <= idx < len(offset):
            ref_mask[idx] = True
    return ref_mask


def _prepare_cpmg_data(points: list[tuple[float, float, float]], time_t2: float) -> Any | None:
    """Convert CPMG intensities to R2eff with deterministic error propagation."""
    ncyc = np.array([p[0] for p in points], dtype=float)
    intensity = np.array([p[1] for p in points], dtype=float)
    error = np.array([p[2] for p in points], dtype=float)

    ref_mask = ncyc == 0
    if not np.any(ref_mask):
        ref_mask[0] = True

    intensity_ref = float(np.mean(intensity[ref_mask]))
    if not np.isfinite(intensity_ref) or intensity_ref == 0:
        return None

    ref_error = _mean_error(error[ref_mask])
    ratio = intensity / intensity_ref
    data_mask = (~ref_mask) & np.isfinite(ratio) & (ratio > 0)
    if not np.any(data_mask):
        return None

    nu_cpmg = np.where(ncyc[data_mask] > 0, ncyc[data_mask] / time_t2, 0.5 / time_t2)
    r2eff = -np.log(ratio[data_mask]) / time_t2
    r2eff_error = (
        _ratio_error(
            numerator=intensity[data_mask],
            numerator_error=error[data_mask],
            denominator=intensity_ref,
            denominator_error=ref_error,
        )
        / time_t2
    )

    dtype = [("nu_cpmg", "f8"), ("r2eff", "f8"), ("error", "f8")]
    return np.array(list(zip(nu_cpmg, r2eff, r2eff_error, strict=True)), dtype=dtype)


def _mean_error(errors: np.ndarray) -> float:
    """Standard error of a mean from independent point errors."""
    if len(errors) == 0:
        return 0.0
    return float(np.sqrt(np.sum(np.square(errors))) / len(errors))


def _ratio_error(
    *,
    numerator: np.ndarray,
    numerator_error: np.ndarray,
    denominator: float,
    denominator_error: float,
) -> np.ndarray:
    """Propagate uncertainty for numerator / denominator."""
    relative_num = np.divide(
        numerator_error,
        np.abs(numerator),
        out=np.zeros_like(numerator_error, dtype=float),
        where=numerator != 0,
    )
    relative_den = abs(denominator_error / denominator) if denominator != 0 else 0.0
    ratio = numerator / denominator
    return np.abs(ratio) * np.sqrt(np.square(relative_num) + relative_den**2)


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
