"""Plotting service for generating visualizations.

This service abstracts plot generation so adapters don't need
to know about matplotlib internals.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from peakfit.core.shared.reporter import NullReporter, Reporter


@dataclass(frozen=True)
class PlotOutput:
    """Result of plot generation.

    Attributes:
        path: Path where the plot was saved
        plot_type: Type of plot generated
        n_plots: Number of individual plots in the output
    """

    path: Path
    plot_type: str
    n_plots: int


class PlotService:
    """Service for generating plots from fitting results.

    This service provides a clean API for generating various types
    of plots from PeakFit results, abstracting away matplotlib
    implementation details.

    Example:
        service = PlotService()
        output = service.generate_intensity_plots(
            results_dir=Path("Fits"),
            output_path=Path("intensity.pdf"),
        )
        print(f"Generated {output.n_plots} plots")
    """

    def __init__(self, reporter: Reporter | None = None) -> None:
        """Initialize the plot service.

        Args:
            reporter: Reporter for status messages (default: silent)
        """
        self._reporter = reporter or NullReporter()

    def _get_result_files(self, results_dir: Path, extension: str = "*.out") -> list[Path]:
        """Get result files from path."""
        if results_dir.is_dir():
            return sorted(results_dir.glob(extension))
        elif results_dir.is_file():
            return [results_dir]
        return []

    def generate_intensity_plots(
        self,
        results_dir: Path,
        output_path: Path | None = None,
        show: bool = False,
    ) -> PlotOutput:
        """Generate intensity profile plots.

        Args:
            results_dir: Directory containing .out result files
            output_path: Output PDF path (auto-generated if None)
            show: Whether to display interactively

        Returns:
            PlotOutput with path and count
        """
        from peakfit.plotting.profiles import make_intensity_figure

        if output_path is None:
            output_path = results_dir / "intensity_profiles.pdf"

        files = self._get_result_files(results_dir)
        n_plots = 0

        with PdfPages(output_path) as pdf:
            for file in files:
                try:
                    data = np.genfromtxt(file, dtype=None, names=("xlabel", "intensity", "error"))
                    fig = make_intensity_figure(file.stem, data)
                    pdf.savefig(fig)
                    plt.close(fig)
                    n_plots += 1

                    if show:
                        fig = make_intensity_figure(file.stem, data)
                        fig.show()
                except Exception:
                    pass

        if show and n_plots > 0:
            plt.show()

        return PlotOutput(
            path=output_path,
            plot_type="intensity",
            n_plots=n_plots,
        )

    def generate_cest_plots(
        self,
        results_dir: Path,
        output_path: Path | None = None,
        reference_indices: list[int] | None = None,
        show: bool = False,
    ) -> PlotOutput:
        """Generate CEST profile plots.

        Args:
            results_dir: Directory containing .out result files
            output_path: Output PDF path (auto-generated if None)
            reference_indices: Reference point indices for normalization
            show: Whether to display interactively

        Returns:
            PlotOutput with path and count
        """
        from peakfit.plotting.profiles import make_cest_figure

        if output_path is None:
            output_path = results_dir / "cest_profiles.pdf"

        ref_points = reference_indices or [-1]
        threshold = 1e4  # Threshold for automatic reference selection
        files = self._get_result_files(results_dir)
        n_plots = 0

        with PdfPages(output_path) as pdf:
            for file in files:
                try:
                    offset, intensity, error = np.loadtxt(file, unpack=True)

                    # Determine reference points
                    if ref_points == [-1]:
                        ref_mask = np.abs(offset) >= threshold
                    else:
                        ref_mask = np.zeros_like(offset, dtype=bool)
                        for idx in ref_points:
                            if 0 <= idx < len(offset):
                                ref_mask[idx] = True

                    if not np.any(ref_mask):
                        continue

                    # Normalize by reference intensity
                    intensity_ref = np.mean(intensity[ref_mask])
                    offset_norm = offset[~ref_mask]
                    intensity_norm = intensity[~ref_mask] / intensity_ref
                    error_norm = error[~ref_mask] / np.abs(intensity_ref)

                    fig = make_cest_figure(file.stem, offset_norm, intensity_norm, error_norm)
                    pdf.savefig(fig)
                    plt.close(fig)
                    n_plots += 1

                    if show:
                        fig = make_cest_figure(file.stem, offset_norm, intensity_norm, error_norm)
                        fig.show()
                except Exception:
                    pass

        if show and n_plots > 0:
            plt.show()

        return PlotOutput(
            path=output_path,
            plot_type="cest",
            n_plots=n_plots,
        )

    def generate_cpmg_plots(
        self,
        results_dir: Path,
        time_t2: float,
        output_path: Path | None = None,
        show: bool = False,
    ) -> PlotOutput:
        """Generate CPMG relaxation dispersion plots.

        Args:
            results_dir: Directory containing .out result files
            time_t2: T2 relaxation time in seconds
            output_path: Output PDF path (auto-generated if None)
            show: Whether to display interactively

        Returns:
            PlotOutput with path and count
        """
        from peakfit.plotting.profiles import (
            intensity_to_r2eff,
            make_cpmg_figure,
            make_intensity_ensemble,
            ncyc_to_nu_cpmg,
        )

        if output_path is None:
            output_path = results_dir / "cpmg_profiles.pdf"

        files = self._get_result_files(results_dir)
        n_plots = 0

        with PdfPages(output_path) as pdf:
            for file in files:
                try:
                    data = np.genfromtxt(file, dtype=None, names=("ncyc", "intensity", "error"))
                    ncyc = data["ncyc"]
                    intensity = data["intensity"]

                    # Reference intensity (ncyc = 0 point or first point)
                    if 0 in ncyc:
                        ref_idx = np.where(ncyc == 0)[0][0]
                    else:
                        ref_idx = np.argmin(ncyc)
                    intensity_ref = intensity[ref_idx]

                    # Convert to CPMG coordinates
                    nu_cpmg = ncyc_to_nu_cpmg(ncyc, time_t2)
                    r2_exp = intensity_to_r2eff(intensity, intensity_ref, time_t2)

                    # Error propagation via Monte Carlo
                    ensemble = make_intensity_ensemble(data, size=1000)
                    r2_ensemble = intensity_to_r2eff(ensemble, intensity_ref, time_t2)
                    r2_std = np.std(r2_ensemble, axis=0)
                    r2_err_down = r2_std
                    r2_err_up = r2_std

                    fig = make_cpmg_figure(file.stem, nu_cpmg, r2_exp, r2_err_down, r2_err_up)
                    pdf.savefig(fig)
                    plt.close(fig)
                    n_plots += 1

                    if show:
                        fig = make_cpmg_figure(file.stem, nu_cpmg, r2_exp, r2_err_down, r2_err_up)
                        fig.show()
                except Exception:
                    pass

        if show and n_plots > 0:
            plt.show()

        return PlotOutput(
            path=output_path,
            plot_type="cpmg",
            n_plots=n_plots,
        )

    def generate_mcmc_diagnostics(
        self,
        chains: Any,  # FloatArray
        parameter_names: list[str],
        output_path: Path | None = None,
        burn_in: int = 0,
    ) -> PlotOutput:
        """Generate MCMC diagnostic plots.

        Args:
            chains: Array of shape (n_chains, n_samples, n_params)
            parameter_names: List of parameter names
            output_path: Output PDF path (auto-generated if None)
            burn_in: Number of burn-in samples to exclude

        Returns:
            PlotOutput with path and count
        """
        from peakfit.plotting.diagnostics import save_diagnostic_plots

        if output_path is None:
            output_path = Path("mcmc_diagnostics.pdf")

        save_diagnostic_plots(
            chains=chains,
            parameter_names=parameter_names,
            output_path=output_path,
            burn_in=burn_in,
        )

        return PlotOutput(
            path=output_path,
            plot_type="mcmc_diagnostics",
            n_plots=len(parameter_names),
        )
