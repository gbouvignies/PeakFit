"""Plotting service for generating visualizations.

This service abstracts plot generation so adapters don't need
to know about matplotlib internals.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        from peakfit.cli.plot_command import plot_intensity_profiles

        if output_path is None:
            output_path = results_dir / "intensity_profiles.pdf"

        # Call underlying implementation
        plot_intensity_profiles(results_dir, output_path, show, verbose=False)

        # Count output files for summary
        n_plots = len(list(results_dir.glob("*.out"))) if results_dir.is_dir() else 1

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
        from peakfit.cli.plot_command import plot_cest_profiles

        if output_path is None:
            output_path = results_dir / "cest_profiles.pdf"

        ref = reference_indices or [-1]
        plot_cest_profiles(results_dir, output_path, show, ref, verbose=False)

        n_plots = len(list(results_dir.glob("*.out"))) if results_dir.is_dir() else 1

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
        from peakfit.cli.plot_command import plot_cpmg_profiles

        if output_path is None:
            output_path = results_dir / "cpmg_profiles.pdf"

        plot_cpmg_profiles(results_dir, output_path, show, time_t2, verbose=False)

        n_plots = len(list(results_dir.glob("*.out"))) if results_dir.is_dir() else 1

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
