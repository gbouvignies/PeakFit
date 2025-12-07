"""Core runner for the fitting pipeline, separating logic from UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from peakfit.core.algorithms.clustering import create_clusters
from peakfit.core.algorithms.noise import prepare_noise_level
from peakfit.core.domain.peaks_io import read_list
from peakfit.core.domain.spectrum import get_shape_names, read_spectra
from peakfit.core.fitting.protocol import FitProtocol, create_protocol_from_config
from peakfit.core.shared import DataIOError
from peakfit.services.fit.fitting import fit_all_clusters

if TYPE_CHECKING:
    from peakfit.core.domain.cluster import Cluster
    from peakfit.core.domain.config import PeakFitConfig
    from peakfit.core.domain.peaks import Peak
    from peakfit.core.domain.scidata import SpectrumData
    from peakfit.core.fitting.parameters import Parameters
    from peakfit.core.shared.events import EventDispatcher
    from peakfit.core.shared.reporter import Reporter


@dataclass
class FitArguments:
    """Arguments for fitting process."""

    path_spectra: Path = field(default_factory=Path)
    path_list: Path = field(default_factory=Path)
    path_z_values: Path | None = None
    path_output: Path = field(default_factory=lambda: Path("Fits"))
    contour_level: float | None = None
    noise: float | None = None
    refine_nb: int = 1
    fixed: bool = False
    jx: bool = False
    phx: bool = False
    phy: bool = False
    exclude: list[int] = field(default_factory=list)
    pvoigt: bool = False
    lorentzian: bool = False
    gaussian: bool = False


def config_to_fit_args(
    config: PeakFitConfig,
    spectrum_path: Path,
    peaklist_path: Path,
    z_values_path: Path | None,
) -> FitArguments:
    """Convert modern config to FitArguments."""
    return FitArguments(
        path_spectra=spectrum_path,
        path_list=peaklist_path,
        path_z_values=z_values_path,
        contour_level=config.clustering.contour_level,
        noise=config.noise_level,
        path_output=config.output.directory,
        refine_nb=config.fitting.refine_iterations,
        fixed=config.fitting.fix_positions,
        jx=config.fitting.fit_j_coupling,
        phx="F3" in config.fitting.fit_phase,
        phy="F2" in config.fitting.fit_phase,
        exclude=config.exclude_planes,
        pvoigt=config.fitting.lineshape == "pvoigt",
        lorentzian=config.fitting.lineshape == "lorentzian",
        gaussian=config.fitting.lineshape == "gaussian",
    )


@dataclass
class PipelineRunner:
    """Orchestrates the fitting process logic, decoupled from UI."""

    config: PeakFitConfig
    clargs: FitArguments

    def load_data(self) -> SpectrumData:
        """Load spectral data."""
        return read_spectra(
            self.clargs.path_spectra, self.clargs.path_z_values, self.clargs.exclude
        )

    def estimate_noise(self, spectra: SpectrumData) -> tuple[float, bool]:
        """Estimate and update noise level.

        Returns
        -------
            Tuple of (noise_value, was_provided_by_user)
        """
        noise_was_provided = self.clargs.noise is not None and self.clargs.noise > 0.0

        # This function updates clargs.noise in-place if it was None
        self.clargs.noise = prepare_noise_level(self.clargs, spectra)

        if self.clargs.noise is None:
            raise DataIOError("Noise must be set by prepare_noise_level")

        return float(self.clargs.noise), noise_was_provided

    def detect_lineshapes(self, spectra: SpectrumData) -> list[str]:
        """Detect lineshapes for dimensions."""
        return get_shape_names(self.clargs, spectra)

    def load_peaks(self, spectra: SpectrumData, shape_names: list[str]) -> list[Peak]:
        """Load peak list."""
        return read_list(spectra, shape_names, self.clargs)

    def cluster_peaks(self, spectra: SpectrumData, peaks: list[Peak]) -> list[Cluster]:
        """Cluster peaks using DBSCAN."""
        if self.clargs.contour_level is None:
            # Should have been set by now or defaulted, ensuring it for safety
            if self.clargs.noise:
                self.clargs.contour_level = 5.0 * self.clargs.noise
            else:
                self.clargs.contour_level = 0.0

        return create_clusters(spectra, peaks, self.clargs.contour_level)

    def get_protocol(self) -> FitProtocol:
        """Get or create the fitting protocol."""
        if self.config.fitting.has_protocol():
            return FitProtocol(steps=self.config.fitting.steps)

        return create_protocol_from_config(
            steps=None,
            refine_iterations=self.clargs.refine_nb,
            fixed=self.clargs.fixed,
        )

    def fit_clusters(
        self,
        clusters: list[Cluster],
        optimizer: str,
        protocol: FitProtocol,
        verbose: bool = False,
        workers: int = -1,
        dispatcher: EventDispatcher | None = None,
        reporter: Reporter | None = None,
        headless: bool = False,
    ) -> Parameters:
        """Run the fitting process on clusters."""
        return fit_all_clusters(
            self.clargs,
            clusters,
            optimizer=optimizer,
            optimizer_seed=self.config.fitting.optimizer_seed,
            max_iterations=self.config.fitting.max_iterations,
            tolerance=self.config.fitting.tolerance,
            verbose=verbose,
            dispatcher=dispatcher,
            parameter_config=self.config.parameters,
            protocol=protocol,
            workers=workers,
            reporter=reporter,
            headless=headless,
        )
