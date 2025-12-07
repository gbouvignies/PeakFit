"""High-level fitting service facade.

This service provides the primary API for fitting operations.
CLI and other adapters should import only from this module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from peakfit.core.domain.config import PeakFitConfig
from peakfit.core.domain.state import FittingState
from peakfit.core.shared.reporter import NullReporter, Reporter


@dataclass(frozen=True)
class FitResult:
    """Result of a fitting operation.

    Attributes
    ----------
        state: The final fitting state
        output_dir: Directory where results were written
        success: Whether fitting completed successfully
        summary: Human-readable summary of results
    """

    state: FittingState
    output_dir: Path
    success: bool
    summary: dict[str, Any]


@dataclass(frozen=True)
class ValidationResult:
    """Result of input validation.

    Attributes
    ----------
        valid: Whether all inputs are valid
        spectrum_info: Information about the spectrum file
        peaklist_info: Information about the peak list file
        errors: List of validation errors
    """

    valid: bool
    spectrum_info: dict[str, Any]
    peaklist_info: dict[str, Any]
    errors: list[str]


class FitService:
    """Service for NMR peak fitting operations.

    This is the primary entry point for fitting workflows.
    It provides a clean API for CLI and other adapters without
    exposing core implementation details.

    Example:
        service = FitService()
        result = service.fit(
            spectrum_path=Path("spectrum.ft2"),
            peaklist_path=Path("peaks.list"),
        )
        print(f"Fitted {len(result.state.peaks)} peaks")
    """

    def __init__(self, reporter: Reporter | None = None) -> None:
        """Initialize the fit service.

        Args:
            reporter: Reporter for status messages (default: silent)
        """
        self._reporter = reporter or NullReporter()

    def fit(
        self,
        spectrum_path: Path,
        peaklist_path: Path,
        z_values_path: Path | None = None,
        config: PeakFitConfig | None = None,
        *,
        optimizer: str = "leastsq",
        save_state: bool = True,
        verbose: bool = False,
        headless: bool | None = None,
    ) -> FitResult:
        """Perform peak fitting on a spectrum.

        Args:
            spectrum_path: Path to NMRPipe spectrum file
            peaklist_path: Path to peak list file
            z_values_path: Optional path to Z-values file
            config: Fitting configuration (uses defaults if not provided)
            optimizer: Optimization algorithm ('leastsq', 'basin-hopping',
                      'differential-evolution')
            save_state: Whether to save state for later analysis
            verbose: Whether to show detailed progress output

        Returns
        -------
            FitResult with fitting state and summary

        Raises
        ------
            FileNotFoundError: If spectrum or peaklist doesn't exist
            ValueError: If configuration is invalid
        """
        from peakfit.services.fit.pipeline import FitPipeline

        if config is None:
            config = PeakFitConfig()

        # Use the pipeline (for now, still uses UI internally)
        # Future refactoring will inject the reporter
        FitPipeline.run(
            spectrum_path=spectrum_path,
            peaklist_path=peaklist_path,
            z_values_path=z_values_path,
            config=config,
            optimizer=optimizer,
            save_state=save_state,
            verbose=verbose,
            reporter=self._reporter,
            headless=headless,
        )

        # Load the state that was saved
        from peakfit.io.state import StateRepository

        state_file = StateRepository.default_path(config.output.directory)
        state = StateRepository.load(state_file)

        return FitResult(
            state=state,
            output_dir=config.output.directory,
            success=True,
            summary={
                "n_peaks": len(state.peaks),
                "n_clusters": len(state.clusters),
                "n_parameters": len(state.params),
            },
        )

    def validate_inputs(
        self,
        spectrum_path: Path,
        peaklist_path: Path,
    ) -> ValidationResult:
        """Validate input files without performing fitting.

        Args:
            spectrum_path: Path to spectrum file
            peaklist_path: Path to peak list file

        Returns
        -------
            ValidationResult with validation details
        """
        errors: list[str] = []
        spectrum_info: dict[str, Any] = {}
        peaklist_info: dict[str, Any] = {}

        # Validate spectrum
        if not spectrum_path.exists():
            errors.append(f"Spectrum file not found: {spectrum_path}")
        else:
            try:
                from peakfit.io.nmrpipe import read_nmrpipe  # type: ignore[import-not-found]

                dic, data = read_nmrpipe(spectrum_path)
                spectrum_info = {
                    "path": str(spectrum_path),
                    "shape": data.shape,
                    "size_mb": spectrum_path.stat().st_size / 1024 / 1024,
                    "ndim": dic["FDDIMCOUNT"],
                    "valid": True,
                }
            except (OSError, ValueError, KeyError) as e:
                errors.append(f"Invalid spectrum file: {e}")
                spectrum_info = {"path": str(spectrum_path), "valid": False, "error": str(e)}

        # Validate peaklist
        if not peaklist_path.exists():
            errors.append(f"Peak list file not found: {peaklist_path}")
        else:
            try:
                from peakfit.io.peaklist import read_peaklist  # type: ignore[import-not-found]

                peaks_df = read_peaklist(peaklist_path)
                peaklist_info = {
                    "path": str(peaklist_path),
                    "n_peaks": len(peaks_df),
                    "columns": list(peaks_df.columns),
                    "valid": True,
                }
            except (OSError, ValueError, KeyError) as e:
                errors.append(f"Invalid peak list file: {e}")
                peaklist_info = {"path": str(peaklist_path), "valid": False, "error": str(e)}

        return ValidationResult(
            valid=len(errors) == 0,
            spectrum_info=spectrum_info,
            peaklist_info=peaklist_info,
            errors=errors,
        )
