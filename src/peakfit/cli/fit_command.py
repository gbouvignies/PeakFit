"""CLI entry point for PeakFit fitting."""

from __future__ import annotations

from pathlib import Path

from peakfit.core.domain.config import PeakFitConfig
from peakfit.services.fit.pipeline import FitArguments, FitPipeline

__all__ = ["FitArguments", "run_fit"]


def run_fit(
    spectrum_path: Path,
    peaklist_path: Path,
    z_values_path: Path | None,
    config: PeakFitConfig,
    optimizer: str = "leastsq",
    save_state: bool = True,
    verbose: bool = False,
) -> None:
    """Delegate CLI fit invocations to the service pipeline."""

    FitPipeline.run(
        spectrum_path,
        peaklist_path,
        z_values_path,
        config,
        optimizer=optimizer,
        save_state=save_state,
        verbose=verbose,
    )
