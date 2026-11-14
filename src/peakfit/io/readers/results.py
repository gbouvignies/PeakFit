"""Read a fit's scientific archive from its results directory."""

from typing import TYPE_CHECKING

from peakfit.io.output_metadata import RunMetadata
from peakfit.io.readers.continuation import load_completed
from peakfit.io.schemas import FitSummarySchema
from peakfit.io.scientific import read_scientific
from peakfit.io.writers.json import _summary_document
from peakfit.shared.paths import format_path

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.scientific.completed import CompletedFit

__all__ = ["ResultsLoader"]


class ResultsLoader:
    """Loader for a results directory whose ``scientific.npz`` is the fit's truth.

    Example:
        >>> loader = ResultsLoader(Path("Fits/20260129_120000"))
        >>> summary = loader.load_summary()
        >>> completed = loader.load_completed()
    """

    def __init__(self, directory: Path) -> None:
        """Initialize the loader.

        Args:
            directory: Path to results directory

        Raises:
            FileNotFoundError: If the directory holds no scientific.npz
        """
        self.directory = directory
        self.archive = directory / "scientific.npz"
        if not self.archive.is_file():
            raise FileNotFoundError(
                f"No scientific.npz in {format_path(directory)}; "
                "refit results written by earlier PeakFit versions."
            )

    def load_summary(self) -> FitSummarySchema:
        """Project the archive into the fit JSON summary schema."""
        outcome, _, context = read_scientific(self.archive)
        return FitSummarySchema.model_validate(
            _summary_document(
                outcome,
                metadata=RunMetadata(**context["metadata"]),
                z_values=context["plane_values"],
            )
        )

    def load_completed(self) -> CompletedFit:
        """The validated completed fit, with its observations, for continuation and MCMC."""
        return load_completed(self.archive)
