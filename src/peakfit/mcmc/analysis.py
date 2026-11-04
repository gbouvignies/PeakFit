"""Post-fit MCMC uncertainty analysis of a fit's results directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.scientific.sampler import SamplerSettings, sample_posterior
from peakfit.engine.scientific.sampling import sampling_problem
from peakfit.io.readers.results import ResultsLoader

if TYPE_CHECKING:
    from collections.abc import Callable

    from peakfit.engine.scientific.sampler import PosteriorSample, SamplerProgress


class PeaksNotFoundError(ValueError):
    """Raised when requested peaks cannot be matched to clusters."""

    def __init__(self, peaks: list[str]) -> None:
        super().__init__(f"No clusters found for peaks: {peaks}")
        self.peaks = peaks


@dataclass(slots=True, frozen=True)
class MCMCAnalysisResult:
    """Every analyzed cluster's posterior, and the entropy that reproduces the run."""

    entropy: int
    posteriors: list[PosteriorSample]


def run_mcmc_analysis(
    results_dir: Path,
    settings: SamplerSettings,
    *,
    target_peaks: list[str] | None = None,
    entropy: int | None = None,
    workers: int = 1,
    progress: Callable[[int, int, SamplerProgress], None] | None = None,
    canceled: Callable[[], bool] | None = None,
) -> MCMCAnalysisResult:
    """Sample each targeted cluster's posterior; only those clusters must be usable.

    One ``SeedSequence`` (fresh entropy unless given) spawns one stream per cluster,
    in cluster order. ``progress`` receives the cluster's position, the number of
    clusters and the sampler's progress.
    """
    completed = ResultsLoader(Path(results_dir)).load_completed()
    peak_set = set(target_peaks or ())
    target_ids = [
        c.cluster_id
        for c in completed.outcome.clusters
        if not peak_set or peak_set.intersection(c.peak_names)
    ]
    if target_peaks and not target_ids:
        raise PeaksNotFoundError(target_peaks)
    seed = np.random.SeedSequence(entropy)
    posteriors = []
    for index, (cluster_id, stream) in enumerate(
        zip(target_ids, seed.spawn(len(target_ids)), strict=True), start=1
    ):

        def report(step: SamplerProgress, index=index) -> None:
            if progress is not None:
                progress(index, len(target_ids), step)

        posteriors.append(
            sample_posterior(
                sampling_problem(completed, cluster_id),
                settings,
                stream,
                progress=report,
                canceled=canceled,
                workers=workers,
            )
        )
    return MCMCAnalysisResult(
        entropy=int(seed.entropy),  # ty: ignore[invalid-argument-type]
        posteriors=posteriors,
    )


__all__ = ["MCMCAnalysisResult", "PeaksNotFoundError", "run_mcmc_analysis"]
