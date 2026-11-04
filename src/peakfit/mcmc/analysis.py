"""Post-fit MCMC uncertainty analysis of a fit's results directory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.scientific.amplitudes import AMPLITUDE_SAMPLES
from peakfit.engine.scientific.sampler import (
    FORMULATION,
    STREAMS,
    SamplerSettings,
    sample_posterior,
)
from peakfit.engine.scientific.sampling import sampling_problem
from peakfit.io.mcmc_record import MCMCRecord, parent_fit, software_versions, write_record
from peakfit.io.readers.results import ResultsLoader

if TYPE_CHECKING:
    from collections.abc import Callable

    from peakfit.engine.scientific.sampler import SamplerProgress


class PeaksNotFoundError(ValueError):
    """Raised when requested peaks cannot be matched to clusters."""

    def __init__(self, peaks: list[str]) -> None:
        super().__init__(f"No clusters found for peaks: {peaks}")
        self.peaks = peaks


@dataclass(slots=True, frozen=True)
class MCMCAnalysisResult:
    """A completed run's record and where it was published."""

    record: MCMCRecord
    path: Path


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
    """Sample each targeted cluster's posterior and record the run in ``results_dir``.

    Only the targeted clusters must be usable. One ``SeedSequence`` (fresh entropy
    unless given) spawns one stream per cluster, in cluster order. ``progress``
    receives the cluster's position, the number of clusters and the sampler's
    progress. The record is published only once every cluster has been sampled.
    """
    loader = ResultsLoader(Path(results_dir))
    parent = parent_fit(loader.archive)
    completed = loader.load_completed()
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
    record = MCMCRecord(
        created=datetime.now(UTC).isoformat(),
        parent=parent,
        formulation=FORMULATION,
        entropy=int(seed.entropy),  # ty: ignore[invalid-argument-type]
        streams=dict(STREAMS),
        amplitude_samples=AMPLITUDE_SAMPLES,
        software=software_versions(),
        posteriors=tuple(posteriors),
    )
    return MCMCAnalysisResult(record, write_record(Path(results_dir), record))


__all__ = ["MCMCAnalysisResult", "PeaksNotFoundError", "run_mcmc_analysis"]
