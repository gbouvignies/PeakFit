"""Whole-dataset execution using the existing pipeline and completed-fit contracts."""

import io
import pickle
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING
from uuid import uuid4

from threadpoolctl import threadpool_limits

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.fit_steps import build_fit_steps
from peakfit.engine.results import FitResult
from peakfit.fit.pipeline import PipelineResult, run_pipeline_iter

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.connection import Connection

    from peakfit.engine.scientific.completed import CompletedFit
    from peakfit.engine.scientific.final_outcome import FinalFitOutcome
    from peakfit.engine.scientific.simulation import FinalModelSnapshot
    from peakfit.gui.fitting import PreparedDatasetFit


@dataclass(frozen=True, slots=True)
class DatasetFitAttempt:
    """Completed scientific record transported only from our disposable child."""

    completed_fit: CompletedFit
    warnings: tuple[str, ...]
    execution_seconds: float


@dataclass(frozen=True, slots=True)
class DatasetFitResult:
    """A completed run bound to the exact reviewed inputs and source-plane order."""

    inputs: PreparedDatasetFit
    completed_fit: CompletedFit
    warnings: tuple[str, ...]
    execution_seconds: float
    software_version: str
    run_id: str
    completed_at: str

    @property
    def outcome(self) -> FinalFitOutcome:
        """The single authority for completed values and optimizer diagnostics."""
        return self.completed_fit.outcome

    @property
    def simulation_snapshot(self) -> FinalModelSnapshot:
        """Owned evaluable geometry; never refit values for display."""
        return self.completed_fit.model_snapshot


def compute_dataset_fit(
    prepared: PreparedDatasetFit,
    progress: Callable[[str], None] = lambda _: None,
) -> DatasetFitAttempt:
    """Run all configured passes, including real cross-cluster correction updates."""
    config = PeakFitConfig.model_validate_json(prepared.snapshot.configuration)
    peaks = [peak for cluster in prepared.clusters for peak in cluster.peaks]
    total = len(prepared.clusters)
    passes = sum(
        step.iterations
        for step in build_fit_steps(
            config.fitting.steps,
            config.fitting.refine_iterations,
        )
    )
    done = 0
    started = perf_counter()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        for item in run_pipeline_iter(
            config,
            prepared.clusters,
            prepared.noise,
            prepared.parameters,
            peaks,
        ):
            if isinstance(item, PipelineResult):
                return DatasetFitAttempt(
                    item.completed_fit,
                    tuple(dict.fromkeys(str(w.message) for w in captured)),
                    perf_counter() - started,
                )
            if isinstance(item, FitResult):
                done += 1
                progress(
                    f"Pass {(done - 1) // total + 1} · cluster {(done - 1) % total + 1}/{total}"
                )
                if done % total == 0:
                    progress(
                        "Finalizing results"
                        if done == total * passes
                        else "Updating cross-cluster corrections"
                    )
    raise RuntimeError("The whole-dataset pipeline did not complete.")


def finish_dataset_fit(
    prepared: PreparedDatasetFit, attempt: DatasetFitAttempt
) -> DatasetFitResult:
    """Retain the reviewed starting problem separately from the completed record."""
    return DatasetFitResult(
        prepared,
        attempt.completed_fit,
        attempt.warnings,
        attempt.execution_seconds,
        version("peakfit"),
        uuid4().hex,
        datetime.now(UTC).isoformat(),
    )


def _mapping(values):
    return MappingProxyType(values)


def _reduce_mapping(value):
    return _mapping, (dict(value),)


def _completed_bytes(attempt: object) -> bytes:
    """Preserve immutable mappings and read-only NumPy buffers across a trusted pipe.

    This private protocol is not a file format or an archive reader. Protocol 5
    retains array write protection; the dispatch table applies only to this writer.
    """
    stream = io.BytesIO()
    writer = pickle.Pickler(stream, protocol=5)
    writer.dispatch_table = {MappingProxyType: _reduce_mapping}
    writer.dump(attempt)
    return stream.getvalue()


def run_dataset_child(connection: Connection) -> None:
    """One serial pipeline in a disposable process, with no nested worker children."""
    try:
        connection.send("ready")
        prepared = connection.recv()
        try:
            attempt = compute_dataset_fit(
                prepared, lambda text: connection.send(("progress", text))
            )
            connection.send(("progress", "Transferring completed result"))
            connection.send(_completed_bytes(attempt))
        except Exception as error:
            connection.send(f"{type(error).__name__}: {error}")
    finally:
        connection.close()


def dataset_summary(result: DatasetFitResult) -> str:
    """Describe numerical completion without claiming scientific validity."""
    clusters = result.outcome.clusters
    usable = sum(cluster.usable for cluster in clusters)
    converged = sum(cluster.optimizer_provenance.converged for cluster in clusters)
    uncertain = sum(
        cluster.analytical_evaluation is not None
        and cluster.analytical_evaluation.uncertainty_warning is not None
        for cluster in clusters
    )
    model = (
        "Complete model"
        if usable == len(clusters)
        else "Incomplete model"
        if usable
        else "No usable model"
    )
    return (
        f"{converged}/{len(clusters)} converged · {usable}/{len(clusters)} usable · {model}"
        f" · {uncertain} uncertainty warnings"
    )
