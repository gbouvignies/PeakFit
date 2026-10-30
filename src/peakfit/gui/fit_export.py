"""Staged scientific export of retained fits, without Qt or continuation state."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.fit.completed import CompletedFit, FinalClusterSamples
from peakfit.fit.export import write_completed_outputs
from peakfit.fit.output_metadata import RunMetadata
from peakfit.gui.dataset_fit import DatasetFitResult
from peakfit.io.writers.config import WriterConfig

if TYPE_CHECKING:
    from peakfit.gui.fit_positions import DatasetCompletion
    from peakfit.gui.local_fit import LocalFitResult


def export_fit_result(
    result: LocalFitResult | DatasetFitResult,
    destination: Path,
    *,
    completion: DatasetCompletion | None = None,
    current: bool | None = None,
) -> Path:
    """Publish all required artifacts together, refusing any existing destination.

    A sibling lock coordinates PeakFit exporters without creating the destination.
    Rename publishes the validated tree on the same filesystem, including Windows.
    The lock cannot exclude unrelated writers: portable Python has no atomic
    no-replace directory rename on POSIX. Never merge into an existing directory.
    """
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Choose a new results directory: {destination} already exists.")
    inputs = result.inputs
    snapshot = inputs.snapshot
    configuration = json.loads(snapshot.configuration)
    metadata = RunMetadata(
        run_id=result.run_id,
        timestamp=result.completed_at,
        software_version=result.software_version,
        input_files={name: json.loads(record) for name, record in snapshot.source_files},
        configuration=configuration,
        run_duration_seconds=result.execution_seconds,
        warnings=result.warnings,
    )
    if completion is not None and completion.result is not result:
        raise ValueError("Export completion belongs to a different run.")
    if isinstance(result, DatasetFitResult):
        completed = result.completed_fit
    else:
        cluster = result.inputs.cluster
        assert cluster is not None
        completed = CompletedFit(
            result.outcome,
            result.simulation_snapshot,
            (
                FinalClusterSamples(
                    cluster.cluster_id,
                    tuple(cluster.grid_indices),
                    cluster.data,
                    cluster.corrections,
                ),
            ),
        )
    settings = PeakFitConfig.model_validate(configuration).output
    with TemporaryDirectory(prefix=".peakfit-export-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "results"
        files = write_completed_outputs(
            completed,
            staged,
            WriterConfig(formats=tuple(settings.formats), precision=17),
            metadata=metadata,
            data_shape=snapshot.data_shape,
            spectral_parameters=[json.loads(axis) for axis in snapshot.spectral_parameters],
            plane_values=np.asarray(snapshot.plane_values),
            source_plane_indices=snapshot.source_plane_indices,
            peak_identities=[
                {
                    "engine_name": identity.engine_name,
                    "peak_id": identity.working_id,
                    "assignment": identity.assignment,
                }
                for identity in inputs.identities
            ],
            save_simulated=settings.save_simulated,
            initial_coordinates={p.id: p.positions for p in snapshot.peaks},
            adopted_coordinates={p.id: p.positions for p in completion.working_peaks}
            if completion
            else None,
            export_state={
                "scope": "dataset" if isinstance(result, DatasetFitResult) else "cluster",
                "currentness": "unknown"
                if current is None
                else ("current" if current else "outdated"),
                "adoption_warnings": completion.warnings if completion else (),
            },
        )
        if not all(path.is_file() and path.stat().st_size for path in files.values()):
            raise OSError("The export did not produce every required artifact.")
        # Keep the destination absent for Windows; reserve a separate sibling name.
        lock = destination.with_name(f".{destination.name}.export-lock")
        lock.mkdir()
        try:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Results directory already exists: {destination}")
            staged.rename(destination)
        finally:
            lock.rmdir()
    return destination
