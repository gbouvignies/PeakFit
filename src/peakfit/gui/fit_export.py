"""Staged scientific export of a retained isolated fit, without Qt or continuation state."""

import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np

from peakfit.fit.output_metadata import RunMetadata
from peakfit.io.readers.results import ResultsLoader
from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.orchestrator import write_fit_outputs

if TYPE_CHECKING:
    from peakfit.gui.local_fit import LocalFitResult


def export_local_fit(result: LocalFitResult, destination: Path) -> Path:
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
        timestamp=result.completed_at,
        software_version=result.software_version,
        input_files={name: json.loads(record) for name, record in snapshot.source_files},
        configuration=configuration,
        run_duration_seconds=result.execution_seconds,
    )
    companion = {
        "schema_version": "1.0.0",
        "kind": "peakfit_gui_local_fit",
        "run_id": result.run_id,
        "completed_at": result.completed_at,
        "software_version": result.software_version,
        "scope": "isolated_cluster_all_original_planes",
        "cross_cluster_corrections": False,
        "n_optimizer_passes": result.outcome.n_optimizer_passes,
        "n_correction_updates": result.outcome.n_correction_updates,
        "cluster_id": inputs.scope.cluster_id,
        "data_shape": snapshot.data_shape,
        "spectral_parameters": [json.loads(axis) for axis in snapshot.spectral_parameters],
        "planes": [
            {"index": index, "value": value} for index, value in enumerate(snapshot.plane_values)
        ],
        "grid_indices": [indices.tolist() for indices in inputs.scope.grid_indices],
        "peak_identities": [asdict(identity) for identity in inputs.identities],
        "original_working_peaks": [asdict(peak) for peak in snapshot.peaks],
        "noise": inputs.noise,
        "contour_threshold": inputs.contour_level,
        "shape_names": inputs.shape_names,
        "configuration": configuration,
        "optimizer_configuration": asdict(result.optimizer_config),
        "warnings": result.warnings,
        "execution_seconds": result.execution_seconds,
        "peak_memory_bytes": result.peak_memory_bytes,
        "source_files": metadata.input_files,
        "source_note": "Original files verified at load; working peaks may be edited. "
        "In-memory spectrum identity is not persistent and is omitted.",
    }
    with TemporaryDirectory(prefix=".peakfit-export-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "results"
        files = write_fit_outputs(
            result.outcome,
            staged,
            WriterConfig(precision=17),
            metadata=metadata,
            z_values=np.asarray(snapshot.plane_values),
        )
        provenance = staged / "summary" / "local_fit.json"
        provenance.write_text(
            json.dumps(companion, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        ResultsLoader(staged).load_summary()
        if not all(
            path.is_file() and path.stat().st_size for path in [*files.values(), provenance]
        ):
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
