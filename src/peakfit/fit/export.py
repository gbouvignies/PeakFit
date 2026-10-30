"""Shared scientific export contract for CLI and retained GUI results."""

from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.scientific.coordinates import coordinate_table
from peakfit.engine.scientific.simulation import simulate_final_outcome
from peakfit.fit.run_models import RunSummary
from peakfit.io.scientific import write_scientific
from peakfit.io.writers.orchestrator import write_fit_outputs

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.scientific.completed import CompletedFit
    from peakfit.io.output_metadata import RunMetadata
    from peakfit.io.writers.config import WriterConfig


def model_records(snapshot):
    """Resolved geometry needed to evaluate the recorded scientific model."""
    return [
        {
            "cluster_id": c.cluster_id,
            "peaks": [
                {
                    "name": p.name,
                    "positions": p.positions,
                    "shapes": [
                        {
                            "name": s.shape_name,
                            "center": s.center,
                            "axis": s.axis,
                            "param_names": s.param_names,
                        }
                        for s in p.shapes
                    ],
                }
                for p in c.peaks
            ],
        }
        for c in snapshot.clusters
    ]


def write_completed_outputs(
    completed: CompletedFit,
    output_dir: Path,
    config: WriterConfig,
    *,
    metadata: RunMetadata,
    data_shape,
    spectral_parameters,
    plane_values,
    source_plane_indices,
    peak_identities=None,
    save_simulated=False,
    detailed_evidence=False,
    initial_coordinates=None,
    adopted_coordinates=None,
    export_state=None,
):
    """Export one authority with exact samples, calibration, identities and provenance.

    The binary scientific record is mandatory. Export-time editor facts are
    separate context; fitted evidence never changes with working-state edits.
    Detailed observations or a verified source support native parent continuation,
    without claiming chain/RNG resumption. Consumers dispatch on kind/schema_version;
    future analyses can define their own kind without changing fit semantics.
    """
    identities = (
        peak_identities
        if peak_identities is not None
        else [
            {"engine_name": p.name, "peak_id": p.name, "assignment": p.name}
            for c in completed.model_snapshot.clusters
            for p in c.peaks
        ]
    )
    context = {
        "metadata": asdict(metadata),
        "data_shape": tuple(data_shape),
        "spectral_parameters": spectral_parameters,
        "plane_values": np.asarray(plane_values),
        "source_plane_indices": np.asarray(source_plane_indices, dtype=np.int64),
        "peak_identities": identities,
        "coordinate_table": coordinate_table(
            completed,
            identities,
            spectral_parameters,
            metadata.run_id,
            initial=initial_coordinates,
            adopted=adopted_coordinates,
        ),
        "export_state": export_state
        if export_state is not None
        else {
            "scope": "dataset",
            "currentness": "not_applicable",
            "adoption_warnings": (),
        },
        "models": model_records(completed.model_snapshot),
    }
    files = {
        "scientific_result": write_scientific(
            output_dir / "scientific.npz",
            completed.outcome,
            completed.samples,
            context=context,
            evidence="detailed" if detailed_evidence else "results",
        )
    }
    files.update(
        write_fit_outputs(
            completed.outcome,
            output_dir,
            config,
            metadata=metadata,
            z_values=np.asarray(plane_values),
            source_plane_indices=np.asarray(source_plane_indices),
            summary=RunSummary.from_outcome(completed.outcome),
        )
    )
    if save_simulated:
        # Only shape is consulted by simulation; a broadcast scalar avoids a
        # redundant full spectrum allocation and works during numerical review.
        shape_only = np.broadcast_to(np.array(0.0), tuple(data_shape))
        simulated = simulate_final_outcome(completed.outcome, completed.model_snapshot, shape_only)
        if simulated is not None:
            path = output_dir / "simulated.npy"
            np.save(path, simulated, allow_pickle=False)
            files["simulated"] = path
    return files
