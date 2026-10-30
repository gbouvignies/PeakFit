"""Strict numerical restoration and evaluable geometry for analysis projects."""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.lineshapes.create import SHAPES, create_shapes
from peakfit.fit.completed import CompletedFit
from peakfit.fit.export import model_records
from peakfit.fit.simulation import FinalModelCluster, FinalModelSnapshot, project_final_outcome
from peakfit.gui.dataset_fit import DatasetFitResult
from peakfit.gui.fitting import FitInputSnapshot, PeakIdentity, PreparedDatasetFit
from peakfit.gui.peaks import WorkingPeak

if TYPE_CHECKING:
    from peakfit.engine.types import Shape


@dataclass
class UnavailableShape:
    """Recorded descriptor for numerical review, deliberately not evaluable geometry."""

    shape_name: str
    center: float
    axis: str
    param_names: list[str]
    cluster_id: int = 0


def read_peaks(records):
    """Validate stable working identities and exact unfolded coordinates."""
    peaks = tuple(WorkingPeak(p["id"], p["name"], tuple(p["positions"])) for p in records)
    if len({p.id for p in peaks}) != len(peaks) or any(not p.id for p in peaks):
        raise ValueError("Project peak identities must be unique and nonempty.")
    return peaks


def restore_result(run, outcome, retained_samples):  # noqa: PLR0912
    """Restore numerical review and construct geometry only from recorded descriptors."""
    raw = run["snapshot"]
    snapshot = FitInputSnapshot(
        0,
        tuple(raw["data_shape"]),
        tuple(raw["spectral_parameters"]),
        tuple(raw["plane_values"]),
        read_peaks(raw["peaks"]),
        raw["configuration"],
        raw["contour_level"],
        tuple(tuple(s) for s in raw["source_files"]),
        tuple(raw["source_plane_indices"]),
    )
    config = PeakFitConfig.model_validate_json(snapshot.configuration)
    # Constructors consult calibration/processing only; there is no synthetic spectrum display.
    metadata = Spectra(
        dic={},
        data=np.empty((1, 1)),
        z_values=np.array(snapshot.plane_values),
        params=[SpectralParameters.model_validate_json(p) for p in snapshot.spectral_parameters],
    )
    geometry, clusters = [], []
    for record, terminal, sample in zip(
        run["models"], outcome.clusters, retained_samples, strict=True
    ):
        cid = terminal.cluster_id
        if record["cluster_id"] != cid:
            raise ValueError("Model cluster identity mismatch.")
        indices = sample.grid_indices
        data, corrections = sample.data, sample.neighbor_corrections
        if len(indices) != len(snapshot.data_shape) - 1 or data.shape[1] != len(
            snapshot.plane_values
        ):
            raise ValueError("Retained samples disagree with the recorded source dimensions.")
        for index, length in zip(indices, snapshot.data_shape[1:], strict=True):
            if np.any(index >= length):
                raise ValueError("Retained sample coordinates exceed the source dimensions.")
        peaks = []
        for p in record["peaks"]:
            if all(s["name"] in SHAPES for s in p["shapes"]):
                shapes = create_shapes(
                    metadata,
                    config.fitting,
                    p["name"],
                    [s["center"] for s in p["shapes"]],
                    [s["name"] for s in p["shapes"]],
                )
            else:
                shapes = [
                    cast(
                        "Shape",
                        UnavailableShape(s["name"], s["center"], s["axis"], s["param_names"]),
                    )
                    for s in p["shapes"]
                ]

            for shape, descriptor in zip(shapes, p["shapes"], strict=True):
                if shape.axis != descriptor["axis"]:
                    raise ValueError("Recorded model axis mismatch.")
                shape.cluster_id = cid
                shape.param_names = descriptor["param_names"]
            peaks.append(Peak(p["name"], np.array(p["positions"]), shapes))
        if tuple(p.name for p in peaks) != terminal.peak_names:
            raise ValueError("Recorded model peak identity mismatch.")
        geometry.append(
            FinalModelCluster(
                cid,
                terminal.peak_names,
                tuple(peaks),
                data.shape[1],
                data.shape[0],
                len(indices),
                tuple(i.shape for i in indices),
            )
        )
        cluster = Cluster(cid, peaks, list(indices), data)
        cluster.corrections = corrections
        clusters.append(cluster)
    model = FinalModelSnapshot(
        tuple(geometry),
        outcome.noise,
        outcome.terminal_correction_revision,
        tuple((p.name, p.value) for p in outcome.final_nonlinear_parameters),
    )
    identities = tuple(PeakIdentity(**p) for p in run["identities"])
    if (
        {i.working_id for i in identities} != {p.id for p in snapshot.peaks}
        or len({i.engine_name for i in identities}) != len(identities)
        or {i.engine_name for i in identities} != {p.name for c in clusters for p in c.peaks}
    ):
        raise ValueError("Invalid run stable-ID mapping.")
    initial = Parameters()
    for p in run["initial_parameters"]:
        if p["name"] in initial:
            raise ValueError("Duplicate initial nonlinear parameter identity.")
        initial.add(
            p["name"],
            value=p["value"],
            min_value=p["min"] if p["min"] is not None else -np.inf,
            max_value=p["max"] if p["max"] is not None else np.inf,
            vary=p["vary"],
            unit=p["unit"],
        )
    prepared = PreparedDatasetFit(
        snapshot,
        identities,
        tuple(run["shape_names"]),
        outcome.noise,
        snapshot.contour_level,
        tuple(clusters),
        initial,
    )
    return DatasetFitResult(
        prepared,
        CompletedFit(outcome, model, retained_samples),
        tuple(run["warnings"]),
        run["execution_seconds"],
        run["software_version"],
        run["run_id"],
        run["completed_at"],
    )


def verify_model(result):
    """Evaluate saved final values at exact samples, with no preparation or solve."""
    if any(
        isinstance(shape, UnavailableShape)
        for cluster in result.simulation_snapshot.clusters
        for peak in cluster.peaks
        for shape in peak.shapes
    ):
        raise ValueError("Unsupported saved model descriptor; numerical review remains available.")
    parameters = Parameters()
    for p in result.outcome.final_nonlinear_parameters:
        parameters.add(p.name, value=p.value, min_value=p.min, max_value=p.max, vary=p.vary)
    for s in result.completed_fit.samples:
        e = result.outcome.cluster(s.cluster_id).analytical_evaluation
        if e is None:
            continue
        geometry = next(
            c for c in result.simulation_snapshot.clusters if c.cluster_id == s.cluster_id
        )
        shapes = Cluster.evaluate_peaks(geometry.peaks, parameters, list(s.grid_indices))
        np.testing.assert_allclose(shapes, e.shapes, rtol=1e-10, atol=1e-12)
        projected = project_final_outcome(
            result.outcome, result.simulation_snapshot, s.grid_indices, cluster_id=s.cluster_id
        )
        assert projected is not None
        np.testing.assert_allclose(projected, e.model_values, rtol=1e-10, atol=1e-9)


def run_record(result):
    """Small immutable run manifest; the canonical summary owns fitted quantities."""
    return {
        "run_id": result.run_id,
        "completed_at": result.completed_at,
        "software_version": result.software_version,
        "warnings": result.warnings,
        "execution_seconds": result.execution_seconds,
        "n_optimizer_passes": result.outcome.n_optimizer_passes,
        "n_correction_updates": result.outcome.n_correction_updates,
        "snapshot": asdict(result.inputs.snapshot),
        "identities": [asdict(i) for i in result.inputs.identities],
        "shape_names": result.inputs.shape_names,
        "initial_parameters": initial_parameter_records(result),
        "models": [
            dict(c, peaks=[dict(p, positions=p["positions"].tolist()) for p in c["peaks"]])
            for c in model_records(result.simulation_snapshot)
        ],
    }


def journal_entry(completion):
    """Compact review survives eviction of full historical arrays."""
    result = completion.result
    return {
        "run_id": result.run_id,
        "completed_at": result.completed_at,
        "status": "completed",
        "software_version": result.software_version,
        "noise": result.inputs.noise,
        "contour_level": result.inputs.contour_level,
        "shape_names": result.inputs.shape_names,
        "n_optimizer_passes": result.outcome.n_optimizer_passes,
        "n_correction_updates": result.outcome.n_correction_updates,
        "terminal_correction_revision": result.outcome.terminal_correction_revision,
        "initial_parameters": initial_parameter_records(result),
        "source_files": result.inputs.snapshot.source_files,
        "spectral_parameters": result.inputs.snapshot.spectral_parameters,
        "plane_values": result.inputs.snapshot.plane_values,
        "source_plane_indices": result.inputs.snapshot.source_plane_indices,
        "identities": [asdict(i) for i in result.inputs.identities],
        "configuration": result.inputs.snapshot.configuration,
        "initial_peaks": [asdict(p) for p in result.inputs.snapshot.peaks],
        "working_peaks": [asdict(p) for p in completion.working_peaks],
        "parameters": [
            {
                "name": p.name,
                "value": p.value,
                "unit": p.unit,
                "vary": p.vary,
                "standard_error": p.standard_error if np.isfinite(p.standard_error) else None,
                "min": p.min if np.isfinite(p.min) else None,
                "max": p.max if np.isfinite(p.max) else None,
            }
            for p in result.outcome.final_nonlinear_parameters
        ],
        "statistics": asdict(result.outcome.statistics),
        "diagnostics": [
            {
                "cluster_id": c.cluster_id,
                "classification": c.classification.value,
                "message": c.optimizer_provenance.termination_message,
                "reason": c.unusable_reason,
                "converged": c.optimizer_provenance.converged,
                "optimizer_kind": c.optimizer_provenance.optimizer_kind,
                "function_evaluations": c.optimizer_provenance.function_evaluations,
                "jacobian_evaluations": c.optimizer_provenance.jacobian_evaluations,
                "iterations": c.optimizer_provenance.iterations,
                "correction_revision": c.correction_revision,
                "uncertainty_warning": (
                    c.analytical_evaluation.uncertainty_warning if c.analytical_evaluation else None
                ),
                "statistics": (
                    asdict(c.analytical_evaluation.statistics) if c.analytical_evaluation else None
                ),
            }
            for c in result.outcome.clusters
        ],
    }


def initial_parameter_records(result):
    """Retain actual submitted values/bounds/freedom separately from fitted parameters."""
    return [
        {
            "name": p.name,
            "value": p.value,
            "min": p.min if np.isfinite(p.min) else None,
            "max": p.max if np.isfinite(p.max) else None,
            "vary": p.vary,
            "unit": p.unit,
        }
        for p in result.inputs.parameters.values()
    ]
