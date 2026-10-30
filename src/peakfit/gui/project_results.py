"""Strict numerical restoration and evaluable geometry for analysis projects."""

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import numpy as np

from peakfit.engine.algorithms.evaluation import AnalyticalFitStatistics, FitOutcomeClassification
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.lineshapes.create import SHAPES, create_shapes
from peakfit.fit.completed import CompletedFit, FinalClusterSamples
from peakfit.fit.final_outcome import (
    FinalAnalyticalEvaluation,
    FinalClusterOutcome,
    FinalFitOutcome,
    FinalFitStatistics,
    FinalParameter,
    OptimizerProvenance,
)
from peakfit.fit.simulation import FinalModelCluster, FinalModelSnapshot, project_final_outcome
from peakfit.gui.dataset_fit import DatasetFitResult
from peakfit.gui.fitting import FitInputSnapshot, PeakIdentity, PreparedDatasetFit
from peakfit.gui.peaks import WorkingPeak
from peakfit.io.schemas import FitSummarySchema

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


def model_records(result):
    """Store resolved per-peak, per-axis geometry without fitted quantities."""
    return [
        {
            "cluster_id": cluster.cluster_id,
            "peaks": [
                {
                    "name": peak.name,
                    "positions": peak.positions.tolist(),
                    "shapes": [
                        {
                            "name": s.shape_name,
                            "center": s.center,
                            "axis": s.axis,
                            "param_names": s.param_names,
                        }
                        for s in peak.shapes
                    ],
                }
                for peak in cluster.peaks
            ],
        }
        for cluster in result.simulation_snapshot.clusters
    ]


def _parameter(p):
    return FinalParameter(
        p.name,
        p.value,
        p.min_bound if p.min_bound is not None else -np.inf,
        p.max_bound if p.max_bound is not None else np.inf,
        p.vary,
        p.unit,
        p.standard_error if p.standard_error is not None else np.nan,
    )


def _array(values):
    array = np.asarray(values, dtype=float)
    array.flags.writeable = False
    return array


def restore_outcome(document, run):
    """Read the canonical 4.1 summary; never recover amplitudes from observations."""
    summary = FitSummarySchema.model_validate(document)
    if summary.noise <= 0:
        raise ValueError("Recorded statistical noise must be positive.")
    clustered = [p.model_dump() for c in summary.clusters for p in c.final_nonlinear_parameters]
    global_parameters = [p.model_dump() for p in summary.final_nonlinear_parameters]
    if sorted(clustered, key=lambda p: p["name"]) != global_parameters:
        raise ValueError("Canonical global and cluster nonlinear parameters disagree.")
    if len({p["name"] for p in clustered}) != len(clustered):
        raise ValueError("Canonical nonlinear parameters have ambiguous identities.")
    clusters = []
    for c in summary.clusters:
        e = c.analytical_evaluation
        evaluation = None
        if e is not None:
            arrays = {
                name: _array(getattr(e, name))
                for name in (
                    "shapes",
                    "amplitudes",
                    "amplitude_standard_errors",
                    "amplitude_covariance",
                    "scaled_amplitude_standard_errors",
                    "model_values",
                    "raw_residuals",
                    "normalized_residuals",
                )
            }
            evaluation = FinalAnalyticalEvaluation(
                c.cluster_id,
                **arrays,
                statistics=AnalyticalFitStatistics(**e.statistics.model_dump()),
                uncertainty_warning=e.uncertainty_warning,
            )
        p = c.optimizer_provenance
        provenance = OptimizerProvenance(
            p.optimizer_kind,
            p.success,
            p.termination_message,
            p.function_evaluations,
            p.jacobian_evaluations,
            p.iterations,
            p.optimality,
            p.final_cost if p.final_cost is not None else np.nan,
            p.correction_revision,
            MappingProxyType(p.metadata),
        )
        clusters.append(
            FinalClusterOutcome(
                c.cluster_id,
                tuple(c.peak_names),
                FitOutcomeClassification(c.classification),
                c.correction_revision,
                provenance,
                tuple(_parameter(p) for p in c.final_nonlinear_parameters),
                evaluation,
                c.unusable_reason,
            )
        )
    return FinalFitOutcome(
        tuple(clusters),
        MappingProxyType({c.cluster_id: c for c in clusters}),
        tuple(_parameter(p) for p in summary.final_nonlinear_parameters),
        summary.terminal_correction_revision,
        summary.noise,
        run["n_optimizer_passes"],
        run["n_correction_updates"],
        all(c.classification is FitOutcomeClassification.CONVERGED for c in clusters),
        FinalFitStatistics(**summary.statistics.model_dump()),
    )


def restore_result(run, document, arrays):  # noqa: PLR0912, PLR0915
    """Restore numerical review and construct geometry only from recorded descriptors."""
    outcome = restore_outcome(document, run)
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
    )
    config = PeakFitConfig.model_validate_json(snapshot.configuration)
    # Constructors consult calibration/processing only; there is no synthetic spectrum display.
    metadata = Spectra(
        dic={},
        data=np.empty((1, 1)),
        z_values=np.array(snapshot.plane_values),
        params=[SpectralParameters.model_validate_json(p) for p in snapshot.spectral_parameters],
    )
    samples, geometry, clusters = [], [], []
    for record, terminal in zip(run["models"], outcome.clusters, strict=True):
        cid = terminal.cluster_id
        if record["cluster_id"] != cid:
            raise ValueError("Model cluster identity mismatch.")
        prefix = f"c{cid}_"
        indices = tuple(arrays[prefix + f"index{dim}"] for dim in range(2))
        data, corrections = arrays[prefix + "data"], arrays[prefix + "corrections"]
        if data.shape != corrections.shape or data.shape != (
            len(indices[0]),
            len(snapshot.plane_values),
        ):
            raise ValueError("Invalid retained sample dimensions.")
        for dim, index in enumerate(indices):
            if (
                index.dtype.kind not in "iu"
                or index.shape != (data.shape[0],)
                or np.any(index < 0)
                or np.any(index >= snapshot.data_shape[dim + 1])
            ):
                raise ValueError("Invalid retained sample coordinates.")
        if not np.all(np.isfinite(data)) or not np.all(np.isfinite(corrections)):
            raise ValueError("Nonfinite retained observations or corrections.")
        e = terminal.analytical_evaluation
        if e is not None:
            if not np.isclose(
                np.sum(e.normalized_residuals**2), e.statistics.chi_squared, rtol=1e-12, atol=1e-12
            ):
                raise ValueError("Retained chi-squared disagrees with normalized residuals.")
            if not np.array_equal(data - corrections - e.model_values, e.raw_residuals):
                raise ValueError("Retained terminal residuals disagree with observations.")
            np.testing.assert_allclose(
                e.shapes.T @ e.amplitudes, e.model_values, rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                e.raw_residuals.ravel() / outcome.noise,
                e.normalized_residuals,
                rtol=1e-12,
                atol=1e-12,
            )
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
        samples.append(FinalClusterSamples(cid, indices, data, corrections))
        geometry.append(
            FinalModelCluster(
                cid,
                terminal.peak_names,
                tuple(peaks),
                data.shape[1],
                data.shape[0],
                2,
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
        CompletedFit(outcome, model, tuple(samples)),
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
        "models": model_records(result),
    }


def journal_entry(completion):
    """Compact review survives eviction of full historical arrays."""
    result = completion.result
    return {
        "run_id": result.run_id,
        "completed_at": result.completed_at,
        "status": "completed",
        "initial_parameters": initial_parameter_records(result),
        "source_files": result.inputs.snapshot.source_files,
        "spectral_parameters": result.inputs.snapshot.spectral_parameters,
        "plane_values": result.inputs.snapshot.plane_values,
        "identities": [asdict(i) for i in result.inputs.identities],
        "configuration": result.inputs.snapshot.configuration,
        "initial_peaks": [asdict(p) for p in result.inputs.snapshot.peaks],
        "working_peaks": [asdict(p) for p in completion.working_peaks],
        "parameters": [
            {
                "name": p.name,
                "value": p.value,
                "unit": p.unit,
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
