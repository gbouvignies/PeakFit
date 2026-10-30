"""Read-only Milestone 3 project evidence; new science is always written natively."""

import gzip
import json
from types import MappingProxyType
from zipfile import ZipFile

import numpy as np

from peakfit.engine.algorithms.evaluation import AnalyticalFitStatistics, FitOutcomeClassification
from peakfit.engine.scientific.completed import FinalClusterSamples
from peakfit.engine.scientific.final_outcome import (
    FinalAnalyticalEvaluation,
    FinalClusterOutcome,
    FinalFitOutcome,
    FinalFitStatistics,
    FinalParameter,
    OptimizerProvenance,
)
from peakfit.io.schemas import FitSummarySchema
from peakfit.io.scientific import _validate_science, read_array_member


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
                cluster_id=c.cluster_id,
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


def read_legacy_project(directory, entry, record):
    """Bound old JSON expansion and sample arrays, then validate their terminal algebra."""
    filename = (
        "summary/fit.json.gz" if "summary/fit.json.gz" in entry["files"] else "summary/fit.json"
    )
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(directory / filename, "rb") as stream:
        payload = stream.read(512 * 1024**2 + 1)
    if len(payload) > 512 * 1024**2:
        raise ValueError("Expanded legacy project JSON exceeds the reader limit.")
    outcome = restore_outcome(json.loads(payload), record)
    with ZipFile(directory / "samples.npz") as archive:
        if (
            len(archive.namelist()) != len(set(archive.namelist()))
            or set(archive.namelist()) != {n + ".npy" for n in record["arrays"]}
            or sum(i.file_size for i in archive.infolist()) > 8 * 1024**3
        ):
            raise ValueError("Invalid legacy project sample members or expansion.")
        arrays = {
            name: read_array_member(archive, name + ".npy", d["dtype"], d["shape"])
            for name, d in record["arrays"].items()
        }
    samples = tuple(
        FinalClusterSamples(
            c.cluster_id,
            tuple(
                arrays[f"c{c.cluster_id}_index{dim}"]
                for dim in range(len(record["snapshot"]["data_shape"]) - 1)
            ),
            arrays[f"c{c.cluster_id}_data"],
            arrays[f"c{c.cluster_id}_corrections"],
        )
        for c in outcome.clusters
    )
    _validate_science(outcome, samples)
    return outcome, samples
