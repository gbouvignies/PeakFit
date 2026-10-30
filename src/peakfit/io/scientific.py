"""Lossless completed science: small metadata and named, typed NumPy arrays.

One immutable ZIP contains metadata.json and NPY members. No pickle, numeric
list projection, solver or external source access is involved. Export-time context
remains separate from the immutable fitted evidence.
"""

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from peakfit.engine.algorithms.evaluation import AnalyticalFitStatistics, FitOutcomeClassification
from peakfit.engine.scientific.completed import FinalClusterInputs, FinalClusterSamples
from peakfit.engine.scientific.coordinates import validate_coordinate_table
from peakfit.engine.scientific.final_outcome import (
    FinalAnalyticalEvaluation,
    FinalAnalyticalResult,
    FinalClusterOutcome,
    FinalFitOutcome,
    FinalFitStatistics,
    FinalParameter,
    OptimizerProvenance,
    freeze_provenance,
)

VERSION = "2.0.0"
MAX_METADATA_BYTES = 16 * 1024**2
MAX_ARRAY_BYTES = 8 * 1024**3
_SAMPLE_DIMENSIONS = 2


def write_scientific(
    path, outcome, samples, *, context=None, evidence="detailed", compression=ZIP_DEFLATED
):
    """Write exact numerical evidence; the caller owns atomic publication/fsync."""
    if evidence not in {"results", "detailed", "full"}:
        raise ValueError("Unknown scientific evidence policy.")
    if evidence != "results" and any(not isinstance(s, FinalClusterSamples) for s in samples):
        raise ValueError("Detailed evidence requires observations and terminal corrections.")
    _validate_science(outcome, samples)
    _validate_context(outcome, samples, context or {})
    arrays = {}

    def encode(value, name):  # noqa: PLR0911
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biufc":
                raise ValueError("Scientific arrays must have a numeric dtype.")
            arrays[name] = value
            return {"array": name, "dtype": value.dtype.str, "shape": value.shape}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, Enum):
            return value.value
        if is_dataclass(value) and not isinstance(value, type):
            return {
                f.name: encode(getattr(value, f.name), name + "/" + f.name)
                for f in fields(value)
                if not (
                    isinstance(value, FinalAnalyticalEvaluation)
                    and f.name
                    in (
                        {"model_values", "raw_residuals", "normalized_residuals"}
                        if evidence == "results"
                        else {"raw_residuals", "normalized_residuals"}
                        if evidence == "detailed"
                        else set()
                    )
                )
                and not (
                    isinstance(value, FinalClusterSamples)
                    and evidence == "results"
                    and f.name == "data"
                )
                and f.name != "by_cluster_id"
                and not (
                    isinstance(value, FinalFitOutcome) and f.name == "final_nonlinear_parameters"
                )
            }
        if isinstance(value, (set, frozenset)):
            return {
                "set": [
                    encode(v, name + "/" + str(i)) for i, v in enumerate(sorted(value, key=repr))
                ]
            }
        if isinstance(value, Mapping):
            return {
                "mapping": {
                    str(k): encode(v, name + "/" + str(i)) for i, (k, v) in enumerate(value.items())
                }
            }
        if isinstance(value, (tuple, list)):
            return [encode(v, name + "/" + str(i)) for i, v in enumerate(value)]
        if isinstance(value, float) and not math.isfinite(value):
            return {"float": struct.pack(">d", value).hex()}
        if value is None or isinstance(value, (str, float, int, bool)):
            return value
        raise TypeError(f"Unsupported scientific metadata: {type(value).__name__}")

    document = {
        "schema_version": VERSION,
        "kind": "peakfit_scientific_result",
        "evidence": evidence,
        "outcome": encode(outcome, "outcome"),
        "samples": encode(samples, "samples"),
        "context": encode(context or {}, "context"),
    }
    if evidence == "results":
        for record, sample in zip(document["samples"], samples, strict=True):
            if isinstance(sample, FinalClusterSamples):
                record["observation_sha256"] = hashlib.sha256(sample.data.tobytes()).hexdigest()
                record["observation_dtype"] = sample.data.dtype.str
    metadata = json.dumps(document, allow_nan=False, separators=(",", ":")).encode()
    if (
        len(metadata) > MAX_METADATA_BYTES
        or sum(a.nbytes for a in arrays.values()) > MAX_ARRAY_BYTES
    ):
        raise ValueError("Scientific evidence exceeds the supported reader limits.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=compression, compresslevel=1) as archive:
        archive.writestr("metadata.json", metadata)
        for name, array in arrays.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, array, allow_pickle=False)
    return path


def read_scientific(path):  # noqa: PLR0915
    """Validate bounded untrusted evidence and restore one immutable outcome."""
    with ZipFile(path) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        if len(names) != len(set(names)) or "metadata.json" not in names:
            raise ValueError("Invalid scientific archive members.")
        if archive.getinfo("metadata.json").file_size > MAX_METADATA_BYTES:
            raise ValueError("Scientific metadata exceeds the reader limit.")
        if sum(m.file_size for m in members) > MAX_ARRAY_BYTES:
            raise ValueError("Expanded scientific arrays exceed the reader limit.")
        document = json.loads(archive.read("metadata.json"))
        used = {"metadata.json"}

        def decode(value):  # noqa: PLR0911
            if isinstance(value, list):
                return [decode(v) for v in value]
            if not isinstance(value, dict):
                return value
            if set(value) == {"set"}:
                return frozenset(decode(v) for v in value["set"])
            if set(value) == {"mapping"}:
                return {k: decode(v) for k, v in value["mapping"].items()}
            if set(value) == {"float"}:
                return struct.unpack(">d", bytes.fromhex(value["float"]))[0]
            if set(value) == {"array", "dtype", "shape"}:
                name = value["array"] + ".npy"
                if name in used:
                    raise ValueError("Duplicate scientific array reference.")
                used.add(name)
                return read_array_member(archive, name, value["dtype"], value["shape"])

            return {k: decode(v) for k, v in value.items()}

        if (
            document.get("schema_version") not in {"1.0.0", VERSION}
            or document.get("kind") != "peakfit_scientific_result"
        ):
            raise ValueError("Unsupported scientific result version.")
        evidence = document.get("evidence", "full")
        if evidence not in {"results", "detailed", "full"}:
            raise ValueError("Unsupported scientific evidence policy.")
        decoded = decode(document)
        if used != set(names):
            raise ValueError("Unreferenced scientific array members.")
    samples = tuple(
        FinalClusterInputs(
            s["cluster_id"],
            tuple(s["grid_indices"]),
            s["neighbor_corrections"],
            s["observation_sha256"],
            s["observation_dtype"],
        )
        if evidence == "results"
        else FinalClusterSamples(
            s["cluster_id"], tuple(s["grid_indices"]), s["data"], s["neighbor_corrections"]
        )
        for s in decoded["samples"]
    )
    raw = decoded["outcome"]
    clusters = []
    for index, c in enumerate(raw.pop("clusters")):
        c["peak_names"] = tuple(c["peak_names"])
        c["classification"] = FitOutcomeClassification(c["classification"])
        c["final_nonlinear_parameters"] = tuple(
            FinalParameter(**p) for p in c["final_nonlinear_parameters"]
        )
        p = c["optimizer_provenance"]
        p["metadata"] = freeze_provenance(p["metadata"])
        c["optimizer_provenance"] = OptimizerProvenance(**p)
        e = c["analytical_evaluation"]
        if e is not None:
            e["statistics"] = AnalyticalFitStatistics(**e["statistics"])
            if evidence == "detailed":
                sample = samples[index]
                if not isinstance(sample, FinalClusterSamples):
                    raise ValueError("Missing detailed sample observations.")
                residual = sample.data - sample.neighbor_corrections - e["model_values"]
                normalized = residual.ravel() / raw["noise"]
                residual.flags.writeable = normalized.flags.writeable = False
                e.update(raw_residuals=residual, normalized_residuals=normalized)
            c["analytical_evaluation"] = (
                FinalAnalyticalResult(**e)
                if evidence == "results"
                else FinalAnalyticalEvaluation(**e)
            )
        clusters.append(FinalClusterOutcome(**c))
    if len({c.cluster_id for c in clusters}) != len(clusters):
        raise ValueError("Duplicate scientific cluster identity.")
    parameters = sorted(
        (p for c in clusters for p in c.final_nonlinear_parameters), key=lambda p: p.name
    )
    if len({p.name for p in parameters}) != len(parameters):
        raise ValueError("Duplicate scientific parameter identity.")
    raw["statistics"] = FinalFitStatistics(**raw["statistics"])
    outcome = FinalFitOutcome(
        clusters=tuple(clusters),
        by_cluster_id=MappingProxyType({c.cluster_id: c for c in clusters}),
        final_nonlinear_parameters=tuple(parameters),
        **raw,
    )
    _validate_science(outcome, samples)
    _validate_context(outcome, samples, decoded["context"])
    return outcome, samples, decoded["context"]


def _validate_science(outcome, samples):  # noqa: PLR0912, PLR0915
    """Check identities, dimensions and terminal algebra without a numerical solve."""
    if not np.isfinite(outcome.noise) or outcome.noise <= 0:
        raise ValueError("Recorded statistical noise must be positive and finite.")
    if tuple(s.cluster_id for s in samples) != tuple(c.cluster_id for c in outcome.clusters):
        raise ValueError("Retained sample cluster identities disagree with the outcome.")
    parameters = tuple(
        sorted(
            (p for c in outcome.clusters for p in c.final_nonlinear_parameters),
            key=lambda p: p.name,
        )
    )
    if outcome.final_nonlinear_parameters != parameters:
        raise ValueError("Global and cluster scientific parameters disagree.")
    if (
        outcome.terminal_correction_revision < 0
        or outcome.n_optimizer_passes < 1
        or outcome.n_correction_updates < 0
    ):
        raise ValueError("Invalid scientific terminal revisions or pass counts.")
    if outcome.overall_converged != all(
        c.classification is FitOutcomeClassification.CONVERGED for c in outcome.clusters
    ):
        raise ValueError("Scientific convergence classifications disagree.")
    peak_names = [name for c in outcome.clusters for name in c.peak_names]
    if len(set(peak_names)) != len(peak_names):
        raise ValueError("Duplicate scientific peak identity.")
    for c, s in zip(outcome.clusters, samples, strict=True):
        if (
            c.correction_revision != outcome.terminal_correction_revision
            or c.optimizer_provenance.correction_revision != c.correction_revision
        ):
            raise ValueError("Scientific correction revisions disagree.")
        for p in c.final_nonlinear_parameters:
            if not np.isfinite(p.value) or not p.min <= p.value <= p.max:
                raise ValueError("Invalid scientific parameter value/bounds.")
        if any(i.dtype.kind not in "iu" or i.ndim != 1 or np.any(i < 0) for i in s.grid_indices):
            raise ValueError("Invalid retained sample coordinates.")
        e = c.analytical_evaluation
        if c.usable != (e is not None):
            raise ValueError("Scientific usability disagrees with retained evaluation.")
        if e is not None:
            k = len(c.peak_names)
            if (
                e.cluster_id != c.cluster_id
                or e.amplitudes.ndim != _SAMPLE_DIMENSIONS
                or e.amplitudes.shape[0] != k
                or e.amplitude_standard_errors.shape != (k,)
                or e.amplitude_covariance.shape != (k, k)
                or e.scaled_amplitude_standard_errors.shape != (k,)
                or not np.all(np.isfinite(e.amplitudes))
                or not np.all(np.isfinite(e.shapes))
            ):
                raise ValueError("Invalid published analytical result dimensions or identity.")
        if (
            not s.grid_indices
            or any(i.shape != s.grid_indices[0].shape for i in s.grid_indices)
            or s.neighbor_corrections.ndim != _SAMPLE_DIMENSIONS
            or s.neighbor_corrections.shape[0] != len(s.grid_indices[0])
            or not np.all(np.isfinite(s.neighbor_corrections))
        ):
            raise ValueError("Invalid terminal corrections or sample mappings.")
        if e is not None and (
            e.shapes.shape != (len(c.peak_names), len(s.grid_indices[0]))
            or e.amplitudes.shape[1] != s.neighbor_corrections.shape[1]
            or e.statistics.n_observations != s.neighbor_corrections.size
        ):
            raise ValueError("Published result disagrees with sample mappings.")
        if not isinstance(s, FinalClusterSamples):
            if (
                re.fullmatch(r"[a-f0-9]{64}", s.observation_sha256) is None
                or np.dtype(s.observation_dtype).kind not in "fiu"
            ):
                raise ValueError("Invalid observation reconstruction identity.")
            if isinstance(e, FinalAnalyticalEvaluation):
                raise ValueError("Sample evidence requires retained inputs.")
            continue
        if s.data.ndim != _SAMPLE_DIMENSIONS or s.data.shape != s.neighbor_corrections.shape:
            raise ValueError("Invalid retained sample dimensions.")
        if not s.grid_indices or any(
            i.dtype.kind not in "iu" or i.shape != (s.data.shape[0],) or np.any(i < 0)
            for i in s.grid_indices
        ):
            raise ValueError("Invalid retained sample coordinates.")
        if not np.all(np.isfinite(s.data)) or not np.all(np.isfinite(s.neighbor_corrections)):
            raise ValueError("Nonfinite retained observations or corrections.")
        e = c.analytical_evaluation
        if c.usable != (e is not None):
            raise ValueError("Scientific usability disagrees with retained evaluation.")
        if e is None:
            continue
        if not isinstance(e, FinalAnalyticalEvaluation):
            raise ValueError("Detailed samples require sample predictions.")
        n_peaks = len(c.peak_names)
        n_points, n_planes = s.data.shape
        expected = {
            "shapes": (n_peaks, n_points),
            "amplitudes": (n_peaks, n_planes),
            "amplitude_standard_errors": (n_peaks,),
            "amplitude_covariance": (n_peaks, n_peaks),
            "scaled_amplitude_standard_errors": (n_peaks,),
            "model_values": s.data.shape,
            "raw_residuals": s.data.shape,
            "normalized_residuals": (s.data.size,),
        }
        if e.cluster_id != c.cluster_id or any(
            getattr(e, name).shape != shape for name, shape in expected.items()
        ):
            raise ValueError("Invalid analytical evaluation dimensions or identity.")
        for name in (
            "shapes",
            "amplitudes",
            "model_values",
            "raw_residuals",
            "normalized_residuals",
        ):
            if not np.all(np.isfinite(getattr(e, name))):
                raise ValueError("Nonfinite analytical evaluation.")
        if not np.array_equal(s.data - s.neighbor_corrections - e.model_values, e.raw_residuals):
            raise ValueError("Retained terminal residuals disagree with observations.")
        np.testing.assert_allclose(e.shapes.T @ e.amplitudes, e.model_values, rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(
            e.raw_residuals.ravel() / outcome.noise, e.normalized_residuals, rtol=1e-10, atol=1e-9
        )
        np.testing.assert_allclose(
            np.sum(e.normalized_residuals**2), e.statistics.chi_squared, rtol=1e-10, atol=1e-9
        )
        if e.statistics.n_observations != s.data.size:
            raise ValueError("Recorded observation count disagrees with samples.")


def _validate_context(outcome, samples, context):  # noqa: PLR0912
    """Validate export coordinate/identity associations when full context is present."""
    if not context:
        return
    shape = tuple(context["data_shape"])
    planes = context["plane_values"]
    if planes.shape != (shape[0],) or not np.all(np.isfinite(planes)):
        raise ValueError("Invalid scientific plane mapping.")
    source_indices = context["source_plane_indices"]
    if (
        source_indices.shape != planes.shape
        or source_indices.dtype.kind not in "iu"
        or np.any(source_indices < 0)
        or len(np.unique(source_indices)) != len(planes)
    ):
        raise ValueError("Invalid scientific source plane indices.")
    if len(context["spectral_parameters"]) != len(shape):
        raise ValueError("Invalid scientific spectral calibration dimensions.")
    if [c["cluster_id"] for c in context["models"]] != [c.cluster_id for c in outcome.clusters]:
        raise ValueError("Scientific model cluster identities disagree.")
    names = {name for c in outcome.clusters for name in c.peak_names}
    identity_records = context["peak_identities"]
    if any(set(i) != {"engine_name", "peak_id", "assignment"} for i in identity_records):
        raise ValueError("Invalid scientific peak identity fields.")
    if len({i["peak_id"] for i in identity_records}) != len(identity_records):
        raise ValueError("Duplicate scientific source peak identity.")
    identities = [i["engine_name"] for i in identity_records]
    if len(set(identities)) != len(identities) or set(identities) != names:
        raise ValueError("Scientific peak identity mapping disagrees with the outcome.")
    state = context["export_state"]
    if (
        set(state) != {"scope", "currentness", "adoption_warnings"}
        or state["scope"] not in {"dataset", "cluster"}
        or state["currentness"] not in {"current", "outdated", "unknown", "not_applicable"}
    ):
        raise ValueError("Invalid scientific export-time state.")
    validate_coordinate_table(context["coordinate_table"], outcome, context)
    for sample in samples:
        if (
            sample.neighbor_corrections.shape[1] != len(planes)
            or len(sample.grid_indices) != len(shape) - 1
        ):
            raise ValueError("Scientific sample dimensions disagree with the source mapping.")
        for index, length in zip(sample.grid_indices, shape[1:], strict=True):
            if np.any(index >= length):
                raise ValueError("Scientific samples are outside the source mapping.")


def read_array_member(archive, name, dtype_name, dimensions):
    """Read a bounded NPY member for native archives and legacy project migration."""
    dtype = np.dtype(dtype_name)
    shape = dimensions
    if dtype.kind not in "biufc" or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Invalid scientific array descriptor.")
    if math.prod(shape) * dtype.itemsize > MAX_ARRAY_BYTES:
        raise ValueError("Scientific array exceeds the reader limit.")
    with archive.open(name) as stream:
        # Check the NPY header before permitting NumPy to allocate.
        version = np.lib.format.read_magic(stream)
        reader = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }.get(version)
        if reader is None:
            raise ValueError("Unsupported scientific array header.")
        actual_shape, _, actual_dtype = reader(stream)
        expected_bytes = math.prod(shape) * dtype.itemsize
        if archive.getinfo(name).file_size != stream.tell() + expected_bytes:
            raise ValueError("Scientific array payload size mismatch.")
        if tuple(shape) != actual_shape or dtype != actual_dtype:
            raise ValueError("Scientific array shape/dtype mismatch.")
    with archive.open(name) as stream:
        array = np.lib.format.read_array(stream, allow_pickle=False)
    array.flags.writeable = False
    return array
