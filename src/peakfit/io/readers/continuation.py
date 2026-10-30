"""Shared, validated scientific reconstruction and continuation without GUI state."""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import numpy as np

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.params_vector import FitParameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.domain.state import FittingState
from peakfit.engine.lineshapes.create import SHAPES, create_shapes
from peakfit.engine.scientific.completed import CompletedFit, FinalClusterSamples
from peakfit.engine.scientific.final_outcome import FinalAnalyticalEvaluation
from peakfit.engine.scientific.simulation import FinalModelCluster, FinalModelSnapshot
from peakfit.io.readers.spectrum import read_spectra
from peakfit.io.scientific import read_scientific

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


def restore_geometry(
    outcome, retained_samples, records, spectral_parameters, plane_values, data_shape, configuration
):
    """Restore recorded shapes and exact terminal inputs; no preparation or solve."""
    config = PeakFitConfig.model_validate(configuration)
    # Constructors consult calibration/processing only; there is no synthetic spectrum display.
    metadata = Spectra(
        dic={},
        data=np.empty((1, 1)),
        z_values=np.array(plane_values),
        params=[SpectralParameters.model_validate_json(p) for p in spectral_parameters],
    )
    geometry, clusters = [], []
    for record, terminal, sample in zip(records, outcome.clusters, retained_samples, strict=True):
        cid = terminal.cluster_id
        if record["cluster_id"] != cid:
            raise ValueError("Model cluster identity mismatch.")
        indices = sample.grid_indices
        data, corrections = sample.data, sample.neighbor_corrections
        if len(indices) != len(data_shape) - 1 or data.shape[1] != len(plane_values):
            raise ValueError("Retained samples disagree with the recorded source dimensions.")
        for index, length in zip(indices, data_shape[1:], strict=True):
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
    return model, clusters


def restore_completed(outcome, samples, context, *, source_data=None):
    """Reconstruct exact inputs and validate the recorded evaluator before continuation."""
    shape = tuple(context["data_shape"])
    full = []
    for sample in samples:
        if isinstance(sample, FinalClusterSamples):
            full.append(sample)
        else:
            if source_data is None:
                raise ValueError(
                    "Compact results require a verified spectrum for continuation. "
                    "Use a detailed-evidence export for source-independent continuation."
                )
            values = np.asarray(
                source_data[(slice(None), *sample.grid_indices)].T,
                dtype=np.dtype(sample.observation_dtype),
            )
            if hashlib.sha256(values.tobytes()).hexdigest() != sample.observation_sha256:
                raise ValueError(
                    "Reconstructed observations disagree with their exact recorded identity."
                )
            values.flags.writeable = False
            full.append(
                FinalClusterSamples(
                    sample.cluster_id, sample.grid_indices, values, sample.neighbor_corrections
                )
            )
    full = tuple(full)
    model, clusters = restore_geometry(
        outcome,
        full,
        context["models"],
        [json.dumps(p) for p in context["spectral_parameters"]],
        context["plane_values"],
        shape,
        context["metadata"]["configuration"],
    )
    if any(isinstance(s, UnavailableShape) for c in clusters for p in c.peaks for s in p.shapes):
        raise ValueError("Unsupported saved model descriptor; continuation is unavailable.")
    params = Parameters()
    for p in outcome.final_nonlinear_parameters:
        params.add(
            p.name,
            value=p.value,
            min_value=p.min,
            max_value=p.max,
            vary=p.vary,
            stderr=p.standard_error,
            unit=p.unit,
        )
    restored = []
    for terminal, cluster in zip(outcome.clusters, clusters, strict=True):
        e = terminal.analytical_evaluation
        if e is None:
            restored.append(terminal)
            continue
        evaluated = Cluster.evaluate_peaks(cluster.peaks, params, cluster.grid_indices)
        np.testing.assert_allclose(
            evaluated,
            e.shapes,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Saved model evaluator no longer reproduces its reference",
        )
        predicted = e.shapes.T @ e.amplitudes
        if isinstance(e, FinalAnalyticalEvaluation):
            np.testing.assert_allclose(predicted, e.model_values, rtol=1e-10, atol=1e-9)
            restored.append(terminal)
            continue
        residual = cluster.corrected_data - predicted
        normalized = residual.ravel() / outcome.noise
        np.testing.assert_allclose(
            np.sum(normalized**2),
            e.statistics.chi_squared,
            rtol=1e-8,
            atol=1e-9,
            err_msg="Reconstructed terminal problem disagrees with recorded fit",
        )
        for array in (predicted, residual, normalized):
            array.flags.writeable = False
        values = {f.name: getattr(e, f.name) for f in fields(e)}
        restored.append(
            replace(
                terminal,
                analytical_evaluation=FinalAnalyticalEvaluation(
                    **values,
                    model_values=predicted,
                    raw_residuals=residual,
                    normalized_residuals=normalized,
                ),
            )
        )
    outcome = replace(
        outcome,
        clusters=tuple(restored),
        by_cluster_id=MappingProxyType({c.cluster_id: c for c in restored}),
    )
    return CompletedFit(outcome, model, full)


def load_completed(path, *, source=None):
    """Load detailed evidence or verify the exact external source for compact results."""
    path = Path(path)
    outcome, samples, context = read_scientific(path)
    source_data = None
    if any(not isinstance(s, FinalClusterSamples) for s in samples):
        info = context["metadata"]["input_files"].get("spectrum", {})
        filename = Path(info.get("path", "__missing_spectrum__")).name
        candidates = (
            [Path(source)]
            if source is not None
            else [
                Path(info.get("path", "__missing_spectrum__")),
                path.parent / filename,
                path.parent.parent / filename,
                path.parent.parent / "data" / filename,
                path.parent.parent.parent / "data" / filename,
            ]
        )
        candidates = [p for p in dict.fromkeys(candidates) if p.is_file()]
        if not candidates:
            raise FileNotFoundError(
                "Compact results require their verified spectrum. "
                "Relink the source or export detailed evidence before moving it."
            )
        candidate = None
        for location in candidates:
            before = location.stat()
            with location.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest == info.get("checksum_sha256"):
                candidate = location
                break
        if candidate is None:
            raise ValueError("Continuation source SHA-256 disagrees with the scientific result.")
        spectra = read_spectra(candidate, None)
        after = candidate.stat()
        if any(
            getattr(before, field) != getattr(after, field)
            for field in ("st_size", "st_mtime_ns", "st_ctime_ns", "st_ino", "st_dev")
        ):
            raise ValueError(
                "Continuation source changed during verification. Retry with a stable source."
            )
        indices = np.asarray(context["source_plane_indices"])
        if np.any(indices >= spectra.data.shape[0]):
            raise ValueError("Continuation plane mapping exceeds the source spectrum.")
        source_data = spectra.data[indices]
        if tuple(source_data.shape) != tuple(context["data_shape"]):
            raise ValueError("Continuation source dimensions disagree.")
        if [p.model_dump() for p in spectra.params] != context["spectral_parameters"]:
            raise ValueError("Continuation spectral calibration disagrees.")
    return restore_completed(outcome, samples, context, source_data=source_data)


def continuation_state(completed):
    """Build mutable solver inputs from validated immutable scientific evidence."""
    params = Parameters()
    for p in completed.outcome.final_nonlinear_parameters:
        params.add(
            p.name,
            value=p.value,
            min_value=p.min,
            max_value=p.max,
            vary=p.vary,
            stderr=p.standard_error,
            unit=p.unit,
        )
    clusters = []
    for geometry, sample in zip(completed.model_snapshot.clusters, completed.samples, strict=True):
        if not completed.outcome.cluster(sample.cluster_id).usable:
            raise ValueError("Unusable parent cluster cannot be continued; adjust and refit it.")
        cluster = Cluster(
            sample.cluster_id,
            list(deepcopy(geometry.peaks)),
            list(sample.grid_indices),
            sample.data.copy(),
        )
        cluster.corrections = sample.neighbor_corrections.copy()
        clusters.append(cluster)
    peaks = [p for c in clusters for p in c.peaks]
    return FittingState(
        clusters=clusters,
        params=FitParameters.from_parameters(params, peaks),
        scalar_params=params,
        noise=completed.outcome.noise,
    )
