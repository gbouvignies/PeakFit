"""Full-grid simulation projected from an authoritative final fit outcome."""

from __future__ import annotations

from concurrent.futures import CancelledError
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.params_scalar import Parameters

_MIN_SPECTRUM_NDIM = 2

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from peakfit.engine.domain.peaks import Peak
    from peakfit.engine.domain.state import FittingState
    from peakfit.engine.scientific.completion import CorrectionSnapshot
    from peakfit.engine.scientific.final_outcome import (
        FinalAnalyticalResult,
        FinalClusterOutcome,
        FinalFitOutcome,
        FinalParameter,
    )
    from peakfit.shared.typing import FloatArray


@dataclass(frozen=True, slots=True)
class FinalModelCluster:
    """Frozen geometry needed to project one terminal cluster onto a full grid."""

    cluster_id: int
    peak_names: tuple[str, ...]
    peaks: tuple[Peak, ...]
    n_series: int
    n_points: int
    n_grid_dimensions: int
    grid_index_shapes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class FinalModelSnapshot:
    """Verified model geometry accompanying one immutable final outcome.

    The snapshot deliberately contains only cloned lineshape geometry and
    identity metadata. Numerical parameters and amplitudes remain authoritative
    only in :class:`FinalFitOutcome`.
    """

    clusters: tuple[FinalModelCluster, ...]

    @classmethod
    def capture(
        cls,
        outcome: FinalFitOutcome,
        state: FittingState,
        correction_snapshot: CorrectionSnapshot,
    ) -> FinalModelSnapshot:
        """Copy the minimal full-grid model geometry at the finalization boundary."""
        _validate_capture_inputs(outcome, state, correction_snapshot)
        state_clusters = {cluster.cluster_id: cluster for cluster in state.clusters}
        clusters = tuple(
            _freeze_cluster_geometry(state_clusters[cluster_outcome.cluster_id])
            for cluster_outcome in outcome.clusters
        )
        return cls(clusters=clusters)


def simulate_final_outcome(
    outcome: FinalFitOutcome,
    snapshot: FinalModelSnapshot,
    data: FloatArray,
) -> FloatArray | None:
    """Project usable final outcomes onto the full spectrum grid.

    This is a deterministic lineshape evaluation, not a fit: it evaluates the
    stored final geometry with frozen final nonlinear values and retained
    analytical amplitudes. ``None`` explicitly represents an all-unusable run.
    """
    output_data = np.asarray(data)
    n_series, grid_shape, positions = _full_grid_positions(output_data, snapshot)
    if any(cluster.n_series != n_series for cluster in snapshot.clusters):
        raise ValueError("Cluster series count does not match output series count")
    model = project_final_outcome(outcome, snapshot, positions)
    if model is None:
        return None
    return model.T.reshape((output_data.shape[0], *grid_shape)).astype(float)


def project_final_outcome(
    outcome: FinalFitOutcome,
    snapshot: FinalModelSnapshot,
    positions: Sequence[np.ndarray],
    *,
    cluster_id: int | None = None,
    planes: Sequence[int] | None = None,
    canceled: Callable[[], bool] | None = None,
) -> FloatArray | None:
    """Evaluate completed contributions at paired fractional grid coordinates.

    Returns (points, requested planes), preserving the supplied plane identities
    and order. Uses final values, retained geometry and analytical amplitudes;
    there is no fitting, parameter preparation, or amplitude recovery here.
    Shape evaluation is chunked to bound temporary memory independently of grid
    extent. Callers choose the coordinates/output extent (a row, tile or grid).
    ``planes`` indexes fitted-series columns, not source-file indices after plane
    exclusion. The caller must retain that mapping. ``None`` denotes no usable
    contribution, never a zero-filled failed result.
    An optional cancellation predicate is checked between bounded numerical chunks;
    cancellation raises ``CancelledError`` and never returns a partial model.
    """
    _validate_snapshot(outcome, snapshot)
    coordinates = [np.asarray(values, dtype=float).ravel() for values in positions]
    if (
        not coordinates
        or not coordinates[0].size
        or any(
            values.shape != coordinates[0].shape or not np.all(np.isfinite(values))
            for values in coordinates
        )
    ):
        raise ValueError("Projection coordinates must be finite, nonempty paired arrays")
    if {c.n_grid_dimensions for c in snapshot.clusters} != {len(coordinates)}:
        raise ValueError("Projection coordinate dimension mismatch")
    n_series = snapshot.clusters[0].n_series
    selected = _selected_planes(planes, n_series)
    if cluster_id is not None and cluster_id not in outcome.by_cluster_id:
        raise ValueError(f"Unknown projection cluster_id {cluster_id}")
    parameters = _parameters_from_outcome(outcome.final_nonlinear_parameters)
    model = np.zeros((coordinates[0].size, selected.size), dtype=float)
    usable_count = 0
    for cluster in snapshot.clusters:
        if canceled is not None and canceled():
            raise CancelledError
        if cluster_id is not None and cluster.cluster_id != cluster_id:
            continue
        terminal = outcome.cluster(cluster.cluster_id)
        if not terminal.usable:
            continue
        evaluation = _validate_usable_projection(terminal, cluster, n_series)
        amplitudes = evaluation.amplitudes[:, selected]
        # At most about 4 MiB of shape/output working arrays per chunk.
        chunk_size = max(1, min(4096, 524288 // max(len(cluster.peaks), selected.size)))
        for start in range(0, model.shape[0], chunk_size):
            if canceled is not None and canceled():
                raise CancelledError
            end = start + chunk_size
            shapes = np.asarray(
                Cluster.evaluate_peaks(
                    cluster.peaks, parameters, [values[start:end] for values in coordinates]
                ),
                dtype=float,
            )
            if shapes.shape != (len(cluster.peaks), len(model[start:end])) or not np.all(
                np.isfinite(shapes)
            ):
                raise ValueError(
                    f"Invalid projected lineshapes for cluster_id {cluster.cluster_id}"
                )
            model[start:end] += shapes.T @ amplitudes
        usable_count += 1
    if not usable_count:
        return None
    if not np.all(np.isfinite(model)):
        raise ValueError("Non-finite completed model projection")
    return model


def project_final_grid(
    outcome: FinalFitOutcome,
    snapshot: FinalModelSnapshot,
    grid_shape: Sequence[int],
    *,
    planes: Sequence[int] | None = None,
    canceled: Callable[[], bool] | None = None,
) -> FloatArray | None:
    """Evaluate completed contributions at every point of the native grid.

    The same model as ``project_final_outcome`` at the grid's integer coordinates,
    computed from per-axis shapes: each peak shape is the product of its
    per-dimension shapes, so the model of a plane is the sum over peaks of the
    amplitude times the outer product of those factors. No cutoffs. Returns
    (requested planes, *grid_shape); ``None`` when no cluster is usable.
    Cancellation is checked between clusters and raises ``CancelledError``.
    """
    _validate_snapshot(outcome, snapshot)
    shape = tuple(int(n) for n in grid_shape)
    if {c.n_grid_dimensions for c in snapshot.clusters} != {len(shape)} or min(shape) <= 0:
        raise ValueError("Projection grid shape must give one positive size per dimension")
    selected = _selected_planes(planes, snapshot.clusters[0].n_series)
    parameters = _parameters_from_outcome(outcome.final_nonlinear_parameters)
    grids = [np.arange(n, dtype=float) for n in shape]
    # Peaks p, planes a, one letter per spectral dimension.
    dims = "ijklmn"[: len(shape)]
    subscripts = ",".join(["pa", *(f"{d}p" for d in dims)]) + f"->a{dims}"
    model = np.zeros((selected.size, *shape), dtype=float)
    usable_count = 0
    for cluster in snapshot.clusters:
        if canceled is not None and canceled():
            raise CancelledError
        terminal = outcome.cluster(cluster.cluster_id)
        if not terminal.usable:
            continue
        evaluation = _validate_usable_projection(terminal, cluster, cluster.n_series)
        factors = Cluster.evaluate_peak_axes(cluster.peaks, parameters, grids)
        if any(
            f.shape != (n, len(cluster.peaks)) or not np.all(np.isfinite(f))
            for f, n in zip(factors, shape, strict=True)
        ):
            raise ValueError(f"Invalid projected lineshapes for cluster_id {cluster.cluster_id}")
        amplitudes = evaluation.amplitudes[:, selected]
        model += np.einsum(subscripts, amplitudes, *factors, optimize=True)
        usable_count += 1
    if not usable_count:
        return None
    if not np.all(np.isfinite(model)):
        raise ValueError("Non-finite completed model projection")
    return model


def project_final_residual(
    outcome: FinalFitOutcome,
    snapshot: FinalModelSnapshot,
    positions: Sequence[np.ndarray],
    observed_values: FloatArray,
    *,
    planes: Sequence[int] | None = None,
    normalize: bool = False,
) -> FloatArray | None:
    """Return observations minus the sum of usable completed cluster models.

    Observations must be (points, requested fitted-series columns), in exactly
    the supplied coordinate/plane order. Source-data provenance is the caller's
    responsibility. This display residual is defined outside fitting masks too;
    it is NOT the recorded optimizer residual against frozen neighbor corrections
    and must not replace that residual or its statistics. No values are written
    back into the outcome. All-unusable results return None.
    """
    model = project_final_outcome(outcome, snapshot, positions, planes=planes)
    if model is None:
        return None
    observed = np.asarray(observed_values, dtype=float)
    if observed.shape != model.shape or not np.all(np.isfinite(observed)):
        raise ValueError("Observed values must be finite with the projected point-by-plane shape")
    residual = observed - model
    return residual / outcome.noise if normalize else residual


def _validate_capture_inputs(
    outcome: FinalFitOutcome,
    state: FittingState,
    correction_snapshot: CorrectionSnapshot,
) -> None:
    if state.noise is None or not np.isfinite(state.noise) or state.noise <= 0.0:
        raise ValueError(
            f"Simulation snapshot noise must be positive and finite, got {state.noise}"
        )
    if state.noise != outcome.noise:
        raise ValueError(
            "Simulation snapshot noise does not match final outcome: "
            f"expected {outcome.noise}, got {state.noise}"
        )
    if correction_snapshot.revision != outcome.terminal_correction_revision:
        raise ValueError(
            "Simulation snapshot correction revision does not match final outcome: "
            f"expected {outcome.terminal_correction_revision}, got {correction_snapshot.revision}"
        )
    state_clusters = _clusters_by_id(state.clusters, label="continuation state")
    outcome_ids = {cluster.cluster_id for cluster in outcome.clusters}
    _require_exact_ids(outcome_ids, set(state_clusters), label="simulation snapshot")
    _require_exact_ids(
        outcome_ids, set(correction_snapshot.corrections), label="correction snapshot"
    )
    for cluster_outcome in outcome.clusters:
        cluster = state_clusters[cluster_outcome.cluster_id]
        _validate_cluster_geometry(cluster_outcome, cluster)
        correction = np.asarray(correction_snapshot.corrections[cluster.cluster_id])
        if correction.shape != cluster.data.shape:
            raise ValueError(
                "Simulation correction shape mismatch for "
                f"cluster_id {cluster.cluster_id}: expected {cluster.data.shape}, "
                f"got {correction.shape}"
            )
        if not np.array_equal(correction, cluster.corrections):
            raise ValueError(
                "Simulation correction snapshot differs from terminal state for "
                f"cluster_id {cluster.cluster_id}"
            )
    _validate_parameter_values(outcome, state.scalar_params)


def _freeze_cluster_geometry(cluster: Cluster) -> FinalModelCluster:
    return FinalModelCluster(
        cluster_id=cluster.cluster_id,
        peak_names=tuple(peak.name for peak in cluster.peaks),
        peaks=tuple(deepcopy(cluster.peaks)),
        n_series=cluster.n_series,
        n_points=cluster.n_points,
        n_grid_dimensions=len(cluster.grid_indices),
        grid_index_shapes=tuple(tuple(indices.shape) for indices in cluster.grid_indices),
    )


def _validate_snapshot(outcome: FinalFitOutcome, snapshot: FinalModelSnapshot) -> None:
    snapshot_ids = {cluster.cluster_id for cluster in snapshot.clusters}
    outcome_ids = {cluster.cluster_id for cluster in outcome.clusters}
    _require_exact_ids(outcome_ids, snapshot_ids, label="simulation snapshot")
    if len(snapshot_ids) != len(snapshot.clusters):
        raise ValueError("Simulation snapshot cluster_id values must be unique")
    for cluster_outcome in outcome.clusters:
        snapshot_cluster = next(
            cluster
            for cluster in snapshot.clusters
            if cluster.cluster_id == cluster_outcome.cluster_id
        )
        if snapshot_cluster.peak_names != cluster_outcome.peak_names:
            raise ValueError(
                "Simulation snapshot peak identity mismatch for "
                f"cluster_id {cluster_outcome.cluster_id}"
            )
        if snapshot_cluster.n_points <= 0 or snapshot_cluster.n_series <= 0:
            raise ValueError(
                "Simulation snapshot has invalid point or series count for "
                f"cluster_id {cluster_outcome.cluster_id}"
            )
        if len(snapshot_cluster.grid_index_shapes) != snapshot_cluster.n_grid_dimensions:
            raise ValueError(
                "Simulation snapshot grid-index dimension mismatch for "
                f"cluster_id {cluster_outcome.cluster_id}"
            )
        if any(
            int(np.prod(shape, dtype=np.intp)) != snapshot_cluster.n_points
            for shape in snapshot_cluster.grid_index_shapes
        ):
            raise ValueError(
                "Simulation snapshot grid-index shape mismatch for "
                f"cluster_id {cluster_outcome.cluster_id}"
            )


def _full_grid_positions(
    data: np.ndarray,
    snapshot: FinalModelSnapshot,
) -> tuple[int, tuple[int, ...], list[np.ndarray]]:
    if data.ndim < _MIN_SPECTRUM_NDIM:
        raise ValueError(
            f"Simulated spectrum data must have series and grid axes, got {data.shape}"
        )
    n_series = data.shape[0]
    if n_series == 0:
        raise ValueError("Simulated spectrum must contain at least one series")
    grid_shape = tuple(data.shape[1:])
    if any(size <= 0 for size in grid_shape):
        raise ValueError(f"Simulated spectrum grid dimensions must be positive, got {grid_shape}")
    dimensions = {cluster.n_grid_dimensions for cluster in snapshot.clusters}
    if dimensions != {len(grid_shape)}:
        raise ValueError(
            "Simulation grid dimension mismatch: "
            f"snapshot has {sorted(dimensions)}, output grid has {len(grid_shape)}"
        )
    grid_indices = np.indices(grid_shape, dtype=np.intp)
    if grid_indices.shape != (len(grid_shape), *grid_shape):
        raise ValueError(
            "Full-grid index shape mismatch: "
            f"expected {(len(grid_shape), *grid_shape)}, got {grid_indices.shape}"
        )
    positions = [index.ravel() for index in grid_indices]
    n_points = int(np.prod(grid_shape, dtype=np.intp))
    if not positions or any(position.shape != (n_points,) for position in positions):
        raise ValueError("Full-grid indices must contain one flat coordinate per output point")
    return n_series, grid_shape, positions


def _selected_planes(planes: Sequence[int] | None, n_series: int) -> np.ndarray:
    selected = np.arange(n_series) if planes is None else np.asarray(planes)
    if (
        selected.ndim != 1
        or not selected.size
        or selected.dtype.kind not in "iu"
        or (np.any(selected < 0) or np.any(selected >= n_series))
    ):
        raise ValueError("Projection planes must be valid original plane indices")
    return selected


def _validate_usable_projection(
    outcome: FinalClusterOutcome,
    snapshot: FinalModelCluster,
    n_series: int,
) -> FinalAnalyticalResult:
    evaluation = outcome.analytical_evaluation
    if evaluation is None:
        raise ValueError(
            "Usable final outcome is missing analytical evaluation for "
            f"cluster_id {outcome.cluster_id}"
        )
    if snapshot.n_series != n_series:
        raise ValueError(
            "Cluster series count does not match spectrum series count for "
            f"cluster_id {outcome.cluster_id}: expected {n_series}, got {snapshot.n_series}"
        )
    expected_amplitudes = (len(snapshot.peaks), n_series)
    if evaluation.amplitudes.shape != expected_amplitudes:
        raise ValueError(
            "Analytical amplitude shape mismatch for "
            f"cluster_id {outcome.cluster_id}: expected {expected_amplitudes}, "
            f"got {evaluation.amplitudes.shape}"
        )
    expected_model = (snapshot.n_points, n_series)
    sampled = outcome.sample_evaluation
    if sampled is not None and sampled.model_values.shape != expected_model:
        raise ValueError(
            "Analytical model shape mismatch for "
            f"cluster_id {outcome.cluster_id}: expected {expected_model}, "
            f"got {sampled.model_values.shape}"
        )
    expected_shapes = (len(snapshot.peaks), snapshot.n_points)
    if evaluation.shapes.shape != expected_shapes:
        raise ValueError(
            "Analytical shape matrix mismatch for "
            f"cluster_id {outcome.cluster_id}: expected {expected_shapes}, "
            f"got {evaluation.shapes.shape}"
        )
    if not np.all(np.isfinite(evaluation.amplitudes)):
        raise ValueError(
            f"Analytical amplitudes are non-finite for cluster_id {outcome.cluster_id}"
        )
    return evaluation


def _parameters_from_outcome(parameters: Sequence[FinalParameter]) -> Parameters:
    result = Parameters()
    for parameter in parameters:
        result.add(
            parameter.name,
            value=parameter.value,
            min_value=parameter.min,
            max_value=parameter.max,
            vary=parameter.vary,
            unit=parameter.unit,
            stderr=parameter.standard_error,
        )
    return result


def _validate_parameter_values(outcome: FinalFitOutcome, parameters: Parameters) -> None:
    for parameter in outcome.final_nonlinear_parameters:
        if parameter.name not in parameters:
            raise ValueError(
                f"Simulation snapshot is missing final nonlinear parameter {parameter.name}"
            )
        if not np.isclose(parameters[parameter.name].value, parameter.value, rtol=1e-12, atol=0.0):
            raise ValueError(
                "Simulation snapshot nonlinear parameter does not match final outcome: "
                f"{parameter.name}"
            )


def _validate_cluster_geometry(outcome: FinalClusterOutcome, cluster: Cluster) -> None:
    if tuple(peak.name for peak in cluster.peaks) != outcome.peak_names:
        raise ValueError(
            f"Simulation snapshot peak identity mismatch for cluster_id {cluster.cluster_id}"
        )
    if cluster.n_series <= 0 or cluster.n_points <= 0:
        raise ValueError(
            f"Simulation snapshot has invalid cluster shape for cluster_id {cluster.cluster_id}"
        )
    if len(cluster.grid_indices) == 0:
        raise ValueError(
            f"Simulation snapshot has no grid indices for cluster_id {cluster.cluster_id}"
        )
    if any(indices.size != cluster.n_points for indices in cluster.grid_indices):
        raise ValueError(
            f"Simulation grid-index shape mismatch for cluster_id {cluster.cluster_id}"
        )


def _clusters_by_id(clusters: Sequence[Cluster], *, label: str) -> Mapping[int, Cluster]:
    by_id = {cluster.cluster_id: cluster for cluster in clusters}
    if len(by_id) != len(clusters):
        raise ValueError(f"Simulation {label} cluster_id values must be unique")
    return MappingProxyType(by_id)


def _require_exact_ids(expected: set[int], actual: set[int], *, label: str) -> None:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {label} cluster_id values: {missing}")
        if unexpected:
            details.append(f"unexpected {label} cluster_id values: {unexpected}")
        raise ValueError("; ".join(details))


__all__ = [
    "FinalModelCluster",
    "FinalModelSnapshot",
    "project_final_outcome",
    "project_final_residual",
    "simulate_final_outcome",
]
