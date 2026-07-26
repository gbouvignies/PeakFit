"""Full-grid simulation projected from an authoritative final fit outcome."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.params_scalar import Parameters

_MIN_SPECTRUM_NDIM = 2

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from peakfit.engine.domain.peaks import Peak
    from peakfit.engine.domain.state import FittingState
    from peakfit.fit.final_outcome import (
        FinalAnalyticalEvaluation,
        FinalClusterOutcome,
        FinalFitOutcome,
        FinalParameter,
    )
    from peakfit.fit.pipeline import CorrectionSnapshot
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
    noise: float
    correction_revision: int
    parameter_values: tuple[tuple[str, float], ...] = ()

    @classmethod
    def capture(
        cls,
        outcome: FinalFitOutcome,
        state: FittingState,
        correction_snapshot: CorrectionSnapshot,
    ) -> FinalModelSnapshot:
        """Copy the minimal full-grid model geometry at the finalization boundary."""
        _validate_capture_inputs(outcome, state, correction_snapshot)
        assert state.noise is not None
        state_clusters = {cluster.cluster_id: cluster for cluster in state.clusters}
        clusters = tuple(
            _freeze_cluster_geometry(state_clusters[cluster_outcome.cluster_id])
            for cluster_outcome in outcome.clusters
        )
        return cls(
            clusters=clusters,
            noise=float(state.noise),
            correction_revision=correction_snapshot.revision,
            parameter_values=tuple(
                (parameter.name, parameter.value)
                for parameter in outcome.final_nonlinear_parameters
            ),
        )


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
    _validate_snapshot(outcome, snapshot)
    output_data = np.asarray(data)
    n_series, grid_shape, positions = _full_grid_positions(output_data, snapshot)
    parameters = _parameters_from_outcome(outcome.final_nonlinear_parameters)
    snapshots = {cluster.cluster_id: cluster for cluster in snapshot.clusters}
    model = np.zeros((len(positions[0]), n_series), dtype=np.float64)
    usable_count = 0

    for cluster_outcome in outcome.clusters:
        if not cluster_outcome.usable:
            continue
        cluster = snapshots[cluster_outcome.cluster_id]
        evaluation = _validate_usable_projection(cluster_outcome, cluster, n_series)
        shapes = np.asarray(
            Cluster.evaluate_peaks(cluster.peaks, parameters, positions), dtype=np.float64
        )
        expected_shapes = (len(cluster.peaks), model.shape[0])
        if shapes.shape != expected_shapes:
            raise ValueError(
                "Full-grid lineshape shape mismatch for "
                f"cluster_id {cluster.cluster_id}: expected {expected_shapes}, got {shapes.shape}"
            )
        if not np.all(np.isfinite(shapes)):
            raise ValueError(
                f"Full-grid lineshape values are non-finite for cluster_id {cluster.cluster_id}"
            )
        model += shapes.T @ evaluation.amplitudes
        usable_count += 1

    if usable_count == 0:
        return None
    return model.T.reshape((n_series, *grid_shape)).astype(float)


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
    if not np.isfinite(snapshot.noise) or snapshot.noise <= 0.0:
        raise ValueError(
            f"Simulation snapshot noise must be positive and finite, got {snapshot.noise}"
        )
    if snapshot.noise != outcome.noise:
        raise ValueError(
            "Simulation snapshot noise does not match final outcome: "
            f"expected {outcome.noise}, got {snapshot.noise}"
        )
    if snapshot.correction_revision != outcome.terminal_correction_revision:
        raise ValueError(
            "Simulation snapshot correction revision does not match final outcome: "
            f"expected {outcome.terminal_correction_revision}, got {snapshot.correction_revision}"
        )
    snapshot_ids = {cluster.cluster_id for cluster in snapshot.clusters}
    outcome_ids = {cluster.cluster_id for cluster in outcome.clusters}
    _require_exact_ids(outcome_ids, snapshot_ids, label="simulation snapshot")
    if len(snapshot_ids) != len(snapshot.clusters):
        raise ValueError("Simulation snapshot cluster_id values must be unique")
    expected_values = tuple(
        (parameter.name, parameter.value) for parameter in outcome.final_nonlinear_parameters
    )
    if snapshot.parameter_values != expected_values:
        raise ValueError("Simulation snapshot nonlinear parameters do not match final outcome")
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


def _validate_usable_projection(
    outcome: FinalClusterOutcome,
    snapshot: FinalModelCluster,
    n_series: int,
) -> FinalAnalyticalEvaluation:
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
    if evaluation.model_values.shape != expected_model:
        raise ValueError(
            "Analytical model shape mismatch for "
            f"cluster_id {outcome.cluster_id}: expected {expected_model}, "
            f"got {evaluation.model_values.shape}"
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


__all__ = ["FinalModelCluster", "FinalModelSnapshot", "simulate_final_outcome"]
