"""Bounded scientific projections and exact sample lookup for desktop comparison.

No Qt objects or optimization live here. Smooth contour coordinates and native
trace coordinates both use the same completed-outcome evaluator.
"""

from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import contourpy
import numpy as np
from threadpoolctl import threadpool_limits

from peakfit.fit.simulation import project_final_outcome

if TYPE_CHECKING:
    from threading import Event

    from peakfit.engine.domain.spectrum import SpectralParameters
    from peakfit.gui.local_fit import LocalFitResult
    from peakfit.gui.spectrum import ContourGeometry


_REFINED_AREA = 65536


@dataclass(frozen=True)
class ModelPlane:
    """One local display grid; never a plane-by-full-spectrum cube."""

    y: np.ndarray
    x: np.ndarray
    values: np.ndarray
    edge_visible: bool

    @property
    def nbytes(self) -> int:
        """Retained numerical payload for the bounded display cache."""
        return self.values.nbytes + self.y.nbytes + self.x.nbytes


@dataclass(frozen=True)
class ModelContours:
    """Per-level paths, split by sampled-cell support."""

    grid: ModelPlane
    sampled: ContourGeometry
    predicted: ContourGeometry

    @property
    def nbytes(self) -> int:
        """Include all retained contour geometry in the cache budget."""
        return self.grid.nbytes + sum(
            points.nbytes + connections.nbytes
            for points, connections in self.sampled + self.predicted
        )


@dataclass(frozen=True)
class ModelSlices:
    """Native row/column predictions for all original planes, plus exact residuals."""

    x: np.ndarray
    y: np.ndarray
    profile: np.ndarray
    support_x: np.ndarray
    support_y: np.ndarray
    residual_x: np.ndarray
    residual_y: np.ndarray
    residual_profile: np.ndarray | None

    @property
    def nbytes(self) -> int:
        """Retained numerical payload for the bounded display cache."""
        return sum(
            array.nbytes
            for array in (
                self.x,
                self.y,
                self.support_x,
                self.support_y,
                self.residual_x,
                self.residual_y,
            )
        )  # Profiles are views into these arrays or into the retained outcome.


def _project(
    result: LocalFitResult, coordinates, planes=None, *, cancel: Event | None = None
) -> np.ndarray:
    if cancel is not None and cancel.is_set():
        raise CancelledError
    values = project_final_outcome(
        result.outcome,
        result.simulation_snapshot,
        coordinates,
        cluster_id=result.inputs.scope.cluster_id,
        planes=planes,
    )
    if values is None:
        raise ValueError("This result has no usable local model")
    return values


def local_model_plane(
    result: LocalFitResult,
    axes: list[SpectralParameters],
    plane: int,
    level: float,
    *,
    cancel: Event | None = None,
) -> ModelPlane:
    """Start with six linewidths, grow until visible tails no longer cross the edge.

    Fractional coordinates give three contour samples per native interval, with
    additional samples around narrow fitted lines. Alias seams remain separate
    physical edges of the loaded grid. Bounds never wrap or connect these edges.
    """
    indices = result.inputs.scope.grid_indices
    params = {p.name: p.value for p in result.outcome.final_nonlinear_parameters}
    bounds = []
    centers = []
    for dimension, axis in enumerate(axes):
        low, high = float(indices[dimension].min()), float(indices[dimension].max())
        refinements = []
        for peak in result.simulation_snapshot.clusters[0].peaks:
            prefix = f"{peak.name}.{peak.shapes[dimension].axis}."
            center = float(axis.ppm2pts(params[prefix + "cs"]))
            if not axis.direct:
                center %= axis.size
            width = max(abs(float(axis.hz2pts_delta(params[prefix + "lw"]))), 0.05)
            coupling = abs(float(axis.hz2pts_delta(params.get(prefix + "j", 0.0)))) / 2
            low = min(low, center - 6 * width - coupling)
            high = max(high, center + 6 * width + coupling)
            refinements.extend(center + width * np.linspace(-2, 2, 21))
        bounds.append([max(0, int(np.floor(low))), min(axis.size - 1, int(np.ceil(high)))])
        centers.append(np.asarray(refinements))
    with threadpool_limits(limits=1):
        while True:
            if cancel is not None and cancel.is_set():
                raise CancelledError
            (y0, y1), (x0, x1) = bounds
            subdivision = 3 if (y1 - y0 + 1) * (x1 - x0 + 1) <= _REFINED_AREA else 1
            y = np.linspace(y0, y1, max(2, subdivision * (y1 - y0) + 1))
            x = np.linspace(x0, x1, max(2, subdivision * (x1 - x0) + 1))
            edge_y = np.concatenate((np.full(len(x), y0), np.full(len(x), y1), y, y))
            edge_x = np.concatenate((x, x, np.full(len(y), x0), np.full(len(y), x1)))
            edge = _project(result, [edge_y, edge_x], [plane])[:, 0]
            visible = bool(np.max(np.abs(edge)) >= level)
            expanded = [
                [max(0, lo - max(2, hi - lo)), min(axis.size - 1, hi + max(2, hi - lo))]
                for (lo, hi), axis in zip(bounds, axes, strict=True)
            ]
            if not visible or expanded == bounds:
                break
            bounds = expanded
        y = np.unique(np.concatenate((y, centers[0][(centers[0] > y0) & (centers[0] < y1)])))
        x = np.unique(np.concatenate((x, centers[1][(centers[1] > x0) & (centers[1] < x1)])))
        # Coordinate and shape temporaries stay bounded independently of grid size.
        values = np.empty((len(y), len(x)))
        for start in range(0, values.size, 4096):
            if cancel is not None and cancel.is_set():
                raise CancelledError
            points = np.arange(start, min(values.size, start + 4096))
            values.ravel()[start : start + len(points)] = _project(
                result, [y[points // len(x)], x[points % len(x)]], [plane]
            )[:, 0]
    values.flags.writeable = False
    return ModelPlane(y, x, values, visible)


def model_contours(
    grid: ModelPlane,
    axes: list[SpectralParameters],
    indices: tuple[np.ndarray, ...],
    levels: np.ndarray,
    *,
    cancel: Event | None = None,
) -> ModelContours:
    """Split smooth paths at sampled-cell boundaries without connecting mask gaps."""
    support = np.zeros(tuple(axis.size for axis in axes), dtype=bool)
    support[indices] = True
    generator = contourpy.contour_generator(
        x=grid.x,
        y=grid.y,
        z=grid.values,
        line_type="Separate",
    )
    sampled, predicted = [], []
    for level in levels:
        if cancel is not None and cancel.is_set():
            raise CancelledError
        inside, outside = [], []
        for segment in generator.lines(float(level)):
            path = np.asarray(segment)
            # Classify edges by their midpoint, then render each class independently.
            middle = (path[:-1] + path[1:]) / 2
            ix = np.clip(np.rint(middle[:, 0]).astype(int), 0, axes[1].size - 1)
            iy = np.clip(np.rint(middle[:, 1]).astype(int), 0, axes[0].size - 1)
            for selected, target in ((support[iy, ix], inside), (~support[iy, ix], outside)):
                starts = np.flatnonzero(selected & ~np.r_[False, selected[:-1]])
                ends = np.flatnonzero(selected & ~np.r_[selected[1:], False])
                target.extend(
                    path[start : end + 2] for start, end in zip(starts, ends, strict=True)
                )
        for paths, target in ((inside, sampled), (outside, predicted)):
            points = np.concatenate(paths) if paths else np.empty((0, 2))
            connect = np.ones(len(points), dtype=np.int32)
            if paths:
                connect[np.cumsum([len(p) for p in paths]) - 1] = 0
            ppm = points.copy()
            ppm[:, 0] = axes[1].pts2ppm(points[:, 0])
            ppm[:, 1] = axes[0].pts2ppm(points[:, 1])
            target.append((ppm, connect))
    # Reused grids can have been grown for a lower threshold. The clipping
    # warning belongs to the current visible levels, not the original request.
    edge_max = max(np.abs(grid.values[[0, -1]]).max(), np.abs(grid.values[:, [0, -1]]).max())
    grid = replace(grid, edge_visible=bool(edge_max >= np.abs(levels).min()))
    return ModelContours(grid, sampled, predicted)


def model_slices(
    result: LocalFitResult,
    axes: list[SpectralParameters],
    iy: int,
    ix: int,
    *,
    cancel: Event | None = None,
) -> ModelSlices:
    """Project native traces, preserving actual sample gaps and original plane order.

    All native row/column coordinates are explicitly evaluated, including tails
    outside the bounded 2D display region. Such points are predictions, not zeros
    or fitting residuals. No interpolation across repeated plane values is used.
    """
    sy, sx = result.inputs.scope.grid_indices
    evaluation = result.outcome.cluster(result.inputs.scope.cluster_id).analytical_evaluation
    assert evaluation is not None
    support_x = np.full(axes[1].size, False)
    support_y = np.full(axes[0].size, False)
    residual_x = np.full((axes[1].size, len(result.inputs.snapshot.plane_values)), np.nan)
    residual_y = np.full((axes[0].size, len(result.inputs.snapshot.plane_values)), np.nan)
    on_x, on_y = sy == iy, sx == ix
    support_x[sx[on_x]] = True
    support_y[sy[on_y]] = True
    residual_x[sx[on_x]] = evaluation.raw_residuals[on_x]
    residual_y[sy[on_y]] = evaluation.raw_residuals[on_y]

    def trace(size, coordinates):
        # Yield between bounded projections so a superseding experimental plane
        # never waits for all original planes along an obsolete crosshair.
        values = np.empty((size, len(result.inputs.snapshot.plane_values)))
        for start in range(0, size, 256):
            end = min(size, start + 256)
            values[start:end] = _project(result, [c[start:end] for c in coordinates], cancel=cancel)
        return values

    with threadpool_limits(limits=1):
        x = trace(axes[1].size, [np.full(axes[1].size, iy), np.arange(axes[1].size)])
        y = trace(axes[0].size, [np.arange(axes[0].size), np.full(axes[0].size, ix)])
    # Use authoritative quantities where the coordinate was actually fitted.
    x[sx[on_x]] = evaluation.model_values[on_x]
    y[sy[on_y]] = evaluation.model_values[on_y]
    point = np.flatnonzero((sy == iy) & (sx == ix))
    residual_profile = evaluation.raw_residuals[point[0]] if point.size else None
    return ModelSlices(x, y, x[ix], support_x, support_y, residual_x, residual_y, residual_profile)
