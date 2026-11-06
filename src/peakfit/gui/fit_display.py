"""Bounded scientific projection of a completed dataset fit for desktop comparison.

No Qt objects or optimization live here. The model plane uses the same
completed-outcome evaluator as scientific exports.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from threadpoolctl import ThreadpoolController

from peakfit.engine.scientific.simulation import project_final_grid

if TYPE_CHECKING:
    from threading import Event

    from peakfit.engine.domain.spectrum import SpectralParameters
    from peakfit.gui.dataset_fit import DatasetFitResult


# NumPy is loaded before this controller. Repeated discovery after Cocoa starts
# scans thousands of unrelated framework paths; limit only the known BLAS pools.
_BLAS = ThreadpoolController()

# Model planes retained for display: the most recent planes, within this many bytes.
_MODEL_PLANES = 8
_MODEL_BYTES = 128 * 1024**2


@dataclass(frozen=True)
class ModelPlane:
    """One display grid; never a plane-by-full-spectrum cube."""

    y: np.ndarray
    x: np.ndarray
    values: np.ndarray
    edge_visible: bool

    @property
    def nbytes(self) -> int:
        """Retained numerical payload for the bounded display cache."""
        return self.values.nbytes + self.y.nbytes + self.x.nbytes


def normalized_residual(observed: np.ndarray, model: np.ndarray, noise: float) -> np.ndarray:
    """Measured minus model, in units of the run's statistical noise (σ_stat)."""
    return (observed - model) / noise


class ModelPlanes:
    """Total-model planes by (run, plane), shared by every view that compares a run.

    Bounded to the most recent planes. Used on the GUI thread only: a worker reads
    an immutable plane from ``get`` and hands a newly computed one back to ``put``.
    """

    def __init__(self) -> None:
        self._planes: OrderedDict[tuple[str, int], ModelPlane] = OrderedDict()

    def get(self, run_id: str, plane: int) -> ModelPlane | None:
        """The cached plane, now most recently used, or None."""
        model = self._planes.get((run_id, plane))
        if model is not None:
            self._planes.move_to_end((run_id, plane))
        return model

    def put(self, run_id: str, plane: int, model: ModelPlane) -> None:
        """Retain a computed plane, evicting the least recently used beyond the bounds."""
        self._planes[run_id, plane] = model
        self._planes.move_to_end((run_id, plane))
        while len(self._planes) > _MODEL_PLANES or self.nbytes > _MODEL_BYTES:
            self._planes.popitem(last=False)

    def keep_only(self, run_id: str | None) -> None:
        """Drop the planes of every other run."""
        for key in [key for key in self._planes if key[0] != run_id]:
            del self._planes[key]

    @property
    def nbytes(self) -> int:
        """Retained numerical payload."""
        return sum(model.nbytes for model in self._planes.values())

    def retained(self) -> list[tuple[str, int]]:
        """The retained (run, plane) keys, least recently used first."""
        return list(self._planes)


def dataset_model_plane(
    result: DatasetFitResult,
    axes: list[SpectralParameters],
    plane: int,
    *,
    cancel: Event,
) -> ModelPlane:
    """Evaluate every usable final contribution at every native point, without cutoffs."""
    ny, nx = (axis.size for axis in axes)
    with _BLAS.limit(limits=1):
        values = project_final_grid(
            result.outcome,
            result.completed_fit.model_snapshot,
            (ny, nx),
            planes=[plane],
            canceled=cancel.is_set,
        )
    if values is None:
        raise ValueError("This run has no usable model contributions.")
    values = values[0]
    values.flags.writeable = False
    return ModelPlane(np.arange(ny), np.arange(nx), values, False)
