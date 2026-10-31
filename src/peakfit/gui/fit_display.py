"""Bounded scientific projection of a completed dataset fit for desktop comparison.

No Qt objects or optimization live here. The model plane uses the same
completed-outcome evaluator as scientific exports.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from threadpoolctl import ThreadpoolController

from peakfit.engine.scientific.simulation import project_final_outcome

if TYPE_CHECKING:
    from threading import Event

    from peakfit.engine.domain.spectrum import SpectralParameters
    from peakfit.gui.dataset_fit import DatasetFitResult


# NumPy is loaded before this controller. Repeated discovery after Cocoa starts
# scans thousands of unrelated framework paths; limit only the known BLAS pools.
_BLAS = ThreadpoolController()


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


def dataset_model_plane(
    result: DatasetFitResult,
    axes: list[SpectralParameters],
    plane: int,
    *,
    cancel: Event,
) -> ModelPlane:
    """Evaluate every usable final contribution at every native point, without cutoffs."""
    ny, nx = (axis.size for axis in axes)
    iy, ix = np.indices((ny, nx))
    with _BLAS.limit(limits=1):
        values = project_final_outcome(
            result.outcome,
            result.simulation_snapshot,
            [iy.ravel(), ix.ravel()],
            planes=[plane],
            canceled=cancel.is_set,
        )
    if values is None:
        raise ValueError("This run has no usable model contributions.")
    values = values[:, 0].reshape(ny, nx)
    values.flags.writeable = False
    return ModelPlane(np.arange(ny), np.arange(nx), values, False)
