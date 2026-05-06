"""Spectral grid computation helper."""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from peakfit.engine.lineshapes.utils import get_axis_label

if TYPE_CHECKING:
    from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
    from peakfit.shared.typing import FloatArray


class SpectralGrid:
    """Helper for spectral grid computations.

    This class handles the creation of DimensionContext and provides
    methods for computing frequency offsets on the grid.
    """

    def __init__(self, spectra: Spectra, dim: int) -> None:
        """Initialize SpectralGrid.

        Args:
            spectra: Spectra object containing parameters
            dim: Dimension index
        """
        self.spec_params: SpectralParameters = spectra.params[dim]
        self.axis_label = self.spec_params.label or get_axis_label(dim)

        self.dim_ctx = self.spec_params.to_dimension_context(label=self.axis_label)

    def compute_offsets(
        self,
        x_grid: npt.ArrayLike,
        positions: npt.ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute (N, K) frequency offset matrix and sign correction.

        Args:
            x_grid: Grid point indices, shape (N,)
            positions: Peak positions in PPM, shape (K,)

        Returns:
            dw_hz: Frequency offset matrix in Hz, shape (N, K)
            sign: Sign correction for P180 alternation, shape (N, K)
        """
        x_arr = np.asarray(x_grid, dtype=np.float64)
        positions_arr = np.asarray(positions, dtype=np.float64)
        return self.dim_ctx.compute_offsets(x_arr, positions_arr)
