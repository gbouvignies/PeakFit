"""Coordinate systems and dimension contexts."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from peakfit.shared.typing import FloatArray

# Mathematical constant
_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True, slots=True)
class DimensionContext:
    """Precomputed, dimension-specific helpers for shapes.

    This class encapsulates both coordinate conversions and spectral geometry
    (aliasing, sign correction for indirect dimensions).

    Responsibilities:
        - Coordinate conversions: PPM ↔ points ↔ Hz
        - Spectral geometry: aliasing and P180 sign corrections
        - Axis metadata: label, size, dimension type

    This separation keeps lineshape physics separate from NMR-specific geometry.

    Note:
        The ``compute_offsets`` method returns frequency offsets in Hz.
        Lineshape kernels receive all parameters in Hz and handle any
        internal conversions to angular frequency (rad/s) as needed.
    """

    label: str
    size: int
    is_direct: bool
    has_p180: bool
    ppm2pts: Callable[[FloatArray], FloatArray]
    ppm2pt_i: Callable[[float], int]
    pts2hz_delta: Callable[[FloatArray], FloatArray]
    ppm2hz: Callable[[FloatArray], FloatArray]

    def compute_offsets(
        self,
        x_grid: FloatArray,
        positions_ppm: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute frequency offsets and sign corrections.

        This method handles the NMR-specific aliasing and phase alternation
        that occurs in indirect dimensions, separating spectral geometry
        from lineshape physics.

        Args:
            x_grid: 1D array of grid point indices, shape (N,)
            positions_ppm: 1D array of peak positions in PPM, shape (K,)

        Returns:
            dw_hz: Frequency offset matrix in Hz, shape (N, K)
            sign: Sign correction matrix for P180 alternation, shape (N, K)

        Notes:
            For direct dimensions: no aliasing, sign is always +1
            For indirect dimensions: positions wrap with aliasing,
                sign alternates if P180 was applied during acquisition
        """
        # Convert positions from PPM to points
        pos_pts = self.ppm2pts(positions_ppm)

        # Compute offset in points: (N, K) matrix
        dx_pt: FloatArray = x_grid[:, None] - pos_pts[None, :]

        if not self.is_direct:
            # Apply aliasing correction for indirect dimensions
            aliasing = (dx_pt + 0.5 * self.size) // self.size
            dx_pt = dx_pt - self.size * aliasing

            # P180 causes sign alternation with aliasing
            if self.has_p180:
                sign: FloatArray = np.power(-1.0, aliasing)
            else:
                sign = np.ones_like(aliasing)
        else:
            # Direct dimension: no aliasing, no sign correction
            sign = np.ones_like(dx_pt)

        # Convert offset from points to Hz
        dw_hz: FloatArray = self.pts2hz_delta(dx_pt)

        return dw_hz, sign
