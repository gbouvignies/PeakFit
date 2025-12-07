"""Utilities for lineshape modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from peakfit.core.shared.typing import FloatArray

# =============================================================================
# Constants
# =============================================================================

_LN2 = np.log(2.0)
_SQRT_PI_4LN2 = np.sqrt(np.pi / (4.0 * _LN2))


def get_axis_label(dim_index: int) -> str:
    """Get the axis label for a dimension using Bruker Topspin convention.

    For pseudo-3D experiments:
    - F1 = pseudo-dimension (intensities, CEST offsets, etc.)
    - F2 = first spectral dimension (indirect, e.g., 15N)
    - F3 = second spectral dimension (direct/acquisition, e.g., 1H)

    Args:
        dim_index: 1-based dimension index (1 = first spectral dim after pseudo)

    Returns
    -------
        Axis label like "F2", "F3", "F4"
    """
    # Offset by 1: F1 is reserved for pseudo-dimension
    return f"F{dim_index + 1}"


# =============================================================================
# Cached Result Container
# =============================================================================


@dataclass(slots=True)
class CachedResult:
    """Container for cached lineshape evaluation results.

    Stores both values and derivatives from a single _core() call,
    allowing reuse when scipy calls fun() then jac() with same parameters.

    Cache is validated by storing both the array's id() and its bytes hash.
    This handles the case where different arrays happen to be allocated at
    the same memory address after garbage collection.
    """

    # Input parameters (for cache validation)
    dx_id: int = 0  # id() of dx array (fast check)
    dx_hash: int = 0  # Hash of dx.tobytes() (content check)
    params_hash: int = 0  # Hash of scalar parameters

    # Cached outputs
    value: FloatArray | None = field(default=None)
    d_dx: FloatArray | None = field(default=None)
    d_fwhm: FloatArray | None = field(default=None)
    d_j: FloatArray | None = field(default=None)
    d_eta: FloatArray | None = field(default=None)  # For PseudoVoigt
    d_r2: FloatArray | None = field(default=None)  # For apodization shapes
    d_phase: FloatArray | None = field(default=None)  # For apodization shapes

    def matches(self, dx: FloatArray, *params: float) -> bool:
        """Check if cache matches the given inputs.

        Uses both id() and content hash to handle array reallocation at same address.
        """
        # Fast path: same array object (same id and same content hash)
        if id(dx) == self.dx_id and hash(params) == self.params_hash:
            # Verify content hasn't changed (in case of in-place modification)
            return hash(dx.tobytes()) == self.dx_hash
        return False

    def update_key(self, dx: FloatArray, *params: float) -> None:
        """Update cache key for new inputs."""
        self.dx_id = id(dx)
        self.dx_hash = hash(dx.tobytes())
        self.params_hash = hash(params)
