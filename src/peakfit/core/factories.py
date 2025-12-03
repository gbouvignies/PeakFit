"""Factories for creating domain objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from peakfit.core.domain.peaks import Peak
from peakfit.core.lineshapes import LineshapeFactory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.core.domain.spectrum import Spectra
    from peakfit.core.shared.typing import FittingOptions


def create_peak(
    name: str,
    positions: Sequence[float],
    shape_names: list[str],
    spectra: Spectra,
    args: FittingOptions,
) -> Peak:
    """Create a `Peak` object with shapes constructed from `shape_names`.

    Args:
        name: Peak name
        positions: Positions per dimension (ppm)
        shape_names: Shape names per dimension
        spectra: Spectra metadata object
        args: CLI fitting options
    """
    factory = LineshapeFactory(spectra, args)
    shapes = factory.create_shapes(name, positions, shape_names)
    return Peak(name=name, positions=np.array(positions), shapes=shapes)
