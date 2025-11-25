"""Shared typing aliases used across PeakFit."""

from typing import Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64 | np.float32]
IntArray = npt.NDArray[np.int_]


class FittingOptions(Protocol):
    """Options required by lineshape and fitting components."""

    jx: bool
    phx: bool
    phy: bool
    noise: float
    pvoigt: bool
    lorentzian: bool
    gaussian: bool
