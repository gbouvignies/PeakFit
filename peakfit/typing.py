from typing import Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64 | np.float32]
IntArray = npt.NDArray[np.int_]


class FittingOptions(Protocol):
    """Protocol for fitting options required by lineshape classes."""

    jx: bool  # Fit J-coupling
    phx: bool  # Fit phase correction in X dimension
    phy: bool  # Fit phase correction in Y dimension
    noise: float  # Noise level
    pvoigt: bool  # Use pseudo-Voigt lineshape
    lorentzian: bool  # Use Lorentzian lineshape
    gaussian: bool  # Use Gaussian lineshape
