"""Shared typing aliases used across PeakFit."""

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type ComplexArray = npt.NDArray[np.complex128]
type IntArray = npt.NDArray[np.int_]
