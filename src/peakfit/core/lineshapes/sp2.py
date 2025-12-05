"""SP2 (sine squared bell) apodization lineshape module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.core.lineshapes.apodization import ApodizationEvaluator, ApodShape
from peakfit.core.lineshapes.registry import register_shape

if TYPE_CHECKING:
    from peakfit.core.domain.spectrum import Spectra
    from peakfit.core.shared.typing import FittingOptions, FloatArray


class SP2Evaluator(ApodizationEvaluator):
    """SP2 (sine squared bell) apodization lineshape evaluator.

    Complex lineshape with sine-squared window function.
    Precomputes acquisition time and apodization parameters at initialization.
    """

    def __init__(self, aq: float, end: float, off: float) -> None:
        super().__init__(aq)
        self.f1 = off * np.pi
        self.f2 = (end - off) * np.pi

        # Precompute exponentials for 2*f1 and 2*f2
        self._e2if1 = np.exp(2j * self.f1)
        self._em2if1 = np.exp(-2j * self.f1)
        self._e2if2 = np.exp(2j * self.f2)
        self._em2if2 = np.exp(-2j * self.f2)
        self._e2if12 = self._e2if1 * self._e2if2
        self._em2if12 = self._em2if1 * self._em2if2

        # Precompute constants
        self._aq_quarter = 0.25 * aq
        self._aq_half = 0.5 * aq
        self._i2f2 = 2j * self.f2

    def _compute_complex(
        self, dx: FloatArray, r2: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        """Compute complex lineshape for single component."""
        aq = self.aq
        z1 = aq * (1j * dx + r2)
        emz1 = np.exp(-z1)
        ez1 = 1.0 / emz1

        denom1 = z1 - self._i2f2
        denom2 = z1 + self._i2f2

        num1 = (self._e2if2 - ez1) * self._e2if1 * emz1
        num2 = (self._em2if2 - ez1) * self._em2if1 * emz1
        num3 = 1 - emz1

        term1 = self._aq_quarter * num1 / denom1
        term2 = self._aq_quarter * num2 / denom2
        term3 = self._aq_half * num3 / z1
        val = term1 + term2 + term3

        if not calc_derivs:
            return val, None, None  # type: ignore[return-value]

        # Derivatives w.r.t. z1
        dnum1_dz1 = -self._e2if12 * emz1
        dnum2_dz1 = -self._em2if12 * emz1

        dterm1_dz1 = self._aq_quarter * (dnum1_dz1 * denom1 - num1) / denom1**2
        dterm2_dz1 = self._aq_quarter * (dnum2_dz1 * denom2 - num2) / denom2**2
        dterm3_dz1 = self._aq_half * (emz1 * (z1 + 1) - 1) / z1**2

        dval_dz1 = dterm1_dz1 + dterm2_dz1 + dterm3_dz1
        d_dx = 1j * aq * dval_dz1
        d_r2 = aq * dval_dz1

        return val, d_dx, d_r2  # type: ignore[return-value]

    def height(self, r2: float, j_hz: float = 0.0) -> float:
        """Compute height at peak position."""
        del j_hz  # Unused but part of interface
        z1 = self.aq * r2
        emz1 = np.exp(-z1)
        ez1 = 1.0 / emz1

        a1 = self._aq_quarter * (self._e2if2 - ez1) * self._e2if1 * emz1 / (z1 - self._i2f2)
        a2 = self._aq_quarter * (self._em2if2 - ez1) * self._em2if1 * emz1 / (z1 + self._i2f2)
        a3 = self._aq_half * (1 - emz1) / z1

        val = a1 + a2 + a3
        return float(val.real)

    def integral(self, j_hz: float = 0.0) -> float:
        """Compute analytical integral."""
        integral = np.pi * np.sin(self.f1) ** 2
        return integral * 2.0 if j_hz != 0.0 else integral


def create_sp2_shape(
    name: str, center: float, spectra: Spectra, dim: int, args: FittingOptions
) -> ApodShape:
    """Factory function to create an SP2 lineshape."""
    spec_params = spectra.params[dim]
    evaluator = SP2Evaluator(
        aq=spec_params.aq_time,
        end=spec_params.apodq2,
        off=spec_params.apodq1,
    )
    shape = ApodShape(name, center, spectra, dim, args, evaluator=evaluator)
    shape.shape_name = "sp2"
    return shape


# Register the factory function
register_shape("sp2")(create_sp2_shape)


def make_sp2_evaluator(aq: float, end: float, off: float) -> SP2Evaluator:
    """Create an SP2 evaluator with the given parameters."""
    return SP2Evaluator(aq, end, off)
