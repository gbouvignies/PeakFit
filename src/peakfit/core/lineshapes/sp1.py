"""SP1 (sine bell) apodization lineshape module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.core.lineshapes.apodization import ApodizationEvaluator, ApodShape
from peakfit.core.lineshapes.registry import register_shape

if TYPE_CHECKING:
    from peakfit.core.domain.spectrum import Spectra
    from peakfit.core.shared.typing import FittingOptions, FloatArray


class SP1Evaluator(ApodizationEvaluator):
    """SP1 (sine bell) apodization lineshape evaluator.

    Complex lineshape with sine-bell window function.
    Precomputes acquisition time and apodization parameters at initialization.
    """

    def __init__(self, aq: float, end: float, off: float) -> None:
        super().__init__(aq)
        self.f1 = off * np.pi
        self.f2 = (end - off) * np.pi

        # Precompute exponentials for f1 and f2
        self._eif1 = np.exp(1j * self.f1)
        self._emif1 = np.exp(-1j * self.f1)
        self._eif2 = np.exp(1j * self.f2)
        self._emif2 = np.exp(-1j * self.f2)
        self._eif12 = self._eif1 * self._eif2
        self._emif12 = self._emif1 * self._emif2

        # Precompute constants
        self._half_i_aq = 0.5j * aq
        self._if2 = 1j * self.f2

    def _compute_complex(
        self, dx: FloatArray, r2: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        """Compute complex lineshape for single component."""
        aq = self.aq
        z1 = aq * (1j * dx + r2)
        emz1 = np.exp(-z1)
        ez1 = 1.0 / emz1

        denom1 = z1 - self._if2
        denom2 = z1 + self._if2

        num1 = (self._eif2 - ez1) * self._eif1 * emz1
        num2 = (ez1 - self._emif2) * self._emif1 * emz1

        term1 = self._half_i_aq * num1 / denom1
        term2 = self._half_i_aq * num2 / denom2
        val = term1 + term2

        if not calc_derivs:
            return val, None, None  # type: ignore[return-value]

        # Derivatives w.r.t. z1
        dnum1_dz1 = -self._eif12 * emz1
        dnum2_dz1 = self._emif12 * emz1

        dterm1_dz1 = self._half_i_aq * (dnum1_dz1 * denom1 - num1) / denom1**2
        dterm2_dz1 = self._half_i_aq * (dnum2_dz1 * denom2 - num2) / denom2**2

        dval_dz1 = dterm1_dz1 + dterm2_dz1
        d_dx = 1j * aq * dval_dz1
        d_r2 = aq * dval_dz1

        return val, d_dx, d_r2  # type: ignore[return-value]

    def height(self, r2: float, j_hz: float = 0.0) -> float:
        """Compute height at peak position."""
        del j_hz  # Unused but part of interface
        z1 = self.aq * r2
        emz1 = np.exp(-z1)
        ez1 = 1.0 / emz1

        term1 = (self._eif2 - ez1) * self._eif1 * emz1 / (z1 - self._if2)
        term2 = (ez1 - self._emif2) * self._emif1 * emz1 / (z1 + self._if2)

        val = self._half_i_aq * (term1 + term2)
        return float(val.real)

    def integral(self, j_hz: float = 0.0) -> float:
        """Compute analytical integral."""
        integral = np.pi * np.sin(self.f1)
        return integral * 2.0 if j_hz != 0.0 else integral


def create_sp1_shape(
    name: str, center: float, spectra: Spectra, dim: int, args: FittingOptions
) -> ApodShape:
    """Factory function to create an SP1 lineshape."""
    spec_params = spectra.params[dim]
    evaluator = SP1Evaluator(
        aq=spec_params.aq_time,
        end=spec_params.apodq2,
        off=spec_params.apodq1,
    )
    shape = ApodShape(name, center, spectra, dim, args, evaluator=evaluator)
    shape.shape_name = "sp1"
    return shape


# Register the factory function
register_shape("sp1")(create_sp1_shape)


def make_sp1_evaluator(aq: float, end: float, off: float) -> SP1Evaluator:
    """Create an SP1 evaluator with the given parameters."""
    return SP1Evaluator(aq, end, off)
