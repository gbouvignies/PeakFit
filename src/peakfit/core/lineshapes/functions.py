"""Pure NumPy lineshape functions for NMR peak fitting.

This module provides vectorized NumPy implementations of common NMR lineshapes.
All functions are optimized using NumPy broadcasting and vectorization for performance.

Architecture
------------
Two families of lineshapes:

1. **FWHM-based** (Lorentzian, Gaussian, PseudoVoigt):
   - Height-normalized to 1.0 at center
   - Real-valued
   - Parameterized by FWHM (full width at half maximum)

2. **R2-based with apodization** (NoApod, SP1, SP2):
   - Complex-valued internally, returning real part after phase correction
   - Parameterized by R2 relaxation rate and phase
   - Require acquisition time (aq) at initialization

Common patterns:
- Unified ``_core()`` method for value + derivatives
- Doublet handling within each evaluator's ``_core()`` method
- Phase correction via ``_apply_phase_and_extract_real()`` for complex lineshapes

Key optimizations:
1. Single ``_core`` function shared between ``evaluate`` and ``evaluate_derivatives``
2. Precomputation of constants (γ², ln(2), etc.) at call time
3. Precomputation of fixed parameters (aq, f1, f2) at initialization for complex lineshapes
4. J-coupling derivative computed "for free" from frequency offset derivatives

Caching Strategy for scipy.optimize.least_squares
-------------------------------------------------
When scipy.optimize.least_squares calls both ``fun`` (residuals) and ``jac`` (Jacobian)
with the same parameters, we avoid redundant computation via a caching mechanism:

- ``evaluate()`` calls ``_core(calc_derivs=True)`` and caches the full result
- ``get_cached_derivatives()`` returns the cached derivatives without recomputation
- Cache is invalidated when input parameters change (compared by array identity/values)

This allows efficient use with scipy's ``jac`` callback without computing lineshapes twice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from peakfit.core.shared.typing import FloatArray

# =============================================================================
# Module-Level Constants
# =============================================================================

_LN2 = np.log(2.0)
_SQRT_PI_4LN2 = np.sqrt(np.pi / (4.0 * _LN2))


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


# =============================================================================
# FWHM-Based Lineshapes (Real-valued, height-normalized)
# =============================================================================


class FWHMLineshapeEvaluator(ABC):
    """Abstract base for FWHM-based lineshapes (Lorentzian, Gaussian, PseudoVoigt).

    These lineshapes are:
    - Real-valued
    - Height-normalized to 1.0 at center for singlets
    - Parameterized by FWHM (full width at half maximum)

    Caching Strategy:
        When evaluate() is called, it computes both values AND derivatives via _core()
        and caches them. Subsequent calls to get_cached_derivatives() with the same
        parameters return the cached derivatives without recomputation.
        This is optimized for scipy.optimize.least_squares which calls fun() then jac().
    """

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache = CachedResult()

    @abstractmethod
    def _compute_single(
        self, dx: FloatArray, fwhm: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        """Compute lineshape for a single component (no J-coupling).

        Returns
        -------
            (value, d_value_dx, d_value_dfwhm) or (value, None, None) if not calc_derivs
        """
        ...

    def _core(
        self, dx: FloatArray, fwhm: float, j_hz: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None, FloatArray | None]:
        """Core computation with doublet handling."""
        if j_hz == 0.0:
            val, d_dx, d_fwhm = self._compute_single(dx, fwhm, calc_derivs)
            if not calc_derivs:
                return val, None, None, None
            return val, d_dx, d_fwhm, np.zeros_like(val)

        # Doublet case
        j_shift = 0.5 * j_hz
        val_p, d_dx_p, d_fwhm_p = self._compute_single(dx + j_shift, fwhm, calc_derivs)
        val_m, d_dx_m, d_fwhm_m = self._compute_single(dx - j_shift, fwhm, calc_derivs)

        val = val_p + val_m
        if not calc_derivs:
            return val, None, None, None

        # d_dx_p and d_dx_m are guaranteed non-None when calc_derivs=True
        assert d_dx_p is not None and d_dx_m is not None
        assert d_fwhm_p is not None and d_fwhm_m is not None
        return (
            val,
            d_dx_p + d_dx_m,
            d_fwhm_p + d_fwhm_m,
            0.5 * (d_dx_p - d_dx_m),  # d/dj = 0.5 * (d/dx_p - d/dx_m)
        )

    def evaluate(self, dx: FloatArray, fwhm: float, j_hz: float = 0.0) -> FloatArray:
        """Evaluate lineshape values.

        Also computes and caches derivatives for efficient subsequent jac() calls.

        Args:
            dx: Frequency offset array (Hz)
            fwhm: Full width at half maximum (Hz)
            j_hz: J-coupling constant (Hz), default 0.0

        Returns
        -------
            Lineshape values at given offsets
        """
        dx_arr = np.asarray(dx)

        # Check cache first
        if self._cache.matches(dx_arr, fwhm, j_hz) and self._cache.value is not None:
            return self._cache.value

        # Compute with derivatives and cache
        val, d_dx, d_fwhm, d_j = self._core(dx_arr, fwhm, j_hz, calc_derivs=True)

        # Update cache
        self._cache.update_key(dx_arr, fwhm, j_hz)
        self._cache.value = val
        self._cache.d_dx = d_dx
        self._cache.d_fwhm = d_fwhm
        self._cache.d_j = d_j

        return val

    def evaluate_derivatives(
        self, dx: FloatArray, fwhm: float, j_hz: float = 0.0
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Evaluate lineshape and all derivatives.

        Returns cached derivatives if available from prior evaluate() call.
        Otherwise computes fresh.

        Args:
            dx: Frequency offset array (Hz)
            fwhm: Full width at half maximum (Hz)
            j_hz: J-coupling constant (Hz), default 0.0

        Returns
        -------
            Tuple of (value, d_value_dx, d_value_dfwhm, d_value_dj)
        """
        dx_arr = np.asarray(dx)

        # Check cache first
        if self._cache.matches(dx_arr, fwhm, j_hz) and self._cache.value is not None:
            assert self._cache.d_dx is not None
            assert self._cache.d_fwhm is not None
            assert self._cache.d_j is not None
            return self._cache.value, self._cache.d_dx, self._cache.d_fwhm, self._cache.d_j

        # Compute fresh and update cache
        val, d_dx, d_fwhm, d_j = self._core(dx_arr, fwhm, j_hz, calc_derivs=True)

        self._cache.update_key(dx_arr, fwhm, j_hz)
        self._cache.value = val
        self._cache.d_dx = d_dx
        self._cache.d_fwhm = d_fwhm
        self._cache.d_j = d_j

        assert d_dx is not None and d_fwhm is not None and d_j is not None
        return val, d_dx, d_fwhm, d_j

    def get_cached_derivatives(
        self, dx: FloatArray, fwhm: float, j_hz: float = 0.0
    ) -> tuple[FloatArray, FloatArray, FloatArray] | None:
        """Get cached derivatives if they match the given parameters.

        This method is designed for use after evaluate() has been called
        with the same parameters, allowing efficient Jacobian computation
        in scipy.optimize.least_squares.

        Args:
            dx: Frequency offset array (Hz)
            fwhm: Full width at half maximum (Hz)
            j_hz: J-coupling constant (Hz), default 0.0

        Returns
        -------
            Tuple of (d_value_dx, d_value_dfwhm, d_value_dj) if cache hit, None otherwise
        """
        dx_arr = np.asarray(dx)
        if self._cache.matches(dx_arr, fwhm, j_hz) and self._cache.d_dx is not None:
            assert self._cache.d_fwhm is not None and self._cache.d_j is not None
            return self._cache.d_dx, self._cache.d_fwhm, self._cache.d_j
        return None

    def clear_cache(self) -> None:
        """Clear the cached results."""
        self._cache = CachedResult()

    def height(self, j_hz: float = 0.0) -> float:
        """Height at peak center (always 1.0 for normalized lineshapes)."""
        return 1.0

    @abstractmethod
    def integral(self, fwhm: float, j_hz: float = 0.0) -> float:
        """Compute analytical integral."""
        ...


class LorentzianEvaluator(FWHMLineshapeEvaluator):
    """Lorentzian lineshape evaluator.

    L(dx) = γ² / (γ² + dx²)  where γ = fwhm/2

    Height-normalized to 1.0 at center.
    """

    def _compute_single(
        self, dx: FloatArray, fwhm: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        gamma = 0.5 * fwhm
        gamma2 = gamma * gamma
        dx2 = dx * dx
        denom = gamma2 + dx2
        lorentz = gamma2 / denom

        if not calc_derivs:
            return lorentz, None, None

        denom_inv2 = 1.0 / (denom * denom)
        d_dx = -2.0 * gamma2 * dx * denom_inv2
        d_fwhm = gamma * dx2 * denom_inv2
        return lorentz, d_dx, d_fwhm

    def integral(self, fwhm: float, j_hz: float = 0.0) -> float:
        """Analytical integral: π * fwhm / 2."""
        integral = np.pi * fwhm / 2.0
        return integral * 2.0 if j_hz != 0.0 else integral


class GaussianEvaluator(FWHMLineshapeEvaluator):
    """Gaussian lineshape evaluator.

    G(dx) = exp(-c * dx²)  where c = 4*ln(2) / fwhm²

    Height-normalized to 1.0 at center.
    """

    def _compute_single(
        self, dx: FloatArray, fwhm: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        c = 4.0 * _LN2 / (fwhm * fwhm)
        dx2 = dx * dx
        gauss = np.exp(-c * dx2)

        if not calc_derivs:
            return gauss, None, None

        c2 = 2.0 * c
        d_dx = -c2 * dx * gauss
        d_fwhm = (c2 / fwhm) * dx2 * gauss
        return gauss, d_dx, d_fwhm

    def integral(self, fwhm: float, j_hz: float = 0.0) -> float:
        """Analytical integral: fwhm * sqrt(π / (4*ln(2)))."""
        integral = fwhm * _SQRT_PI_4LN2
        return integral * 2.0 if j_hz != 0.0 else integral


class PseudoVoigtEvaluator:
    """Pseudo-Voigt lineshape evaluator.

    V(dx) = η * L(dx) + (1-η) * G(dx)

    Linear combination of Lorentzian and Gaussian with mixing parameter η ∈ [0,1].
    Height-normalized to 1.0 at center.

    Note: Does not inherit from FWHMLineshapeEvaluator due to extra eta parameter.

    Caching Strategy:
        Similar to FWHMLineshapeEvaluator, caches results from evaluate() calls
        for efficient subsequent get_cached_derivatives() calls.
    """

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache = CachedResult()

    def _compute_single(
        self, dx: FloatArray, fwhm: float, eta: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None, FloatArray | None]:
        """Compute PseudoVoigt for single component."""
        gamma = 0.5 * fwhm
        gamma2 = gamma * gamma
        c = 4.0 * _LN2 / (fwhm * fwhm)
        dx2 = dx * dx

        # Component values
        denom = gamma2 + dx2
        lorentz = gamma2 / denom
        gauss = np.exp(-c * dx2)
        pvoigt = eta * lorentz + (1.0 - eta) * gauss

        if not calc_derivs:
            return pvoigt, None, None, None

        # Lorentzian derivatives
        denom_inv2 = 1.0 / (denom * denom)
        d_lor_dx = -2.0 * gamma2 * dx * denom_inv2
        d_lor_fwhm = gamma * dx2 * denom_inv2

        # Gaussian derivatives
        c2 = 2.0 * c
        d_gau_dx = -c2 * dx * gauss
        d_gau_fwhm = (c2 / fwhm) * dx2 * gauss

        # Combined derivatives
        one_minus_eta = 1.0 - eta
        d_dx = eta * d_lor_dx + one_minus_eta * d_gau_dx
        d_fwhm = eta * d_lor_fwhm + one_minus_eta * d_gau_fwhm
        d_eta = lorentz - gauss

        return pvoigt, d_dx, d_fwhm, d_eta

    def _core(
        self, dx: FloatArray, fwhm: float, eta: float, j_hz: float, calc_derivs: bool
    ) -> tuple[
        FloatArray, FloatArray | None, FloatArray | None, FloatArray | None, FloatArray | None
    ]:
        """Core computation with doublet handling."""
        if j_hz == 0.0:
            val, d_dx, d_fwhm, d_eta = self._compute_single(dx, fwhm, eta, calc_derivs)
            if not calc_derivs:
                return val, None, None, None, None
            return val, d_dx, d_fwhm, d_eta, np.zeros_like(val)

        # Doublet case
        j_shift = 0.5 * j_hz
        val_p, d_dx_p, d_fwhm_p, d_eta_p = self._compute_single(
            dx + j_shift, fwhm, eta, calc_derivs
        )
        val_m, d_dx_m, d_fwhm_m, d_eta_m = self._compute_single(
            dx - j_shift, fwhm, eta, calc_derivs
        )

        val = val_p + val_m
        if not calc_derivs:
            return val, None, None, None, None

        assert d_dx_p is not None and d_dx_m is not None
        assert d_fwhm_p is not None and d_fwhm_m is not None
        assert d_eta_p is not None and d_eta_m is not None
        return (
            val,
            d_dx_p + d_dx_m,
            d_fwhm_p + d_fwhm_m,
            d_eta_p + d_eta_m,
            0.5 * (d_dx_p - d_dx_m),
        )

    def evaluate(self, dx: FloatArray, fwhm: float, eta: float, j_hz: float = 0.0) -> FloatArray:
        """Evaluate Pseudo-Voigt lineshape.

        Also computes and caches derivatives for efficient subsequent jac() calls.

        Args:
            dx: Frequency offset array (Hz)
            fwhm: Full width at half maximum (Hz)
            eta: Mixing parameter (0=Gaussian, 1=Lorentzian)
            j_hz: J-coupling constant (Hz), default 0.0

        Returns
        -------
            Lineshape values at given offsets
        """
        dx_arr = np.asarray(dx)

        # Check cache first
        if self._cache.matches(dx_arr, fwhm, eta, j_hz) and self._cache.value is not None:
            return self._cache.value

        # Compute with derivatives and cache
        val, d_dx, d_fwhm, d_eta, d_j = self._core(dx_arr, fwhm, eta, j_hz, calc_derivs=True)

        # Update cache
        self._cache.update_key(dx_arr, fwhm, eta, j_hz)
        self._cache.value = val
        self._cache.d_dx = d_dx
        self._cache.d_fwhm = d_fwhm
        self._cache.d_eta = d_eta
        self._cache.d_j = d_j

        return val

    def evaluate_derivatives(
        self, dx: FloatArray, fwhm: float, eta: float, j_hz: float = 0.0
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Evaluate Pseudo-Voigt and all derivatives.

        Returns cached derivatives if available from prior evaluate() call.
        Otherwise computes fresh.

        Args:
            dx: Frequency offset array (Hz)
            fwhm: Full width at half maximum (Hz)
            eta: Mixing parameter (0=Gaussian, 1=Lorentzian)
            j_hz: J-coupling constant (Hz), default 0.0

        Returns
        -------
            Tuple of (value, d_dx, d_fwhm, d_eta, d_j)
        """
        dx_arr = np.asarray(dx)

        # Check cache first
        if self._cache.matches(dx_arr, fwhm, eta, j_hz) and self._cache.value is not None:
            assert self._cache.d_dx is not None
            assert self._cache.d_fwhm is not None
            assert self._cache.d_eta is not None
            assert self._cache.d_j is not None
            return (
                self._cache.value,
                self._cache.d_dx,
                self._cache.d_fwhm,
                self._cache.d_eta,
                self._cache.d_j,
            )

        # Compute fresh and update cache
        val, d_dx, d_fwhm, d_eta, d_j = self._core(dx_arr, fwhm, eta, j_hz, calc_derivs=True)

        self._cache.update_key(dx_arr, fwhm, eta, j_hz)
        self._cache.value = val
        self._cache.d_dx = d_dx
        self._cache.d_fwhm = d_fwhm
        self._cache.d_eta = d_eta
        self._cache.d_j = d_j

        assert d_dx is not None and d_fwhm is not None and d_eta is not None and d_j is not None
        return val, d_dx, d_fwhm, d_eta, d_j

    def get_cached_derivatives(
        self, dx: FloatArray, fwhm: float, eta: float, j_hz: float = 0.0
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None:
        """Get cached derivatives if they match the given parameters.

        Args:
            dx: Frequency offset array (Hz)
            fwhm: Full width at half maximum (Hz)
            eta: Mixing parameter (0=Gaussian, 1=Lorentzian)
            j_hz: J-coupling constant (Hz), default 0.0

        Returns
        -------
            Tuple of (d_dx, d_fwhm, d_eta, d_j) if cache hit, None otherwise
        """
        dx_arr = np.asarray(dx)
        if self._cache.matches(dx_arr, fwhm, eta, j_hz) and self._cache.d_dx is not None:
            assert self._cache.d_fwhm is not None
            assert self._cache.d_eta is not None
            assert self._cache.d_j is not None
            return self._cache.d_dx, self._cache.d_fwhm, self._cache.d_eta, self._cache.d_j
        return None

    def clear_cache(self) -> None:
        """Clear the cached results."""
        self._cache = CachedResult()

    def height(self, j_hz: float = 0.0) -> float:
        """Height at peak center (always 1.0)."""
        del j_hz  # Unused but part of interface
        return 1.0

    def integral(self, fwhm: float, eta: float, j_hz: float = 0.0) -> float:
        """Analytical integral as weighted sum of components."""
        integral = eta * (np.pi * fwhm / 2.0) + (1.0 - eta) * (fwhm * _SQRT_PI_4LN2)
        return integral * 2.0 if j_hz != 0.0 else integral


# =============================================================================
# R2-Based Lineshapes (Complex-valued with apodization)
# =============================================================================


class ApodizationEvaluator(ABC):
    """Abstract base for apodization-based lineshapes (NoApod, SP1, SP2).

    These lineshapes are:
    - Complex-valued internally, returning real part after phase correction
    - Parameterized by R2 relaxation rate and phase
    - Require acquisition time (aq) at initialization

    Caching Strategy:
        Similar to FWHM-based evaluators, caches results from evaluate() calls
        for efficient subsequent get_cached_derivatives() calls.
    """

    def __init__(self, aq: float) -> None:
        """Initialize with acquisition time.

        Args:
            aq: Acquisition time in seconds
        """
        self.aq = aq
        self._cache = CachedResult()

    @abstractmethod
    def _compute_complex(
        self, dx: FloatArray, r2: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        """Compute complex lineshape for single component.

        Returns
        -------
            (complex_value, d_value_dx, d_value_dr2) - all complex arrays
        """
        ...

    def _apply_phase(
        self,
        val: FloatArray,
        phase: float,
        d_dx: FloatArray | None = None,
        d_r2: FloatArray | None = None,
        d_jhz: FloatArray | None = None,
    ) -> tuple[
        FloatArray, FloatArray | None, FloatArray | None, FloatArray | None, FloatArray | None
    ]:
        """Apply phase correction and extract real parts.

        Parameters
        ----------
        val : FloatArray
            Complex lineshape values
        phase : float
            Phase correction in degrees
        d_dx, d_r2, d_jhz : FloatArray | None
            Optional complex derivatives

        Returns
        -------
        tuple
            Real parts of (value, d_dx, d_r2, d_phase, d_jhz)
        """
        if d_dx is None:
            # No derivatives case
            if phase != 0.0:
                val = val * np.exp(1j * np.deg2rad(phase))
            return val.real, None, None, None, None

        # d_r2 is guaranteed non-None when d_dx is non-None
        assert d_r2 is not None

        d_phase = 1j * val

        if phase != 0.0:
            pf = np.exp(1j * np.deg2rad(phase))
            val = val * pf
            d_dx = d_dx * pf
            d_r2 = d_r2 * pf
            d_phase = d_phase * pf
            if d_jhz is not None:
                d_jhz = d_jhz * pf

        d_jhz_real = d_jhz.real if d_jhz is not None else np.zeros_like(val.real)
        # After multiplication by pf, d_dx and d_r2 are guaranteed to be arrays
        d_dx_real: FloatArray = d_dx.real  # type: ignore[union-attr]
        d_r2_real: FloatArray = d_r2.real  # type: ignore[union-attr]
        return val.real, d_dx_real, d_r2_real, d_phase.real, d_jhz_real

    def _core(
        self, dx: FloatArray, r2: float, phase: float, j_hz: float, calc_derivs: bool
    ) -> tuple[
        FloatArray, FloatArray | None, FloatArray | None, FloatArray | None, FloatArray | None
    ]:
        """Core computation with doublet and phase handling."""
        if j_hz == 0.0:
            val, d_dx, d_r2 = self._compute_complex(dx, r2, calc_derivs)
            if not calc_derivs:
                return self._apply_phase(val, phase)
            return self._apply_phase(val, phase, d_dx, d_r2, np.zeros_like(val))

        # Doublet case
        j_rad = np.pi * j_hz
        val_p, d_dx_p, d_r2_p = self._compute_complex(dx + j_rad, r2, calc_derivs)
        val_m, d_dx_m, d_r2_m = self._compute_complex(dx - j_rad, r2, calc_derivs)

        val = val_p + val_m
        if not calc_derivs:
            return self._apply_phase(val, phase)

        assert d_dx_p is not None and d_dx_m is not None
        assert d_r2_p is not None and d_r2_m is not None
        d_dx = d_dx_p + d_dx_m
        d_r2 = d_r2_p + d_r2_m
        d_jhz = np.pi * (d_dx_p - d_dx_m)

        return self._apply_phase(val, phase, d_dx, d_r2, d_jhz)

    def evaluate(
        self, dx: FloatArray, r2: float, phase: float = 0.0, j_hz: float = 0.0
    ) -> FloatArray:
        """Evaluate lineshape values.

        Also computes and caches derivatives for efficient subsequent jac() calls.

        Args:
            dx: Frequency offset array (radians/s)
            r2: R2 relaxation rate (Hz)
            phase: Phase correction in degrees, default 0.0
            j_hz: J-coupling constant (Hz), default 0.0

        Returns
        -------
            Lineshape values at given offsets
        """
        dx_arr = np.asarray(dx)

        # Check cache first
        if self._cache.matches(dx_arr, r2, phase, j_hz) and self._cache.value is not None:
            return self._cache.value

        # Compute with derivatives and cache
        val, d_dx, d_r2, d_phase, d_j = self._core(dx_arr, r2, phase, j_hz, calc_derivs=True)

        # Update cache
        self._cache.update_key(dx_arr, r2, phase, j_hz)
        self._cache.value = val
        self._cache.d_dx = d_dx
        self._cache.d_r2 = d_r2
        self._cache.d_phase = d_phase
        self._cache.d_j = d_j

        return val

    def evaluate_derivatives(
        self, dx: FloatArray, r2: float, phase: float = 0.0, j_hz: float = 0.0
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Evaluate lineshape and all derivatives.

        Returns cached derivatives if available from prior evaluate() call.
        Otherwise computes fresh.

        Args:
            dx: Frequency offset array (radians/s)
            r2: R2 relaxation rate (Hz)
            phase: Phase correction in degrees, default 0.0
            j_hz: J-coupling constant (Hz), default 0.0

        Returns
        -------
            Tuple of (value, d_dx, d_r2, d_phase, d_j)
        """
        dx_arr = np.asarray(dx)

        # Check cache first
        if self._cache.matches(dx_arr, r2, phase, j_hz) and self._cache.value is not None:
            assert self._cache.d_dx is not None
            assert self._cache.d_r2 is not None
            assert self._cache.d_phase is not None
            assert self._cache.d_j is not None
            return (
                self._cache.value,
                self._cache.d_dx,
                self._cache.d_r2,
                self._cache.d_phase,
                self._cache.d_j,
            )

        # Compute fresh and update cache
        val, d_dx, d_r2, d_phase, d_j = self._core(dx_arr, r2, phase, j_hz, calc_derivs=True)

        self._cache.update_key(dx_arr, r2, phase, j_hz)
        self._cache.value = val
        self._cache.d_dx = d_dx
        self._cache.d_r2 = d_r2
        self._cache.d_phase = d_phase
        self._cache.d_j = d_j

        assert d_dx is not None and d_r2 is not None and d_phase is not None and d_j is not None
        return val, d_dx, d_r2, d_phase, d_j

    def get_cached_derivatives(
        self, dx: FloatArray, r2: float, phase: float = 0.0, j_hz: float = 0.0
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None:
        """Get cached derivatives if they match the given parameters.

        Args:
            dx: Frequency offset array (radians/s)
            r2: R2 relaxation rate (Hz)
            phase: Phase correction in degrees, default 0.0
            j_hz: J-coupling constant (Hz), default 0.0

        Returns
        -------
            Tuple of (d_dx, d_r2, d_phase, d_j) if cache hit, None otherwise
        """
        dx_arr = np.asarray(dx)
        if self._cache.matches(dx_arr, r2, phase, j_hz) and self._cache.d_dx is not None:
            assert self._cache.d_r2 is not None
            assert self._cache.d_phase is not None
            assert self._cache.d_j is not None
            return self._cache.d_dx, self._cache.d_r2, self._cache.d_phase, self._cache.d_j
        return None

    def clear_cache(self) -> None:
        """Clear the cached results."""
        self._cache = CachedResult()

    def __call__(
        self, dx: FloatArray, r2: float, phase: float = 0.0, j_hz: float = 0.0
    ) -> FloatArray:
        """Allow calling the evaluator as a function.

        This provides backward compatibility with factory function usage patterns.

        Args:
            dx: Frequency offset array (radians/s)
            r2: R2 relaxation rate (Hz)
            phase: Phase correction in degrees, default 0.0
            j_hz: J-coupling constant (Hz), default 0.0

        Returns
        -------
            Lineshape values at given offsets
        """
        return self.evaluate(dx, r2, phase, j_hz)

    @abstractmethod
    def height(self, r2: float, j_hz: float = 0.0) -> float:
        """Compute height at peak center."""
        ...

    def integral(self, j_hz: float = 0.0) -> float:
        """Analytical integral (π for singlet, 2π for doublet)."""
        return 2.0 * np.pi if j_hz != 0.0 else np.pi


class NoApodEvaluator(ApodizationEvaluator):
    """Non-apodized lineshape evaluator.

    Complex lineshape from pure FID Fourier transform without window function::

        F(dx) = aq * (1 - exp(-z)) / z  where z = aq * (i*dx + r2)

    Returns real part after optional phase correction.
    """

    def __init__(self, aq: float) -> None:
        """Initialize with acquisition time.

        Parameters
        ----------
        aq : float
            Acquisition time in seconds
        """
        super().__init__(aq)

    def _compute_complex(
        self, dx: FloatArray, r2: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        """Compute complex lineshape for single component."""
        aq = self.aq
        z1 = aq * (1j * dx + r2)
        emz1 = np.exp(-z1)
        val = aq * (1.0 - emz1) / z1

        if not calc_derivs:
            return val, None, None  # type: ignore[return-value]

        dval_dz1 = aq * (emz1 * (z1 + 1) - 1) / z1**2
        d_dx = 1j * aq * dval_dz1
        d_r2 = aq * dval_dz1
        return val, d_dx, d_r2  # type: ignore[return-value]

    def height(self, r2: float, j_hz: float = 0.0) -> float:
        """Compute height at peak position."""
        del j_hz  # Unused but part of interface
        z1 = self.aq * r2
        height = np.real(self.aq * (1.0 - np.exp(-z1)) / z1)
        return float(height)


class SP1Evaluator(ApodizationEvaluator):
    """SP1 (sine bell) apodization lineshape evaluator.

    Complex lineshape with sine-bell window function.
    Precomputes acquisition time and apodization parameters at initialization.
    """

    def __init__(self, aq: float, end: float, off: float) -> None:
        """Initialize with acquisition and apodization parameters.

        Parameters
        ----------
        aq : float
            Acquisition time in seconds
        end : float
            End point of sine bell (fraction of π)
        off : float
            Offset of sine bell (fraction of π)
        """
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


class SP2Evaluator(ApodizationEvaluator):
    """SP2 (sine squared bell) apodization lineshape evaluator.

    Complex lineshape with sine-squared window function.
    Precomputes acquisition time and apodization parameters at initialization.
    """

    def __init__(self, aq: float, end: float, off: float) -> None:
        """Initialize with acquisition and apodization parameters.

        Parameters
        ----------
        aq : float
            Acquisition time in seconds
        end : float
            End point of sine bell (fraction of π)
        off : float
            Offset of sine bell (fraction of π)
        """
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


# =============================================================================
# Convenience Functions (for backward compatibility and simple usage)
# =============================================================================

# Module-level evaluator instances for convenience functions
_lorentzian_evaluator = LorentzianEvaluator()
_gaussian_evaluator = GaussianEvaluator()
_pvoigt_evaluator = PseudoVoigtEvaluator()


def lorentzian(dx: FloatArray, fwhm: float, j_hz: float = 0.0) -> FloatArray:
    """Evaluate Lorentzian lineshape.

    Convenience function for simple evaluation without managing evaluator instances.
    For repeated evaluations or when derivatives are needed, use LorentzianEvaluator
    directly to benefit from caching.

    Args:
        dx: Frequency offset array (Hz)
        fwhm: Full width at half maximum (Hz)
        j_hz: J-coupling constant (Hz), default 0.0

    Returns
    -------
        Lineshape values at given offsets
    """
    return _lorentzian_evaluator.evaluate(np.asarray(dx), fwhm, j_hz)


def gaussian(dx: FloatArray, fwhm: float, j_hz: float = 0.0) -> FloatArray:
    """Evaluate Gaussian lineshape.

    Convenience function for simple evaluation without managing evaluator instances.
    For repeated evaluations or when derivatives are needed, use GaussianEvaluator
    directly to benefit from caching.

    Args:
        dx: Frequency offset array (Hz)
        fwhm: Full width at half maximum (Hz)
        j_hz: J-coupling constant (Hz), default 0.0

    Returns
    -------
        Lineshape values at given offsets
    """
    return _gaussian_evaluator.evaluate(np.asarray(dx), fwhm, j_hz)


def pvoigt(dx: FloatArray, fwhm: float, eta: float, j_hz: float = 0.0) -> FloatArray:
    """Evaluate Pseudo-Voigt lineshape.

    Convenience function for simple evaluation without managing evaluator instances.
    For repeated evaluations or when derivatives are needed, use PseudoVoigtEvaluator
    directly to benefit from caching.

    Args:
        dx: Frequency offset array (Hz)
        fwhm: Full width at half maximum (Hz)
        eta: Mixing parameter (0=Gaussian, 1=Lorentzian)
        j_hz: J-coupling constant (Hz), default 0.0

    Returns
    -------
        Lineshape values at given offsets
    """
    return _pvoigt_evaluator.evaluate(np.asarray(dx), fwhm, eta, j_hz)


def no_apod(
    dx: FloatArray, r2: float, aq: float, phase: float = 0.0, j_hz: float = 0.0
) -> FloatArray:
    """Evaluate non-apodized lineshape.

    Convenience function that creates a temporary NoApodEvaluator.
    For repeated evaluations with the same aq, use NoApodEvaluator directly.

    Args:
        dx: Frequency offset array (radians/s)
        r2: R2 relaxation rate (Hz)
        aq: Acquisition time in seconds
        phase: Phase correction in degrees, default 0.0
        j_hz: J-coupling constant (Hz), default 0.0

    Returns
    -------
        Lineshape values at given offsets
    """
    evaluator = NoApodEvaluator(aq)
    return evaluator.evaluate(np.asarray(dx), r2, phase, j_hz)


def make_sp1_evaluator(aq: float, end: float, off: float) -> SP1Evaluator:
    """Create an SP1 evaluator with the given parameters.

    This is a factory function for creating SP1Evaluator instances
    with pre-computed exponentials. The returned evaluator can be called
    directly or via its evaluate() method.

    Args:
        aq: Acquisition time in seconds
        end: End point of sine bell (fraction of π)
        off: Offset of sine bell (fraction of π)

    Returns
    -------
        SP1Evaluator instance ready for evaluation

    Example:
        sp1 = make_sp1_evaluator(0.1, 1.0, 0.35)
        result = sp1.evaluate(dx, r2, phase)
        # Or using __call__:
        result = sp1(dx, r2, phase=0.0)
    """
    return SP1Evaluator(aq, end, off)


def make_sp2_evaluator(aq: float, end: float, off: float) -> SP2Evaluator:
    """Create an SP2 evaluator with the given parameters.

    This is a factory function for creating SP2Evaluator instances
    with pre-computed exponentials. The returned evaluator can be called
    directly or via its evaluate() method.

    Args:
        aq: Acquisition time in seconds
        end: End point of sine bell (fraction of π)
        off: Offset of sine bell (fraction of π)

    Returns
    -------
        SP2Evaluator instance ready for evaluation

    Example:
        sp2 = make_sp2_evaluator(0.1, 1.0, 0.35)
        result = sp2.evaluate(dx, r2, phase)
        # Or using __call__:
        result = sp2(dx, r2, phase=0.0)
    """
    return SP2Evaluator(aq, end, off)
