import numpy as np
import sys
import os

# Define FloatArray for type hinting
FloatArray = np.ndarray


class NoApodEvaluator:
    """Evaluator for non-apodized lineshape."""

    def __init__(self, aq: float) -> None:
        self.aq = aq

    def evaluate(
        self, dx: FloatArray, r2: float, phase: float = 0.0, j_hz: float = 0.0
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Evaluate non-apodized lineshape and derivatives.

        Args:
            dx: Frequency offset from peak center (rad/s)
            r2: Transverse relaxation rate (Hz)
            phase: Phase correction (degrees)
            j_hz: Scalar coupling constant (Hz)

        Returns
        -------
            Tuple of (spec, d_spec_dx, d_spec_dr2, d_spec_dphase, d_spec_djhz)
        """
        # Handle scalar coupling
        if j_hz == 0.0:
            j_rads = np.array([[0.0]])
        else:
            j_rads = j_hz * np.pi * np.array([[1.0, -1.0]])

        # Reshape dx to broadcast with j_rads: (n_freq,) -> (n_freq, 1)
        dx_in = np.atleast_1d(dx)
        original_shape = dx_in.shape
        dx_expanded = dx_in.reshape(-1, 1) + j_rads

        # Complex arithmetic implementation
        z1 = self.aq * (1j * dx_expanded + r2)
        exp_nz1 = np.exp(-z1)

        # Lineshape
        L = self.aq * (1.0 - exp_nz1) / z1

        # dL/dz1 = aq * (exp(-z1)*(z1 + 1) - 1) / z1^2
        dL_dz1 = self.aq * (exp_nz1 * (z1 + 1.0) - 1.0) / z1**2

        # Derivatives
        dL_ddx = 1j * self.aq * dL_dz1
        dL_dr2 = self.aq * dL_dz1
        dL_dphase = 1j * L

        # Apply phase correction
        if phase != 0.0:
            phase_rad = np.deg2rad(phase)
            phase_factor = np.exp(1j * phase_rad)
            L = L * phase_factor
            dL_ddx = dL_ddx * phase_factor
            dL_dr2 = dL_dr2 * phase_factor
            dL_dphase = dL_dphase * phase_factor

        # d(L)/d(phase_deg) = d(L)/d(phase_rad) * (pi/180)
        deg2rad = np.pi / 180.0
        dL_dphase *= deg2rad

        # Extract real parts
        L_re = L.real
        dL_ddx_re = dL_ddx.real
        dL_dr2_re = dL_dr2.real
        dL_dphase_re = dL_dphase.real

        # Handle J coupling combination
        if j_hz == 0.0:
            dL_djhz_re = np.zeros_like(L_re)

            L_re = L_re.reshape(original_shape)
            dL_ddx_re = dL_ddx_re.reshape(original_shape)
            dL_dr2_re = dL_dr2_re.reshape(original_shape)
            dL_dphase_re = dL_dphase_re.reshape(original_shape)
        else:
            # dL/djhz = pi * (dL/dx_p - dL/dx_m)
            dL_djhz_re = np.pi * (dL_ddx_re[:, 0] - dL_ddx_re[:, 1])

            L_re = L_re.sum(axis=-1).reshape(original_shape)
            dL_ddx_re = dL_ddx_re.sum(axis=-1).reshape(original_shape)
            dL_dr2_re = dL_dr2_re.sum(axis=-1).reshape(original_shape)
            dL_dphase_re = dL_dphase_re.sum(axis=-1).reshape(original_shape)
            dL_djhz_re = dL_djhz_re.reshape(original_shape)

        return L_re, dL_ddx_re, dL_dr2_re, dL_dphase_re, dL_djhz_re


def no_apod_lineshape_analytical(dw, r2, aq, phase=0.0, j_hz=0.0):
    """
    No-apodization lineshape with analytical derivatives including j_hz.
    """
    # Handle scalar coupling: compute at dx ± π*j_hz
    if j_hz == 0.0:
        # Simple case: no coupling
        z1 = aq * (1j * dw + r2)
        exp_nz1 = np.exp(-z1)

        L = aq * (1.0 - exp_nz1) / z1
        dL_dz1 = aq * (exp_nz1 * (z1 + 1) - 1) / z1**2

        dL_ddx = 1j * aq * dL_dz1
        dL_dr2 = aq * dL_dz1
        dL_dphase = 1j * L
        dL_djhz = np.zeros_like(L)  # No dependence when j_hz=0

        if phase != 0.0:
            pf = np.exp(1j * np.deg2rad(phase))
            L, dL_ddx, dL_dr2, dL_dphase = (
                L * pf,
                dL_ddx * pf,
                dL_dr2 * pf,
                dL_dphase * pf,
            )

        # d(L)/d(phase_deg) = d(L)/d(phase_rad) * (pi/180)
        deg2rad = np.pi / 180.0
        dL_dphase *= deg2rad

        return L.real, dL_ddx.real, dL_dr2.real, dL_dphase.real, dL_djhz.real

    # Doublet case: compute at both offset frequencies
    j_rad = np.pi * j_hz
    dx_plus = dw + j_rad
    dx_minus = dw - j_rad

    # Compute lineshape at +πJ
    z1_p = aq * (1j * dx_plus + r2)
    exp_nz1_p = np.exp(-z1_p)
    L_p = aq * (1.0 - exp_nz1_p) / z1_p
    dL_dz1_p = aq * (exp_nz1_p * (z1_p + 1) - 1) / z1_p**2
    dL_ddx_p = 1j * aq * dL_dz1_p
    dL_dr2_p = aq * dL_dz1_p

    # Compute lineshape at -πJ
    z1_m = aq * (1j * dx_minus + r2)
    exp_nz1_m = np.exp(-z1_m)
    L_m = aq * (1.0 - exp_nz1_m) / z1_m
    dL_dz1_m = aq * (exp_nz1_m * (z1_m + 1) - 1) / z1_m**2
    dL_ddx_m = 1j * aq * dL_dz1_m
    dL_dr2_m = aq * dL_dz1_m

    # Total lineshape is sum of both components
    L = L_p + L_m
    dL_ddx = dL_ddx_p + dL_ddx_m
    dL_dr2 = dL_dr2_p + dL_dr2_m
    dL_dphase = 1j * L

    # Key insight: dL_total/d(j_hz) = π * (dL/ddx|+πJ - dL/ddx|-πJ)
    # Because d(dx±πJ)/d(j_hz) = ±π
    dL_djhz = np.pi * (dL_ddx_p - dL_ddx_m)

    if phase != 0.0:
        pf = np.exp(1j * np.deg2rad(phase))
        L = L * pf
        dL_ddx = dL_ddx * pf
        dL_dr2 = dL_dr2 * pf
        dL_dphase = dL_dphase * pf
        dL_djhz = dL_djhz * pf

    # d(L)/d(phase_deg) = d(L)/d(phase_rad) * (pi/180)
    deg2rad = np.pi / 180.0
    dL_dphase *= deg2rad

    return L.real, dL_ddx.real, dL_dr2.real, dL_dphase.real, dL_djhz.real


def test_analytical_derivatives():
    print("Testing analytical derivatives...")
    r2_test = 30.0
    aq_test = 0.15
    phase_test = 15.0  # degrees
    dw_test = np.linspace(-150, 150, 100)

    # Get analytical derivatives
    L_anal, dL_ddw_anal, dL_dr2_anal, dL_dphase_anal, dL_djhz_anal = no_apod_lineshape_analytical(
        dw_test, r2_test, aq_test, phase=phase_test, j_hz=0.0
    )

    # Finite difference checks
    eps_dw = 1e-5
    eps_r2 = 1e-5
    eps_phase = 1e-5  # degrees

    # d/ddw
    L_plus, _, _, _, _ = no_apod_lineshape_analytical(
        dw_test + eps_dw, r2_test, aq_test, phase=phase_test
    )
    L_minus, _, _, _, _ = no_apod_lineshape_analytical(
        dw_test - eps_dw, r2_test, aq_test, phase=phase_test
    )
    dL_ddw_fd = (L_plus - L_minus) / (2 * eps_dw)

    # d/dr2
    L_plus, _, _, _, _ = no_apod_lineshape_analytical(
        dw_test, r2_test + eps_r2, aq_test, phase=phase_test
    )
    L_minus, _, _, _, _ = no_apod_lineshape_analytical(
        dw_test, r2_test - eps_r2, aq_test, phase=phase_test
    )
    dL_dr2_fd = (L_plus - L_minus) / (2 * eps_r2)

    # d/dphase
    L_plus, _, _, _, _ = no_apod_lineshape_analytical(
        dw_test, r2_test, aq_test, phase=phase_test + eps_phase
    )
    L_minus, _, _, _, _ = no_apod_lineshape_analytical(
        dw_test, r2_test, aq_test, phase=phase_test - eps_phase
    )
    dL_dphase_fd = (L_plus - L_minus) / (2 * eps_phase)

    print(f"d/ddw max diff: {np.max(np.abs(dL_ddw_anal - dL_ddw_fd))}")
    print(f"d/dr2 max diff: {np.max(np.abs(dL_dr2_anal - dL_dr2_fd))}")
    print(f"d/dphase max diff: {np.max(np.abs(dL_dphase_anal - dL_dphase_fd))}")

    # Check J coupling
    print("\nTesting J coupling derivatives...")
    j_hz = 30.0
    L_j, dL_ddw_j, dL_dr2_j, dL_dphase_j, dL_djhz_j = no_apod_lineshape_analytical(
        dw_test, r2_test, aq_test, phase=phase_test, j_hz=j_hz
    )

    eps_j = 1e-5
    L_plus, _, _, _, _ = no_apod_lineshape_analytical(
        dw_test, r2_test, aq_test, phase=phase_test, j_hz=j_hz + eps_j
    )
    L_minus, _, _, _, _ = no_apod_lineshape_analytical(
        dw_test, r2_test, aq_test, phase=phase_test, j_hz=j_hz - eps_j
    )
    dL_djhz_fd = (L_plus - L_minus) / (2 * eps_j)

    print(f"d/djhz max diff: {np.max(np.abs(dL_djhz_j - dL_djhz_fd))}")


def test_class_implementation():
    print("\nTesting NoApodEvaluator class implementation...")

    r2_test = 30.0
    aq_test = 0.15
    phase_test = 15.0  # degrees
    dw_test = np.linspace(-150, 150, 100)
    j_hz = 30.0

    # Instantiate class
    evaluator = NoApodEvaluator(aq=aq_test)

    # Evaluate using class
    L_cls, dL_ddw_cls, dL_dr2_cls, dL_dphase_cls, dL_djhz_cls = evaluator.evaluate(
        dw_test, r2_test, phase=phase_test, j_hz=j_hz
    )

    # Evaluate using analytical function
    L_anal, dL_ddw_anal, dL_dr2_anal, dL_dphase_anal, dL_djhz_anal = no_apod_lineshape_analytical(
        dw_test, r2_test, aq_test, phase=phase_test, j_hz=j_hz
    )

    print(f"L match: {np.allclose(L_cls, L_anal)}")
    print(f"dL_ddx match: {np.allclose(dL_ddw_cls, dL_ddw_anal)}")
    print(f"dL_dr2 match: {np.allclose(dL_dr2_cls, dL_dr2_anal)}")
    print(f"dL_dphase match: {np.allclose(dL_dphase_cls, dL_dphase_anal)}")
    print(f"dL_djhz match: {np.allclose(dL_djhz_cls, dL_djhz_anal)}")


if __name__ == "__main__":
    test_analytical_derivatives()
    test_class_implementation()
