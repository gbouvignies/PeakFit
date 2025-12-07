"""Linear algebra utilities for variable projection optimization.

This module encapsulates the lower-level linear algebra operations required
for the VarPro algorithm, specifically QR decomposition and solving for
linear parameters (amplitudes).
"""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.linalg import solve_triangular


class LinearAlgebraHelper:
    """Helper class for linear algebra operations in VarPro."""

    @staticmethod
    def qr_decomposition(shapes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Perform reduced QR decomposition on the shapes matrix.

        Args:
            shapes: Matrix of shape (n_peaks, n_points)

        Returns
        -------
            Tuple of (Q, R) where:
            - Q has shape (n_points, n_peaks)
            - R has shape (n_peaks, n_peaks)
        """
        # Transpose shapes to (n_points, n_peaks) for QR
        return np.linalg.qr(shapes.T, mode="reduced")

    @staticmethod
    def solve_amplitudes(q: np.ndarray, r: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Solve for linear amplitudes using QR factors.

        Solves the system: R @ amplitudes = Q.T @ data

        Args:
            q: Q matrix from QR decomposition
            r: R matrix from QR decomposition
            data: Data vector or matrix

        Returns
        -------
            Amplitudes vector or matrix
        """
        qty = q.T @ data

        # Use simple try-except block for the solver
        try:
            return cast("np.ndarray", solve_triangular(r, qty, check_finite=False))
        except np.linalg.LinAlgError:
            # Fallback for singular R (rank deficient cases)
            result, *_ = np.linalg.lstsq(r, qty, rcond=None)
            return cast("np.ndarray", result)

    @staticmethod
    def compute_phi_pinv(q: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Compute the pseudo-inverse helper for Jacobian correction.

        Computes phi_pinv = R^-1 @ Q.T

        Args:
            q: Q matrix
            r: R matrix

        Returns
        -------
            phi_pinv matrix
        """
        try:
            return cast("np.ndarray", solve_triangular(r, q.T, check_finite=False))
        except np.linalg.LinAlgError:
            return cast("np.ndarray", np.linalg.pinv(r) @ q.T)

    @staticmethod
    def project_residuals(data: np.ndarray, q: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
        """Compute residuals by projecting data efficiently.

        residuals = data - Q @ (Q.T @ data)
                  = data - Q @ qty (but we use amplitudes to verify)

        Actually for VarPro, residuals = data - model
        model = shapes.T @ amplitudes
        Since shapes.T = Q @ R, model = Q @ R @ amplitudes
        And R @ amplitudes = Q.T @ data (approx)

        Args:
            data: Original data
            q: Q matrix
            amplitudes: Calculated amplitudes (used if we want strict model subtraction)

        Returns
        -------
            Residuals
        """
        # More stable to use the definition: residuals = P_perp @ data
        # P_perp = I - Q @ Q.T
        # residuals = data - Q @ (Q.T @ data)
        # Note: (Q.T @ data) is 'qty' which we computed in solve_amplitudes

        qty = q.T @ data
        return cast("np.ndarray", data - q @ qty)
