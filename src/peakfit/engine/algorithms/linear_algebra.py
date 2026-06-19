"""Linear algebra utilities for variable projection optimization.

This module encapsulates the lower-level linear algebra operations required
for the VarPro algorithm and general amplitude solution.
"""

import warnings
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.linalg import solve_triangular

if TYPE_CHECKING:
    from peakfit.shared.typing import FloatArray


def qr_decomposition(shapes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perform reduced QR decomposition on the shapes matrix.

    Args:
        shapes: Matrix of shape (n_peaks, n_points)

    Returns:
    -------
        Tuple of (Q, R) where:
        - Q has shape (n_points, n_peaks)
        - R has shape (n_peaks, n_peaks)
    """
    # Shapes are (n_peaks, n_points), but QR needs (n_points, n_peaks) for A*x=b
    return np.linalg.qr(shapes.T, mode="reduced")


def solve_amplitudes(q: np.ndarray, r: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Solve for linear amplitudes using QR factors.

    Solves the system: R @ amplitudes = Q.T @ data

    Args:
        q: Q matrix from QR decomposition
        r: R matrix from QR decomposition
        data: Data vector (n_points,) or matrix (n_points, n_series)

    Returns:
    -------
        Amplitudes vector or matrix (n_peaks, ...)
    """
    qty = q.T @ data

    try:
        return cast("np.ndarray", solve_triangular(r, qty, check_finite=False))
    except (np.linalg.LinAlgError, ValueError):
        # Fallback for rank deficient cases
        result, *_ = np.linalg.lstsq(r, qty, rcond=None)
        return cast("np.ndarray", result)


def compute_phi_pinv(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Compute the pseudo-inverse helper for Jacobian correction.

    Computes phi_pinv = R^-1 @ Q.T
    """
    try:
        return cast("np.ndarray", solve_triangular(r, q.T, check_finite=False))
    except (np.linalg.LinAlgError, ValueError):
        return cast("np.ndarray", np.linalg.pinv(r) @ q.T)


def project_residuals(data: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute residuals by projecting data efficiently.

    residuals = data - model
    model = shapes.T @ amplitudes = Q @ R @ amplitudes = Q @ (Q.T @ data)
    """
    projection = q @ (q.T @ data)
    return cast("np.ndarray", data - projection)


def calculate_amplitudes(shapes: FloatArray, data: FloatArray) -> FloatArray:
    """Calculate peak amplitudes via QR decomposition (more stable than Normal Equations)."""
    if not np.all(np.isfinite(shapes)) or not np.all(np.isfinite(data)):
        n_peaks = shapes.shape[0]
        extra_dims = data.shape[1:] if data.ndim > 1 else ()
        return np.full((n_peaks, *extra_dims), np.nan)

    q, r = qr_decomposition(shapes)
    if np.linalg.matrix_rank(r) < r.shape[0]:
        warnings.warn(
            "Rank-deficient design matrix detected while solving amplitudes; "
            "falling back to least-squares solution.",
            RuntimeWarning,
            stacklevel=2,
        )
        qty = q.T @ data
        result, *_ = np.linalg.lstsq(r, qty, rcond=None)
        return cast("FloatArray", result)
    return solve_amplitudes(q, r, data)


def calculate_amplitude_covariance(shapes: FloatArray, noise: float) -> FloatArray:
    """Calculate covariance matrix for amplitudes using QR factors.

    Cov = (J.T J)^-1 * sigma^2
    Here J = Shapes.T. so J.T J = Shapes @ Shapes.T
    Using QR: Shapes.T = Q R.  Shapes @ Shapes.T = R.T Q.T Q R = R.T R.
    Cov = (R.T R)^-1 * sigma^2 = R^-1 (R.T)^-1 * sigma^2
    """
    _, r = qr_decomposition(shapes)

    try:
        r_inv = np.linalg.inv(r)
        cov = r_inv @ r_inv.T
    except np.linalg.LinAlgError:
        warnings.warn(
            "Rank-deficient design matrix detected while computing amplitude covariance; "
            "falling back to pseudo-inverse.",
            RuntimeWarning,
            stacklevel=2,
        )
        # Fallback to pseudo-inverse of STS
        sts = shapes @ shapes.T
        cov = np.linalg.pinv(sts)

    return np.asarray(cov * (noise**2), dtype=np.float64)


def calculate_amplitudes_with_uncertainty(
    shapes: FloatArray, data: FloatArray, noise: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Calculate amplitudes and their uncertainties."""
    if not np.all(np.isfinite(shapes)) or not np.all(np.isfinite(data)):
        n_peaks = shapes.shape[0]
        extra_dims = data.shape[1:] if data.ndim > 1 else ()
        return (
            np.full((n_peaks, *extra_dims), np.nan),
            np.full((n_peaks,), np.nan),
            np.full((n_peaks, n_peaks), np.nan),
        )

    q, r = qr_decomposition(shapes)
    if np.linalg.matrix_rank(r) < r.shape[0]:
        warnings.warn(
            "Rank-deficient design matrix detected while solving amplitudes; "
            "falling back to least-squares solution.",
            RuntimeWarning,
            stacklevel=2,
        )
        qty = q.T @ data
        amplitudes, *_ = np.linalg.lstsq(r, qty, rcond=None)
    else:
        amplitudes = solve_amplitudes(q, r, data)

    # Covariance
    try:
        r_inv = np.linalg.inv(r)
        covariance = (r_inv @ r_inv.T) * (noise**2)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Rank-deficient design matrix detected while computing amplitude covariance; "
            "falling back to pseudo-inverse.",
            RuntimeWarning,
            stacklevel=2,
        )
        sts = shapes @ shapes.T
        covariance = np.linalg.pinv(sts) * (noise**2)

    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    return (
        np.asarray(amplitudes, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
        np.asarray(covariance, dtype=np.float64),
    )
