"""Rank-aware linear least squares for variable projection and analytical output."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import solve_triangular

if TYPE_CHECKING:
    from peakfit.shared.typing import FloatArray


@dataclass(frozen=True)
class ShapeProjection:
    """One factorization defines the amplitudes, model space and pseudoinverse.

    QR handles ordinary groups; an SVD of the small R factor detects numerical
    rank with the same relative cutoff as ``lstsq(shapes.T, data, rcond=None)``.
    Only retained left singular vectors may contribute to the projection.
    """

    basis: np.ndarray
    pinv: np.ndarray
    singular_values: np.ndarray
    covariance: np.ndarray
    rank: int

    def solve(self, data: np.ndarray) -> np.ndarray:
        """Return the minimum-norm amplitudes for this model space."""
        return self.pinv @ data


def factorize_shapes(shapes: np.ndarray) -> ShapeProjection:
    """Factor a (peak, point) shape matrix without squaring its condition number."""
    q, r = np.linalg.qr(shapes.T, mode="reduced")
    u, singular, vt = np.linalg.svd(r, full_matrices=False)
    cutoff = np.finfo(float).eps * max(shapes.shape) * singular[0]
    rank = int(np.count_nonzero(singular > cutoff))
    full_rank = rank == shapes.shape[0]
    # Keep the existing triangular path when it is numerically safe. The SVD
    # path also avoids unstable triangular solves before actual rank loss.
    if full_rank and singular[-1] > np.sqrt(np.finfo(float).eps) * singular[0]:
        r_inv = solve_triangular(r, np.eye(r.shape[0]), check_finite=False)
        pinv = r_inv @ q.T
        basis = q
    else:
        basis = q @ u[:, :rank]
        pinv = (vt[:rank].T / singular[:rank]) @ basis.T
    # A pseudoinverse covariance would assign false precision to null directions.
    # Conservatively withhold the whole joint covariance when it is unreliable.
    if full_rank and singular[-1] > np.sqrt(np.finfo(float).eps) * singular[0]:
        inverse = vt.T / singular
        with np.errstate(over="ignore", invalid="ignore"):
            covariance = inverse @ inverse.T
    else:
        covariance = np.full((shapes.shape[0], shapes.shape[0]), np.nan)
    return ShapeProjection(basis, pinv, singular, covariance, rank)


def calculate_amplitudes(shapes: FloatArray, data: FloatArray) -> FloatArray:
    """Solve the actual model, including duplicated or nearly dependent shapes."""
    if not np.all(np.isfinite(shapes)) or not np.all(np.isfinite(data)):
        return _nan_amplitude_solution(shapes, data)[0]
    with np.errstate(over="ignore", invalid="ignore"):
        return factorize_shapes(shapes).solve(data)


def calculate_amplitude_covariance(shapes: FloatArray, noise: float) -> FloatArray:
    """Return conditional amplitude covariance, or NaNs if not identifiable."""
    with np.errstate(over="ignore", invalid="ignore"):
        return factorize_shapes(shapes).covariance * np.square(noise)


def calculate_amplitudes_with_uncertainty(
    shapes: FloatArray, data: FloatArray, noise: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Solve amplitudes with noise-based, conditional standard errors.

    These errors condition on the nonlinear parameters. They do not include
    uncertainty in fitted positions or linewidths.
    """
    if not np.all(np.isfinite(shapes)) or not np.all(np.isfinite(data)):
        return _nan_amplitude_solution(shapes, data)
    with np.errstate(over="ignore", invalid="ignore"):
        projection = factorize_shapes(shapes)
        amplitudes = projection.solve(data)
        if not np.all(np.isfinite(amplitudes)):
            return _nan_amplitude_solution(shapes, data)
        covariance = projection.covariance * np.square(noise)
        errors = np.sqrt(np.diag(covariance))
    return amplitudes, errors, covariance


def _nan_amplitude_solution(
    shapes: np.ndarray,
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return correctly shaped non-finite amplitude results."""
    n_peaks = shapes.shape[0]
    extra_dims = data.shape[1:] if data.ndim > 1 else ()
    return (
        np.full((n_peaks, *extra_dims), np.nan),
        np.full((n_peaks,), np.nan),
        np.full((n_peaks, n_peaks), np.nan),
    )
