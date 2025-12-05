"""Jacobian computation for NMR peak fitting using VarPro.

This module provides the specialized Jacobian calculation for the least-squares
optimization using the Kaufman approximation.
"""

from typing import TYPE_CHECKING

import numpy as np

from peakfit.core.fitting.computation import calculate_amplitudes

if TYPE_CHECKING:
    from peakfit.core.domain.cluster import Cluster
    from peakfit.core.fitting.parameters import Parameters


def compute_jacobian(
    x: np.ndarray,
    names: list[str],
    params_template: "Parameters",
    cluster: "Cluster",
    noise: float,
) -> np.ndarray:
    """Compute Jacobian for scipy.optimize.least_squares using VarPro.

    Approximates as: J ~ - (I - P_S) (dS/dtheta) c - (Phi^dagger)^T (dS/dtheta)^T r

    Args:
        x: Current parameter values
        names: Parameter names
        params_template: Template Parameters
        cluster: Cluster being fit
        noise: Noise level

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (n_residuals, n_params)
    """
    # Update parameters
    for i, name in enumerate(names):
        params_template[name].value = x[i]

    # 1. Evaluate shapes and derivatives
    shapes_list = []
    derivs_list = []
    positions = cluster.positions

    for peak in cluster.peaks:
        val, derivs = peak.evaluate_derivatives(positions, params_template)
        shapes_list.append(val)
        derivs_list.append(derivs)

    shapes = np.array(shapes_list)  # (n_peaks, n_points)
    n_points = shapes.shape[1]

    # 2. Compute amplitudes and residuals
    data = cluster.corrected_data
    amplitudes = calculate_amplitudes(shapes, data)

    # Handle dimensions (broadcasting prep)
    if amplitudes.ndim == 1:
        amplitudes = amplitudes[:, np.newaxis]
    if data.ndim == 1:
        data = data[:, np.newaxis]

    n_planes = amplitudes.shape[1]
    residuals = data - shapes.T @ amplitudes  # (n_points, n_planes)

    # 3. Compute Projection matrices
    # phi_pinv = (S^T S)^-1 S^T
    sts = shapes @ shapes.T
    try:
        sts_inv = np.linalg.inv(sts)
    except np.linalg.LinAlgError:
        sts_inv = np.linalg.pinv(sts)

    phi_pinv = sts_inv @ shapes  # (n_peaks, n_points)

    # 4. Accumulate V (unprojected Jacobian) and Correction terms
    n_params = len(names)
    param_map = {name: i for i, name in enumerate(names)}

    # Initialize tensors
    v_tensor = np.zeros((n_points, n_planes, n_params))
    correction = np.zeros((n_points, n_planes, n_params))

    for i, peak_derivs in enumerate(derivs_list):
        amp = amplitudes[i]  # (n_planes,)
        phi_row = phi_pinv[i]  # (n_points,)

        for name, d_val in peak_derivs.items():
            if name in param_map:
                idx = param_map[name]

                # V term: dS/dtheta * c
                v_tensor[:, :, idx] += d_val[:, np.newaxis] * amp[np.newaxis, :]

                # Correction term: (Phi^dagger)^T * (dS/dtheta)^T * r
                # w = d_val @ residuals -> scalar or (n_planes,)
                w = d_val @ residuals
                correction[:, :, idx] += phi_row[:, np.newaxis] * w[np.newaxis, :]

    # 5. Project V onto orthogonal complement of S: P_perp V = V - S (phi_pinv V)
    # Flatten planes/params for matrix multiplication
    v_flat = v_tensor.reshape(n_points, -1)

    # Project: S (S^T S)^-1 S^T V
    projection = shapes.T @ (phi_pinv @ v_flat)
    p_perp_v = (v_flat - projection).reshape(n_points, n_planes, n_params)

    # 6. Combine and normalize
    # J = - (P_perp V + Correction)
    j_tensor = -(p_perp_v + correction)

    # Final reshape to (n_residuals, n_params) where n_residuals = n_points * n_planes
    return j_tensor.reshape(-1, n_params) / noise
