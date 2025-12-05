"""Least-squares optimization for NMR peak fitting.

This module provides fitting functions that directly interface with
scipy.optimize.least_squares for efficient parameter optimization.
"""

from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


from peakfit.core.fitting.parameters import Parameters
from peakfit.core.fitting.results import FitResult

if TYPE_CHECKING:
    from peakfit.core.domain.cluster import Cluster


class ScipyOptimizerError(Exception):
    """Exception raised for errors in scipy optimization."""


@dataclass
class VarProOptimizer:
    """Variable Projection Optimizer state manager.

    This class manages the state for Variable Projection optimization,
    caching intermediate results to avoid duplicate calculations between
    residuals and Jacobian computations.
    """

    cluster: "Cluster"
    names: list[str]
    params_template: Parameters
    noise: float
    # Cache fields
    _last_x: np.ndarray | None = None
    _shapes: np.ndarray | None = None
    _q: np.ndarray | None = None
    _r: np.ndarray | None = None
    _amplitudes: np.ndarray | None = None
    _residuals: np.ndarray | None = None

    def _update_state(self, x: np.ndarray) -> None:
        """Update cached state if parameters have changed."""
        if self._last_x is not None and np.array_equal(x, self._last_x):
            return

        # Update parameters
        for i, name in enumerate(self.names):
            self.params_template[name].value = x[i]

        # 1. Evaluate shapes
        positions = self.cluster.positions
        # We only need values here, derivatives are computed in Jacobian
        # But BaseShape handles caching, so evaluate() is efficient
        shapes_list = [
            peak.evaluate(positions, self.params_template) for peak in self.cluster.peaks
        ]

        # (n_peaks, n_points)
        shapes = np.array(shapes_list)

        # 2. QR Decomposition of Transposed Shapes (Features matrix)
        # We need to solve: shapes.T @ amplitudes = data
        # Let A = shapes.T (n_points, n_peaks)
        # A = Q R
        # Q (n_points, K), R (K, n_peaks), where K = min(n_points, n_peaks) usually n_peaks

        # Note: numpy's qr returns Q(M, K), R(K, N) for mode='reduced'
        # shapes.T is (n_points, n_peaks)
        q, r = np.linalg.qr(shapes.T, mode="reduced")

        # 3. Solve for amplitudes: R @ amplitudes = Q.T @ data
        data = self.cluster.corrected_data

        # Handle broadcasting for data (n_points,) or (n_points, n_planes)
        data_matrix = data[:, np.newaxis] if data.ndim == 1 else data

        qty = q.T @ data_matrix

        # Solve R @ amp = qty
        # R is upper triangular (n_peaks, n_peaks)
        try:
            amplitudes = np.linalg.solve(r, qty)
        except np.linalg.LinAlgError:
            # Fallback for singular R
            amplitudes = np.linalg.lstsq(r, qty, rcond=None)[0]

        if data.ndim == 1:
            amplitudes = amplitudes.flatten()

        # 4. Compute residuals: r = data - A @ amplitudes
        # Using Q: proj_y = Q @ (Q.T @ y) = Q @ qty
        # residuals = y - proj_y
        # This is strictly correct for least squares residual vector

        proj_data = q @ qty
        if data.ndim == 1:
            proj_data = proj_data.flatten()

        residuals = data - proj_data

        # Cache results
        self._last_x = x.copy()
        self._shapes = shapes
        self._q = q
        self._r = r
        self._amplitudes = amplitudes
        self._residuals = residuals

    def compute_residuals(self, x: np.ndarray) -> np.ndarray:
        """Compute residuals for optimization."""
        self._update_state(x)
        assert self._residuals is not None
        # Normalize by noise for least_squares
        return self._residuals.ravel() / self.noise

    def compute_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Compute Jacobian for optimization."""
        self._update_state(x)

        shapes = self._shapes
        q = self._q
        r = self._r
        residuals = self._residuals
        amplitudes = self._amplitudes

        # 1. Evaluate shapes and derivatives
        # Logic from previous jacobian.py
        # If shapes provided (cached), we still need derivatives.
        # BaseShape caches, so calling evaluate_derivatives should be fast if evaluate was just called.
        derivs_list = []
        positions = self.cluster.positions

        # We assume _update_state called evaluate(), so cache is hot
        for peak in self.cluster.peaks:
            # We don't need the value if we have shapes, but evaluate_derivatives returns both
            _, derivs = peak.evaluate_derivatives(positions, self.params_template)
            derivs_list.append(derivs)

        n_points = shapes.shape[1]

        # 2. Amplitudes and residuals are cached

        # Dimensions check
        amplitudes_mat = amplitudes[:, np.newaxis] if amplitudes.ndim == 1 else amplitudes
        n_planes = amplitudes_mat.shape[1]

        # Ensure residuals is 2D
        if residuals.ndim == 1:
            residuals = residuals[:, np.newaxis]

        # 3. Compute Projection matrices helpers
        # phi_pinv = (S^T)+ = R^-1 Q^T

        try:
            # r is upper triangular
            # solve r X = Q.T
            # We want R^-1 Q^T.
            # Let X = R^-1 Q^T. Then R X = Q.T.
            phi_pinv = np.linalg.solve(r, q.T)
        except np.linalg.LinAlgError:
            # Fallback
            r_inv = np.linalg.pinv(r)
            phi_pinv = r_inv @ q.T

        # 4. Accumulate V (unprojected Jacobian) and Correction terms
        n_params = len(self.names)
        param_map = {name: i for i, name in enumerate(self.names)}

        # Initialize tensors
        v_tensor = np.zeros((n_points, n_planes, n_params))
        correction = np.zeros((n_points, n_planes, n_params))

        for i, peak_derivs in enumerate(derivs_list):
            amp = amplitudes_mat[i]  # (n_planes,)
            phi_row = phi_pinv[i]  # (n_points,)

            for name, d_val in peak_derivs.items():
                if name in param_map:
                    idx = param_map[name]

                    # V term: (dS/dtheta) * c
                    v_tensor[:, :, idx] += d_val[:, np.newaxis] * amp[np.newaxis, :]

                    # Correction term: (Phi^dagger)^T * (dS/dtheta)^T * r
                    w = d_val @ residuals
                    correction[:, :, idx] += phi_row[:, np.newaxis] * w[np.newaxis, :]

        # 5. Project V onto orthogonal complement of S: P_perp V = V - P_S V
        # P_S = Q Q^T
        v_flat = v_tensor.reshape(n_points, -1)

        # Project: Q (Q^T V)
        projection = q @ (q.T @ v_flat)
        p_perp_v = (v_flat - projection).reshape(n_points, n_planes, n_params)

        # 6. Combine and normalize
        # J = - (P_perp V + Correction)
        j_tensor = -(p_perp_v + correction)

        # Final reshape to (n_residuals, n_params)
        return j_tensor.reshape(-1, n_params) / self.noise


def fit_cluster(
    params: Parameters,
    cluster: "Cluster",
    noise: float,
    max_nfev: int = 1000,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    verbose: int = 0,
) -> FitResult:
    """Fit a single cluster using scipy.optimize.least_squares.

    Args:
        params: Initial parameters
        cluster: Cluster to fit
        noise: Noise level (must be positive)
        max_nfev: Maximum number of function evaluations
        ftol: Tolerance for termination by change of cost function
        xtol: Tolerance for termination by change of variables
        gtol: Tolerance for termination by gradient norm
        verbose: Verbosity level

    Returns
    -------
    FitResult
        FitResult containing optimized parameters and fit statistics
    """
    if noise <= 0:
        raise ValueError(f"Noise must be positive, got {noise}")

    if not cluster.peaks:
        raise ScipyOptimizerError("Cluster has no peaks to fit")

    # Get initial values and bounds
    x0 = params.get_vary_values()
    lower, upper = params.get_vary_bounds()
    vary_names = params.get_vary_names()

    # Calculate number of amplitude parameters for DOF
    n_amplitude_params = cluster.n_amplitude_params

    # Initialize Optimizer
    optimizer = VarProOptimizer(
        cluster=cluster,
        names=vary_names,
        params_template=params,
        noise=noise,
    )

    # Run optimization
    result = least_squares(
        optimizer.compute_residuals,
        x0,
        jac=optimizer.compute_jacobian,
        bounds=(lower, upper),
        method="trf",
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
        verbose=verbose,
    )

    # Update parameters
    params.set_vary_values(result.x)

    return FitResult(
        params=params,
        residual=result.fun,
        cost=result.cost,
        nfev=result.nfev,
        njev=getattr(result, "njev", 0),
        success=result.success,
        message=result.message,
        optimality=getattr(result, "optimality", 0.0),
        n_amplitude_params=n_amplitude_params,
    )


def fit_clusters(
    clusters: list["Cluster"],
    noise: float,
    refine_iterations: int = 1,
    *,
    fixed: bool = False,
    verbose: bool = False,
) -> Parameters:
    """Fit all clusters using direct scipy optimization.

    Args:
        clusters: List of clusters
        noise: Noise level
        refine_iterations: Number of refinement passes
        fixed: Whether to fix positions
        verbose: Print progress

    Returns
    -------
    Parameters
        Combined fitted parameters
    """
    from peakfit.core.domain.peaks import create_params
    from peakfit.core.fitting.computation import update_cluster_corrections

    params_all = Parameters()

    # Initialize with all parameters
    for cluster in clusters:
        try:
            cluster_params = create_params(cluster.peaks, fixed=fixed)
            for name, param in cluster_params.items():
                params_all.add(
                    name, value=param.value, vary=param.vary, min=param.min, max=param.max
                )
        except (ValueError, TypeError):
            continue

    for iteration in range(refine_iterations + 1):
        if iteration > 0:
            update_cluster_corrections(params_all, clusters)

        for cluster in clusters:
            try:
                # Synchronize current global parameters to cluster
                cluster_params = create_params(cluster.peaks, fixed=fixed)
                for name in cluster_params:
                    if name in params_all:
                        cluster_params[name].value = params_all[name].value

                result = fit_cluster(cluster_params, cluster, noise)

                # Update global parameters from result
                for name, param in result.params.items():
                    target = (
                        params_all[name]
                        if name in params_all
                        else params_all.add(name, value=param.value)
                    )

                    target.value = param.value
                    target.stderr = param.stderr
                    target.vary = param.vary
                    target.min = param.min
                    target.max = param.max
                    target.computed = param.computed

            except ScipyOptimizerError:
                if verbose:
                    print("Skipping cluster with error")
                continue

    return params_all
