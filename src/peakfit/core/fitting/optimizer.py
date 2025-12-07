"""Least-squares optimization for NMR peak fitting.

This module provides fitting functions that directly interface with
scipy.optimize.least_squares for efficient parameter optimization.

The VarProOptimizer implements Variable Projection (VarPro), which
analytically solves for linear parameters (amplitudes) while optimizing
nonlinear parameters (positions, linewidths). This reduces the problem
dimensionality and improves convergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from peakfit.core.fitting.linear_algebra import LinearAlgebraHelper
from peakfit.core.fitting.parameters import Parameters
from peakfit.core.fitting.results import FitResult

if TYPE_CHECKING:
    from peakfit.core.domain.cluster import Cluster


class ScipyOptimizerError(Exception):
    """Exception raised for errors in scipy optimization."""


@dataclass
class VarProOptimizer:
    """Variable Projection Optimizer with efficient caching.

    This class manages state for Variable Projection optimization,
    caching intermediate results to avoid duplicate calculations between
    residuals and Jacobian computations.

    The key optimization is computing shapes and derivatives together
    via evaluate_derivatives(), which avoids redundant lineshape evaluations.
    Results are cached and reused when parameters haven't changed.
    """

    cluster: Cluster
    names: list[str]
    params_template: Parameters
    noise: float

    # Cache fields - use field(default=None) for mutable defaults
    _cache_hash: int | None = field(default=None, init=False, repr=False)
    _shapes: np.ndarray | None = field(default=None, init=False, repr=False)
    _derivs_list: list[dict[str, np.ndarray]] | None = field(default=None, init=False, repr=False)
    _q: np.ndarray | None = field(default=None, init=False, repr=False)
    _r: np.ndarray | None = field(default=None, init=False, repr=False)
    _amplitudes: np.ndarray | None = field(default=None, init=False, repr=False)
    _residuals: np.ndarray | None = field(default=None, init=False, repr=False)
    _phi_pinv: np.ndarray | None = field(default=None, init=False, repr=False)

    # Pre-computed constants
    _param_map: dict[str, int] | None = field(default=None, init=False, repr=False)
    _data_matrix: np.ndarray | None = field(default=None, init=False, repr=False)
    _data_is_1d: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Pre-compute constants that don't change during optimization."""
        # Parameter index mapping
        self._param_map = {name: i for i, name in enumerate(self.names)}

        # Pre-process data matrix (doesn't change during optimization)
        data = self.cluster.corrected_data
        self._data_is_1d = data.ndim == 1
        self._data_matrix = data[:, np.newaxis] if self._data_is_1d else data

    def _compute_cache_hash(self, x: np.ndarray) -> int:
        """Compute hash for parameter array to detect changes."""
        # Use tobytes() for fast comparison - faster than np.array_equal
        return hash(x.tobytes())

    def _update_state(self, x: np.ndarray) -> None:
        """Update cached state if parameters have changed.

        This method:
        1. Evaluates shapes AND derivatives in a single pass per peak
        2. Performs QR decomposition for amplitude solution
        3. Caches all intermediate results for Jacobian computation
        """
        cache_hash = self._compute_cache_hash(x)
        if self._cache_hash == cache_hash:
            return

        # Update parameter values
        params = self.params_template
        for i, name in enumerate(self.names):
            params[name].value = x[i]

        # Evaluate shapes AND derivatives together (single pass per peak)
        positions = self.cluster.positions
        shapes_list = []
        derivs_list = []

        for peak in self.cluster.peaks:
            shape_val, derivs = peak.evaluate_derivatives(positions, params)
            shapes_list.append(shape_val)
            derivs_list.append(derivs)

        # Stack shapes: (n_peaks, n_points)
        shapes = np.vstack(shapes_list)

        # 1. QR Decomposition
        q, r = LinearAlgebraHelper.qr_decomposition(shapes)

        # 2. Solve for amplitudes
        amplitudes = LinearAlgebraHelper.solve_amplitudes(q, r, self._data_matrix)

        # 3. Compute residuals
        # Using helper method which is more stable: data - Q @ (Q.T @ data)
        residuals = LinearAlgebraHelper.project_residuals(self._data_matrix, q, amplitudes)

        # 4. Compute pseudo-inverse helper for Jacobian
        phi_pinv = LinearAlgebraHelper.compute_phi_pinv(q, r)

        # Cache all results
        self._cache_hash = cache_hash
        self._shapes = shapes
        self._derivs_list = derivs_list
        self._q = q
        self._r = r
        self._amplitudes = amplitudes
        self._residuals = residuals
        self._phi_pinv = phi_pinv

    def compute_residuals(self, x: np.ndarray) -> np.ndarray:
        """Compute residuals for optimization."""
        self._update_state(x)
        assert self._residuals is not None
        # Flatten and normalize by noise
        return self._residuals.ravel() / self.noise

    def compute_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Compute Jacobian for optimization.

        Uses cached shapes, derivatives, and QR factors from compute_residuals.
        The Jacobian for Variable Projection includes correction terms
        accounting for the implicit dependence of amplitudes on parameters.
        """
        self._update_state(x)

        # Retrieve cached values (all guaranteed non-None after _update_state)
        assert self._q is not None
        assert self._shapes is not None
        assert self._residuals is not None
        assert self._amplitudes is not None
        assert self._phi_pinv is not None
        assert self._derivs_list is not None
        assert self._param_map is not None

        n_points = self._shapes.shape[1]
        n_params = len(self.names)
        n_planes = self._amplitudes.shape[1]
        n_peaks = len(self._derivs_list)

        # Build per-peak derivative matrix: (n_peaks, n_params, n_points)
        deriv_by_peak = np.zeros((n_peaks, n_params, n_points))
        for peak_idx, peak_derivs in enumerate(self._derivs_list):
            for name, d_val in peak_derivs.items():
                param_idx = self._param_map.get(name)
                if param_idx is not None:
                    deriv_by_peak[peak_idx, param_idx, :] = d_val

        # V term: V[pt, plane, param] = sum_peak(deriv[peak, param, pt] * amp[peak, plane])
        # Using einsum: deriv(k,p,x) @ amp(k,y) -> result(x,y,p)
        v_tensor = np.einsum("kpx,ky->xyp", deriv_by_peak, self._amplitudes)

        # Correction term for VarPro:
        # w[peak, param, plane] = deriv[peak, param, :] @ residuals[:, plane]
        w = np.einsum("kpx,xy->kpy", deriv_by_peak, self._residuals)
        # correction[pt, plane, param] = sum_peak(phi_pinv[peak, pt] * w[peak, param, plane])
        correction = np.einsum("kx,kpy->xyp", self._phi_pinv, w)

        # Project V onto orthogonal complement of column space of shapes
        # P_perp = I - Q @ Q.T
        v_flat = v_tensor.reshape(n_points, -1)
        projection = self._q @ (self._q.T @ v_flat)
        p_perp_v = (v_flat - projection).reshape(n_points, n_planes, n_params)

        # Combine: J = -(P_perp @ V + Correction)
        j_tensor = -(p_perp_v + correction)

        # Reshape and normalize
        return j_tensor.reshape(-1, n_params) / self.noise


def fit_cluster(
    params: Parameters,
    cluster: Cluster,
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


def _init_cluster_params(cluster: Cluster, params_all: Parameters, fixed: bool) -> bool:
    """Initialize parameters for a cluster, adding them to params_all.

    Returns True if successful, False on error.
    """
    from peakfit.core.domain.peaks import create_params

    try:
        cluster_params = create_params(cluster.peaks, fixed=fixed)
    except (ValueError, TypeError):
        return False

    for name, param in cluster_params.items():
        params_all.add(name, value=param.value, vary=param.vary, min=param.min, max=param.max)
    return True


def _sync_and_fit_cluster(
    cluster: Cluster, params_all: Parameters, noise: float, fixed: bool
) -> FitResult | None:
    """Synchronize parameters and fit a single cluster.

    Returns FitResult on success, None on error.
    """
    from peakfit.core.domain.peaks import create_params

    cluster_params = create_params(cluster.peaks, fixed=fixed)

    # Synchronize current global parameters to cluster
    for name in cluster_params:
        if name in params_all:
            cluster_params[name].value = params_all[name].value

    return fit_cluster(cluster_params, cluster, noise)


def _update_params_from_result(params_all: Parameters, result: FitResult) -> None:
    """Update global parameters from a fit result."""
    for name, param in result.params.items():
        target = params_all[name] if name in params_all else params_all.add(name, value=param.value)

        if target is not None:
            target.value = param.value
            target.stderr = param.stderr
            target.vary = param.vary
            target.min = param.min
            target.max = param.max
            target.computed = param.computed


def fit_clusters(
    clusters: list[Cluster],
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
        Combined fitted parameters
    """
    from peakfit.core.fitting.computation import update_cluster_corrections

    params_all = Parameters()

    # Initialize parameters from all clusters
    for cluster in clusters:
        _init_cluster_params(cluster, params_all, fixed)

    # Iterate: fit all clusters, optionally refine corrections
    for iteration in range(refine_iterations + 1):
        if iteration > 0:
            update_cluster_corrections(params_all, clusters)

        for cluster in clusters:
            try:
                result = _sync_and_fit_cluster(cluster, params_all, noise, fixed)
                if result is not None:
                    _update_params_from_result(params_all, result)
            except ScipyOptimizerError:
                if verbose:
                    print("Skipping cluster with error")

    return params_all
