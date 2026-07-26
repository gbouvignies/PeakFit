"""Least-squares optimization using Variable Projection (VarPro)."""

import os
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import scipy.optimize

from peakfit.engine.algorithms.linear_algebra import (
    compute_phi_pinv,
    project_residuals,
    qr_decomposition,
    solve_amplitudes,
)
from peakfit.engine.domain.param_id import ParameterId
from peakfit.engine.domain.param_map import ParameterMap
from peakfit.engine.domain.params_vector import FitParameters
from peakfit.engine.results import FitResult

if TYPE_CHECKING:
    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.shared.typing import FloatArray


class ScipyOptimizerError(Exception):
    """Exception raised for errors in scipy optimization."""


def _sync_alias_values(params: Parameters, shared_aliases: dict[str, str] | None) -> None:
    """Copy source parameter values into aliased target parameters."""
    if not shared_aliases:
        return

    for target_name, source_name in shared_aliases.items():
        if target_name in params and source_name in params:
            params[target_name].value = params[source_name].value


@dataclass
class VarProOptimizer:
    """Variable Projection Optimizer state manager.

    Separates linear and nonlinear parameters, managing caching of shape matrices
    and factorizations to efficiently compute residuals and Jacobians.
    """

    cluster: Cluster
    names: list[str]
    params_template: Parameters
    fit_params: FitParameters
    noise: float
    shared_aliases: dict[str, str] | None = None

    # Cache fields
    _cache_hash: int | None = field(default=None, init=False, repr=False)
    _shapes: np.ndarray | None = field(default=None, init=False, repr=False)
    _q: np.ndarray | None = field(default=None, init=False, repr=False)
    _r: np.ndarray | None = field(default=None, init=False, repr=False)
    _amplitudes: np.ndarray | None = field(default=None, init=False, repr=False)
    _residuals: np.ndarray | None = field(default=None, init=False, repr=False)
    _phi_pinv: np.ndarray | None = field(default=None, init=False, repr=False)

    # Pre-computed constants
    _param_map: ParameterMap | None = field(default=None, init=False, repr=False)
    _param_peak_indices: np.ndarray | None = field(default=None, init=False, repr=False)
    _data_matrix: np.ndarray | None = field(default=None, init=False, repr=False)
    _d_matrix: np.ndarray | None = field(default=None, init=False, repr=False)
    _grid_dims: list[np.ndarray] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Pre-compute constants that don't change during optimization."""
        # Parameter index mapping
        self._param_map = self.fit_params.map or ParameterMap.from_names(self.names)
        self._param_peak_indices = np.zeros(len(self.names), dtype=int)

        # Map each parameter index to its peak index
        for peak_idx, peak in enumerate(self.cluster.peaks):
            peak_params = peak.create_params()
            for name in peak_params:
                param_idx = self._param_map.index_of(name)
                if param_idx is not None:
                    self._param_peak_indices[param_idx] = peak_idx

        # Pre-process data matrix
        # SciPy's least_squares expects real-valued residuals/Jacobians.
        # PeakFit's objective is defined on the real part of the corrected data
        # (consistent with core.algorithms.common.residuals).
        data = np.asarray(self.cluster.corrected_data.real, dtype=np.float64)
        self._data_matrix = data

        # Grid dimensions and d_matrix pre-allocation
        grid_indices = self.cluster.grid_indices
        if isinstance(grid_indices, list):
            self._grid_dims = grid_indices
        else:
            self._grid_dims = [grid_indices]
        n_points = self._grid_dims[0].size
        n_dims = len(self._grid_dims)

        self._d_matrix = np.zeros((len(self.names), n_points))

        # Validate peak shapes
        for p in self.cluster.peaks:
            if len(p.shapes) != n_dims:
                raise ScipyOptimizerError(
                    f"Peak {p.name} shape mismatch: expected {n_dims} dimensions, "
                    f"got {len(p.shapes)}"
                )

    def _compute_cache_hash(self, x: FloatArray) -> int:
        """Compute hash for parameter array to detect changes."""
        return hash(x.tobytes())

    def _update_state(self, x: FloatArray) -> None:
        """Update cached state if parameters have changed."""
        cache_hash = self._compute_cache_hash(x)
        if self._cache_hash == cache_hash:
            return

        # Update parameter values in template
        for i, name in enumerate(self.names):
            self.params_template[name].value = x[i]

        # Keep aliased parameters synchronized with their source parameter.
        _sync_alias_values(self.params_template, self.shared_aliases)

        assert self._d_matrix is not None
        assert self._grid_dims is not None
        assert self._data_matrix is not None

        # Reset derivative matrix
        self._d_matrix.fill(0.0)

        shapes = self.cluster.evaluate(
            self.params_template,
            self._grid_dims,
            d_matrix=self._d_matrix,
            param_map=self._param_map,
        )

        # 1. QR Decomposition
        q, r = qr_decomposition(shapes)

        # 2. Solve for amplitudes
        amplitudes = solve_amplitudes(q, r, self._data_matrix)

        # 3. Compute residuals
        residuals = project_residuals(self._data_matrix, q)

        # 4. Compute pseudo-inverse helper for Jacobian
        phi_pinv = compute_phi_pinv(q, r)

        # Cache results
        self._cache_hash = cache_hash
        self._shapes = shapes
        self._q = q
        self._r = r
        self._amplitudes = amplitudes
        self._residuals = residuals
        self._phi_pinv = phi_pinv

    def compute_residuals(self, x: FloatArray) -> FloatArray:
        """Compute residuals for optimization."""
        self._update_state(x)
        assert self._residuals is not None
        return cast("FloatArray", self._residuals.ravel() / self.noise)

    def get_optimized_amplitudes(self) -> FloatArray:
        """Recover the linear amplitude parameters for the current state."""
        if self._amplitudes is None:
            raise RuntimeError("Optimizer state not initialized. Run compute_residuals first.")
        return self._amplitudes

    def compute_jacobian(self, x: FloatArray) -> FloatArray:
        r"""Compute the Jacobian of the Variable Projection functional."""
        self._update_state(x)

        assert self._q is not None
        assert self._residuals is not None
        assert self._amplitudes is not None
        assert self._phi_pinv is not None
        assert self._d_matrix is not None
        assert self._param_peak_indices is not None

        n_params = len(self.names)
        n_points_grid = self._d_matrix.shape[1]

        # 1. Expand Amplitudes
        amps_expanded = self._amplitudes[self._param_peak_indices, :]

        # 2. V term: D (params, pts) * Amps (params, planes) -> (pts, planes, params)
        v_tensor = self._d_matrix.T[:, np.newaxis, :] * amps_expanded.T[np.newaxis, :, :]

        # 3. Correction term: W = D @ residuals
        w = self._d_matrix @ self._residuals

        pinv_expanded = self._phi_pinv[self._param_peak_indices, :]
        correction = (pinv_expanded.T)[:, np.newaxis, :] * (w.T)[np.newaxis, :, :]

        # Project V onto orthogonal complement
        v_flat = v_tensor.reshape(n_points_grid, -1)

        # P_perp = I - Q Q^T
        # v_perp = v_flat - Q (Q^T v_flat)
        projection = self._q @ (self._q.T @ v_flat)
        v_perp = v_flat - projection

        # Jacobian J is negative of (P_perp(V) + Correction)
        c_flat = correction.reshape(n_points_grid, -1)
        j_flat = -(v_perp + c_flat)

        j_final = j_flat.reshape(-1, n_params)

        return cast("FloatArray", j_final / self.noise)


def _calculate_errors(
    optimizer: VarProOptimizer, x: FloatArray, noise: float
) -> tuple[FloatArray, FloatArray]:
    """Calculate parameter standard errors."""
    # 1. Nonlinear Parameter Errors
    jacobian = optimizer.compute_jacobian(x)
    try:
        # Covariance equals inverse of (J.T @ J)
        cov = np.linalg.inv(jacobian.T @ jacobian)
        with np.errstate(invalid="ignore"):
            stderrs = np.sqrt(np.diag(cov))
    except (np.linalg.LinAlgError, ValueError):
        stderrs = np.zeros_like(x)

    # 2. Linear (Amplitude) Parameter Errors
    final_amplitudes = optimizer.get_optimized_amplitudes()
    n_amps = final_amplitudes.shape[0]

    try:
        if optimizer._r is not None:
            # Var(a) = (R^T R)^-1 * sigma_noise^2
            amp_cov = np.linalg.inv(optimizer._r.T @ optimizer._r) * (noise**2)

            # Ensure safe sqrt
            diag = np.diag(amp_cov)
            diag[diag < 0] = 0.0
            amp_stderrs = np.sqrt(diag)
        else:
            amp_stderrs = np.zeros(n_amps)
    except (np.linalg.LinAlgError, ValueError):
        amp_stderrs = np.zeros(n_amps)

    return stderrs, amp_stderrs


def _update_amplitude_params(
    cluster: Cluster,
    params: Parameters,
    amplitudes: FloatArray,
    amp_stderrs: FloatArray,
) -> None:
    """Update amplitude parameters in the params object."""
    n_series = amplitudes.shape[1]

    for i, peak in enumerate(cluster.peaks):
        axis = peak.shapes[0].axis if peak.shapes else "pseudo"
        peak_err = amp_stderrs[i] if i < len(amp_stderrs) else 0.0

        for series_idx in range(n_series):
            val = amplitudes[i, series_idx]

            if n_series == 1:
                pid = ParameterId(
                    peak_name=peak.name,
                    axis=axis,
                    label="I",
                    index=0,
                )
            else:
                pid = ParameterId(
                    peak_name=peak.name,
                    axis=axis,
                    label="I",
                    index=series_idx,
                )

            if pid.name in params:
                p = params[pid.name]
                p.value = val
                p.stderr = peak_err
                p.computed = True
                p.vary = False
            else:
                params.add(pid, value=val, computed=True, vary=False, stderr=peak_err)


def _apply_shared_aliases_to_params(
    params: Parameters,
    shared_aliases: dict[str, str] | None,
) -> None:
    """Freeze target aliases and synchronize their values from source names."""
    if not shared_aliases:
        return

    _sync_alias_values(params, shared_aliases)
    for target_name in shared_aliases:
        if target_name in params:
            params[target_name].vary = False


def _inject_shared_aliases_into_map(
    fit_params: FitParameters,
    params: Parameters,
    shared_aliases: dict[str, str] | None,
) -> None:
    """Map aliased names to the same fitted index as their source parameter."""
    if not shared_aliases or fit_params.map is None:
        return

    name_to_index = dict(fit_params.map.name_to_index)
    for target_name, source_name in shared_aliases.items():
        source_index = name_to_index.get(source_name)
        if source_index is not None and target_name in params:
            name_to_index[target_name] = source_index
    fit_params.map = ParameterMap(name_to_index=name_to_index)


def fit_cluster(
    params: Parameters,
    cluster: Cluster,
    noise: float,
    shared_aliases: dict[str, str] | None = None,
    max_nfev: int = 1000,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    verbose: int = 0,
) -> FitResult:
    """Fit a single cluster using the Trust Region Reflective algorithm and VarPro."""
    if noise <= 0:
        raise ValueError(f"Noise must be positive, got {noise}")
    if not cluster.peaks:
        raise ScipyOptimizerError("Cluster has no peaks to fit")

    _apply_shared_aliases_to_params(params, shared_aliases)

    x0 = params.get_vary_values()
    lower, upper = params.get_vary_bounds()
    vary_names = params.get_vary_names()

    fit_params = FitParameters.from_parameters(params, cluster.peaks)
    _inject_shared_aliases_into_map(fit_params, params, shared_aliases)

    optimizer = VarProOptimizer(
        cluster=cluster,
        names=vary_names,
        params_template=params,
        fit_params=fit_params,
        noise=noise,
        shared_aliases=shared_aliases,
    )

    try:
        result = scipy.optimize.least_squares(
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
    except Exception as e:
        if os.environ.get("PEAKFIT_TRACEBACK"):
            tb = traceback.format_exc()
            raise ScipyOptimizerError(
                f"Optimization failed: {e}\n\nTraceback (most recent call last):\n{tb}"
            ) from e
        raise ScipyOptimizerError(f"Optimization failed: {e}") from e

    # Update parameters & clamp
    result.x = np.clip(result.x, lower, upper)
    params.set_vary_values(result.x)
    _apply_shared_aliases_to_params(params, shared_aliases)

    # Ensure final state is computed and amplitudes are recoverable
    optimizer.compute_residuals(result.x)
    final_amplitudes = optimizer.get_optimized_amplitudes()

    stderrs, amp_stderrs = _calculate_errors(optimizer, result.x, noise)
    params.set_errors(stderrs)

    _update_amplitude_params(cluster, params, final_amplitudes, amp_stderrs)

    return FitResult(
        params=params,
        residual=result.fun,
        cost=result.cost,
        nfev=result.nfev,
        njev=getattr(result, "njev", 0),
        success=result.success,
        message=result.message,
        optimality=getattr(result, "optimality", 0.0),
        n_amplitude_params=cluster.n_amplitude_params,
    )
