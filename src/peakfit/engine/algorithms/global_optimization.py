"""Global optimization methods for NMR peak fitting."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize

from peakfit.engine.algorithms.common import residuals
from peakfit.engine.algorithms.varpro import VarProOptimizer
from peakfit.engine.domain.param_id import ParameterId
from peakfit.engine.domain.params_vector import FitParameters
from peakfit.engine.results import (
    compute_chi_squared,
    compute_reduced_chi_squared,
)
from peakfit.shared.constants import (
    BASIN_HOPPING_LOCAL_MAXITER,
    BASIN_HOPPING_NITER,
    BASIN_HOPPING_STEPSIZE,
    BASIN_HOPPING_TEMPERATURE,
    DIFF_EVOLUTION_INIT,
    DIFF_EVOLUTION_MAXITER,
    DIFF_EVOLUTION_MUTATION,
    DIFF_EVOLUTION_RECOMBINATION,
    DIFF_EVOLUTION_STRATEGY,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.shared.typing import FloatArray


@dataclass
class GlobalFitResult:
    """Extended fit result with global optimization info."""

    params: Parameters
    residual: FloatArray
    cost: float
    nfev: int
    success: bool
    message: str
    global_iterations: int = 0
    local_minimizations: int = 0
    global_minimum_found: bool = False
    basin_hopping_temperature: float = 1.0
    covar: FloatArray | None = None
    n_amplitude_params: int = 0  # Number of analytically computed amplitude parameters

    @property
    def chisqr(self) -> float:
        """Chi-squared value."""
        return compute_chi_squared(self.residual)

    @property
    def redchi(self) -> float:
        """Reduced chi-squared."""
        ndata = len(self.residual)
        nvarys = len(self.params.get_vary_names())
        n_total_fitted = nvarys + self.n_amplitude_params
        return compute_reduced_chi_squared(self.chisqr, ndata, n_total_fitted)


def residuals_global(
    x: FloatArray,
    params: Parameters,
    cluster: Cluster,
    noise: float,
) -> float:
    """Compute sum of squared residuals for global optimization."""
    params.set_vary_values(x)
    res_vec = residuals(params, cluster, noise)
    return float(np.sum(res_vec**2))


def _compute_numerical_hessian(
    func: Callable[[FloatArray], float],
    x: FloatArray,
    bounds: list[tuple[float, float]],
    epsilon: float = 1e-8,
) -> FloatArray:
    """Compute numerical Hessian matrix using central finite differences."""
    n = len(x)
    hessian = np.zeros((n, n))
    f0 = func(x)

    for i in range(n):
        # Adaptive step size
        hi = epsilon * max(1.0, abs(x[i]))
        if x[i] + hi > bounds[i][1]:
            hi = -hi
        if x[i] + hi < bounds[i][0]:
            hi = epsilon

        # Compute diagonal term
        xi_plus = x.copy()
        xi_plus[i] += hi
        xi_minus = x.copy()
        xi_minus[i] -= hi
        fi_plus = func(xi_plus)
        fi_minus = func(xi_minus)
        hessian[i, i] = (fi_plus - 2 * f0 + fi_minus) / hi**2

        # Compute off-diagonal terms
        for j in range(i + 1, n):
            hj = epsilon * max(1.0, abs(x[j]))
            if x[j] + hj > bounds[j][1]:
                hj = -hj
            if x[j] + hj < bounds[j][0]:
                hj = epsilon

            # 4-point stencil for mixed partial
            # f(x+h, y+k) - f(x+h, y-k) - f(x-h, y+k) + f(x-h, y-k) / 4hk
            xij_pp = x.copy()
            xij_pp[i] += hi
            xij_pp[j] += hj
            xij_pm = x.copy()
            xij_pm[i] += hi
            xij_pm[j] -= hj
            xij_mp = x.copy()
            xij_mp[i] -= hi
            xij_mp[j] += hj
            xij_mm = x.copy()
            xij_mm[i] -= hi
            xij_mm[j] -= hj

            term = func(xij_pp) - func(xij_pm) - func(xij_mp) + func(xij_mm)
            hessian[i, j] = term / (4 * hi * hj)
            hessian[j, i] = hessian[i, j]

    return hessian


def _compute_covariance_and_errors(
    objective: Callable[[FloatArray], float],
    x: FloatArray,
    bounds: list[tuple[float, float]],
    params: Parameters,
) -> FloatArray | None:
    """Compute covariance matrix and set parameter errors."""
    try:
        hessian = _compute_numerical_hessian(objective, x, bounds)
        # Cov = 2 * Hessian^-1 (approximation for sum-squared prob)
        covar = np.asarray(np.linalg.inv(hessian) * 2.0, dtype=np.float64)

        with np.errstate(invalid="ignore"):
            variances = np.diag(covar)
            # Ensure positive variance
            variances[variances < 0] = np.nan
            std_errors = np.sqrt(variances)

        params.set_errors(std_errors)
        return covar
    except (np.linalg.LinAlgError, ValueError):
        return None


def fit_basin_hopping(
    params: Parameters,
    cluster: Cluster,
    noise: float,
    n_iterations: int = BASIN_HOPPING_NITER,
    temperature: float = BASIN_HOPPING_TEMPERATURE,
    step_size: float = BASIN_HOPPING_STEPSIZE,
    seed: int | None = None,
) -> GlobalFitResult:
    """Fit cluster using basin-hopping global optimization."""
    amplitude_params_state: dict[str, bool] = {}

    # Identify amplitude parameters and temporarily set vary=False (VarPro principle)
    for param in params.values():
        is_amplitude = (
            param.param_id
            and param.param_id.label.startswith("I")
            and param.param_id.label[1:].isdigit()
        )
        if is_amplitude and param.vary:
            amplitude_params_state[param.name] = True
            param.vary = False

    try:
        vary_names = params.get_vary_names()
        fit_params = FitParameters.from_parameters(params, cluster.peaks)
        varpro_optimizer = VarProOptimizer(
            cluster=cluster,
            names=vary_names,
            params_template=params,
            fit_params=fit_params,
            noise=noise,
        )

        n_amplitude_params = cluster.n_amplitude_params
        x0 = params.get_vary_values()
        bounds = params.get_vary_bounds_list()

        def objective(x: FloatArray) -> float:
            r = varpro_optimizer.compute_residuals(x)
            return float(np.sum(r**2))

        def jacobian(x: FloatArray) -> FloatArray:
            r = varpro_optimizer.compute_residuals(x)
            jac = varpro_optimizer.compute_jacobian(x)
            return 2.0 * r @ jac

        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds,
            "options": {"maxiter": BASIN_HOPPING_LOCAL_MAXITER},
            "jac": jacobian,
        }

        result = optimize.basinhopping(
            objective,
            x0,
            niter=n_iterations,
            T=temperature,
            minimizer_kwargs=minimizer_kwargs,
            disp=False,
            seed=seed,
        )

        # Finalize
        lower, upper = params.get_vary_bounds()
        result.x = np.clip(result.x, lower, upper)
        params.set_vary_values(result.x)

        # Restore amplitudes
        _restore_amplitude_params(
            params, cluster, varpro_optimizer, amplitude_params_state, result.x
        )

        final_residuals = residuals(params, cluster, noise)
        covar = _compute_covariance_and_errors(objective, result.x, bounds, params)

        # Simplified success check
        success = result.lowest_optimization_result.success
        if not success:
            msg = str(result.message)
            if "completed successfully" in msg or "requested number" in msg:
                success = True

        return GlobalFitResult(
            params=params,
            residual=final_residuals,
            cost=result.fun,
            nfev=result.nfev,
            success=success,
            message=str(result.message),
            global_iterations=n_iterations,
            local_minimizations=result.nit,
            global_minimum_found=result.lowest_optimization_result.success,
            basin_hopping_temperature=temperature,
            covar=covar,
            n_amplitude_params=n_amplitude_params,
        )

    finally:
        # Ensure vary flags are restored even on error
        for name in amplitude_params_state:
            if name in params:
                params[name].vary = True


def _restore_amplitude_params(
    params: Parameters,
    cluster: Cluster,
    varpro_optimizer: VarProOptimizer,
    amplitude_params_state: dict[str, bool],
    optimized_x: FloatArray,
) -> None:
    """Restore amplitude parameters from VarPro optimization."""
    varpro_optimizer.compute_residuals(optimized_x)
    final_amplitudes = varpro_optimizer.get_optimized_amplitudes()

    final_amplitudes = varpro_optimizer.get_optimized_amplitudes()
    for i, peak in enumerate(cluster.peaks):
        axis = peak.shapes[0].axis if peak.shapes else "pseudo"
        for plane_idx in range(final_amplitudes.shape[1]):
            val = final_amplitudes[i, plane_idx]

            if final_amplitudes.shape[1] == 1:
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
                    index=plane_idx,
                )

            if pid.name in params:
                params[pid.name].value = val
                params[pid.name].computed = True


def fit_differential_evolution(
    params: Parameters,
    cluster: Cluster,
    noise: float,
    max_iterations: int = DIFF_EVOLUTION_MAXITER,
    mutation: tuple[float, float] = DIFF_EVOLUTION_MUTATION,
    recombination: float = DIFF_EVOLUTION_RECOMBINATION,
    strategy: str = DIFF_EVOLUTION_STRATEGY,
    init: str = DIFF_EVOLUTION_INIT,
    polish: bool = True,
    seed: int | None = None,
) -> GlobalFitResult:
    """Fit cluster using differential evolution."""
    bounds = params.get_vary_bounds_list()
    # Note: DE optimizes all 'vary' parameters directly without VarPro in this
    # implementation.

    def objective(x: FloatArray) -> float:
        return residuals_global(x, params, cluster, noise)

    result = optimize.differential_evolution(
        objective,
        bounds,
        maxiter=max_iterations,
        mutation=mutation,
        recombination=recombination,
        strategy=strategy,
        init=init,
        polish=False,
        disp=False,
        workers=1,
        seed=seed,
    )

    if polish:
        vary_names = params.get_vary_names()
        fit_params = FitParameters.from_parameters(params, cluster.peaks)
        varpro_optimizer = VarProOptimizer(
            cluster=cluster,
            names=vary_names,
            params_template=params,
            fit_params=fit_params,
            noise=noise,
        )

        def polish_objective(x: FloatArray) -> float:
            r = varpro_optimizer.compute_residuals(x)
            return float(np.sum(r**2))

        def polish_jacobian(x: FloatArray) -> FloatArray:
            r = varpro_optimizer.compute_residuals(x)
            jac = varpro_optimizer.compute_jacobian(x)
            return 2.0 * r @ jac

        polish_result = optimize.minimize(
            polish_objective,
            result.x,
            method="L-BFGS-B",
            bounds=bounds,
            jac=polish_jacobian,
        )

        if polish_result.success:
            result.x = polish_result.x
            result.fun = float(polish_result.fun)
            result.nfev += polish_result.nfev

    lower, upper = params.get_vary_bounds()
    result.x = np.clip(result.x, lower, upper)
    params.set_vary_values(result.x)

    final_residuals = residuals(params, cluster, noise)
    covar = _compute_covariance_and_errors(objective, result.x, bounds, params)

    success = result.success
    if not success and "Maximum number of iterations" in str(result.message):
        success = True

    return GlobalFitResult(
        params=params,
        residual=final_residuals,
        cost=result.fun,
        nfev=result.nfev,
        success=success,
        message=str(result.message),
        global_iterations=result.nit,
        local_minimizations=1 if polish else 0,
        global_minimum_found=result.success,
        covar=covar,
        n_amplitude_params=cluster.n_amplitude_params,
    )
