"""Profile likelihood estimation for NMR peak fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize

from peakfit.core.fitting.computation import residuals
from peakfit.core.algorithms.global_optimization import residuals_global
from peakfit.core.fitting.parameters import Parameters  # noqa: TC001
from peakfit.core.shared.constants import (
    PROFILE_LIKELIHOOD_DELTA_CHI2,
    PROFILE_LIKELIHOOD_NPOINTS,
)

if TYPE_CHECKING:
    from peakfit.core.domain.cluster import Cluster
    from peakfit.core.shared.typing import FloatArray


def compute_profile_likelihood(
    params: Parameters,
    cluster: Cluster,
    noise: float,
    param_name: str,
    n_points: int = PROFILE_LIKELIHOOD_NPOINTS,
    delta_chi2: float = PROFILE_LIKELIHOOD_DELTA_CHI2,  # 95% CI for 1 parameter
) -> tuple[FloatArray, FloatArray, tuple[float, float]]:
    """Compute profile likelihood confidence interval.

    Profile likelihood provides more accurate confidence intervals than
    the covariance matrix, especially for non-linear parameters.

    Args:
        params: Fitted parameters
        cluster: Cluster data
        noise: Noise level
        param_name: Parameter to profile
        n_points: Number of profile points
        delta_chi2: Chi-squared threshold (3.84 for 95% CI)

    Returns
    -------
        Tuple of (parameter_values, chi_squared_values, confidence_interval)
    """
    # Get current best-fit chi-squared
    best_chi2 = float(np.sum(residuals(params, cluster, noise) ** 2))
    param = params[param_name]
    best_value = param.value

    # Determine profile range
    if param.stderr > 0:
        # Use 3 sigma range
        range_min = max(param.min, best_value - 3 * param.stderr)
        range_max = min(param.max, best_value + 3 * param.stderr)
    else:
        # Use 10% of allowed range
        span = param.max - param.min
        range_min = max(param.min, best_value - 0.1 * span)
        range_max = min(param.max, best_value + 0.1 * span)

    profile_values = np.linspace(range_min, range_max, n_points)
    chi2_values = np.zeros(n_points)

    # Compute profile
    for i, val in enumerate(profile_values):
        # Fix parameter at this value
        params_copy = params.copy()
        params_copy[param_name].value = val
        params_copy[param_name].vary = False

        # Re-optimize other parameters
        x0 = params_copy.get_vary_values()
        if len(x0) > 0:
            bounds_list = params_copy.get_vary_bounds_list()

            def objective(x: FloatArray, params: Parameters = params_copy) -> float:
                return residuals_global(x, params, cluster, noise)

            result = optimize.minimize(objective, x0, method="L-BFGS-B", bounds=bounds_list)
            params_copy.set_vary_values(result.x)

        chi2_values[i] = float(np.sum(residuals(params_copy, cluster, noise) ** 2))

    # Find confidence interval
    threshold = best_chi2 + delta_chi2
    below_threshold = chi2_values <= threshold

    if np.any(below_threshold):
        indices = np.where(below_threshold)[0]
        ci_low = float(profile_values[indices[0]])
        ci_high = float(profile_values[indices[-1]])
    else:
        ci_low = range_min
        ci_high = range_max

    return profile_values, chi2_values, (ci_low, ci_high)
