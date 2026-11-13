"""Profile σ in a downstream fit with a free scale (profile-uncertainty §4).

A cluster's first peak decays as I_s = e^(−t_s) over 8 planes; a downstream program
fits A·e^(−R·t_s) to its amplitudes by weighted least squares, with the profile σ as
independent per-plane errors. Scatter / stated σ of R is the §4 calibration.

- **Isolated line, Monte Carlo:** each realization adds noise with the processed
  correlation, fits the cluster and then the decay.
- **Overlap, first order:** the scatter is √(KΣKᵀ) with Σ the full covariance between
  planes (validated against the sandwich, T1) and K the decay fit's linear response.
  Monte Carlo at 2000 realizations scatters by ±3% here, too much for a 5% gate.
- **Overlap and the correlated mode:** for these Gaussian lines the first-order ratio
  is 1.02 at a largest correlated mode of 0.05c, 1.04 at 0.12c and 1.09 at 0.26c; the
  §4 synthetic pair gave 1.05 at 0.26c. The mapping depends on the lineshapes and the
  profiles, which is why T3 sets the correlated-profile threshold by consequence.
"""

import numpy as np
import pytest
from builders import peak_pair
from scipy.optimize import least_squares

from peakfit.engine.algorithms.covariance_factors import (
    covariance_between_planes,
    profile_standard_errors,
    relative_profile_covariance,
)
from peakfit.engine.algorithms.evaluation import (
    AnalyticalModelEvaluation,
    evaluate_analytical_model,
)
from peakfit.engine.algorithms.noise_correlation import (
    AxisProcessing,
    NoiseModel,
    cluster_correlation,
)
from peakfit.engine.algorithms.varpro import fit_cluster
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.params_scalar import Parameters

MODEL = NoiseModel(axes=(AxisProcessing(40, 128, "sine_bell", offset=0.4, end=0.98, power=2.0),))
NOISE = 0.02
TIMES = np.linspace(0.0, 1.5, 8)
RATE = 1.0
REALIZATIONS = 4000


def _cluster(separation: float | None):
    """The decaying peak alone (None), or with a slower partner ``separation`` apart."""
    profiles = np.array([np.exp(-RATE * TIMES), 0.8 * np.exp(-0.2 * TIMES)])
    cluster, params = peak_pair(separation or 40.0, amplitudes=profiles)
    if separation is None:
        peak = cluster.peaks[0]
        params = Parameters.from_peaks([peak])
        cluster = Cluster(1, [peak], cluster.grid_indices, np.zeros_like(cluster.data))
        cluster.data = cluster.evaluate(params).T @ profiles[:1]
    return cluster, params


def _decay_rate(amplitudes: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    """R and its stated σ from a weighted free-scale decay fit."""
    fit = least_squares(lambda p: (p[0] * np.exp(-p[1] * TIMES) - amplitudes) / errors, [1.0, RATE])
    return float(fit.x[1]), float(np.sqrt(np.linalg.inv(fit.jac.T @ fit.jac)[1, 1]))


def _largest_mode(evaluation: AnalyticalModelEvaluation) -> float:
    """The first peak's largest correlated mode of Σ_rel, relative to c."""
    factors, amplitudes = evaluation.covariance_factors, evaluation.amplitudes
    conditional = factors.conditional[0, 0]
    relative = relative_profile_covariance(factors, amplitudes)[0]
    return float(np.linalg.eigvalsh(relative - conditional * np.eye(TIMES.size)).max())


def _monte_carlo(separation: float | None) -> tuple[float, float]:
    """Scatter / stated σ of R, and the median largest correlated mode / c."""
    cluster, start = _cluster(separation)
    clean = cluster.data.copy()
    correlation = cluster_correlation(MODEL, cluster.grid_indices)
    values, vectors = np.linalg.eigh(correlation)
    root = vectors * np.sqrt(np.clip(values, 0.0, None))  # R is singular: 40 of 128 points
    rng = np.random.default_rng(161)
    rates, stated, modes = [], [], []
    for _ in range(REALIZATIONS):
        params = start.copy(deep=True)
        cluster.data = clean + NOISE * root @ rng.normal(size=clean.shape)
        fit_cluster(params, cluster, NOISE, correlation=correlation)
        evaluation = evaluate_analytical_model(cluster, params, NOISE, MODEL)
        assert isinstance(evaluation, AnalyticalModelEvaluation)
        factors, amplitudes = evaluation.covariance_factors, evaluation.amplitudes
        rate, error = _decay_rate(amplitudes[0], profile_standard_errors(factors, amplitudes)[0])
        rates.append(rate)
        stated.append(error)
        modes.append(_largest_mode(evaluation) / factors.conditional[0, 0])
    return float(np.std(rates) / np.median(stated)), float(np.median(modes))


def _first_order(separation: float) -> tuple[float, float]:
    """Scatter / stated σ of R at the noiseless fit, and the largest correlated mode / c."""
    cluster, params = _cluster(separation)
    correlation = cluster_correlation(MODEL, cluster.grid_indices)
    fit_cluster(params, cluster, NOISE, correlation=correlation)
    evaluation = evaluate_analytical_model(cluster, params, NOISE, MODEL)
    assert isinstance(evaluation, AnalyticalModelEvaluation)
    factors, amplitudes = evaluation.covariance_factors, evaluation.amplitudes
    full = covariance_between_planes(factors, amplitudes)[0, :, 0, :]
    weights = np.diag(profile_standard_errors(factors, amplitudes)[0] ** -2.0)
    decay = np.exp(-RATE * TIMES)
    jacobian = np.column_stack((decay, -TIMES * decay))  # ∂/∂A, ∂/∂R at A = 1
    stated = np.linalg.inv(jacobian.T @ weights @ jacobian)
    response = stated @ jacobian.T @ weights
    scatter = response @ full @ response.T
    ratio = float(np.sqrt(scatter[1, 1] / stated[1, 1]))
    return ratio, _largest_mode(evaluation) / factors.conditional[0, 0]


@pytest.mark.slow
def test_the_profile_sigma_is_calibrated_for_an_isolated_line() -> None:
    ratio, mode = _monte_carlo(None)
    assert mode < 1e-10
    assert ratio == pytest.approx(1.0, abs=0.05)


def test_the_profile_sigma_is_within_five_percent_for_moderate_overlap() -> None:
    ratio, mode = _first_order(5.5)
    assert 0.05 < mode < 0.25  # within the §6 range for the flag threshold, 0.1–0.25
    assert 1.0 < ratio < 1.05


def test_strong_overlap_leaves_the_profile_sigma_short() -> None:
    # §4: correlated modes of Σ_rel beyond c make any per-plane σ wrong; the flag marks it.
    ratio, mode = _first_order(4.0)
    assert mode > 1.0
    assert ratio > 1.2
