"""Ratios and R2eff from the covariance between planes, against repeated fits (§4).

Each realization adds correlated noise to one cluster, refits its lineshapes and
amplitudes, and states ratio and R2eff errors from the stored covariance factors. The
scatter of the fitted ratios over realizations is the truth the stated σ must match.
"""

import numpy as np
import pytest
from builders import peak_pair

from peakfit.engine.algorithms.covariance_factors import (
    covariance_between_planes,
    peak_covariances_between_planes,
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
from peakfit.engine.scientific.profiles import Normalized, normalize, r2eff

MODEL = NoiseModel(axes=(AxisProcessing(40, 128, "sine_bell", offset=0.4, end=0.98, power=2.0),))
NOISE = 0.1
TIME_T2 = 0.04
REALIZATIONS = 1500
# Profiles that differ between the two peaks, so overlap mixes lineshape errors
# differently into each plane.
AMPLITUDES = np.array([[10.0, 6.0, 3.0, 9.5], [10.0, 2.0, 8.0, 10.5]])


def _realizations(separation: float):
    """Fitted amplitudes and their evaluations over noise realizations."""
    cluster, truth = peak_pair(separation)
    clean = cluster.evaluate(truth).T @ AMPLITUDES
    correlation = cluster_correlation(MODEL, cluster.grid_indices)
    values, vectors = np.linalg.eigh(correlation)
    root = vectors * np.sqrt(np.clip(values, 0.0, None))
    rng = np.random.default_rng(2026)
    for _ in range(REALIZATIONS):
        cluster.data = clean + NOISE * root @ rng.standard_normal(clean.shape)
        params = truth.copy(deep=True)
        fit_cluster(params, cluster, NOISE, correlation=correlation)
        evaluation = evaluate_analytical_model(cluster, params, NOISE, MODEL)
        assert isinstance(evaluation, AnalyticalModelEvaluation)
        assert evaluation.covariance_factors.available
        yield evaluation


def _ratios(evaluation: AnalyticalModelEvaluation, reference: np.ndarray):
    covariances = peak_covariances_between_planes(
        evaluation.covariance_factors, evaluation.amplitudes
    )
    shown = [
        normalize(amplitudes, covariance, reference)
        for amplitudes, covariance in zip(evaluation.amplitudes, covariances, strict=True)
    ]
    assert all(isinstance(s, Normalized) for s in shown)
    return shown


@pytest.mark.parametrize(
    ("separation", "independent_low"),
    # Isolated: the lineshape term is a common scale and cancels from ratios, so
    # independent conditional errors are already right. Overlapping (3 points): they
    # understate the scatter by about 1.5×; the covariance is calibrated.
    [(30.0, None), (3.0, 1.3)],
    ids=["isolated", "overlapping pair"],
)
def test_ratio_and_r2eff_errors_agree_with_repeated_fits(separation, independent_low) -> None:
    reference_mean = np.array([0, 3])
    reference_one = np.array([0])
    ratios, stated, independent, rates, stated_rates = [], [], [], [], []
    for evaluation in _realizations(separation):
        mean = _ratios(evaluation, reference_mean)
        ratios.append([s.values for s in mean])
        stated.append([s.errors for s in mean])
        conditional = np.diag(evaluation.amplitude_covariance)
        today = [
            normalize(a, np.eye(a.size) * c, reference_mean)
            for a, c in zip(evaluation.amplitudes, conditional, strict=True)
        ]
        assert all(isinstance(t, Normalized) for t in today)
        independent.append([t.errors for t in today if isinstance(t, Normalized)])
        one = _ratios(evaluation, reference_one)
        pairs = [r2eff(s.values[1:], s.errors[1:], TIME_T2) for s in one]
        rates.append([rate for rate, _ in pairs])
        stated_rates.append([sigma for _, sigma in pairs])
    # Scatter over realizations / median stated σ; 1500 draws give an SE of about 2%.
    calibration = np.std(ratios, axis=0) / np.median(stated, axis=0)
    rate_calibration = np.std(rates, axis=0) / np.median(stated_rates, axis=0)
    assert np.all((calibration > 0.93) & (calibration < 1.07)), calibration
    assert np.all((rate_calibration > 0.93) & (rate_calibration < 1.07)), rate_calibration
    understated = np.std(ratios, axis=0) / np.median(independent, axis=0)
    if independent_low is None:
        assert np.all((understated > 0.93) & (understated < 1.07)), understated
    else:
        # Only outside the reference: within it, the reference mean shares the errors.
        assert np.all(understated[:, 1:3] > independent_low), understated


def test_each_peaks_block_is_its_own_covariance_between_planes() -> None:
    evaluation = next(_realizations(3.0))
    full = covariance_between_planes(evaluation.covariance_factors, evaluation.amplitudes)
    blocks = peak_covariances_between_planes(evaluation.covariance_factors, evaluation.amplitudes)
    for peak, block in enumerate(blocks):
        np.testing.assert_allclose(block, full[peak, :, peak, :], rtol=1e-13, atol=0)
