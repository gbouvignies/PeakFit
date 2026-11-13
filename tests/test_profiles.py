"""Derived profiles: reference ratios and R2eff, shared by the GUI and CLI plots.

Ratios take the covariance of one peak's amplitudes between planes (specification
§4); independent per-plane errors are the diagonal special case.
"""

import numpy as np
import pytest

from peakfit.engine.scientific.profiles import (
    RELIABILITY_LIMIT,
    Normalized,
    Unavailable,
    normalize,
    r2eff,
)


def _normalized(*args, **kwargs) -> Normalized:
    result = normalize(*args, **kwargs)
    assert isinstance(result, Normalized), result
    return result


def test_without_covariance_between_planes_ratios_reduce_to_the_independent_formula():
    amplitudes = np.array([10.0, 8.0, 5.0, -2.0, 9.5])
    errors = np.array([0.2, 0.3, 0.25, 0.4, 0.1])
    reference = np.array([0, 4])
    shown = _normalized(amplitudes, np.diag(errors**2), reference)
    mean = (10.0 + 9.5) / 2
    sigma_r = np.sqrt(0.2**2 + 0.1**2) / 2
    assert shown.reference == mean
    assert shown.reference_error == pytest.approx(sigma_r)
    outside = ~shown.in_reference
    ratio = amplitudes / mean
    expected = np.sqrt((errors**2 + ratio**2 * sigma_r**2) / mean**2)
    np.testing.assert_allclose(shown.values, ratio)
    np.testing.assert_allclose(shown.errors[outside], expected[outside], rtol=1e-12)


def test_a_single_reference_plane_is_exactly_one_without_error():
    amplitudes = np.array([3.7, 1.2, -0.4])
    shown = _normalized(amplitudes, np.diag(np.full(3, 0.05**2)), np.array([0]))
    assert shown.exact
    assert shown.values[0] == 1.0
    assert shown.errors[0] == 0.0
    assert shown.values[1] == pytest.approx(1.2 / 3.7)


def test_full_gradient_matches_monte_carlo_inside_and_outside_the_reference():
    rng = np.random.default_rng(7)
    amplitudes = np.array([100.0, 96.0, 104.0, 40.0, -15.0])
    errors = np.array([1.0, 1.5, 0.8, 1.2, 0.9])
    reference = np.array([0, 1, 2])
    shown = _normalized(amplitudes, np.diag(errors**2), reference)
    draws = amplitudes + rng.standard_normal((400_000, amplitudes.size)) * errors
    ratios = draws / draws[:, reference].mean(axis=1, keepdims=True)
    # Members of S are correlated with R; the full gradient accounts for it.
    np.testing.assert_allclose(shown.errors, ratios.std(axis=0), rtol=0.02)
    naive = np.sqrt(
        (errors**2 + (amplitudes / amplitudes[reference].mean()) ** 2 * shown.reference_error**2)
        / shown.reference**2
    )
    assert not np.allclose(naive[reference], ratios.std(axis=0)[reference], rtol=0.02)


@pytest.mark.parametrize(
    ("amplitudes", "errors", "reason", "undefined"),
    [
        ([0.0, 1.0], [0.1, 0.1], "zero or not finite", True),
        ([np.nan, 1.0], [0.1, 0.1], "zero or not finite", True),
        ([1.0, 2.0], [np.nan, np.nan], "σ unavailable", True),
        ([1.0, 2.0], [0.34, 0.34], "σ_R/|R| = 0.34", False),
    ],
)
def test_unavailable_ratios_say_why_and_whether_they_are_undefined(
    amplitudes, errors, reason, undefined
):
    result = normalize(np.array(amplitudes), np.diag(np.array(errors) ** 2), np.array([0]))
    assert isinstance(result, Unavailable)
    assert reason in result.reason
    assert result.undefined is undefined


def test_the_limit_is_inclusive_and_displayed():
    limit = normalize(np.array([1.0, 2.0]), np.diag([RELIABILITY_LIMIT**2] * 2), np.array([0]))
    assert isinstance(limit, Normalized)


def test_negative_reference_keeps_signs_and_never_alters_recorded_amplitudes():
    amplitudes = np.array([-4.0, -2.0, 1.0])
    original = amplitudes.copy()
    shown = _normalized(amplitudes, np.diag(np.full(3, 0.1**2)), np.array([0]))
    assert shown.reference == -4.0
    np.testing.assert_array_equal(shown.values, [1.0, 0.5, -0.25])
    np.testing.assert_array_equal(amplitudes, original)


def test_r2eff_error_carries_the_logarithm_derivative():
    # σ(−ln r / T) = σ_r / (|r| T): the 1/r factor matters most for strongly
    # attenuated planes, where omitting it underestimated σ by the factor r.
    rng = np.random.default_rng(11)
    amplitudes = np.array([100.0, 80.0, 30.0])
    errors = np.array([1.0, 1.0, 1.0])
    shown = _normalized(amplitudes, np.diag(errors**2), np.array([0]))
    rates, sigma = r2eff(shown.values[1:], shown.errors[1:], 0.04)
    np.testing.assert_allclose(rates, -np.log([0.8, 0.3]) / 0.04)
    draws = amplitudes + rng.standard_normal((400_000, 3)) * errors
    sampled = -np.log(draws[:, 1:] / draws[:, :1]) / 0.04
    np.testing.assert_allclose(sigma, sampled.std(axis=0), rtol=0.02)


def _correlated_case():
    """A peak whose lineshape uncertainty correlates its amplitudes between planes."""
    amplitudes = np.array([100.0, 60.0, 30.0, 95.0, 70.0])
    rng = np.random.default_rng(5)
    modes = rng.standard_normal((5, 2))
    covariance = (
        np.diag(np.full(5, 1.2**2)) + modes @ modes.T + 0.8 * np.outer(amplitudes, amplitudes) / 1e4
    )
    return amplitudes, covariance


def test_ratio_errors_follow_the_covariance_between_planes():
    amplitudes, covariance = _correlated_case()
    reference = np.array([0, 3])
    shown = _normalized(amplitudes, covariance, reference)
    draws = np.random.default_rng(13).multivariate_normal(amplitudes, covariance, 400_000)
    ratios = draws / draws[:, reference].mean(axis=1, keepdims=True)
    np.testing.assert_allclose(shown.errors, ratios.std(axis=0), rtol=0.02)
    # The diagonal alone misstates them: the covariance is not decoration here.
    independent = _normalized(amplitudes, np.diag(np.diag(covariance)), reference)
    assert not np.allclose(independent.errors, ratios.std(axis=0), rtol=0.1)


def test_r2eff_errors_follow_the_covariance_between_planes():
    amplitudes, covariance = _correlated_case()
    shown = _normalized(amplitudes, covariance, np.array([0]))
    rates, sigma = r2eff(shown.values[1:], shown.errors[1:], 0.04)
    draws = np.random.default_rng(17).multivariate_normal(amplitudes, covariance, 400_000)
    sampled = -np.log(draws[:, 1:] / draws[:, :1]) / 0.04
    np.testing.assert_allclose(rates, -np.log(amplitudes[1:] / amplitudes[0]) / 0.04)
    np.testing.assert_allclose(sigma, sampled.std(axis=0), rtol=0.02)


def test_the_reliability_limit_applies_to_the_total_sigma_of_the_reference_mean():
    # Two reference planes, σ = 0.12 each: independent, σ_R/|R| = 0.085 is within the
    # limit; fully correlated, the mean does not average them down and σ_R/|R| = 0.12.
    amplitudes = np.array([1.0, 1.0, 0.5])
    variance = 0.12**2
    independent = np.diag([variance, variance, variance])
    correlated = independent.copy()
    correlated[0, 1] = correlated[1, 0] = variance
    reference = np.array([0, 1])
    assert _normalized(amplitudes, independent, reference).reference_error == pytest.approx(
        0.12 / np.sqrt(2)
    )
    result = normalize(amplitudes, correlated, reference)
    assert isinstance(result, Unavailable)
    assert "σ_R/|R| = 0.12" in result.reason
    assert not result.undefined


def test_an_unavailable_covariance_between_planes_gives_no_ratio():
    result = normalize(np.array([1.0, 2.0]), np.full((2, 2), np.nan), np.array([0]))
    assert isinstance(result, Unavailable)
    assert "σ unavailable" in result.reason
    assert result.undefined
