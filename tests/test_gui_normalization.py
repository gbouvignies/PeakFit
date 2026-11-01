"""Reference-normalized amplitude profiles: a display transform with exact special cases."""

import numpy as np
import pytest

from peakfit.gui.editor.profile import RELIABILITY_LIMIT, Normalized, normalize


def _normalized(*args, **kwargs) -> Normalized:
    result = normalize(*args, **kwargs)
    assert isinstance(result, Normalized), result
    return result


def test_outside_the_reference_reduces_to_the_independent_ratio_formula():
    amplitudes = np.array([10.0, 8.0, 5.0, -2.0, 9.5])
    errors = np.array([0.2, 0.3, 0.25, 0.4, 0.1])
    reference = np.array([0, 4])
    shown = _normalized(amplitudes, errors, reference)
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
    shown = _normalized(amplitudes, np.full(3, 0.05), np.array([0]))
    assert shown.exact
    assert shown.values[0] == 1.0
    assert shown.errors[0] == 0.0
    assert shown.values[1] == pytest.approx(1.2 / 3.7)


def test_full_gradient_matches_monte_carlo_inside_and_outside_the_reference():
    rng = np.random.default_rng(7)
    amplitudes = np.array([100.0, 96.0, 104.0, 40.0, -15.0])
    errors = np.array([1.0, 1.5, 0.8, 1.2, 0.9])
    reference = np.array([0, 1, 2])
    shown = _normalized(amplitudes, errors, reference)
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
    ("amplitudes", "errors", "reason"),
    [
        ([0.0, 1.0], [0.1, 0.1], "zero or not finite"),
        ([np.nan, 1.0], [0.1, 0.1], "zero or not finite"),
        ([1.0, 2.0], [np.nan, np.nan], "σ unavailable"),
        ([1.0, 2.0], [0.34, 0.34], "σ_R/|R| = 0.34"),
    ],
)
def test_unreliable_references_are_unavailable_with_a_reason(amplitudes, errors, reason):
    result = normalize(np.array(amplitudes), np.array(errors), np.array([0]))
    assert isinstance(result, str)
    assert reason in result


def test_the_limit_is_inclusive_and_displayed():
    limit = normalize(np.array([1.0, 2.0]), np.array([RELIABILITY_LIMIT] * 2), np.array([0]))
    assert isinstance(limit, Normalized)


def test_negative_reference_keeps_signs_and_never_alters_recorded_amplitudes():
    amplitudes = np.array([-4.0, -2.0, 1.0])
    original = amplitudes.copy()
    shown = _normalized(amplitudes, np.full(3, 0.1), np.array([0]))
    assert shown.reference == -4.0
    np.testing.assert_array_equal(shown.values, [1.0, 0.5, -0.25])
    np.testing.assert_array_equal(amplitudes, original)
