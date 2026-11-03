"""One "At bound" rule: a varied value within 1e-6 of its bound interval of a bound."""

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from builders import completed_result

from peakfit.engine.diagnostics.bounds import AT_BOUND_FRACTION, at_bound
from peakfit.engine.domain.params_scalar import Parameter
from peakfit.engine.scientific.final_outcome import FinalParameter
from peakfit.fit.fitting import find_review_clusters
from peakfit.gui.editor.review import _warnings

if TYPE_CHECKING:
    from peakfit.engine.scientific.final_outcome import FinalFitOutcome


def _parameter(value, minimum, maximum, *, vary=True):
    return FinalParameter("p", value, minimum, maximum, vary, "", math.nan)


@pytest.mark.parametrize(("minimum", "maximum"), [(0.0, 2e-3), (-1.0, 1.0), (1e4, 3e4)])
def test_the_tolerance_is_the_same_fraction_of_the_interval_at_every_scale(minimum, maximum):
    span = maximum - minimum
    tolerance = AT_BOUND_FRACTION * span
    assert at_bound(_parameter(minimum, minimum, maximum))
    assert at_bound(_parameter(maximum, minimum, maximum))
    assert at_bound(_parameter(minimum + 0.5 * tolerance, minimum, maximum))
    assert at_bound(_parameter(maximum - 0.5 * tolerance, minimum, maximum))
    assert not at_bound(_parameter(minimum + 2 * tolerance, minimum, maximum))
    assert not at_bound(_parameter(maximum - 2 * tolerance, minimum, maximum))
    assert not at_bound(_parameter(minimum + span / 2, minimum, maximum))


def test_a_distance_of_exactly_the_tolerance_is_at_bound():
    tolerance = AT_BOUND_FRACTION * 4.0
    assert at_bound(_parameter(tolerance, 0.0, 4.0))
    assert not at_bound(_parameter(math.nextafter(tolerance, math.inf), 0.0, 4.0))


def test_the_tolerance_scales_with_the_interval_not_the_value():
    # A ppm position near 120 in a ±0.1 ppm window, 5e-5 ppm from its lower bound: the
    # former value-relative CLI rule (1e-6 × (1 + |v|) ≈ 1.2e-4) flagged it.
    assert not at_bound(_parameter(119.9 + 5e-5, 119.9, 120.1))
    assert at_bound(_parameter(119.9 + 1e-7, 119.9, 120.1))


def test_fixed_parameters_are_never_at_bound():
    assert not at_bound(_parameter(0.0, 0.0, 1.0, vary=False))
    assert not at_bound(Parameter(name="p", value=1.0, min=0.0, max=1.0, vary=False))
    assert at_bound(Parameter(name="p", value=1.0, min=0.0, max=1.0, vary=True))


@pytest.mark.parametrize(
    ("value", "minimum", "maximum", "flagged"),
    [
        (0.0, 0.0, math.inf, False),  # one-sided: no interval to scale the tolerance
        (0.0, -math.inf, 0.0, False),
        (0.0, -math.inf, math.inf, False),
        (math.nan, 0.0, 1.0, False),
        (-0.5, 0.0, 1.0, True),  # outside its bounds
        (2.0, 2.0, 2.0, True),  # zero-width interval
        (0.5, 1.0, 0.0, False),  # inverted, invalid interval
    ],
)
def test_bounds_that_are_not_a_finite_interval(value, minimum, maximum, flagged):
    assert at_bound(_parameter(value, minimum, maximum)) is flagged


def test_cli_review_and_gui_warnings_flag_the_same_recorded_parameters():
    _, _, _, result = completed_result([], False)
    cluster = result.outcome.clusters[0]
    first, second, *rest = cluster.final_nonlinear_parameters
    span = first.max - first.min
    parameters = (
        replace(first, value=first.max - 0.5 * AT_BOUND_FRACTION * span),
        replace(second, value=second.min, vary=False),
        *rest,
    )
    cluster = replace(cluster, final_nonlinear_parameters=parameters)
    outcome = cast("FinalFitOutcome", SimpleNamespace(clusters=[cluster]))
    (review,) = find_review_clusters(outcome)
    (warning,) = [w for w in _warnings(cluster) if w.kind == "At bound"]
    assert review.at_bounds == [first.name]
    assert warning.detail == first.name
