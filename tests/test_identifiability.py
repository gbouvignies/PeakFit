"""Per-member amplitude identifiability from the recorded shape factorization.

Compact-support (triangular) shapes make "no projection" exact, so each case tests a
mathematical property rather than a tolerance: estimability within a rank-deficient
matrix, ill-conditioning without rank loss, and the near-collinear rule.
"""

import numpy as np
import pytest

from peakfit.engine.algorithms.linear_algebra import factorize_shapes
from peakfit.engine.diagnostics.identifiability import (
    AmplitudeStatus,
    Cause,
    amplitude_identifiability,
)

X = np.linspace(-20.0, 20.0, 801)
OK, NOT, NONE = (
    AmplitudeStatus.IDENTIFIABLE,
    AmplitudeStatus.NOT_ESTABLISHED,
    AmplitudeStatus.UNIDENTIFIABLE,
)


def tent(center, width=2.0):
    """A triangle of half-width ``width``: exactly zero beyond it."""
    return np.clip(1 - np.abs(X - center) / width, 0, None)


def lorentzian(center, width=2.0):
    return 1 / (1 + ((X - center) / (width / 2)) ** 2)


def statuses(shapes, errors=None):
    return [(m.status, m.cause) for m in amplitude_identifiability(np.array(shapes), errors=errors)]


def test_separated_members_are_identifiable():
    assert statuses([tent(-8), tent(0), tent(8)]) == [(OK, Cause.NONE)] * 3


def test_a_member_outside_the_null_space_is_identifiable_in_a_rank_deficient_cluster():
    # Two identical shapes: their split is undetermined, the third amplitude is not.
    members = amplitude_identifiability(np.array([tent(0), tent(0), tent(8)]))
    assert members[0].rank == 2
    assert [m.status for m in members] == [NONE, NONE, OK]
    assert members[0].cause is Cause.NULL_SPACE
    assert members[0].null_projection == pytest.approx(np.sqrt(0.5))
    assert members[2].null_projection == 0.0


def test_a_three_way_dependence_without_a_collinear_pair_is_unidentifiable():
    # s3 = s1 + s2: no two shapes are collinear, yet none of the three is estimable.
    shapes = [tent(-1), tent(1), tent(-1) + tent(1), tent(10)]
    members = amplitude_identifiability(np.array(shapes))
    assert members[0].rank == 3
    assert members[0].partners == ()
    assert [m.status for m in members] == [NONE, NONE, NONE, OK]
    assert members[0].null_projection == pytest.approx(1 / np.sqrt(3))


def test_ill_conditioning_without_rank_loss_is_never_called_unidentifiable():
    # Full numerical rank, but s_min/s₀ ≈ 1e-10 < √ε: covariance withheld, not identifiability.
    shapes = np.array([tent(0), tent(0) + 1e-10 * tent(0.5), tent(10)])
    projection = factorize_shapes(shapes)
    assert projection.rank == 3
    assert not np.all(np.isfinite(projection.covariance))
    members = amplitude_identifiability(shapes)
    assert [(m.status, m.cause) for m in members[:2]] == [(NOT, Cause.ILL_CONDITIONED)] * 2
    assert members[2].status is OK
    assert all(m.status is not NONE for m in members)


def test_near_collinear_members_are_not_established_with_their_partner():
    # Well conditioned (covariance finite) but within the position-adoption rule.
    shapes = np.array([lorentzian(0), lorentzian(0.004), lorentzian(9)])
    assert np.all(np.isfinite(factorize_shapes(shapes).covariance))
    members = amplitude_identifiability(shapes)
    assert [(m.status, m.cause) for m in members] == [
        (NOT, Cause.NEAR_COLLINEAR),
        (NOT, Cause.NEAR_COLLINEAR),
        (OK, Cause.NONE),
    ]
    assert members[0].partners == (1,)
    assert members[0].dissimilarity < 1e-4


def test_a_missing_sigma_leaves_the_value_determined():
    members = amplitude_identifiability(
        np.array([tent(-8), tent(8)]), errors=np.array([1.0, np.nan])
    )
    assert members[0].status is OK
    assert (members[1].status, members[1].cause) == (NOT, Cause.UNCERTAINTY_UNAVAILABLE)
    assert members[1].determined
    assert not amplitude_identifiability(np.array([tent(0), tent(0)]))[0].determined


def test_invalid_shapes_are_not_established():
    members = amplitude_identifiability(np.array([tent(0), np.zeros_like(X)]))
    assert {(m.status, m.cause) for m in members} == {(NOT, Cause.INVALID_SHAPE)}


def test_fewer_samples_than_members_leaves_every_member_unidentifiable():
    shapes = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert [m.status for m in amplitude_identifiability(shapes)] == [NONE] * 3


@pytest.mark.parametrize("seed", range(5))
def test_withheld_covariance_always_has_a_member_that_is_not_identifiable(seed):
    # The classification uses the engine's conventions: whenever the engine withholds
    # the joint covariance, at least one member projects onto the withheld directions.
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(3, 60))
    shapes = np.vstack([base, base[0] + rng.choice([0.0, 1e-12]) * base[1]])
    withheld = not np.all(np.isfinite(factorize_shapes(shapes).covariance))
    members = amplitude_identifiability(shapes)
    assert withheld == any(m.status is not OK for m in members)


def test_amplitudes_report_cancellation_without_changing_status():
    shapes = np.array([lorentzian(0), lorentzian(0.004)])
    plain = amplitude_identifiability(shapes)
    weighted = amplitude_identifiability(shapes, np.array([[1e6, 2e6], [-1e6, -2e6]]))
    assert [m.status for m in plain] == [m.status for m in weighted]
