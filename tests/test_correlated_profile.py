"""The correlated-profile flag (profile-uncertainty §6, thresholds from T3 §8.1).

A peak is flagged where the largest correlated mode of Σ_rel exceeds 0.1c, or where
its within-plane amplitude correlation with a cluster partner exceeds 0.15 in
magnitude. Both diagnostics read only the stored factors and amplitudes.

The synthetic clusters below are built at the level of the factors, so that each
diagnostic sits exactly where the test puts it: with orthogonal unit profiles, one
lineshape parameter of variance v, no cross term and peak A's gain the partner's
profile, A's only correlated mode is v·‖a_B‖² = v, relative to c = 1.
"""

import numpy as np
import pytest
from builders import peak_pair

from peakfit.engine.algorithms.covariance_factors import (
    CORRELATED_MODE_THRESHOLD,
    PARTNER_CORRELATION_THRESHOLD,
    CovarianceFactors,
    ProfileCorrelation,
    profile_correlations,
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

# Orthogonal unit profiles over four planes.
AMPLITUDES = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]) / np.sqrt(2.0)


def _pair(mode: float, rho: float) -> CovarianceFactors:
    """Two peaks: A's largest correlated mode ``mode``·c, and within-plane ρ ``rho``."""
    shape_gain = np.zeros((2, 2, 1))
    shape_gain[1, 0, 0] = 1.0  # A's amplitudes move with B's profile
    return CovarianceFactors(
        np.array([[1.0, rho], [rho, 1.0]]),
        shape_gain,
        np.zeros((2, 2, 1)),
        np.array([[mode]]),
    )


def test_the_thresholds_are_those_set_by_t3() -> None:
    assert CORRELATED_MODE_THRESHOLD == 0.1
    assert PARTNER_CORRELATION_THRESHOLD == 0.15


@pytest.mark.parametrize(("mode", "flagged"), [(0.095, False), (0.105, True), (0.5, True)])
def test_a_correlated_mode_above_a_tenth_of_c_flags_the_peak(mode, flagged) -> None:
    peak, partner = profile_correlations(_pair(mode, 0.0), AMPLITUDES)
    assert peak.mode == pytest.approx(mode, rel=1e-12)
    assert peak.status is (
        ProfileCorrelation.CORRELATED if flagged else ProfileCorrelation.WITHIN_TOLERANCE
    )
    # B's gain is zero: it has no correlated mode of its own.
    assert partner.status is ProfileCorrelation.WITHIN_TOLERANCE


@pytest.mark.parametrize(
    ("rho", "flagged"), [(0.14, False), (-0.15, False), (0.16, True), (-0.16, True)]
)
def test_a_partner_correlation_above_fifteen_hundredths_flags_both_peaks(rho, flagged) -> None:
    flags = profile_correlations(_pair(0.0, rho), AMPLITUDES)
    expected = ProfileCorrelation.CORRELATED if flagged else ProfileCorrelation.WITHIN_TOLERANCE
    assert [f.status for f in flags] == [expected, expected]
    assert [f.partner_correlation for f in flags] == pytest.approx([abs(rho)] * 2)
    assert [f.partner for f in flags] == [1, 0]


def test_withheld_lineshape_factors_leave_the_profile_not_assessed() -> None:
    # Even a strong partner correlation: without the lineshape terms there is no
    # profile σ to qualify.
    factors = CovarianceFactors(np.array([[1.0, 0.5], [0.5, 1.0]]), None, None, None)
    flags = profile_correlations(factors, AMPLITUDES)
    assert [f.status for f in flags] == [ProfileCorrelation.NOT_ASSESSED] * 2
    assert all(np.isnan(f.mode) for f in flags)


def test_one_plane_has_no_profile_to_flag() -> None:
    factors = _pair(0.5, 0.5)
    flags = profile_correlations(factors, np.array([[1.0], [1.0]]))
    assert [f.status for f in flags] == [ProfileCorrelation.NO_PROFILE] * 2


MODEL = NoiseModel(axes=(AxisProcessing(40, 128, "sine_bell", offset=0.4, end=0.98, power=2.0),))
NOISE = 0.02
# Differing profiles over eight planes, as in the profile-σ calibration.
TIMES = np.linspace(0.0, 1.5, 8)
PROFILES = np.array([np.exp(-TIMES), 0.8 * np.exp(-0.2 * TIMES)])


def _fitted(cluster: Cluster, params: Parameters) -> AnalyticalModelEvaluation:
    fit_cluster(
        params, cluster, NOISE, correlation=cluster_correlation(MODEL, cluster.grid_indices)
    )
    evaluation = evaluate_analytical_model(cluster, params, NOISE, MODEL)
    assert isinstance(evaluation, AnalyticalModelEvaluation)
    return evaluation


def test_an_isolated_peak_is_never_flagged() -> None:
    # §3.2: an isolated peak's lineshape term is a common scale, so Σ_rel = cI.
    pair, _ = peak_pair(40.0, amplitudes=PROFILES)
    peak = pair.peaks[0]
    params = Parameters.from_peaks([peak])
    cluster = Cluster(1, [peak], pair.grid_indices, np.zeros_like(pair.data))
    cluster.data = cluster.evaluate(params).T @ PROFILES[:1]
    evaluation = _fitted(cluster, params)
    (flag,) = profile_correlations(evaluation.covariance_factors, evaluation.amplitudes)
    assert flag.status is ProfileCorrelation.WITHIN_TOLERANCE
    assert flag.mode == pytest.approx(0.0, abs=1e-9)
    assert flag.partner_correlation == 0.0
    assert flag.partner is None


def test_a_resolved_pair_is_not_flagged_and_an_overlapping_pair_is() -> None:
    resolved = _fitted(*peak_pair(8.0, amplitudes=PROFILES))
    flags = profile_correlations(resolved.covariance_factors, resolved.amplitudes)
    assert [f.status for f in flags] == [ProfileCorrelation.WITHIN_TOLERANCE] * 2
    overlapping = _fitted(*peak_pair(4.0, amplitudes=PROFILES))
    flags = profile_correlations(overlapping.covariance_factors, overlapping.amplitudes)
    assert flags[0].status is ProfileCorrelation.CORRELATED
    assert flags[0].mode > CORRELATED_MODE_THRESHOLD
