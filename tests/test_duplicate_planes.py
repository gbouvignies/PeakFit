"""Duplicate planes: contrasts between copies against the conditional variance.

The duplicate check of ``peakfit replicates`` (issue #174, spec #172): q = |Ca|²/c.

The hand-built factors below describe one peak with a single lineshape parameter
(``shape_gain`` g, parameter covariance v, cross gain x) and conditional variance c:
every expected value is |Ca|²/c, worked out by hand, whatever g, v and x.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from click.utils import strip_ansi
from typer.testing import CliRunner

from peakfit.cli.app import app
from peakfit.engine.algorithms.covariance_factors import (
    CovarianceFactors,
    covariance_between_planes,
    profile_correlations,
)
from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.engine.scientific.replicates import (
    ExclusionReason,
    RecordedCluster,
    duplicate_groups,
    duplicate_plane_check,
    orthonormal_contrasts,
    run_duplicate_check,
)
from peakfit.io.scientific import read_scientific

DATA = Path(__file__).resolve().parents[1] / "examples" / "04-uncertainty-analysis" / "data"


def _factors(conditional=1.0, gain=1.0, variance=1.0, cross=0.0) -> CovarianceFactors:
    """One peak, one lineshape parameter: Σ_st = cδ_st + a_s a_t (g²v − 2gx)."""
    return CovarianceFactors(
        conditional=np.array([[conditional]]),
        shape_gain=np.array([[[gain]]]),
        cross_gain=np.array([[[cross]]]),
        parameter_covariance=np.array([[variance]]),
    )


def _cluster(plane_values, amplitudes, factors=None, *, flagged=False, name="A"):
    return RecordedCluster(
        peak_names=(name,),
        plane_values=np.asarray(plane_values, dtype=float),
        amplitudes=np.asarray([amplitudes], dtype=float),
        factors=_factors() if factors is None else factors,
        flagged=(flagged,),
    )


def test_one_pair_is_standardized_by_the_conditional_variance():
    # Contrast d = (3 − 1)/√2 = √2 and c = 0.5, so q = d²/c = 4 over k = 1, whatever
    # the lineshape factors state.
    check = duplicate_plane_check(
        [_cluster([10.0, 10.0, 20.0], [3.0, 1.0, 7.0], _factors(conditional=0.5))]
    )
    pool = check.unflagged
    assert pool.contrasts == 1
    assert pool.peaks == 1
    assert pool.clusters == 1
    assert np.isclose(pool.q, 4.0)
    assert np.isclose(pool.rms, 2.0)
    assert check.flagged.contrasts == 0
    assert check.excluded == {}


def test_q_does_not_depend_on_the_lineshape_factors():
    # Two pairs, d = (√2, 3/√2): q = |d|²/c = 6.5 for every gain, parameter
    # covariance and cross gain, and with the lineshape factors withheld.
    planes, amplitudes = [1.0, 1.0, 2.0, 2.0], [3.0, 1.0, 5.0, 2.0]
    variants = [
        _factors(gain=0.0, variance=0.0),
        _factors(gain=3.0, variance=2.0, cross=0.4),
        _factors(variance=0.0, cross=1.0),
        CovarianceFactors(np.array([[1.0]]), None, None, None),
    ]
    for factors in variants:
        check = duplicate_plane_check([_cluster(planes, amplitudes, factors)])
        assert check.excluded == {}
        assert check.unflagged.contrasts == 2
        assert np.isclose(check.unflagged.q, 6.5)


def test_three_copies_give_two_contrasts_and_use_every_copy():
    # Centred copies (−2, −1, 3): |d|² = 14 over k = 2, with c = 2: q = 7.
    check = duplicate_plane_check(
        [_cluster([5.0, 5.0, 5.0, 9.0], [1.0, 2.0, 6.0, 4.0], _factors(conditional=2.0))]
    )
    assert check.unflagged.contrasts == 2
    assert np.isclose(check.unflagged.q, 7.0)


def test_each_peak_is_standardized_by_its_own_conditional_variance():
    # Two peaks, one pair: d = √2 for A and 2√2 for B, c = 2 and 4 (the off-diagonal
    # term plays no part): q = 1 + 2 = 3 over k = 2.
    factors = CovarianceFactors(np.array([[2.0, 1.5], [1.5, 4.0]]), None, None, None)
    cluster = RecordedCluster(
        ("A", "B"),
        np.array([0.0, 0.0, 1.0]),
        np.array([[3.0, 1.0, 0.0], [5.0, 1.0, 0.0]]),
        factors,
        (False, False),
    )
    check = duplicate_plane_check([cluster])
    assert (check.unflagged.contrasts, check.unflagged.peaks) == (2, 2)
    assert np.isclose(check.unflagged.q, 3.0)


def _random_cluster(rng, plane_values, *, n_peaks=2, n_theta=2):
    """A cluster with generic factors and shared parameters, positive definite."""
    n_planes = len(plane_values)
    conditional = np.array([[2.0, 0.3], [0.3, 1.5]])[:n_peaks, :n_peaks]
    root = rng.normal(size=(n_theta, n_theta))
    factors = CovarianceFactors(
        conditional=conditional,
        shape_gain=rng.normal(size=(n_peaks, n_peaks, n_theta)),
        cross_gain=np.zeros((n_peaks, n_peaks, n_theta)),
        parameter_covariance=0.1 * root @ root.T,
    )
    return RecordedCluster(
        peak_names=tuple(f"P{i}" for i in range(n_peaks)),
        plane_values=np.asarray(plane_values, dtype=float),
        amplitudes=rng.normal(10.0, 3.0, size=(n_peaks, n_planes)),
        factors=factors,
        flagged=(False,) * n_peaks,
    )


def test_the_result_depends_neither_on_plane_order_nor_on_the_contrast_basis():
    rng = np.random.default_rng(174)
    cluster = _random_cluster(rng, [0.0, 3.0, 0.0, 7.0, 3.0, 0.0, 1.0, 3.0, 0.0])
    reference = duplicate_plane_check([cluster])
    assert reference.unflagged.contrasts == 2 * (3 + 2)
    for _ in range(5):
        # A permutation reorders the groups and the copies within each group, which
        # changes the contrast basis as well as the plane order.
        order = rng.permutation(cluster.plane_values.size)
        shuffled = RecordedCluster(
            cluster.peak_names,
            cluster.plane_values[order],
            cluster.amplitudes[:, order],
            cluster.factors,
            cluster.flagged,
        )
        assert np.isclose(duplicate_plane_check([shuffled]).unflagged.q, reference.unflagged.q)


def test_a_run_without_repeated_plane_values_has_nothing_to_check():
    check = duplicate_plane_check([_cluster([1.0, 2.0, 3.0], [3.0, 1.0, 7.0])])
    assert check.unflagged.contrasts == check.flagged.contrasts == 0
    assert np.isnan(check.unflagged.rms)
    assert check.excluded == {}


def test_each_excluded_peak_is_listed_with_its_reason():
    planes = [1.0, 1.0, 2.0, 2.0, 3.0]
    amplitudes = [3.0, 1.0, 5.0, 2.0, 4.0]
    unusable = RecordedCluster(("U1", "U2"), np.asarray(planes), None, None, (False, False))
    # A NaN amplitude on a plane without copies still excludes the peak.
    check = duplicate_plane_check(
        [
            unusable,
            _cluster(planes, [3.0, 1.0, 5.0, 2.0, np.nan], name="A"),
            _cluster(planes, amplitudes, _factors(conditional=np.nan), name="W"),
            _cluster(planes, amplitudes, _factors(conditional=np.inf), name="I"),
            _cluster(planes, amplitudes, _factors(conditional=0.0), name="Z"),
            _cluster(planes, amplitudes, _factors(conditional=-1.0), name="N"),
            _cluster(planes, amplitudes, name="OK"),
        ]
    )
    assert check.excluded == {
        "U1": ExclusionReason.UNUSABLE_CLUSTER,
        "U2": ExclusionReason.UNUSABLE_CLUSTER,
        "A": ExclusionReason.NON_FINITE_AMPLITUDE,
        "W": ExclusionReason.NO_CONDITIONAL_VARIANCE,
        "I": ExclusionReason.NO_CONDITIONAL_VARIANCE,
        "Z": ExclusionReason.NO_CONDITIONAL_VARIANCE,
        "N": ExclusionReason.NO_CONDITIONAL_VARIANCE,
    }
    assert (check.unflagged.peaks, check.unflagged.clusters) == (1, 1)
    assert np.isclose(check.unflagged.q, 6.5)
    assert np.isfinite(check.unflagged.rms)


def test_a_withheld_conditional_covariance_excludes_every_peak_of_its_cluster():
    # Where the shapes are rank-deficient or too ill-conditioned, the fit withholds
    # the whole conditional covariance as NaN (``linear_algebra.factorize_shapes``).
    withheld = CovarianceFactors(np.full((2, 2), np.nan), None, None, None)
    cluster = RecordedCluster(
        ("C1", "C2"),
        np.array([0.0, 0.0, 1.0]),
        np.array([[3.0, 1.0, 0.0], [5.0, 1.0, 0.0]]),
        withheld,
        (False, False),
    )
    check = duplicate_plane_check([cluster])
    assert check.excluded == dict.fromkeys(("C1", "C2"), ExclusionReason.NO_CONDITIONAL_VARIANCE)
    assert check.unflagged.peaks == check.flagged.peaks == 0


def test_exclusion_reasons_keep_their_stated_wording():
    assert {reason.value for reason in ExclusionReason} == {
        "unusable cluster",
        "non-finite amplitude",
        "conditional variance withheld or not positive",
    }


def test_signed_zeros_are_one_plane_value_and_nan_matches_nothing():
    assert duplicate_groups(np.array([0.0, -0.0, 1.0])) == ((0, 1),)
    assert duplicate_groups(np.array([np.nan, np.nan, 1.0, 1.0])) == ((2, 3),)
    assert duplicate_groups(np.array([np.nan, 2.0, np.nan])) == ()


def test_the_contrasts_are_orthonormal_and_blind_to_equal_copies():
    planes = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    contrasts = orthonormal_contrasts(duplicate_groups(planes), planes.size)
    assert contrasts.shape == (3, 7)
    np.testing.assert_allclose(contrasts @ contrasts.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(contrasts @ (planes**2 + 5.0), 0.0, atol=1e-12)


def test_a_fitted_cluster_carries_the_correlated_profile_flag_of_each_peak():
    # Within-plane amplitude correlation 0.5 > 0.15 with the partner flags both peaks;
    # withheld lineshape factors leave the flag unassessed, hence unset.
    planes = np.array([0.0, 0.0, 1.0])
    amplitudes = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    conditional = np.array([[1.0, 0.5], [0.5, 1.0]])
    zeros = np.zeros((2, 2, 1))
    correlated = CovarianceFactors(conditional, zeros, zeros, np.array([[0.0]]))
    withheld = CovarianceFactors(conditional, None, None, None)
    fitted = RecordedCluster.fitted(("A", "B"), planes, amplitudes, correlated)
    assert fitted.flagged == (True, True)
    assert RecordedCluster.fitted(("A", "B"), planes, amplitudes, withheld).flagged == (
        False,
        False,
    )


def test_flagged_peaks_are_pooled_apart_with_their_own_counts():
    rng = np.random.default_rng(7)
    clusters = [_random_cluster(rng, [0.0, 0.0, 1.0, 1.0, 1.0]) for _ in range(3)]
    last = clusters[2]
    flagged = RecordedCluster(
        last.peak_names, last.plane_values, last.amplitudes, last.factors, (True, False)
    )
    check = duplicate_plane_check([clusters[0], clusters[1], flagged])
    unflagged, correlated = check.unflagged, check.flagged
    assert (unflagged.contrasts, unflagged.peaks, unflagged.clusters) == (15, 5, 3)
    assert (correlated.contrasts, correlated.peaks, correlated.clusters) == (3, 1, 1)
    # Flagging a peak moves its contrasts from one pool to the other, nothing else.
    total = duplicate_plane_check(clusters).unflagged.q
    assert np.isclose(unflagged.q + correlated.q, total)


def test_fewer_than_fifty_contrasts_carry_a_low_sample_warning():
    def pool(pairs):
        planes = np.repeat(np.arange(pairs, dtype=float), 2)
        return duplicate_plane_check([_cluster(planes, np.arange(2.0 * pairs))]).unflagged

    small, enough = pool(49), pool(50)
    assert small.low_sample
    assert np.isclose(small.standard_error_bound, 1 / np.sqrt(98))
    assert not enough.low_sample
    assert np.isclose(enough.standard_error_bound, 0.1)


# Validation T8, fast cases (profile-uncertainty §8.2): amplitudes drawn from the
# covariance the factors state, around true duplicates, then checked as recorded. The
# end-to-end cases, through the processing operators and the engine fit, are the tool's
# (tools/profile_uncertainty/duplicate_check.py).
REALIZATIONS = 2000


def _draws(cluster: RecordedCluster, rng, *, copies_correlation: float = 0.0) -> list:
    """Realizations of ``cluster`` whose amplitudes scatter as its factors state.

    The true amplitudes are equal within each group of equal plane values. A nonzero
    ``copies_correlation`` adds what the factors omit: the conditional noise of the
    copies of each group, correlated with that coefficient.
    """
    factors = cluster.factors
    assert factors is not None
    planes = cluster.plane_values
    truth = np.array(cluster.amplitudes, dtype=float)
    for group in duplicate_groups(planes):
        truth[:, list(group)] = truth[:, [group[0]]]
    n_peaks, n_planes = truth.shape
    stated = covariance_between_planes(factors, truth)
    stated = stated.reshape(n_peaks * n_planes, n_peaks * n_planes)
    copies = np.zeros((n_planes, n_planes))
    for group in duplicate_groups(planes):
        for s in group:
            for t in group:
                copies[s, t] = copies_correlation if s != t else 0.0
    actual = stated + np.kron(factors.conditional, copies)
    root = np.linalg.cholesky(actual)
    draws = truth.ravel() + rng.standard_normal((REALIZATIONS, truth.size)) @ root.T
    return [replace(cluster, amplitudes=a.reshape(n_peaks, n_planes)) for a in draws]


def _measured_cluster(rng, plane_values) -> RecordedCluster:
    """Two peaks sharing two lineshape parameters, at a signal-to-noise ratio near 70.

    The lineshape term of each amplitude's variance, a²·gᵀC_θg, is of the order of the
    conditional c, as in a fit: shape parameters are known about as well as amplitudes.
    """
    cluster = _random_cluster(rng, plane_values)
    factors = cluster.factors
    assert factors.parameter_covariance is not None
    return replace(
        cluster,
        amplitudes=10.0 * cluster.amplitudes,
        factors=replace(factors, parameter_covariance=1e-3 * factors.parameter_covariance),
    )


def _mean_q_over_k(clusters) -> tuple[float, float]:
    """The mean over realizations of Σq/Σk, every peak pooled, and its standard error."""
    values = []
    for cluster in clusters:
        check = duplicate_plane_check([cluster])
        assert check.excluded == {}
        q = check.unflagged.q + check.flagged.q
        values.append(q / (check.unflagged.contrasts + check.flagged.contrasts))
    values = np.asarray(values)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


@pytest.mark.parametrize(
    "plane_values",
    [
        pytest.param([0.0, 1.0, 1.0, 2.0, 3.0, 3.0], id="two copies"),
        pytest.param([0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0], id="three copies"),
    ],
)
def test_contrasts_scattering_as_stated_give_a_mean_q_over_k_of_one(plane_values):
    rng = np.random.default_rng(175)
    cluster = _measured_cluster(rng, plane_values)
    mean, se = _mean_q_over_k(_draws(cluster, rng))
    assert abs(mean - 1.0) <= 3.0 * se


def test_lineshape_parameter_noise_leaves_q_unchanged():
    # Copies with equal true amplitudes a*, the conditional noise e, and lineshape
    # parameter errors δθ, 100 times larger in σ than the conditional noise, entering
    # through the gain: a_s = a*_s + e_s + G_s(a*)δθ. The contrasts annihilate G(a*)δθ,
    # so q, and its mean of k, do not move. Standardizing by CΣCᵀ built from the
    # fitted amplitudes would let that noise return along the contrast, biasing q low.
    plane_values = np.array([0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0])
    rng = np.random.default_rng(176)
    cluster = _measured_cluster(rng, plane_values)
    factors = cluster.factors
    assert factors is not None
    assert factors.shape_gain is not None
    truth = np.array(cluster.amplitudes, dtype=float)
    for group in duplicate_groups(plane_values):
        truth[:, list(group)] = truth[:, [group[0]]]
    gains = np.einsum("ms,mlp->lsp", truth, factors.shape_gain)  # G_s(a*), by peak
    root = np.linalg.cholesky(factors.conditional)
    contrasts = orthonormal_contrasts(duplicate_groups(plane_values), plane_values.size)
    quiet, loud = [], []
    for _ in range(REALIZATIONS):
        noise = root @ rng.standard_normal(truth.shape)
        theta = 100.0 * rng.standard_normal(gains.shape[2])
        quiet.append(replace(cluster, amplitudes=truth + noise))
        loud.append(replace(cluster, amplitudes=truth + noise + gains @ theta))
    assert not np.allclose(quiet[0].amplitudes, loud[0].amplitudes)
    np.testing.assert_allclose(
        loud[0].amplitudes @ contrasts.T, quiet[0].amplitudes @ contrasts.T, atol=1e-8
    )
    mean, se = _mean_q_over_k(loud)
    assert mean == pytest.approx(_mean_q_over_k(quiet)[0], rel=1e-9)
    assert abs(mean - 1.0) <= 3.0 * se


@pytest.mark.parametrize(("rho", "below"), [(0.3, True), (-0.3, False)])
def test_unmodelled_correlation_between_copies_moves_the_result_away_from_one(rho, below):
    # Positive correlation shrinks the contrasts below what the covariance states;
    # negative correlation enlarges them.
    rng = np.random.default_rng(1750)
    cluster = _measured_cluster(rng, [0.0, 1.0, 1.0, 2.0, 3.0, 3.0])
    mean, se = _mean_q_over_k(_draws(cluster, rng, copies_correlation=rho))
    assert (mean < 1.0 - 3.0 * se) if below else (mean > 1.0 + 3.0 * se)


def _fit_example(output: Path, z_values: Path, peaks: Path = DATA / "pseudo3d.list") -> Path:
    result = CliRunner().invoke(
        app,
        [
            "fit",
            str(DATA / "pseudo3d.ft2"),
            str(peaks),
            "--z-values",
            str(z_values),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    (run,) = output.iterdir()
    return run


@pytest.fixture(scope="module")
def repeated_run(tmp_path_factory):
    """The bundled CEST example, its plane values edited into three pairs and a triple.

    These planes are not real duplicates: the run tests wiring, not calibration.
    """
    folder = tmp_path_factory.mktemp("repeated")
    offsets = np.loadtxt(DATA / "b1_offsets.txt")
    for copy, original in ((2, 1), (4, 3), (6, 5), (8, 7), (9, 7)):
        offsets[copy] = offsets[original]
    np.savetxt(folder / "offsets.txt", offsets)
    return _fit_example(folder / "Fits", folder / "offsets.txt")


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """Four peaks of the bundled example and one pair of equal plane values."""
    folder = tmp_path_factory.mktemp("small")
    offsets = np.loadtxt(DATA / "b1_offsets.txt")
    offsets[2] = offsets[1]
    np.savetxt(folder / "offsets.txt", offsets)
    lines = (DATA / "pseudo3d.list").read_text().splitlines()[:4]
    (folder / "peaks.list").write_text("\n".join(lines) + "\n")
    return _fit_example(folder / "Fits", folder / "offsets.txt", folder / "peaks.list")


@pytest.fixture(scope="module")
def plain_run(tmp_path_factory):
    """The bundled CEST example as distributed: no repeated plane values."""
    folder = tmp_path_factory.mktemp("plain")
    return _fit_example(folder / "Fits", DATA / "b1_offsets.txt")


@pytest.mark.slow
def test_a_published_run_accounts_for_every_peak_of_its_duplicates(repeated_run):
    outcome, _samples, context = read_scientific(repeated_run / "scientific.npz")
    check = run_duplicate_check(outcome, context)
    names = [name for cluster in outcome.clusters for name in cluster.peak_names]
    pools = (check.unflagged, check.flagged)
    # Every peak is pooled or excluded with a reason; each pooled peak has 3 + 2 contrasts.
    assert sum(pool.peaks for pool in pools) + len(check.excluded) == len(names)
    assert set(check.excluded) <= set(names)
    for pool in pools:
        assert pool.contrasts == 5 * pool.peaks
        assert pool.clusters <= len(outcome.clusters)
    assert check.unflagged.peaks > 100
    assert 0 < check.unflagged.rms < np.inf
    # The pools follow the correlated-profile flag published with the run.
    flagged = {
        name
        for cluster in outcome.clusters
        if cluster.usable and cluster.analytical_evaluation is not None
        for name, flag in zip(
            cluster.peak_names,
            profile_correlations(
                cluster.analytical_evaluation.covariance_factors,
                cluster.analytical_evaluation.amplitudes,
            ),
            strict=True,
        )
        if flag.flagged
    }
    assert check.flagged.peaks == len(flagged - set(check.excluded))
    # Unusable clusters are excluded as such.
    for cluster in outcome.clusters:
        if not cluster.usable:
            for name in cluster.peak_names:
                assert check.excluded[name] is ExclusionReason.UNUSABLE_CLUSTER


@pytest.mark.slow
def test_a_published_run_lists_peaks_it_cannot_check(repeated_run):
    outcome, _samples, context = read_scientific(repeated_run / "scientific.npz")
    first, second, *rest = outcome.clusters
    evaluation = first.analytical_evaluation
    assert evaluation is not None
    # A conditional covariance withheld as the fit withholds it: all NaN.
    conditional = np.full_like(evaluation.amplitude_covariance, np.nan)
    withheld = CovarianceFactors(conditional, None, None, None)
    edited = replace(
        outcome,
        clusters=(
            replace(
                first,
                analytical_evaluation=replace(evaluation, covariance_factors=withheld),
            ),
            replace(
                second,
                classification=FitOutcomeClassification.UNUSABLE,
                analytical_evaluation=None,
                unusable_reason="edited for the test",
            ),
            *rest,
        ),
    )
    check = run_duplicate_check(edited, context)
    expected = dict.fromkeys(first.peak_names, ExclusionReason.NO_CONDITIONAL_VARIANCE)
    expected |= dict.fromkeys(second.peak_names, ExclusionReason.UNUSABLE_CLUSTER)
    assert check.excluded == expected
    whole = run_duplicate_check(outcome, context)
    pooled = check.unflagged.peaks + check.flagged.peaks
    assert pooled == whole.unflagged.peaks + whole.flagged.peaks - len(expected)


def _replicates(*args: str):
    result = CliRunner().invoke(app, ["replicates", *args])
    return result, " ".join(strip_ansi(result.output).split())


LIMITS = (
    "It measures repeatability, not accuracy.",
    "It tests the conditional σ only, the noise part: it cannot detect a "
    "miscalibration of the lineshape-parameter part of the marginal or profile σ.",
    "It checks the uncertainty of contrasts between copies, not marginal σ.",
    "It checks raw amplitudes, not normalized profiles.",
    "Equal plane values do not prove independent, equivalent acquisitions.",
    "It is not calibrated for copies reconstructed jointly.",
    "No error is changed.",
)


@pytest.mark.slow
def test_the_command_checks_duplicate_planes_when_the_run_has_them(repeated_run):
    result, text = _replicates(str(repeated_run))
    assert result.exit_code == 0, result.output
    assert "Method: duplicate planes" in text
    assert "4 groups of equal plane values, 5 contrasts per peak" in text
    assert (
        "Peaks without the correlated-profile flag: RMS of standardized contrasts 1.09 "
        "(690 contrasts, 138 peaks, 114 clusters)" in text
    )
    assert (
        "Peaks with the correlated-profile flag: RMS of standardized contrasts 1.24 "
        "(140 contrasts, 28 peaks, 13 clusters)" in text
    )
    assert "Contrasts from peaks of one cluster are correlated" in text
    assert "Few contrasts" not in text
    assert "Observed / stated" not in text
    for limit in LIMITS:
        assert limit in text


@pytest.mark.slow
def test_a_small_sample_carries_its_warning(small_run):
    result, text = _replicates(str(small_run))
    assert result.exit_code == 0, result.output
    assert "1 group of equal plane values, 1 contrast per peak" in text
    assert "(4 contrasts, 4 peaks," in text
    assert (
        "Few contrasts: the sampling uncertainty of the RMS is at least 1/√(2n) = 0.35, "
        "a lower bound" in text
    )


@pytest.mark.slow
def test_the_baseline_method_can_be_chosen_on_a_run_with_duplicates(repeated_run):
    result, text = _replicates(str(repeated_run), "--baseline")
    assert result.exit_code == 0, result.output
    assert "Method: CEST baseline planes" in text
    assert "Observed / stated profile σ" in text
    assert "standardized contrasts" not in text


@pytest.mark.slow
@pytest.mark.parametrize("option", [["--ref", "0"], ["--window", "300"]])
def test_baseline_options_are_an_error_when_duplicates_are_checked(repeated_run, option):
    result, text = _replicates(str(repeated_run), *option)
    assert result.exit_code == 2
    assert "--ref and --window belong to the CEST baseline method" in text
    assert "--baseline" in text
    assert "standardized contrasts" not in text


@pytest.mark.slow
def test_a_run_without_duplicates_says_so_and_checks_its_baseline(plain_run):
    result, text = _replicates(str(plain_run))
    assert result.exit_code == 0, result.output
    assert "No duplicate planes" in text
    assert "Method: CEST baseline planes" in text
    assert "Observed / stated profile σ: median 1.07" in text
