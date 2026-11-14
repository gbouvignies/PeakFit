"""Duplicate planes: contrasts between copies against the covariance the factors predict.

The duplicate check of ``peakfit replicates`` (issue #174, spec #172).

The hand-built factors below describe one peak with a single lineshape parameter, so
that its covariance between planes is Σ = cI + g²v·aaᵀ (``shape_gain`` g, parameter
covariance v, no cross term): every expected value can be worked out by hand.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from peakfit.cli.app import app
from peakfit.engine.algorithms.covariance_factors import (
    CovarianceFactors,
    profile_correlations,
)
from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.engine.scientific.replicates import (
    RecordedCluster,
    duplicate_plane_check,
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


def test_one_pair_is_standardized_by_its_predicted_variance():
    # Contrast d = (3 − 1)/√2 = √2; Var d = c + g²v·d² = 1 + 2 = 3, so q = 2/3.
    check = duplicate_plane_check([_cluster([10.0, 10.0, 20.0], [3.0, 1.0, 7.0])])
    pool = check.unflagged
    assert pool.contrasts == 1
    assert pool.peaks == 1
    assert pool.clusters == 1
    assert np.isclose(pool.rms, np.sqrt(2.0 / 3.0))
    assert check.flagged.contrasts == 0
    assert check.excluded == {}


def test_groups_sharing_a_parameter_are_whitened_jointly():
    # Two pairs: d = (√2, 3/√2) and V = I + ddᵀ, so q = |d|²/(1 + |d|²) = 6.5/7.5.
    # Whitening each pair alone would give 2/3 + 4.5/5.5 instead.
    check = duplicate_plane_check([_cluster([1.0, 1.0, 2.0, 2.0], [3.0, 1.0, 5.0, 2.0])])
    assert check.unflagged.contrasts == 2
    assert np.isclose(check.unflagged.rms, np.sqrt(6.5 / 7.5 / 2))


def test_three_copies_give_two_contrasts_and_use_every_copy():
    # Centred copies (−2, −1, 3): |d|² = 14, so q = 14/15 over k = 2.
    check = duplicate_plane_check([_cluster([5.0, 5.0, 5.0, 9.0], [1.0, 2.0, 6.0, 4.0])])
    assert check.unflagged.contrasts == 2
    assert np.isclose(check.unflagged.rms, np.sqrt(14.0 / 15.0 / 2))


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
    planes = [1.0, 1.0, 2.0, 2.0]
    amplitudes = [3.0, 1.0, 5.0, 2.0]
    withheld = CovarianceFactors(np.array([[1.0]]), None, None, None)
    unusable = RecordedCluster(("U1", "U2"), np.asarray(planes), None, None, (False, False))
    # V = cI − 2gx·ddᵀ: with x = 1 and |d|² = 6.5, negative along d.
    negative = _factors(variance=0.0, cross=1.0)
    # V = cI + ddᵀ: its eigenvalues c and c + |d|² differ by far more than 1/√ε.
    weak = _factors(conditional=1e-12)
    check = duplicate_plane_check(
        [
            unusable,
            _cluster(planes, amplitudes, withheld, name="W"),
            _cluster(planes, amplitudes, negative, name="N"),
            _cluster(planes, amplitudes, weak, name="D"),
            _cluster(planes, amplitudes, name="OK"),
        ]
    )
    assert check.excluded == {
        "U1": "unusable cluster",
        "U2": "unusable cluster",
        "W": "covariance factors withheld",
        "N": "contrast covariance not positive definite",
        "D": "contrast covariance has a direction weaker than √ε of the largest",
    }
    assert (check.unflagged.peaks, check.unflagged.clusters) == (1, 1)
    assert np.isclose(check.unflagged.rms, np.sqrt(6.5 / 7.5 / 2))


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


def _fit_example(output: Path, z_values: Path) -> Path:
    result = CliRunner().invoke(
        app,
        [
            "fit",
            str(DATA / "pseudo3d.ft2"),
            str(DATA / "pseudo3d.list"),
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
                assert check.excluded[name] == "unusable cluster"


@pytest.mark.slow
def test_a_published_run_lists_peaks_it_cannot_check(repeated_run):
    outcome, _samples, context = read_scientific(repeated_run / "scientific.npz")
    first, second, *rest = outcome.clusters
    evaluation = first.analytical_evaluation
    assert evaluation is not None
    withheld = CovarianceFactors(evaluation.amplitude_covariance, None, None, None)
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
    expected = dict.fromkeys(first.peak_names, "covariance factors withheld")
    expected |= dict.fromkeys(second.peak_names, "unusable cluster")
    assert check.excluded == expected
    whole = run_duplicate_check(outcome, context)
    pooled = check.unflagged.peaks + check.flagged.peaks
    assert pooled == whole.unflagged.peaks + whole.flagged.peaks - len(expected)
