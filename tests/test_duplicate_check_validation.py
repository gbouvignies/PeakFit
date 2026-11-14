"""The building blocks of validation T8 (tools/profile_uncertainty/duplicate_check.py)."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from peakfit.engine.algorithms.covariance_factors import CovarianceFactors
from peakfit.engine.scientific.replicates import (
    ExclusionReason,
    RecordedCluster,
    duplicate_plane_check,
)

TOOL = Path(__file__).resolve().parents[1] / "tools" / "profile_uncertainty"
RECORD = (
    Path(__file__).resolve().parents[1] / "docs/validation/profile-uncertainty/duplicate_check.json"
)


@pytest.fixture(scope="module")
def study():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(TOOL))
        patch.syspath_prepend(str(TOOL.parent))
        yield importlib.import_module("duplicate_check")


@pytest.mark.parametrize("rho", [0.3, -0.3])
def test_copies_mixing_correlates_only_the_copies_of_each_group(study, rho):
    planes = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    mixing = study.copies_mixing(planes, rho)
    expected = np.eye(planes.size)
    for group in ((1, 2), (4, 5, 6)):
        for s in group:
            for t in group:
                if s != t:
                    expected[s, t] = rho
    np.testing.assert_allclose(mixing @ mixing.T, expected, atol=1e-12)


def test_the_gate_accepts_calibrated_and_rejects_understated_contrasts(study):
    # q/k of k = 2 standardized contrasts is χ²₂/2: mean 1 when calibrated; contrasts
    # 10% larger than stated give 1.21.
    rng = np.random.default_rng(8)
    calibrated = rng.chisquare(2, 4000) / 2
    assert study.gate(calibrated)["passes"]
    understated = study.gate(1.21 * calibrated)
    assert not understated["passes"]
    assert understated["rms"] == pytest.approx(1.1, abs=0.02)


def test_blocks_replicate_the_first_block_and_detect_inconsistent_blocks(study):
    rng = np.random.default_rng(9)
    calibrated = rng.chisquare(2, 40000) / 2
    consistent = study.blocks(calibrated, 10)
    assert len(consistent["deviation_se"]) == 10
    assert consistent["replication"] == study.gate(calibrated[4000:])
    assert consistent["p_value"] > 0.001
    shifted = calibrated.copy()
    shifted[:4000] *= 0.9  # one block 10% low: far outside the others
    assert study.blocks(shifted, 10)["p_value"] < 1e-4


def test_one_realization_reads_the_fit_as_a_published_run(study):
    case = study.CASES["sp2, pair 3 points, three copies"]
    row = study.realization(case, (1, 2, 3))
    unflagged = row.pools["unflagged"]
    q, k, peaks = unflagged.q, unflagged.contrasts, unflagged.peaks
    assert (k, peaks) == (8, 2)  # two triples: 2 contrasts each, for both peaks
    assert q > 0
    assert row.excluded == {}
    assert row.coincident == {}
    assert row.withheld == {}
    # The stated contrast covariance is each peak's conditional variance times I.
    assert row.stated.shape == (2, 4, 4)
    for stated in row.stated:
        assert stated[0, 0] > 0
        np.testing.assert_array_equal(stated, stated[0, 0] * np.eye(4))
    variances = np.array([stated[0, 0] for stated in row.stated])
    assert q == pytest.approx(float(np.sum(row.contrasts**2 / variances[:, None])))


def test_the_exclusions_case_records_where_the_coincident_pair_went(study):
    case = study.CASES["sp2, isolated with a coincident pair (exclusions), two copies"]
    row = study.realization(case, (1, 2, 3))
    # Each peak of the pair is either excluded or pooled, never both.
    pooled = sum(pool.peaks for pool in row.coincident.values())
    excluded = sum(name in row.excluded for name in ("C1", "C2"))
    assert pooled + excluded == 2
    assert set(row.excluded) <= {"C1", "C2"}


def test_peaks_excluded_for_withheld_lineshape_covariance_keep_their_q(study):
    # One pair, d = (3 − 1)/√2 and c = 0.5: q = 4 over k = 1, in the unflagged pool
    # where the run puts a peak whose flag is not assessed.
    factors = CovarianceFactors(np.array([[0.5]]), None, None, None)
    cluster = RecordedCluster.fitted(
        ("L",), np.array([10.0, 10.0, 20.0]), np.array([[3.0, 1.0, 7.0]]), factors
    )
    excluded = duplicate_plane_check([cluster]).excluded
    assert excluded == {"L": ExclusionReason.LINESHAPE_COVARIANCE_WITHHELD}
    sums = study.withheld_sums([cluster], excluded)
    assert set(sums) == {"unflagged"}
    assert (sums["unflagged"].contrasts, sums["unflagged"].peaks) == (1, 1)
    assert sums["unflagged"].q == pytest.approx(4.0)


def test_the_record_passes_its_gate_and_shows_the_illustration(study):
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    cases = record["cases"]
    gated = {name: case for name, case in cases.items() if case["gated"]}
    assert set(gated) == {name for name, case in study.CASES.items() if case.gated}
    assert record["gate"]["passes"] == all(case["passes"] for case in gated.values())
    assert record["gate"]["passes"]
    for case in gated.values():
        statistic = case["statistic"]
        assert abs(statistic["mean_q_over_k"] - 1.0) <= 3.0 * statistic["se"]
    # Unmodelled correlation between copies: RMS below 1 if positive, above if negative.
    for name, case in cases.items():
        rho = case["copies_correlation"]
        if rho:
            assert (case["statistic"]["rms"] < 1.0) == (rho > 0), name
