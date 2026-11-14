"""The building blocks of validation T8 (tools/profile_uncertainty/duplicate_check.py)."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

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


def test_one_realization_reads_the_fit_as_a_published_run(study):
    case = study.CASES["sp2, pair 3 points, three copies"]
    row = study.realization(case, (1, 2, 3))
    unflagged = row.pools["unflagged"]
    q, k, peaks = unflagged.q, unflagged.contrasts, unflagged.peaks
    assert (k, peaks) == (8, 2)  # two triples: 2 contrasts each, for both peaks
    assert q > 0
    assert row.excluded == {}
    assert row.stated.shape == (2, 4, 4)


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
