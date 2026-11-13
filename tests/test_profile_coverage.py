"""The building blocks of the T3 coverage study (tools/profile_uncertainty/profile_coverage.py)."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "profile_uncertainty"
RECORD = (
    Path(__file__).resolve().parents[1]
    / "docs/validation/profile-uncertainty/profile_coverage.json"
)


@pytest.fixture(scope="module")
def study():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(TOOL))
        patch.syspath_prepend(str(TOOL.parent))
        yield importlib.import_module("profile_coverage")


@pytest.mark.parametrize("profile", ["3-plane decay", "30-plane CPMG", "30-plane CEST"])
@pytest.mark.parametrize("lineshape", ["sp1", "sp2", "no_apod"])
def test_the_profile_sigma_is_exact_downstream_for_an_isolated_peak(study, lineshape, profile):
    # §3.2: the lineshape term of an isolated peak is a common scale, so Σ_rel = cI and
    # independent profile σ give every free-scale downstream fit its exact σ.
    result = study.first_order(study.Case(lineshape, None, profile))
    assert result["modes"] == [pytest.approx(0.0, abs=1e-10)]
    assert result["partner_correlation"] == [0.0]
    for misstatement in result["misstatement"].values():
        assert misstatement == pytest.approx(1.0, abs=1e-6)


def test_strong_overlap_misstates_downstream_errors_and_is_flagged(study):
    # §4: a pair 2 points apart on both axes leaves correlated modes the per-plane
    # profile σ cannot carry.
    result = study.first_order(study.Case("sp2", (2.0, 2.0), "3-plane decay"))
    assert all(study.flagged(result["modes"], result["partner_correlation"]))
    assert max(abs(m - 1.0) for m in result["misstatement"].values()) > study.TOLERANCE


def test_a_resolved_pair_is_unflagged_and_within_tolerance(study):
    result = study.first_order(study.Case("sp2", (4.0, 4.0), "30-plane CPMG"))
    assert not any(study.flagged(result["modes"], result["partner_correlation"]))
    assert max(abs(m - 1.0) for m in result["misstatement"].values()) < study.TOLERANCE


def test_calibration_recognizes_correct_and_understated_errors(study):
    rng = np.random.default_rng(3)
    sigma = np.array([0.5, 2.0])
    errors = rng.normal(size=(20000, 2)) * sigma
    right = study.calibration(errors, np.broadcast_to(sigma, errors.shape))
    assert right["scatter_within_gate"]
    assert right["coverage_95"] == pytest.approx(0.95, abs=3 * right["coverage_se"])
    short = study.calibration(errors, np.broadcast_to(0.8 * sigma, errors.shape))
    assert not short["scatter_within_gate"]
    assert not short["coverage_within_2se"]
    assert short["max"] == pytest.approx(1.25, rel=0.03)


def _covered(rows: int, columns: int, outside: int) -> np.ndarray:
    """Errors with unit stated σ, of which exactly ``outside`` lie beyond 1.96σ."""
    rng = np.random.default_rng(1)
    errors = rng.uniform(-1.5, 1.5, rows * columns)
    errors[rng.choice(errors.size, outside, replace=False)] = 3.0
    return errors.reshape(rows, columns)


def test_a_single_quantity_passes_coverage_within_three_standard_errors(study):
    # 4000 realizations, coverage 0.9415: 2.5 SE low. Strictly outside ±2 SE (reported),
    # within criterion (b)'s ±3 SE.
    errors = _covered(4000, 1, 234)
    result = study.calibration(errors, np.ones_like(errors))
    assert result["coverage_95"] == pytest.approx(0.9415)
    assert not result["coverage_within_2se"]
    assert result["coverage_within_criterion"]
    four_se = study.calibration(_covered(4000, 1, 256), np.ones((4000, 1)))  # 0.936: 4 SE
    assert not four_se["coverage_within_criterion"]


def test_pooled_quantities_pass_coverage_within_one_point(study):
    # Pooled over planes, the SE is small enough to resolve effects of no practical
    # weight; criterion (b) asks for 95% coverage within ±0.01 absolute.
    near = study.calibration(_covered(4000, 30, 5160), np.ones((4000, 30)))  # 0.957
    assert not near["coverage_within_2se"]
    assert near["coverage_within_criterion"]
    far = study.calibration(_covered(4000, 30, 7800), np.ones((4000, 30)))  # 0.935
    assert not far["coverage_within_criterion"]


def test_the_record_states_criterion_b_and_its_verdict(study):
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    gate = record["gate"]
    assert gate["criterion"]["coverage_single"] == "±3 SE"
    assert gate["criterion"]["coverage_pooled"] == "±0.01"
    assert gate["criterion"]["scatter"] == [0.95, 1.05] == list(study.SCATTER_GATE)
    gated = [
        q
        for case in record["cases"].values()
        if case["passes"] is not None
        for name, q in case["quantities"].items()
        if not name.endswith("(joint covariance)")
    ]
    assert gate["passes"] == all(q["passes"] for q in gated)
    assert "strict_2se" in gate  # the stricter count, kept as information


def test_the_record_uses_the_published_thresholds(study):
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    thresholds = record["thresholds"]
    assert thresholds["mode"] == study.MODE_THRESHOLD
    assert thresholds["partner_correlation"] == study.PARTNER_THRESHOLD
    assert thresholds["tolerance"] == study.TOLERANCE
    # The thresholds lie below the sweep's boundary, and leave no swept peak beyond it.
    assert thresholds["mode"] <= thresholds["boundary"]["mode"]
    assert thresholds["partner_correlation"] <= thresholds["boundary"]["partner_correlation"]
    assert thresholds["sweep"]["unflagged_beyond_tolerance"] == 0
