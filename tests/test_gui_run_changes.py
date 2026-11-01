"""Run-to-run changes describe two recorded runs without refitting or judging either."""

from copy import deepcopy
from dataclasses import replace

import pytest
from test_gui_dataset import _dataset

from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.changes import compare_runs
from peakfit.gui.editor.review import ClusterReview, NumericalWarning, review_clusters
from peakfit.gui.editor.workspace import _change_entries


@pytest.fixture(scope="module")
def run():
    spectra, _peaks, _config, prepared = _dataset()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return spectra, result


def _titles(spectra):
    return [f"axis {i}" for i in range(len(spectra.spectral_params))]


def _reviews(spectra, result):
    return review_clusters(result, list(spectra.spectral_params), _titles(spectra))


def _compare(spectra, active, parent, active_reviews=None, parent_reviews=None):
    return compare_runs(
        active,
        _reviews(spectra, active) if active_reviews is None else active_reviews,
        parent,
        _reviews(spectra, parent) if parent_reviews is None else parent_reviews,
        list(spectra.spectral_params),
        _titles(spectra),
    )


def _with_outcome(result, clusters):
    outcome = replace(result.outcome, clusters=tuple(clusters))
    return replace(result, completed_fit=replace(result.completed_fit, outcome=outcome))


def _review(ids, warnings=(), chi2=1.0):
    return ClusterReview(
        0,
        tuple(ids),
        tuple(i.upper() for i in ids),
        tuple(NumericalWarning(kind, "") for kind in warnings),
        chi2,
        3.0,
        100,
        float("nan"),
        (),
    )


def test_identical_runs_report_nothing_to_investigate(run):
    spectra, result = run
    changes = _compare(spectra, result, result)
    assert changes.changes == ()
    assert changes.displacements == ()
    assert changes.regrouped == 0
    assert changes.clusters[0] == changes.clusters[1]
    assert changes.median_chi2[0] == changes.median_chi2[1]
    assert (changes.added, changes.removed, changes.uncompared) == (0, 0, 0)


def test_position_differences_are_reported_in_ppm_and_linewidths(run):
    spectra, result = run
    cluster = next(c for c in result.outcome.clusters if c.usable)
    moved = next(p for p in cluster.final_nonlinear_parameters if p.name.endswith(".cs") and p.vary)
    engine, axis, _kind = moved.name.split(".")
    width = next(
        p.value for p in cluster.final_nonlinear_parameters if p.name == f"{engine}.{axis}.lw"
    )
    index = int(axis[1:]) - 2
    obs = float(spectra.spectral_params[index].obs)
    step = 0.02 * width / obs  # 0.02 linewidth, in ppm
    parent = _with_outcome(
        result,
        [
            replace(
                c,
                final_nonlinear_parameters=tuple(
                    replace(p, value=p.value - step) if p is moved else p
                    for p in c.final_nonlinear_parameters
                ),
            )
            if c is cluster
            else c
            for c in result.outcome.clusters
        ],
    )
    changes = _compare(spectra, result, parent)
    (move,) = changes.displacements
    working = next(i.working_id for i in result.inputs.identities if i.engine_name == engine)
    assert move.working_id == working
    assert move.axis == f"axis {index}"
    assert move.delta == pytest.approx(step)
    assert move.linewidths == pytest.approx(0.02)
    # Neutral information: nothing about position differences is a warning or highlight.
    assert changes.changes == ()


def test_warning_on_the_same_members_is_reported_neutrally(run):
    spectra, result = run
    active = [_review(["a", "b"])]
    parent = [_review(["a", "b"], ["Uncertainty unreliable"])]
    changes = _compare(spectra, result, result, active, parent)
    (change,) = changes.changes
    assert change.regrouped_from == ()
    entries = _change_entries(changes, {})
    detail = next(e[2] for e in entries if e[0] == "link" and e[1] == "A +1")
    assert detail == "▲ Uncertainty unreliable no longer reported"
    assert "resolved" not in detail


def test_regrouped_clusters_show_before_and_now_without_claiming_a_resolution(run):
    spectra, result = run
    active = [_review(["a", "b"])]
    parent = [_review(["a"], ["Uncertainty unreliable"]), _review(["b"])]
    changes = _compare(spectra, result, result, active, parent)
    (change,) = changes.changes
    assert change.regrouped_from == ("A", "B")
    assert changes.regrouped == 1
    detail = next(e[2] for e in _change_entries(changes, {}) if e[0] == "link")
    assert detail == "Regrouped from A, B\nBefore regrouping ▲ Uncertainty unreliable"
    assert "no longer reported" not in detail


def test_fit_and_clustering_summary(run):
    spectra, result = run
    active = [_review(["a", "b"], chi2=2.0), _review(["c"], ["Not converged"], chi2=4.0)]
    parent = [_review(["a"], chi2=1.0), _review(["b"], chi2=3.0), _review(["c"], chi2=5.0)]
    changes = _compare(spectra, result, result, active, parent)
    assert changes.clusters == (3, 2)
    assert changes.warned == (0, 1)
    assert changes.median_chi2 == (3.0, 3.0)
    lines = [e[1] for e in _change_entries(changes, {}) if e[0] == "line"]
    assert lines == [
        "Median χ²ᵣ 3.00 → 3.00",
        "Clusters with ▲ warnings 0 → 1",
        "Clusters 3 → 2 · 1 regrouped",
    ]


def test_peaks_only_in_one_run_and_unusable_peaks_are_counted_not_compared(run):
    spectra, result = run
    dropped = result.inputs.identities[0]
    parent = replace(
        result,
        inputs=replace(
            result.inputs,
            identities=tuple(i for i in result.inputs.identities if i is not dropped),
        ),
    )
    changes = _compare(spectra, result, parent)
    assert (changes.added, changes.removed) == (1, 0)
    unusable = next(c for c in result.outcome.clusters if c.usable)
    broken = _with_outcome(
        result,
        [
            replace(c, classification=FitOutcomeClassification.UNUSABLE) if c is unusable else c
            for c in result.outcome.clusters
        ],
    )
    changes = _compare(spectra, result, broken)
    assert changes.uncompared == len(unusable.peak_names)
    engines = {i.engine_name: i.working_id for i in result.inputs.identities}
    excluded = {engines[name] for name in unusable.peak_names}
    assert not excluded & {move.working_id for move in changes.displacements}
