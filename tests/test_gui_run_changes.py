"""Run-to-run changes compare two recorded runs without refitting either."""

from copy import deepcopy
from dataclasses import replace

import pytest
from test_gui_dataset import _dataset

from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.changes import compare_runs
from peakfit.gui.editor.review import review_clusters


@pytest.fixture(scope="module")
def run():
    spectra, _peaks, _config, prepared = _dataset()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return spectra, result


def _with_outcome(result, clusters):
    outcome = replace(result.outcome, clusters=tuple(clusters))
    return replace(result, completed_fit=replace(result.completed_fit, outcome=outcome))


def _reviews(spectra, result):
    axes = list(spectra.spectral_params)
    return review_clusters(result, axes, [f"axis {i}" for i in range(len(axes))])


def _compare(spectra, active, parent):
    titles = [f"axis {i}" for i in range(len(spectra.spectral_params))]
    return compare_runs(
        active, _reviews(spectra, active), parent, _reviews(spectra, parent), titles
    )


def test_identical_runs_report_no_changes(run):
    spectra, result = run
    changes = _compare(spectra, result, result)
    assert changes.clusters == ()
    assert changes.shifts == ()
    assert (changes.added, changes.removed, changes.unjudged) == (0, 0, 0)


def test_a_position_moved_beyond_sigma_is_reported_in_units_of_sigma(run):
    spectra, result = run
    cluster = next(c for c in result.outcome.clusters if c.usable)
    moved = next(p for p in cluster.final_nonlinear_parameters if p.name.endswith(".cs") and p.vary)
    # The parent run recorded this position 3 σ lower; the active σ is the yardstick.
    parent = _with_outcome(
        result,
        [
            replace(
                c,
                final_nonlinear_parameters=tuple(
                    replace(p, value=p.value - 3 * moved.standard_error) if p is moved else p
                    for p in c.final_nonlinear_parameters
                ),
            )
            if c is cluster
            else c
            for c in result.outcome.clusters
        ],
    )
    changes = _compare(spectra, result, parent)
    engine = moved.name.split(".")[0]
    working = next(i.working_id for i in result.inputs.identities if i.engine_name == engine)
    (shift,) = [s for s in changes.shifts if s.working_id == working]
    assert shift.ratio == pytest.approx(3.0)
    assert shift.delta == pytest.approx(3 * moved.standard_error)
    assert shift.sigma == moved.standard_error


def test_gained_and_resolved_warnings_are_named(run):
    spectra, result = run
    cluster = next(c for c in result.outcome.clusters if c.usable and c.analytical_evaluation)
    evaluation = cluster.analytical_evaluation
    warned = replace(
        cluster, analytical_evaluation=replace(evaluation, uncertainty_warning="synthetic")
    )
    parent = _with_outcome(result, [warned if c is cluster else c for c in result.outcome.clusters])
    resolved = _compare(spectra, result, parent)
    (change,) = resolved.clusters
    assert change.lost == ("▲ Uncertainty unreliable",)
    assert change.gained == ()
    appeared = _compare(spectra, parent, result)
    assert appeared.clusters[0].gained == ("▲ Uncertainty unreliable",)


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
    assert changes.unjudged == len(unusable.peak_names)
    engines = {i.engine_name: i.working_id for i in result.inputs.identities}
    excluded = {engines[name] for name in unusable.peak_names}
    assert not excluded & {shift.working_id for shift in changes.shifts}
