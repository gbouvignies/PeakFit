"""The next fit milestone's run-owned I → W contract, without GUI fitting features."""

from dataclasses import replace

import pytest
from test_gui_dataset import _dataset, _example

from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.fit_positions import dataset_completion
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.peaks import PeakEdits


@pytest.fixture(scope="module")
def completed():
    _, _, _, prepared = _dataset()
    return finish_dataset_fit(prepared, compute_dataset_fit(prepared))


def test_atomic_transition_preserves_seeds_outcome_and_currentness(completed):
    submitted = completed.inputs.snapshot
    initial = submitted.peaks
    transition = dataset_completion(completed, submitted, submitted_revision=7, manual_revision=7)
    assert transition.result is completed
    assert transition.result.inputs.snapshot.peaks == initial
    assert transition.working_peaks != initial
    post = replace(submitted, peaks=transition.working_peaks)
    assert transition.is_current(post, 7)
    assert not transition.is_current(submitted, 7)
    edits = PeakEdits(transition.working_peaks, revision=transition.manual_revision)
    assert transition.is_current(post, edits.revision)
    peak = edits.peaks[0]
    edits.update(peak.id, peak.name, (peak.positions[0] + 1e-6, peak.positions[1]))
    edits.undo()
    assert edits.peaks == transition.working_peaks
    assert not transition.is_current(post, edits.revision)
    assert not transition.is_current(replace(post, source_files=(("spectrum", "changed"),)), 7)
    assert [p.id for p in initial] == [p.id for p in transition.working_peaks]
    assert [p.name for p in initial] == [p.name for p in transition.working_peaks]
    for cluster in completed.outcome.clusters:
        assert not cluster.analytical_evaluation.model_values.flags.writeable


def changed_clusters(result, clusters):
    outcome = replace(
        result.outcome, clusters=tuple(clusters), by_cluster_id={c.cluster_id: c for c in clusters}
    )
    return replace(result, completed_fit=replace(result.completed_fit, outcome=outcome))


def test_usable_nonconvergence_and_uncertainty_do_not_block_positions(completed):
    clusters = [
        replace(
            c,
            classification=FitOutcomeClassification.USABLE_NON_CONVERGED,
            optimizer_provenance=replace(c.optimizer_provenance, converged=False),
            analytical_evaluation=replace(
                c.analytical_evaluation, uncertainty_warning="Unidentifiable"
            ),
        )
        for c in completed.outcome.clusters
    ]
    warning_result = changed_clusters(completed, clusters)
    expected = dataset_completion(
        completed, completed.inputs.snapshot, submitted_revision=0, manual_revision=0
    )
    actual = dataset_completion(
        warning_result, completed.inputs.snapshot, submitted_revision=0, manual_revision=0
    )
    assert actual.working_peaks == expected.working_peaks
    assert actual.result.outcome.clusters[0].analytical_evaluation is not None
    assert (
        actual.result.outcome.clusters[0].analytical_evaluation.uncertainty_warning
        == "Unidentifiable"
    )


def test_mixed_and_all_unusable_keep_whole_pairs(completed):
    first, *others = completed.outcome.clusters
    unusable = replace(
        first, classification=FitOutcomeClassification.UNUSABLE, analytical_evaluation=None
    )
    result = changed_clusters(completed, [unusable, *others])
    transition = dataset_completion(
        result, result.inputs.snapshot, submitted_revision=0, manual_revision=0
    )
    first_ids = {
        i.working_id for i in result.inputs.identities if i.engine_name in first.peak_names
    }
    for before, after in zip(result.inputs.snapshot.peaks, transition.working_peaks, strict=True):
        if before.id in first_ids:
            assert after is before
    assert len(transition.warnings) == len(first_ids)
    all_unusable = changed_clusters(
        completed,
        [
            replace(c, classification=FitOutcomeClassification.UNUSABLE, analytical_evaluation=None)
            for c in completed.outcome.clusters
        ],
    )
    all_transition = dataset_completion(
        all_unusable, all_unusable.inputs.snapshot, submitted_revision=0, manual_revision=0
    )
    assert all_transition.working_peaks == completed.inputs.snapshot.peaks
    assert all_transition.is_current(completed.inputs.snapshot, 0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1e6])
def test_invalid_axis_retains_both_coordinates(completed, bad):
    first, *others = completed.outcome.clusters
    target = next(p for p in first.final_nonlinear_parameters if p.name.endswith(".cs"))
    parameters = tuple(
        replace(p, value=bad) if p.name == target.name else p
        for p in first.final_nonlinear_parameters
    )
    result = changed_clusters(
        completed, [replace(first, final_nonlinear_parameters=parameters), *others]
    )
    transition = dataset_completion(
        result, result.inputs.snapshot, submitted_revision=0, manual_revision=0
    )
    identity = next(
        i.working_id
        for i in result.inputs.identities
        if target.name.startswith(i.engine_name + ".")
    )
    before = next(p for p in result.inputs.snapshot.peaks if p.id == identity)
    assert next(p for p in transition.working_peaks if p.id == identity) is before
    assert any(identity in warning for warning in transition.warnings)


def test_fatal_revision_source_and_identity_mismatch_publish_nothing(completed):
    inputs = completed.inputs.snapshot
    for current, revision in ((inputs, 1), (replace(inputs, configuration="changed"), 0)):
        with pytest.raises(ValueError, match="changed"):
            dataset_completion(completed, current, submitted_revision=0, manual_revision=revision)
    broken = replace(
        completed, inputs=replace(completed.inputs, identities=completed.inputs.identities[:-1])
    )
    with pytest.raises(ValueError, match="mapping"):
        dataset_completion(broken, inputs, submitted_revision=0, manual_revision=0)


def test_custom_axis_labels_follow_coordinate_order():
    spectra, working, config = _example()
    # Labels deliberately sort in the opposite order to working coordinates.
    spectra.params[1].label = "F3"
    spectra.params[2].label = "F2"
    config.clustering.contour_level = 0.01
    prepared = prepare_dataset_fit(spectra, working, config)
    result = finish_dataset_fit(prepared, compute_dataset_fit(prepared))
    submitted = result.inputs.snapshot
    transition = dataset_completion(result, submitted, submitted_revision=0, manual_revision=0)
    assert not transition.warnings
    final = {p.name: p.value for p in result.outcome.final_nonlinear_parameters}
    adopted = {p.id: p.positions for p in transition.working_peaks}
    for identity in prepared.identities:
        assert adopted[identity.working_id] == (
            final[f"{identity.engine_name}.F3.cs"],
            final[f"{identity.engine_name}.F2.cs"],
        )
