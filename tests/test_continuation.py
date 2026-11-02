"""A refit continues unchanged peaks from the active run's complete fitted state.

See docs/validation/repeat-fit-collapse.md: adopting fitted positions while
restarting every linewidth at its default produced starts that matched no
solution, and on the bundled data collapsed the resolved pair 52N-H/102N-H.
"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from test_gui_dataset import _dataset

from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.domain.constraints import ParameterConfig, apply_constraints
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.fit_positions import continuation_source, dataset_completion
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.spectrum import read_dataset

DATA = Path(__file__).resolve().parents[1] / "examples/02-advanced-fitting/data"


def _fit(spectra, peaks, settings, completion=None):
    source = continuation_source(completion) if completion is not None else None
    prepared = prepare_dataset_fit(spectra, tuple(peaks), settings, source)
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return dataset_completion(
        result, result.inputs.snapshot, submitted_revision=0, manual_revision=0
    )


def _provenance(prepared):
    continuation = prepared.continuation
    assert continuation is not None
    return continuation


def _by_members(result):
    names = {i.engine_name: i.assignment for i in result.inputs.identities}
    return {
        tuple(sorted(names[p] for p in cluster.peak_names)): cluster
        for cluster in result.outcome.clusters
    }


def _fitted(result, identity):
    engine = next(i.engine_name for i in result.inputs.identities if i.working_id == identity)
    return {
        p.name.partition(".")[2]: p.value
        for c in result.outcome.clusters
        for p in c.final_nonlinear_parameters
        if p.name.partition(".")[0] == engine
    }


@pytest.fixture(scope="module")
def synthetic():
    spectra, peaks, config, _prepared = _dataset()
    return spectra, peaks, config, _fit(spectra, peaks, config)


@pytest.fixture(scope="module")
def bundled():
    dataset = read_dataset(DATA / "pseudo3d.ft2", DATA / "pseudo3d.list", DATA / "b1_offsets.txt")
    settings = PeakFitConfig(fitting=FitConfig(refine_iterations=2))
    first = _fit(dataset.spectra, dataset.peaks, settings)
    return dataset, settings, first


@pytest.mark.slow
def test_unedited_refit_continues_every_adopted_peak_of_the_bundled_run(bundled):
    dataset, settings, first = bundled
    second = _fit(dataset.spectra, first.working_peaks, settings, first)
    continuation = _provenance(second.result.inputs)
    assert continuation.run_id == first.result.run_id
    # Every adopted peak continues; peaks whose positions the run retained
    # (near-collinear fitted shapes, tests/test_coincidence.py) start from defaults.
    assert set(continuation.continued) == set(first.adopted)
    assert set(continuation.initialized) == {i for i, _ in first.retained}
    before, after = _by_members(first.result), _by_members(second.result)
    assert set(after) == set(before)
    retained = {i for i, _ in first.retained}
    names = {p.id: p.name for p in dataset.peaks}
    for members, cluster in after.items():
        if retained & {i for i, n in names.items() if n in members}:
            continue
        # No cluster of continued peaks gains a warning.
        assert bool(cluster.analytical_evaluation.uncertainty_warning) <= bool(
            before[members].analytical_evaluation.uncertainty_warning
        ), members
    pair = after[("102N-H", "52N-H")].analytical_evaluation
    assert pair.uncertainty_warning is None
    assert pair.statistics.reduced_chi_squared == pytest.approx(1.5296, abs=1e-4)


@pytest.mark.slow
def test_positions_only_refit_is_the_collapse_continuation_prevents(bundled):
    """The defect itself: adopted positions with default widths collapse the pair."""
    dataset, settings, first = bundled
    second = _fit(dataset.spectra, first.working_peaks, settings)
    pair = _by_members(second.result)[("102N-H", "52N-H")].analytical_evaluation
    assert pair.uncertainty_warning is not None
    assert pair.statistics.reduced_chi_squared == pytest.approx(2.3154, abs=1e-4)


def test_identical_prepared_inputs_give_identical_outcomes(synthetic):
    spectra, _peaks, config, first = synthetic
    source = continuation_source(first)
    prepared = prepare_dataset_fit(spectra, first.working_peaks, config, source)
    a = compute_dataset_fit(deepcopy(prepared)).completed_fit.outcome
    b = compute_dataset_fit(deepcopy(prepared)).completed_fit.outcome
    assert [c.final_nonlinear_parameters for c in a.clusters] == [
        c.final_nonlinear_parameters for c in b.clusters
    ]


def test_unchanged_peaks_continue_and_edited_or_new_peaks_start_from_defaults(synthetic):
    spectra, _peaks, config, first = synthetic
    defaults = prepare_dataset_fit(spectra, first.working_peaks, config)
    moved, kept, renamed = first.working_peaks[0], first.working_peaks[1], first.working_peaks[2]
    peaks = (
        replace(moved, positions=(moved.positions[0] + 0.01, moved.positions[1])),
        kept,
        replace(renamed, name="Renamed"),
        *first.working_peaks[3:],
    )
    added = replace(kept, id="added", name="Added")
    prepared = prepare_dataset_fit(spectra, (*peaks, added), config, continuation_source(first))
    continuation = _provenance(prepared)
    assert continuation.run_id == first.result.run_id
    assert set(continuation.initialized) == {moved.id, "added"}
    assert set(continuation.continued) == {p.id for p in first.working_peaks} - {moved.id}
    engine = {i.working_id: i.engine_name for i in prepared.identities}
    # A continued peak starts from its fitted width, mapped by stable ID;
    # the moved and the new peak keep the configured default width.
    for identity in (kept.id, renamed.id):
        for suffix, value in _fitted(first.result, identity).items():
            if prepared.parameters[f"{engine[identity]}.{suffix}"].vary:
                assert prepared.parameters[f"{engine[identity]}.{suffix}"].value == value
    default = {i.working_id: i.engine_name for i in defaults.identities}
    for identity in (moved.id, "added"):
        lw = f"{engine[identity]}.F2.lw"
        reference = f"{default[kept.id]}.F2.lw"
        assert prepared.parameters[lw].value == defaults.parameters[reference].value


def test_an_incompatible_model_never_inherits_fitted_values(synthetic):
    spectra, _peaks, config, first = synthetic
    other = config.model_copy(deep=True)
    other.fitting.lineshape = "gaussian" if config.fitting.lineshape != "gaussian" else "lorentzian"
    prepared = prepare_dataset_fit(spectra, first.working_peaks, other, continuation_source(first))
    assert _provenance(prepared).continued == ()
    assert _provenance(prepared).reason == "lineshape models differ from that run"
    fresh = prepare_dataset_fit(spectra, first.working_peaks, other)
    assert [(p.name, p.value) for p in prepared.parameters.values()] == [
        (p.name, p.value) for p in fresh.parameters.values()
    ]


def test_first_fit_records_that_every_peak_started_from_defaults(synthetic):
    spectra, peaks, config, _first = synthetic
    prepared = prepare_dataset_fit(spectra, peaks, config)
    continuation = _provenance(prepared)
    assert continuation.run_id is None
    assert continuation.continued == ()
    assert continuation.initialized == tuple(p.id for p in peaks)


def test_explicit_configured_starts_take_precedence_and_are_the_recorded_starts(synthetic):
    spectra, peaks, config, _first = synthetic
    settings = config.model_copy(deep=True)
    settings.parameters = ParameterConfig.model_validate({"defaults": {"*.*.lw": {"value": 15.0}}})
    first = _fit(spectra, peaks, settings)
    prepared = prepare_dataset_fit(
        spectra, first.working_peaks, settings, continuation_source(first)
    )
    assert _provenance(prepared).continued == tuple(p.id for p in first.working_peaks)
    engine = {i.working_id: i.engine_name for i in prepared.identities}
    for peak in first.working_peaks:
        fitted = _fitted(first.result, peak.id)
        for axis in ("F2", "F3"):
            # The configured start wins over the fitted width; the position continues.
            assert prepared.parameters[f"{engine[peak.id]}.{axis}.lw"].value == 15.0
            assert prepared.parameters[f"{engine[peak.id]}.{axis}.cs"].value == fitted[f"{axis}.cs"]
    # What is recorded is what the pipeline starts from after its own constraint pass.
    resolved = apply_constraints(prepared.parameters.copy(deep=True), settings.parameters)
    assert [(p.name, p.value) for p in resolved.values()] == [
        (p.name, p.value) for p in prepared.parameters.values()
    ]
