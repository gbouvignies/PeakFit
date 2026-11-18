"""The Run page compares the active run with any retained run, and changes nothing.

Four real runs of the two-cluster dataset form one refit chain: run 2 changes the
lineshape family, run 3 starts from a peak moved by hand, and before run 4 another
peak was moved and moved back, so run 4 refits unchanged. A failed attempt between
runs 2 and 3 is journaled but never numbered.
"""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import pytest
from builders import two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtWidgets

from peakfit.engine.domain.config import AxisModel
from peakfit.engine.domain.constraints import ParameterConfig
from peakfit.gui import formatting, project
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.axis_table import AxisInfo
from peakfit.gui.editor.changes import compare_runs, edited_since
from peakfit.gui.editor.model_page import Setting, setting_differences
from peakfit.gui.editor.review import NumericalWarning, review_clusters
from peakfit.gui.editor.window import EditorWindow
from peakfit.gui.editor.workspace import _change_entries
from peakfit.gui.fit_positions import continuation_source, dataset_completion
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.peaks import ParameterLimit
from peakfit.gui.project import Analysis, dataset_header, open_analysis, save_analysis
from peakfit.gui.project_results import journal_entry, run_numbers
from peakfit.gui.spectrum import Dataset

pytestmark = pytest.mark.gui

MOVED = "remote"  # moved by hand before run 3
FITTED = "0"  # never moved by hand; its recorded position changes with every fit


def _fit(spectra, peaks, settings, completion=None):
    source = continuation_source(completion) if completion is not None else None
    prepared = prepare_dataset_fit(spectra, tuple(peaks), settings, source)
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return dataset_completion(
        result, result.inputs.snapshot, submitted_revision=0, manual_revision=0
    )


def _lorentzian(config):
    settings = config.model_copy(deep=True)
    settings.fitting.family = "lorentzian"
    return settings


@pytest.fixture(scope="module")
def chain():
    spectra, peaks, config, _prepared = two_cluster_dataset()
    first = _fit(spectra, peaks, config)
    second = _fit(spectra, first.working_peaks, _lorentzian(config), first)
    moved = tuple(
        replace(p, positions=(p.positions[0] + 0.01, p.positions[1])) if p.id == MOVED else p
        for p in second.working_peaks
    )
    third = _fit(spectra, moved, _lorentzian(config), second)
    # An intermediate edit, reverted before the fit: the working peaks are run 3's again.
    edit = tuple(
        replace(p, positions=(p.positions[0], p.positions[1] + 0.02)) if p.id == FITTED else p
        for p in third.working_peaks
    )
    reverted = tuple(
        next(q for q in third.working_peaks if q.id == p.id) if p.id == FITTED else p for p in edit
    )
    assert edit != third.working_peaks
    assert reverted == third.working_peaks
    fourth = _fit(spectra, reverted, _lorentzian(config), third)
    runs = (first, second, third, fourth)
    journal = []
    parent = None
    for index, run in enumerate(runs):
        if index == 2:  # a failed attempt from run 2, before run 3 completed
            journal.append(
                {
                    "run_id": "f" * 32,
                    "parent_run_id": parent,
                    "status": "failed",
                    "configuration": config.model_dump_json(),
                    "parameters": [],
                }
            )
        journal.append(dict(journal_entry(run), parent_run_id=parent))
        parent = run.result.run_id
    return spectra, config, runs, tuple(journal)


def _titles(spectra):
    return [f"axis {i}" for i in range(len(spectra.spectral_params))]


def _reviews(spectra, result):
    return review_clusters(result, list(spectra.spectral_params), _titles(spectra))


def _compare(spectra, active, baseline, journal=None, active_reviews=None, base_reviews=None):
    edited = edited_since(journal, active.run_id, baseline.run_id) if journal is not None else None
    return compare_runs(
        active,
        _reviews(spectra, active) if active_reviews is None else active_reviews,
        baseline,
        _reviews(spectra, baseline) if base_reviews is None else base_reviews,
        list(spectra.spectral_params),
        _titles(spectra),
        edited=edited,
    )


# ----------------------------------------------------------------- run labels


def test_runs_are_numbered_in_journal_order_and_failed_attempts_are_not(chain):
    _spectra, _config, runs, journal = chain
    numbers = run_numbers(journal)
    assert numbers == {run.result.run_id: index for index, run in enumerate(runs, 1)}
    assert "f" * 32 not in numbers


# ----------------------------------------------------------- edits and settings


def test_edits_are_those_recorded_by_refits_after_the_baseline(chain):
    _spectra, _config, runs, journal = chain
    ids = [run.result.run_id for run in runs]
    assert edited_since(journal, ids[3], ids[2]) == frozenset()  # the parent: none
    assert edited_since(journal, ids[3], ids[1]) == {MOVED}  # run 3 started from it
    assert edited_since(journal, ids[3], ids[0]) == {MOVED}
    # Not an earlier run of the active one: no edit can be traced, none is claimed.
    assert edited_since(journal, ids[1], ids[3]) is None
    # Peak 0, moved and moved back before run 4, differs between runs 2 and 4 only by
    # fitting: it is never "edited", though its recorded positions differ.
    position = lambda run: next(p for p in run.working_peaks if p.id == FITTED).positions  # noqa: E731
    assert position(runs[1]) != position(runs[3])
    assert FITTED not in (edited_since(journal, ids[3], ids[1]) or {FITTED})


def test_a_peak_moved_back_where_the_baseline_left_it_is_not_edited(chain):
    # Run B starts from a peak moved by hand; before run C it is moved back by hand to
    # where run A left it. Both refits record the move, yet against run A it is not
    # edited: run C started it exactly where a continuation from run A would have.
    spectra, config, runs, _journal = chain
    first = runs[0]
    moved = tuple(
        replace(p, positions=(p.positions[0] + 0.01, p.positions[1])) if p.id == MOVED else p
        for p in first.working_peaks
    )
    second = _fit(spectra, moved, config, first)
    left = next(p for p in first.working_peaks if p.id == MOVED)
    back = tuple(left if p.id == MOVED else p for p in second.working_peaks)
    third = _fit(spectra, back, config, second)
    reasons = dict(third.result.inputs.continuation.reasons)
    assert reasons[MOVED].startswith("moved since run")
    journal = []
    parent = None
    for run in (first, second, third):
        journal.append(dict(journal_entry(run), parent_run_id=parent))
        parent = run.result.run_id
    a, b, c = (run.result.run_id for run in (first, second, third))
    assert edited_since(journal, c, a) == frozenset()
    assert edited_since(journal, c, b) == {MOVED}
    assert edited_since(journal, b, a) == {MOVED}
    # Also after a save: positions read back from JSON lists compare the same.
    reread = json.loads(json.dumps(journal))
    assert edited_since(reread, c, a) == frozenset()


def test_a_peak_given_its_own_constraints_is_edited_until_they_are_set_back(chain):
    # Run B starts from a peak given a linewidth range of its own; before run C the
    # range is removed again. Against run A the peak is edited in B, not in C.
    spectra, config, runs, _journal = chain
    first = runs[0]
    limits = (("F2.lw", ParameterLimit(lower=1.0, upper=100.0)),)
    constrained = tuple(
        replace(p, limits=limits) if p.id == FITTED else p for p in first.working_peaks
    )
    second = _fit(spectra, constrained, config, first)
    released = tuple(replace(p, limits=()) for p in second.working_peaks)
    third = _fit(spectra, released, config, second)
    assert dict(second.result.inputs.continuation.reasons)[FITTED].startswith(
        "constraints changed since run"
    )
    journal = []
    parent = None
    for run in (first, second, third):
        journal.append(dict(journal_entry(run), parent_run_id=parent))
        parent = run.result.run_id
    a, b, c = (run.result.run_id for run in (first, second, third))
    assert edited_since(journal, b, a) == {FITTED}
    assert edited_since(journal, c, b) == {FITTED}
    assert edited_since(journal, c, a) == frozenset()
    reread = json.loads(json.dumps(journal))  # limits read back from JSON lists
    assert edited_since(reread, b, a) == {FITTED}
    assert edited_since(reread, c, a) == frozenset()


def test_a_non_parent_comparison_labels_only_recorded_edits(chain):
    spectra, _config, runs, journal = chain
    active, baseline = runs[3].result, runs[1].result
    changes = _compare(spectra, active, baseline, journal)
    edited = {name for change in changes.changes for name in change.edited}
    assert edited == {next(p.name for p in runs[3].working_peaks if p.id == MOVED)}
    # Without the journal's chain, run 4's own record names no edit against run 2.
    assert not any(c.edited for c in _compare(spectra, active, baseline).changes)


def test_settings_differences_read_each_runs_recorded_settings(chain):
    spectra, config, runs, _journal = chain
    axes = [
        AxisInfo(f"F{i + 2}", formatting.axis_name(a), "gaussian")
        for i, a in enumerate(spectra.spectral_params)
    ]
    settings = [run.result.inputs.snapshot.configuration for run in runs]
    read = [config.model_validate_json(s) for s in settings]
    assert setting_differences(read[3], read[1], axes, {}) == ({}, {})
    by_axis, rows = setting_differences(read[3], read[0], axes, {})
    assert rows == {}
    assert set(by_axis) == {a.label for a in axes}
    assert all(pair == ("Gaussian", "Lorentzian") for pair in by_axis.values())
    changed = read[3].model_copy(deep=True)
    changed.clustering.contour_level = None
    changed.clustering.contour_factor = 8.0
    changed.exclude_planes = [1]
    _axes, rows = setting_differences(changed, read[3], axes, {})
    assert rows[Setting.THRESHOLD] == ("level 0.1", "8 × noise")
    assert rows[Setting.PLANES] == ("every plane", "planes 2 excluded")


# --------------------------------------------------------------- comparability


def _first_change(spectra, active, baseline):
    """The first cluster's change, made visible by a warning only the baseline had."""
    reviews = _reviews(spectra, baseline)
    warned = [
        replace(reviews[0], warnings=(NumericalWarning("Uncertainty unreliable", ""),)),
        *reviews[1:],
    ]
    changes = _compare(spectra, active, baseline, base_reviews=warned)
    return next(c for c in changes.changes if set(c.ids) == set(reviews[0].ids))


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda s: setattr(s, "exclude_planes", [1]), "fitted planes changed"),
        (lambda s: setattr(s, "noise_level", 2 * s.noise_level), "σ_stat changed"),
        (lambda s: setattr(s.fitting, "family", "lorentzian"), "lineshape model changed"),
        (
            lambda s: setattr(s.fitting, "axes", {"F2": AxisModel(family="lorentzian")}),
            "lineshape model changed",
        ),
        (
            lambda s: setattr(
                s,
                "parameters",
                ParameterConfig.model_validate({"defaults": {"*.F2.lw": {"max": 200.0}}}),
            ),
            "constraints changed",
        ),
    ],
)
def test_measures_of_differently_defined_runs_are_not_compared(chain, change, reason):
    spectra, config, runs, _journal = chain
    baseline = runs[0]
    settings = config.model_copy(deep=True)
    change(settings)
    other = _fit(spectra, baseline.result.inputs.snapshot.peaks, settings)
    found = _first_change(spectra, other.result, baseline.result)
    assert found.measures is None
    assert found.not_compared == reason
    detail = "\n".join(e[2] for e in _change_entries_for(found) if e[0] == "link")
    assert f"Measures not compared: {reason}" in detail
    assert "χ²ᵣ" not in detail


def _change_entries_for(change):
    from peakfit.gui.editor.changes import RunChanges  # noqa: PLC0415

    return _change_entries(RunChanges("x", (1, 1), (0, 0), 0, (change,), (), 0, 0, 0))


def test_a_different_threshold_compares_no_measure_on_different_samples(chain):
    spectra, config, runs, _journal = chain
    baseline = runs[0]
    settings = config.model_copy(deep=True)
    settings.clustering.contour_level = 0.03
    other = _fit(spectra, baseline.result.inputs.snapshot.peaks, settings)
    found = _first_change(spectra, other.result, baseline.result)
    assert found.measures is None
    assert found.not_compared.startswith(("fitting samples changed", "members changed"))


# ------------------------------------------------------- numerical correctness


def test_each_shown_difference_is_the_difference_of_the_recorded_values(chain):
    spectra, _config, runs, journal = chain
    active, baseline = runs[3].result, runs[0].result
    changes = _compare(spectra, active, baseline, journal)
    assert changes.displacements
    for move in changes.displacements:
        engine = {i.working_id: i.engine_name for i in active.inputs.identities}[move.working_id]
        old_engine = {i.working_id: i.engine_name for i in baseline.inputs.identities}[
            move.working_id
        ]
        label = next(f"F{i + 2}" for i, t in enumerate(_titles(spectra)) if t == move.axis)
        now = active.index.fitted(engine, label, "cs", usable=True).value
        before = baseline.index.fitted(old_engine, label, "cs", usable=True).value
        assert move.delta == now - before
    same = runs[3].result
    found = _first_change(spectra, same, same)
    before, after = found.measures
    review = _reviews(spectra, same)[0]
    assert (before.chi2, after.chi2) == (review.chi2, review.chi2)


# ------------------------------------------------- the Run page, saved and reopened


def _analysis(chain, tmp_path):
    spectra, _config, runs, journal = chain
    source = tmp_path / "source.ft2"
    source.write_bytes(b"External spectrum stand-in; the reader is mocked")
    stat = source.stat()
    record = json.dumps(
        {
            "path": str(source),
            "checksum_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "size": str(stat.st_size),
            "mtime_ns": str(stat.st_mtime_ns),
            "ctime_ns": str(stat.st_ctime_ns),
            "inode": str(stat.st_ino),
            "device": str(stat.st_dev),
        }
    )
    sources = (("spectrum", record),)

    def linked(completion):
        result = completion.result
        snapshot = replace(result.inputs.snapshot, source_files=sources)
        return replace(
            completion, result=replace(result, inputs=replace(result.inputs, snapshot=snapshot))
        )

    runs = tuple(linked(run) for run in runs)
    journal = tuple(
        dict(e, source_files=sources) if e.get("status") == "completed" else e for e in journal
    )
    last = runs[-1]
    dataset = Dataset(source, spectra, last.working_peaks, sources, True, 0.01)
    return Analysis(
        dataset,
        dataset_header(dataset),
        last.working_peaks,
        last.result.inputs.snapshot.configuration,
        0,
        runs,
        {"plane": 1},
        journal=journal,
    )


@pytest.fixture
def window(qapp, chain, tmp_path, monkeypatch):
    analysis = _analysis(chain, tmp_path)
    # Three full runs: run 1 is kept as a summary only.
    monkeypatch.setattr(project, "MAX_RUNS", 3)
    path = save_analysis(analysis, tmp_path / "chain.peakfit")
    monkeypatch.setattr(project, "read_dataset", lambda _, **_kw: analysis.dataset)
    opened = []

    def reopen(where=path):
        loaded = open_analysis(where)
        assert loaded.dataset is not None, loaded.reason
        shown = EditorWindow()
        shown.install_analysis(loaded)
        shown.resize(1280, 800)
        shown.show()
        opened.append(shown)
        workspace = shown.workspace
        assert workspace is not None
        workspace.select_peak(None)
        workspace.render_fitting()
        return shown

    yield reopen
    for shown in opened:
        shown._saved_state = shown._analysis_state()
        shown.close()
    qapp.processEvents()


def _state(window):
    """Everything a baseline choice must leave untouched, hashed."""
    w = window.workspace
    fitting = w.fitting
    state = {
        **w.analysis_state(),
        "current": bool(fitting.current),
        "undo": (w.edits.can_undo, w.edits.can_redo),
        "modified": window._analysis_state() != window._saved_state,
    }
    return hashlib.sha256(repr(sorted(state.items(), key=str)).encode()).hexdigest()


def _texts(window):
    page = window.workspace.panel.analysis_page
    return [label.text() for label in page.findChildren(QtWidgets.QLabel) if label.isVisible()]


def _changes(window):
    """The texts of the changes section only: the baseline, settings and changes."""
    found = []

    def walk(layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() is not None:
                found.extend(
                    w.text()
                    for w in [item.widget(), *item.widget().findChildren(QtWidgets.QLabel)]
                    if isinstance(w, (QtWidgets.QLabel, QtWidgets.QPushButton))
                )
            elif item.layout() is not None:
                walk(item.layout())

    walk(window.workspace.panel.run_page.changes)
    return found


def _row(window, label):
    from test_gui_editor_fitting import _review_row  # noqa: PLC0415

    return _review_row(window.workspace.panel.run_page.changes.itemAt(0).layout(), label)


def _choose(window, number):
    menu = window.workspace.panel.run_page.baseline
    index = next(i for i, c in enumerate(menu.choices) if c[0] == f"Run {number}")
    menu.setCurrentIndex(index)
    fitting = window.workspace.fitting
    wait_until(lambda: not any(v == "loading" for v in fitting._runs_read.values()))
    window.workspace.render_fitting()
    QtWidgets.QApplication.processEvents()  # rows added to a shown page show on the next pass


def test_the_parent_is_the_default_baseline_and_runs_keep_their_numbers(chain, window):
    _spectra, _config, runs, _journal = chain
    shown = window()
    page = shown.workspace.panel.run_page
    ids = [run.result.run_id for run in runs]
    wait_until(lambda: f"{ids[2][:8]} · parent" in _texts(shown))
    assert page.baseline.text().startswith("Run 3")
    assert [c[0] for c in page.baseline.choices] == ["Run 3", "Run 2", "Run 1"]
    menu = [c[1] for c in page.baseline.choices]
    assert menu[0].endswith("· parent")
    assert menu[2].endswith("· summary\u00a0only")
    assert _row(shown, "Baseline") == f"{ids[2][:8]} · parent"
    from test_gui_editor_fitting import _review_row  # noqa: PLC0415

    assert _review_row(page.run_rows, "Run").startswith(f"4 · {ids[3][:8]} · ")
    assert _row(shown, "Settings") == "identical"


def test_choosing_a_baseline_changes_nothing_and_survives_save_and_reopen(chain, window, tmp_path):
    _spectra, _config, runs, _journal = chain
    shown = window()
    w = shown.workspace
    ids = [run.result.run_id for run in runs]
    before = _state(shown)
    # Run 2: full evidence, settings identical, the hand-moved peak listed as edited.
    _choose(shown, 2)
    assert _state(shown) == before
    assert _row(shown, "Baseline") == ids[1][:8]
    texts = _texts(shown)
    assert _row(shown, "Settings") == "identical"
    moved = next(p.name for p in w.peaks if p.id == MOVED)
    assert any(f"Edited: {moved}" in t for t in texts)
    assert not any("Edited: " in t and moved not in t for t in texts)
    # Run 1: kept as a summary; its settings compare, nothing else is reconstructed.
    _choose(shown, 1)
    assert _state(shown) == before
    texts = _texts(shown)
    assert _row(shown, "Baseline") == f"{ids[0][:8]} · summary\u00a0only"
    assert any("Gaussian → Lorentzian" in t for t in texts)
    assert any(t.startswith("Run 1 is kept as a summary") for t in texts)
    assert not any("χ²ᵣ" in t or "ppm ·" in t for t in texts)
    rows = shown.workspace.panel.run_page.changes
    labels = [
        rows.itemAt(i).layout().itemAtPosition(r, 0).widget().text()
        for i in range(rows.count())
        if isinstance(rows.itemAt(i).layout(), QtWidgets.QGridLayout)
        for r in range(rows.itemAt(i).layout().rowCount())
        if rows.itemAt(i).layout().itemAtPosition(r, 0) is not None
    ]
    assert "With ▲ warnings" not in labels
    assert "Clusters" in labels
    # The choice is not analysis state: saving records none of it.
    saved = save_analysis(w.capture(), tmp_path / "again.peakfit")
    assert _state(shown) == before
    again = window(saved)
    assert _state(again) == before
    page = again.workspace.panel.run_page
    wait_until(lambda: page.baseline.text().startswith("Run 3"))
    assert [c[0] for c in page.baseline.choices] == ["Run 3", "Run 2", "Run 1"]
    assert _row(again, "Baseline") == f"{ids[2][:8]} · parent"
    # A working edit, fitted or not, is not part of either record: the comparison of
    # run 4 with run 2 is the same with peak 0 moved, and after Undo.
    _choose(again, 2)
    assert _state(again) == before
    compared = _changes(again)
    w = again.workspace
    fitted = next(p for p in w.peaks if p.id == FITTED)
    w.move_peak(fitted.id, (fitted.positions[0], fitted.positions[1] + 0.02))
    assert not w.fitting.current
    _choose(again, 2)
    assert _changes(again) == compared
    w.history(False)  # Undo restores the position; it never recertifies the run
    _choose(again, 2)
    assert _changes(again) == compared
