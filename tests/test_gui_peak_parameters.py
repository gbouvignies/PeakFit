"""The Peak page's Parameters: one peak's own starts, bounds, windows and Hold.

A peak's own limit is kept on its working peak, committed as one undoable edit, marked
against the active run, and fitted as ``[parameters.peaks."<name>"]`` is by
``peakfit fit``; an invalid value is explained under its parameter and blocks Fit.
"""

import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from builders import two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtWidgets

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.constraints import ParameterConstraint
from peakfit.gui import theme as themes
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.axis_table import DIFFERS
from peakfit.gui.editor.peak_parameters import NOTE, ParameterRow, PeakParameters
from peakfit.gui.editor.run_page import submitted_rows
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.fitting import PeakIdentity, engine_configuration
from peakfit.gui.peaks import ParameterLimit, WorkingEdits, WorkingPeak
from peakfit.gui.project_results import run_numbers
from peakfit.gui.spectrum import Dataset
from peakfit.io.config import load_config

pytestmark = pytest.mark.usefixtures("qapp")
INSPECTOR_CONTENT = 288  # logical px: a 320 px inspector less its margins at 1280×800
ROWS = [
    ParameterRow("F3.cs", "δ ¹H F3", "ppm", 8.25, 8.21, 8.29, True, 0.04),
    ParameterRow("F2.cs", "δ ¹⁵N F2", "ppm", 120.5, 120.46, 120.54, True, 0.04),
    ParameterRow("F3.lw", "LW ¹H F3", "Hz", 25.0, 0.1, 200.0, True),
    ParameterRow("F2.lw", "LW ¹⁵N F2", "Hz", 25.0, 0.1, 200.0, False),  # held by a rule
]


_WINDOWS: list[QtWidgets.QWidget] = []


def _editor(peak: WorkingPeak, rows=ROWS, recorded=None):
    """The section in a themed window, following its own commits as the workspace does."""
    window = QtWidgets.QWidget()
    themes.apply(window, themes.LIGHT)
    layout = QtWidgets.QVBoxLayout(window)
    editor = PeakParameters()
    layout.addWidget(editor)
    shown = [peak]

    def follow(identity, limits):
        assert identity == shown[-1].id
        shown.append(replace(shown[-1], limits=limits))
        editor.show_peak(shown[-1], rows, NOTE, recorded)

    editor.changed.connect(follow)
    editor.show_peak(peak, rows, NOTE, recorded)
    window.resize(320, 700)
    window.show()
    _WINDOWS.append(window)  # alive for the test
    return editor, shown


def _row(editor: PeakParameters, key: str):
    return next(row for row in editor.rows if row.key == key)


def _type(edit: QtWidgets.QLineEdit, text: str) -> None:
    edit.setText(text)
    edit.editingFinished.emit()


PEAK = WorkingPeak("a1", "G45N-HN", (120.5, 8.25))


def _range(editor):
    row = _row(editor, "F3.lw")
    _type(row.low, "5")
    _type(row.high, "50")


def _start_and_hold(editor):
    row = _row(editor, "F3.lw")
    _type(row.start, "1.5e1")
    row.hold.setChecked(True)


def _window(editor):
    _type(_row(editor, "F2.cs").window_edit, "0.05")


def _fit_against_a_held_rule(editor):
    _row(editor, "F2.lw").hold.setChecked(False)


def _hold_position(editor):
    _row(editor, "F3.cs").hold.setChecked(True)


@pytest.mark.parametrize(
    ("edit", "toml"),
    [
        pytest.param(_range, '"F3.lw" = { min = 5, max = 50 }', id="range"),
        pytest.param(_start_and_hold, '"F3.lw" = { value = 15, vary = false }', id="start-hold"),
        pytest.param(_window, "position_windows = { F2 = 0.05 }", id="window"),
        pytest.param(_fit_against_a_held_rule, '"F2.lw" = { vary = true }', id="fit-held"),
        pytest.param(_hold_position, '"F3.cs" = { vary = false }', id="hold-position"),
    ],
)
def test_each_peak_edit_gives_the_engine_config_of_the_equivalent_toml(tmp_path, edit, toml):
    editor, shown = _editor(PEAK)
    edit(editor)
    path = tmp_path / "peakfit.toml"
    path.write_text(f'[parameters.peaks."G45N-HN"]\n{toml}\n')
    expected = load_config(path).parameters.peaks
    engine = engine_configuration(
        PeakFitConfig(), (shown[-1],), (PeakIdentity("p_0", PEAK.id, PEAK.name),)
    )
    assert engine.parameters.peaks == {"p_0": expected["G45N-HN"]}
    assert not editor.invalid


def test_an_empty_field_shows_what_the_model_settings_give():
    editor, _shown = _editor(PEAK)
    width = _row(editor, "F3.lw")
    assert [e.placeholderText() for e in (width.start, width.low, width.high)] == [
        "25",
        "0.1",
        "200",
    ]
    position = _row(editor, "F2.cs")
    assert position.at.text() == "120.500"
    assert position.window_edit.placeholderText() == "0.04"
    assert position.window_edit.toolTip().endswith("the Model settings give ± 0.04 ppm")
    assert not width.hold.isChecked()
    assert _row(editor, "F2.lw").hold.isChecked()  # the rule holds it
    assert not width.reset.isVisible()


@pytest.mark.parametrize(
    ("field", "typed", "reason"),
    [
        ("start", "abc", "Enter the start and bounds as numbers"),
        ("low", "60", "The range 60–50 Hz is empty"),
        ("start", "70", "Start 70 Hz is outside its range −∞–50 Hz"),
        ("window", "-1", "positive number of ppm"),
    ],
)
def test_an_invalid_entry_is_explained_and_never_committed(field, typed, reason):
    editor, shown = _editor(PEAK)
    row = _row(editor, "F2.cs" if field == "window" else "F3.lw")
    if field != "window":
        _type(row.high, "50")
    committed = shown[-1].limits
    _type(row.window_edit if field == "window" else getattr(row, field), typed)
    assert reason in row.message.text()
    assert row.message.text().startswith("▲")
    edits = row.fields
    assert any(e.property("invalid") for e in edits)
    assert editor.invalid.startswith(f"{PEAK.name}: {row.row.title}: ")
    assert shown[-1].limits == committed
    # Another peak, or Undo, drops the draft and its block.
    editor.discard_drafts()
    assert not editor.invalid
    assert not row.message.isVisible()


def test_hold_and_reset():
    editor, shown = _editor(PEAK)
    row = _row(editor, "F3.lw")
    row.hold.setChecked(True)
    assert shown[-1].limits == (("F3.lw", ParameterLimit(vary=False)),)
    assert row.reset.isVisible()
    row.reset.click()
    assert shown[-1].limits == ()
    assert not row.reset.isVisible()


def test_hold_back_to_what_the_model_settings_give_leaves_no_limit_of_its_own():
    # Hold on then off is the Model settings again: no hidden vary=True that would
    # restart the peak from defaults or keep a later Model-wide Hold off it.
    editor, shown = _editor(PEAK)
    width = _row(editor, "F3.lw")
    width.hold.setChecked(True)
    width.hold.setChecked(False)
    assert shown[-1].limits == ()
    assert not width.reset.isVisible()
    # Against a rule that holds it, fitting is the peak's own; holding again is not.
    held = _row(editor, "F2.lw")
    held.hold.setChecked(False)
    assert shown[-1].limits == (("F2.lw", ParameterLimit(vary=True)),)
    held.hold.setChecked(True)
    assert shown[-1].limits == ()


def test_a_position_bounded_by_a_rules_range_names_it_rather_than_no_window():
    rows = [ParameterRow("F2.cs", "δ ¹⁵N F2", "ppm", 120.5, 118.0, 122.0, True, None)]
    editor, _shown = _editor(PEAK, rows=rows)
    window = _row(editor, "F2.cs").window_edit
    assert window.placeholderText() == "range"
    assert window.toolTip().startswith("The Model settings bound it to 118.000–122.000 ppm")


def test_each_own_limit_is_marked_when_it_differs_from_the_run():
    recorded = ("3", (("F3.lw", ParameterLimit(lower=5.0, upper=50.0)),))
    peak = replace(PEAK, limits=recorded[1])
    editor, _shown = _editor(peak, recorded=recorded)
    assert not editor.differs
    row = _row(editor, "F3.lw")
    _type(row.high, "40")
    assert editor.differs
    assert row.title.text() == f"LW ¹H F3 {DIFFERS}"
    assert row.title.toolTip() == "Run 3 used range 5–50 Hz"
    window = _row(editor, "F2.cs")
    _type(window.window_edit, "0.05")
    assert window.title.toolTip() == "Run 3 used the Model settings"


def test_a_preparation_problem_is_shown_under_its_parameter():
    peak = replace(PEAK, limits=(("F3.lw", ParameterLimit(start=300.0)),))
    rows = [
        replace(r, problem="LW ¹H F3 start 300 Hz is outside its range 0.1–200 Hz.") for r in ROWS
    ]
    editor, _shown = _editor(peak, rows)
    row = _row(editor, "F3.lw")
    assert "outside its range 0.1–200 Hz" in row.message.text()
    assert not any(e.property("invalid") for e in row.fields)  # nothing typed is wrong
    assert not editor.invalid  # the preparation blocks the fit, with the peak's name


def test_the_section_in_its_widest_state_fits_the_inspector():
    editor, _shown = _editor(PEAK)
    row = _row(editor, "F3.lw")
    _type(row.start, "−1.234 × 10⁻⁵")
    _type(row.low, "1.5e5")
    _type(row.high, "-12345")
    _type(_row(editor, "F2.cs").window_edit, "0.0125")
    assert editor.minimumSizeHint().width() <= INSPECTOR_CONTENT


# ------------------------------------------------------------- the working history
def test_limits_belong_to_the_working_peak_not_its_assignment():
    one = WorkingPeak.new("Duplicate", (120.0, 8.0))
    other = WorkingPeak.new("Duplicate", (121.0, 8.5))
    edits = WorkingEdits((one, other), PeakFitConfig())
    limits = (("F3.lw", ParameterLimit(lower=5.0)),)
    edits.constrain(one.id, limits)
    assert edits.undo_label == "constraints of Duplicate"
    assert [p.limits for p in edits.peaks] == [limits, ()]
    edits.update(one.id, "G45N-HN", one.positions)  # a rename keeps them
    edits.update(one.id, "G45N-HN", (120.1, 8.0))  # so does a move
    assert edits.peaks[0].limits == limits
    edits.delete(one.id)
    edits.undo()
    assert edits.peaks[0].limits == limits
    for _ in range(3):
        edits.undo()
    assert edits.peaks[0].limits == ()


def test_a_limit_is_kept_only_in_a_form_a_fit_can_use():
    with pytest.raises(ValueError, match="positive"):
        ParameterLimit(window=0.0)
    with pytest.raises(ValueError, match="finite"):
        ParameterLimit(start=math.inf)
    with pytest.raises(ValueError, match="position only"):
        WorkingPeak("a", "A", (1.0, 2.0), (("F3.lw", ParameterLimit(window=0.1)),))
    with pytest.raises(ValueError, match="empty"):
        WorkingPeak("a", "A", (1.0, 2.0), (("F3.lw", ParameterLimit()),))


# ------------------------------------------------------------- the workspace
@pytest.fixture(scope="module")
def scientific():
    spectra, peaks, config, prepared = two_cluster_dataset()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return spectra, peaks, config, result


@pytest.fixture
def workspace(qapp, scientific):
    spectra, peaks, config, _ = scientific
    dataset = Dataset(Path("synthetic.ft2"), spectra, peaks, (), True, 0.01)
    widget = EditorWorkspace(dataset)
    widget.edits = WorkingEdits(widget.peaks, config.model_copy(deep=True))
    widget.fitting.model_seen = True
    widget.resize(1280, 750)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.stop()
    widget.close()
    widget.deleteLater()


def _publish(w, result):
    fitting = w.fitting
    fitting._submitted_revision = w.edits.revision
    fitting.process.state = "completed"
    fitting._finished(0, result)
    wait_until(lambda: not fitting.busy)
    assert fitting.result is result, fitting.failure


def _open(w, identity):
    w.select_peak(identity)
    w.panel.parameters_toggle.setChecked(True)
    wait_until(lambda: w.fitting.prepared is not None and w.panel.parameters.rows)
    w._update_inspector()


def test_a_peak_edit_is_one_undoable_edit_that_outdates_the_run(workspace, scientific):
    w = workspace
    _publish(w, scientific[-1])
    assert w.fitting.current
    peak = w.peaks[0]
    _open(w, peak.id)
    row = _row(w.panel.parameters, "F3.lw")
    _type(row.low, "10")
    assert w.peaks[0].limits == (("F3.lw", ParameterLimit(lower=10.0)),)
    assert w.undo_action.text() == f"Undo constraints of {peak.name}"
    assert not w.fitting.current  # Outdated
    w._update_inspector()
    run = scientific[-1].run_id[:8]
    number = run_numbers(w.fitting.journal)[scientific[-1].run_id]
    row = _row(w.panel.parameters, "F3.lw")
    # Named by its number, as every caption names a run; the reason keeps the short ID.
    assert row.title.toolTip() == f"Run {number} used the Model settings"
    assert w.panel.parameters_toggle.text() == f"▾  Parameters · 1 set for this peak {DIFFERS}"
    # The Model page counts the peaks with their own; the next fit does not continue them.
    assert "1 peak with its own" in w.panel.model_page.constraints_summary.text()
    wait_until(lambda: w.fitting.prepared is not None)
    reasons = dict(w.fitting.prepared.continuation.reasons)
    assert reasons[peak.id] == f"constraints changed since run {run}"
    w.history(redo=False)
    assert w.peaks[0].limits == ()
    w._update_inspector()
    assert not _row(w.panel.parameters, "F3.lw").low.text()
    assert not w.fitting.current  # Undo never recertifies a run
    # Back to the limits the run used, the peak continues from its fitted values again.
    wait_until(lambda: w.fitting.prepared is not None)
    continuation = w.fitting.prepared.continuation
    assert peak.id in continuation.continued
    assert peak.id not in dict(continuation.reasons)


def test_an_invalid_draft_blocks_fit_until_another_peak_is_chosen(workspace):
    w = workspace
    _open(w, w.peaks[0].id)
    assert w.fit_button.isEnabled()
    row = _row(w.panel.parameters, "F3.lw")
    _type(row.low, "60")
    _type(row.high, "50")
    assert not w.fit_button.isEnabled()
    assert "The range 60–50 Hz is empty" in w.fit_button.toolTip()
    assert w.peaks[0].limits == (("F3.lw", ParameterLimit(lower=60.0)),)  # the valid step
    w.select_peak(w.peaks[1].id)
    w._update_inspector()
    assert not w.panel.parameters.invalid
    assert w.fit_button.isEnabled()


def test_a_start_outside_the_settings_range_blocks_fit_and_leads_to_its_peak(workspace):
    w = workspace
    peak = w.peaks[2]
    _open(w, peak.id)
    _type(_row(w.panel.parameters, "F3.lw").start, "500")
    wait_until(lambda: bool(w.fitting.blocked))
    assert w.fitting.blocked.startswith(f"{peak.name}: LW")
    assert "LW F3: Start 500 Hz is outside its range" in w.fitting.blocked
    w._update_inspector()
    assert "outside its range" in _row(w.panel.parameters, "F3.lw").message.text()
    assert not w.fit_button.isEnabled()
    # Fit leads to the peak, its Parameters open, rather than to the Model page.
    w.select_peak(None)
    w.fitting.fit()
    assert w.selected_id == peak.id
    assert w.panel.parameters_toggle.isChecked()
    assert not w.fitting.busy


def test_the_run_page_lists_each_peaks_own_constraints(scientific):
    result = scientific[-1]
    snapshot = result.inputs.snapshot
    peaks = (
        replace(snapshot.peaks[0], limits=(("F3.lw", ParameterLimit(lower=5.0, vary=False)),)),
        *snapshot.peaks[1:],
    )
    inputs = replace(result.inputs, snapshot=replace(snapshot, peaks=peaks))
    names = {"F2": "¹⁵N F2", "F3": "¹H F3"}
    (row,) = [
        r for r in submitted_rows(replace(result, inputs=inputs), names) if r.label == "Constraints"
    ]
    assert "1 peak with its own" in row.value
    assert f"{peaks[0].name} LW ¹H F3: range 5–∞ Hz · held" in row.detail


def test_the_greyed_values_stay_the_model_settings_beside_the_peaks_own(workspace):
    # Through a real preparation: own values are typed, the greyed ones are what the
    # Model settings alone give, and the fit uses the peak's own.
    w = workspace
    peak = w.peaks[0]
    w.constrain(
        peak.id,
        (
            ("F2.cs", ParameterLimit(window=0.01)),
            ("F3.lw", ParameterLimit(start=50.0, lower=40.0)),
        ),
    )
    _open(w, peak.id)
    width = _row(w.panel.parameters, "F3.lw")
    fields = (width.start, width.low, width.high)
    assert [e.placeholderText() for e in fields] == ["25", "0.1", "200"]
    assert [e.text() for e in fields] == ["50", "40", ""]
    position = _row(w.panel.parameters, "F2.cs").window_edit
    assert (position.placeholderText(), position.text()) == ("0.04", "0.01")
    engine = next(i.engine_name for i in w.fitting.prepared.identities if i.working_id == peak.id)
    fitted = w.fitting.prepared.parameters[f"{engine}.F3.lw"]
    assert (fitted.value, fitted.min, fitted.max) == (50.0, 40.0, 200.0)
    # A Model rule's absolute range on positions is named, not shown as no window.
    settings = w.edits.settings.model_copy(deep=True)
    settings.parameters.position_window = None
    settings.parameters.defaults.patterns["*.F2.cs"] = ParameterConstraint(min=-5.0, max=5.0)
    w.edits.configure(settings)
    w._refresh()
    w.constrain(peak.id, ())
    _open(w, peak.id)
    position = _row(w.panel.parameters, "F2.cs").window_edit
    assert position.placeholderText() == "range"
    assert "bound it to −5.000–5.000 ppm" in position.toolTip()


def test_a_failed_attempt_records_the_peaks_own_limits_as_a_run_does(workspace):
    w = workspace
    peak = replace(w.peaks[0], limits=(("F3.lw", ParameterLimit(lower=10.0)),))
    fitting = w.fitting
    fitting._attempt_id, fitting._attempt_parent = "a" * 32, None
    fitting._attempt_initial = (peak, *w.peaks[1:])
    fitting._record_attempt("failed", "simulated")
    entry = fitting.journal[-1]
    assert entry["status"] == "failed"
    assert entry["initial_peaks"][0]["limits"] == (
        ("F3.lw", {"start": None, "lower": 10.0, "upper": None, "vary": None, "window": None}),
    )


def test_a_long_default_window_is_hinted_short_and_given_exactly_in_its_tooltip():
    # The model's default ± 50 Hz on ¹H at 600 MHz: 0.0833153 ppm does not fit the field.
    rows = [
        ParameterRow(
            "F3.cs",
            "δ ¹H F3",
            "ppm",
            8.25,
            8.25 - 50 / 600.13,
            8.25 + 50 / 600.13,
            True,
            50 / 600.13,
        )
    ]
    editor, _shown = _editor(PEAK, rows=rows)
    window = _row(editor, "F3.cs").window_edit
    assert window.placeholderText() == "0.083"
    assert window.toolTip().endswith("the Model settings give ± 0.0833153 ppm")
