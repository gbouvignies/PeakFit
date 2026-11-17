"""Failed and canceled jobs tell the scientist what happened; tracebacks reach the log."""

import logging
from concurrent.futures import CancelledError
from copy import deepcopy
from pathlib import Path

import pytest
from builders import two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from peakfit.gui import fit_process
from peakfit.gui import theme as themes
from peakfit.gui.background import ViewerComputation
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.fitting import FailureStage
from peakfit.gui.editor.inspector import Page
from peakfit.gui.editor.window import EditorWindow
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.failures import (
    WorkerError,
    child_failure,
    close_log,
    failure_message,
    install_log,
    raise_child_failure,
)
from peakfit.gui.peaks import WorkingEdits
from peakfit.gui.project_process import project_operation
from peakfit.gui.spectrum import Dataset


def test_worker_failure_shows_its_message_and_logs_the_child_traceback(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="peakfit.gui")
    with pytest.raises(WorkerError) as error:
        project_operation(
            "open", tmp_path / "missing.peakfit", progress=lambda _: None, canceled=lambda: False
        )
    assert "missing.peakfit" in str(error.value)
    assert not str(error.value).startswith("FileNotFoundError")
    (record,) = caplog.records
    assert "Analysis open failed in its worker process" in record.getMessage()
    assert "Traceback (most recent call last)" in record.getMessage()
    assert "FileNotFoundError" in record.getMessage()


def test_child_records_round_trip_cancellation_and_unexpected_errors(caplog):
    try:
        raise CancelledError("stopped")
    except CancelledError as error:
        canceled = child_failure(error)
    with pytest.raises(CancelledError, match="stopped"):
        raise_child_failure(canceled, "Analysis save")
    assert not caplog.records

    try:
        {}["runs"]
    except KeyError as error:
        unexpected = child_failure(error)
    with pytest.raises(WorkerError, match=r"Unexpected KeyError: 'runs'.*Please report it"):
        raise_child_failure(unexpected, "Analysis open")
    assert "KeyError: 'runs'" in caplog.records[-1].getMessage()


def test_expected_failures_are_shown_as_raised():
    assert failure_message(ValueError("Refit the dataset.")) == "Refit the dataset."
    assert failure_message(OSError("disk full")) == "disk full"


@pytest.mark.usefixtures("qapp")
def test_background_job_failure_logs_its_traceback(caplog):
    caplog.set_level(logging.ERROR, logger="peakfit.gui")
    owner = QtCore.QObject()
    worker = ViewerComputation(owner)
    delivered = []
    worker.completed.connect(lambda *args: delivered.append(args))
    worker.submit("model", 0, lambda: {}["missing"])
    wait_until(lambda: delivered)
    ((_kind, _revision, outcome),) = delivered
    assert isinstance(outcome, KeyError)
    assert "Background model job failed" in caplog.records[-1].getMessage()
    assert caplog.records[-1].exc_info is not None
    worker.close()


@pytest.mark.parametrize(
    ("kind", "text"),
    [
        ("save", "Save canceled"),
        ("analysis", "Opening canceled"),
        ("relink", "Relink canceled"),
    ],
)
def test_canceled_analysis_operation_names_what_was_canceled(qapp, kind, text):
    window = EditorWindow()
    window._analysis_finished(kind, 0, CancelledError())
    assert window.loading_status.text().startswith(text)
    window.close()
    qapp.processEvents()


def test_close_refused_during_an_analysis_operation_explains_why(qapp, monkeypatch):
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: shown.append(args))
    window = EditorWindow()
    window._busy = True
    assert not window.close()
    assert len(shown) == 1
    assert "Wait for it to finish, or cancel it" in shown[0][2]
    window._busy = False
    assert window.close()
    qapp.processEvents()


def test_unexpected_failure_message_has_no_doubled_punctuation():
    message = failure_message(RuntimeError("Dataset fit worker exited with code 0."))
    assert message.startswith("Unexpected RuntimeError: Dataset fit worker exited with code 0. ")
    assert ".." not in message


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
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    qapp.processEvents()


def _crash(*_args, **_kwargs):
    """A defect in the fit job: an unexpected error with a traceback."""
    return {}["runs"]


def _failed_fit(workspace, monkeypatch):
    monkeypatch.setattr(fit_process, "run_child", _crash)
    fitting = workspace.fitting
    wait_until(lambda: fitting.prepared is not None)
    fitting.fit()
    wait_until(lambda: not fitting.busy)


def _publish(workspace, result):
    fitting = workspace.fitting
    fitting._submitted_revision = workspace.edits.revision
    fitting.process.state = "completed"
    fitting._finished(0, result)
    wait_until(lambda: not fitting.busy)
    assert fitting.result is result, fitting.failure


def test_a_failed_first_fit_says_there_is_no_result_yet(workspace, monkeypatch):
    seeds = workspace.peaks
    assert not workspace.fitting.failed
    _failed_fit(workspace, monkeypatch)
    fitting = workspace.fitting
    assert fitting.failed
    assert fitting.failure_stage is FailureStage.FITTING
    assert fitting.attempt == "Failed · no result yet"
    assert "Unexpected KeyError: 'runs'." in fitting.failure
    assert fitting.result is None
    assert workspace.peaks == seeds


def test_a_failed_refit_says_the_previous_result_is_retained(workspace, scientific, monkeypatch):
    result = scientific[3]
    _publish(workspace, result)
    seeds = workspace.peaks
    _failed_fit(workspace, monkeypatch)
    fitting = workspace.fitting
    assert fitting.attempt == "Failed · previous result retained"
    assert fitting.result is result
    assert fitting.current
    assert workspace.peaks == seeds


def test_a_canceled_fit_says_whether_a_result_is_retained(workspace, scientific):
    fitting = workspace.fitting
    fitting.cancel()
    assert fitting.attempt == "Canceled · no result yet"
    assert not fitting.failed
    _publish(workspace, scientific[3])
    fitting.cancel()
    assert fitting.attempt == "Canceled · previous result retained"


def _foreground(label: QtWidgets.QLabel) -> str:
    return label.palette().color(label.foregroundRole()).name()


@pytest.mark.parametrize("theme", [themes.LIGHT, themes.DARK])
def test_the_run_line_shows_a_failure_first_in_the_error_role(
    workspace, scientific, monkeypatch, theme
):
    themes.apply(workspace, theme)
    summary = workspace.summary
    _failed_fit(workspace, monkeypatch)
    assert summary.text() == "▲ Failed · no result yet"
    assert _foreground(summary) == QtGui.QColor(theme.error).name()

    _publish(workspace, scientific[3])
    assert summary.text().startswith("Run ")
    assert _foreground(summary) != QtGui.QColor(theme.error).name()

    _failed_fit(workspace, monkeypatch)
    assert summary.text().startswith("▲ Failed · previous result retained · Run ")
    assert _foreground(summary) == QtGui.QColor(theme.error).name()


def _analysis_sections(panel):
    """The Run page's sections on show, in page order; empty while Model is shown."""
    if panel.run_page.isHidden():
        return []
    layout = panel.run_page.layout()
    widgets = (layout.itemAt(i).widget() for i in range(layout.count()))
    return [w for w in widgets if w is not None and not w.isHidden()]


def _row_values(grid: QtWidgets.QGridLayout) -> dict[str, str]:
    values = {}
    for row in range(grid.rowCount()):
        label, value = (grid.itemAtPosition(row, c) for c in (0, 1))
        if label is not None and value is not None:
            title = label.widget()
            # A value with an ⓘ shares its cell with the button: the label comes first.
            cell = value.layout()
            shown = value.widget() or (cell.itemAt(0).widget() if cell else None)  # ty: ignore[possibly-missing-attribute]
            assert isinstance(title, QtWidgets.QLabel)
            assert isinstance(shown, QtWidgets.QLabel)
            values[title.text()] = shown.text()
    return values


def test_the_failure_is_the_first_section_of_the_run_page(workspace, scientific, monkeypatch):
    panel = workspace.panel
    assert panel.run_page.failure_block not in _analysis_sections(panel)
    assert not panel.run_tab.isEnabled()  # nothing to show on Run before an attempt
    _failed_fit(workspace, monkeypatch)
    # A failed first attempt brings the Run page forward, the failure leading it.
    assert panel.page is Page.RUN
    assert _analysis_sections(panel)[0] is panel.run_page.failure_block
    rows = _row_values(panel.run_page.failure_rows)
    assert rows["Status"] == "▲ Failed · no result yet"
    assert rows["Stage"] == "Fitting"
    assert "Unexpected KeyError: 'runs'." in rows["Reason"]

    _publish(workspace, scientific[3])
    assert panel.run_page.failure_block not in _analysis_sections(panel)
    _failed_fit(workspace, monkeypatch)
    sections = _analysis_sections(panel)
    assert sections[:2] == [panel.run_page.failure_block, panel.run_page.run_block]
    assert (
        _row_values(panel.run_page.failure_rows)["Status"] == "▲ Failed · previous result retained"
    )
    # The failure is no longer repeated among the active run's rows.
    assert not any(label.startswith("Failed") for label in _row_values(panel.run_page.run_rows))


@pytest.fixture
def log_file(tmp_path, capsys):  # noqa: ARG001 - capture stderr before the handler binds it
    """An installed log; its terminal handler writes to the captured stderr."""
    path = install_log(tmp_path / "logs" / "peakfit.log")
    yield path
    close_log()


def test_copy_details_and_the_log_file_report_a_failure_without_a_terminal(
    workspace, monkeypatch, log_file, capsys
):
    _failed_fit(workspace, monkeypatch)
    panel = workspace.panel
    assert _row_values(panel.run_page.failure_rows)["Log"] == str(log_file)
    logged = log_file.read_text(encoding="utf-8")
    assert "Dataset fit failed" in logged
    assert "Traceback (most recent call last)" in logged
    assert "KeyError: 'runs'" in logged
    # Logging to the terminal continues.
    assert "KeyError: 'runs'" in capsys.readouterr().err

    QtGui.QGuiApplication.clipboard().clear()
    QtTest.QTest.mouseClick(panel.run_page.copy_details, QtCore.Qt.MouseButton.LeftButton)
    copied = QtGui.QGuiApplication.clipboard().text()
    assert "Status: ▲ Failed · no result yet" in copied
    assert "Stage: Fitting" in copied
    assert "Message: Unexpected KeyError: 'runs'." in copied
    assert "Dataset: synthetic.ft2" in copied
    assert "Active run: none" in copied
    assert f"Log: {log_file}" in copied
