"""Failed and canceled jobs tell the scientist what happened; tracebacks reach the log."""

import logging
from concurrent.futures import CancelledError

import pytest
from gui_helpers import wait_until
from PySide6 import QtCore, QtWidgets

from peakfit.gui.background import ViewerComputation
from peakfit.gui.editor.window import EditorWindow
from peakfit.gui.failures import WorkerError, child_failure, failure_message, raise_child_failure
from peakfit.gui.project_process import project_operation


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
