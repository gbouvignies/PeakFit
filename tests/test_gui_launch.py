"""`launch` opens datasets and saved analyses in the desktop application."""

from pathlib import Path

import pytest
from PySide6 import QtCore, QtWidgets
from test_gui_editor import wait_for

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.gui.editor.window import EditorWindow, launch
from peakfit.gui.project import Analysis, dataset_header, save_analysis
from peakfit.gui.spectrum import read_dataset

DATA = Path(__file__).resolve().parents[1] / "examples" / "02-advanced-fitting" / "data"
pytestmark = pytest.mark.gui


@pytest.fixture
def started(monkeypatch):
    """Run launch with an event loop that returns once startup loading settles."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    windows: list[EditorWindow] = []
    show = EditorWindow.show

    def record(window):
        windows.append(window)
        show(window)

    def settle(_app):
        wait_for(lambda: not windows[0]._busy, timeout=30)
        return 0

    monkeypatch.setattr(EditorWindow, "show", record)
    monkeypatch.setattr(QtWidgets.QApplication, "exec", settle)
    # Keep the test session's style; launch sets Fusion for the real application.
    monkeypatch.setattr(QtWidgets.QApplication, "setStyle", lambda *_args: None)
    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning", lambda *args, **_kwargs: warnings.append(args[1:])
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Open,
    )

    def run(*args, **kwargs):
        launch(*args, **kwargs)
        return windows[0], warnings

    yield run
    for window in windows:
        window.close()
        window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()


def test_empty_window(started) -> None:
    window, warnings = started()
    assert window.workspace is None
    assert warnings == []


@pytest.mark.parametrize("swap", [False, True])
def test_dataset_with_peaks_values_and_mapping(started, swap) -> None:
    peaks = DATA / "pseudo3d.list"
    values = DATA / "b1_offsets.txt"
    window, warnings = started(DATA / "pseudo3d.ft2", peaks, values, swap_peak_axes=swap)
    assert warnings == []
    assert window.workspace is not None
    expected = read_dataset(DATA / "pseudo3d.ft2", peaks, values, swap_peak_axes=swap)
    assert [p.positions for p in window.workspace.peaks] == [p.positions for p in expected.peaks]
    assert [p.name for p in window.workspace.peaks] == [p.name for p in expected.peaks]
    assert window.workspace.dataset.spectra.z_values.tolist() == expected.spectra.z_values.tolist()


def test_saved_analysis_reopens_with_its_recorded_inputs(started, tmp_path) -> None:
    dataset = read_dataset(DATA / "pseudo3d.ft2", DATA / "pseudo3d.list")
    analysis = Analysis(
        dataset, dataset_header(dataset), dataset.peaks, PeakFitConfig().model_dump_json(), 0
    )
    path = save_analysis(analysis, tmp_path / "Study.peakfit")
    window, warnings = started(path)
    assert warnings == []
    assert window.workspace is not None
    assert window.analysis_path == path
    assert window.workspace.peaks == dataset.peaks
    assert not window.isWindowModified()


def test_startup_load_error_is_reported_and_window_stays_usable(started, tmp_path) -> None:
    spectrum = tmp_path / "broken.ft2"
    spectrum.write_bytes(b"not an NMRPipe spectrum")
    window, warnings = started(spectrum)
    assert window.workspace is None
    assert [title for title, _message in warnings] == ["Cannot load inputs"]
    assert "Load failed" in window.loading_status.text()
    assert window.open_action.isEnabled()


def test_startup_corrupt_analysis_is_reported(started, tmp_path) -> None:
    path = tmp_path / "Broken.peakfit"
    path.mkdir()
    window, warnings = started(path)
    assert window.workspace is None
    assert [title for title, _message in warnings] == ["Cannot complete analysis operation"]
    assert window.analysis_path is None
