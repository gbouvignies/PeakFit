"""The measurements of the GUI acceptance tool (tools/validate_gui_layout.py)."""

import importlib
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from builders import two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtCore, QtWidgets

from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.peaks import WorkingEdits
from peakfit.gui.spectrum import Dataset

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture(scope="module")
def tool():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(TOOLS))
        yield importlib.import_module("validate_gui_layout")


def test_contrast_records_text_and_control_border_pairs_of_each_appearance(tool):
    from peakfit.gui.theme import DARK, LIGHT, contrast_ratio  # noqa: PLC0415

    record = tool.theme_contrast(LIGHT)
    assert record["text/panel"] == pytest.approx(contrast_ratio(LIGHT.text, LIGHT.panel))
    assert record["border/panel"] == pytest.approx(contrast_ratio(LIGHT.border, LIGHT.panel))
    assert {"text/chrome", "secondary/panel", "faint/panel", "border/chrome"} <= set(record)
    # Editable controls are bounded by the outline token.
    assert {"outline/panel", "outline/chrome", "outline/plot"} <= set(record)
    assert tool.theme_contrast(DARK)["text/panel"] > 10


@pytest.mark.gui
def test_window_record_names_logical_size_and_device_pixel_ratio(qapp, tool):
    widget = QtWidgets.QWidget()
    widget.resize(640, 480)
    record = tool.window_record(widget)
    dpr = widget.devicePixelRatioF()
    assert record["logical"] == [640, 480]
    assert record["dpr"] == dpr
    assert record["label"] == f"640×480 logical @ DPR {tool.dpr_text(dpr)}"
    assert record["qt_platform"] == qapp.platformName()
    assert record["platform"]


@pytest.fixture(scope="module")
def scientific():
    spectra, peaks, config, prepared = two_cluster_dataset()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return spectra, peaks, config, result


@pytest.fixture
def fitted(qapp, scientific):
    """A workspace showing a completed fit, as the tool drives it."""
    spectra, peaks, config, result = scientific
    w = EditorWorkspace(Dataset(Path("synthetic.ft2"), spectra, peaks, (), True, 0.01))
    w.edits = WorkingEdits(w.peaks, config.model_copy(deep=True))
    w.fitting.model_seen = True
    w.resize(1280, 500)
    w.show()
    qapp.processEvents()
    w.fitting._submitted_revision = w.edits.revision
    w.fitting.process.state = "completed"
    w.fitting._finished(0, result)
    wait_until(lambda: not w.fitting.busy)
    assert w.fitting.result is result, w.fitting.failure
    yield w
    w.stop()
    w.close()
    w.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


@pytest.mark.gui
def test_peak_page_reports_the_fitted_table_in_page_coordinates(tool, fitted):
    w = fitted
    w.select_peak(w.peaks[0].id)
    inspector = w.panel
    measured = tool.steady(lambda: tool.peak_page_measures(inspector))
    assert measured["viewport_height"] == inspector.viewport().height()
    title = inspector.fitted_title
    title_top = title.mapTo(inspector.widget(), QtCore.QPoint(0, 0)).y()
    # The table starts below its "Fitted" heading and has at least one row.
    assert measured["fitted_top"] > title_top
    assert measured["fitted_bottom"] > measured["fitted_top"]
    # The table is its value rows (positions, linewidths, amplitude ± marginal σ),
    # ending with the amplitude row; captions below it are not table rows.
    amplitude = next(
        label
        for label in inspector.peak_page.findChildren(QtWidgets.QLabel)
        if label.text().split(" (")[0] == "Amplitude"
    )
    body = inspector.widget()
    bottom = amplitude.mapTo(body, QtCore.QPoint(0, 0)).y() + amplitude.height()
    assert measured["fitted_bottom"] == bottom
    assert measured["fitted_notes_bottom"] > bottom
    # The peak page starts below the inspector's identity header; the audit of
    # 14 November 2026 quoted the table relative to the peak page.
    header = inspector.peak_page.mapTo(inspector.widget(), QtCore.QPoint(0, 0)).y()
    assert measured["peak_page_top"] == header > 0
    # Positions are on the page, not the viewport: scrolling does not move them.
    bar = inspector.verticalScrollBar()
    assert bar.maximum() > 0
    bar.setValue(bar.maximum())
    tool.settle(50)
    assert tool.peak_page_measures(inspector) == measured


@pytest.mark.gui
def test_spectral_measures_report_plot_slices_and_only_the_lanes_shown(tool, fitted):
    w = fitted
    w.resize(1280, 800)
    w.set_lanes(False)
    c = w.canvas
    hidden = tool.steady(lambda: tool.spectral_measures(c))
    width, height = hidden["plot"]
    assert width >= 360
    assert height >= 300
    # The row slice shares the plot's width, the column slice its height.
    assert hidden["row_slice"][0] == width
    assert hidden["column_slice"][1] == height
    assert hidden["row_lane"] is None
    assert hidden["column_lane"] is None
    w.set_lanes(True)
    shown = tool.steady(lambda: tool.spectral_measures(c))
    assert shown["row_lane"][0] == shown["plot"][0]
    assert shown["column_lane"][1] == shown["plot"][1]


@pytest.mark.gui
def test_profiles_measures_report_columns_and_the_drawn_tile_size(tool, fitted):
    w = fitted
    w.resize(1280, 800)
    w.show_view(profiles=True)
    wait_until(lambda: "Computing" not in w.profiles.caption.text())
    wide = tool.steady(lambda: tool.profiles_measures(w.profiles))
    view = w.profiles.list
    first = view.visualRect(view.model().index(0, 0))
    assert wide["tile"] == [first.width(), first.height()]
    assert wide["columns"] >= 1
    assert wide["columns"] * wide["tile"][0] <= view.viewport().width()
    assert (wide["columns"] + 1) * wide["tile"][0] > view.viewport().width()
    w.resize(2000, 800)
    wider = tool.steady(lambda: tool.profiles_measures(w.profiles))
    assert wider["columns"] > wide["columns"]


@pytest.mark.gui
def test_clipped_labels_reports_visible_text_cut_by_its_fixed_size(qapp, tool):
    root = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(root)
    narrow = QtWidgets.QLabel("Amplitude plane 1 of 60")
    narrow.setFixedWidth(30)
    short = QtWidgets.QLabel("Wrapped caption text that needs several lines at this width")
    short.setWordWrap(True)
    short.setFixedSize(80, 14)
    roomy = QtWidgets.QLabel("Fits")
    hidden = QtWidgets.QLabel("Hidden text that would not fit")
    hidden.setFixedWidth(10)
    for label in (narrow, short, roomy, hidden):
        layout.addWidget(label)
    root.show()
    hidden.hide()
    qapp.processEvents()
    clipped = tool.clipped_labels(root)
    assert [entry["text"] for entry in clipped] == [narrow.text(), short.text()]
    assert clipped[0]["needed"][0] > clipped[0]["size"][0] == 30
    assert clipped[1]["needed"][1] > clipped[1]["size"][1] == 14
    root.close()


@pytest.mark.parametrize(
    ("dpr", "text"), [(1.0, "1.0"), (2.0, "2.0"), (1.25, "1.25"), (1.5, "1.5"), (1.75, "1.75")]
)
def test_device_pixel_ratio_text_keeps_fractional_scale_factors(tool, dpr, text):
    # Linux fractional scaling (1.25, 1.75) must not read as 1.2 or 1.8.
    assert tool.dpr_text(dpr) == text


def test_watchdog_ends_a_blocked_run_with_its_tracebacks():
    # A modal dialog or native deadlock blocks the main thread where no wait can see it.
    script = (
        "import sys, time; sys.path.insert(0, sys.argv[1]);"
        "import validate_gui_layout as tool; tool.arm_watchdog(0.5); time.sleep(60)"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(TOOLS)],
        capture_output=True,
        text=True,
        timeout=50,
        check=False,
    )
    assert done.returncode != 0
    assert "Timeout" in done.stderr
    assert 'File "<string>"' in done.stderr  # where the main thread was blocked
