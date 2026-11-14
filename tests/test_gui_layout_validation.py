"""The measurements of the GUI acceptance tool (tools/validate_gui_layout.py)."""

import importlib
from copy import deepcopy
from pathlib import Path

import pytest
from builders import two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtCore, QtWidgets

from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.spectrum import Dataset

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture(scope="module")
def tool():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(TOOLS))
        yield importlib.import_module("validate_gui_layout")


@pytest.mark.parametrize(
    ("foreground", "background", "ratio"),
    [
        ("#000000", "#ffffff", 21.0),
        ("#ffffff", "#000000", 21.0),
        ("#5f6a76", "#5f6a76", 1.0),
        # WCAG's documented threshold grey: #767676 on white is 4.54:1.
        ("#767676", "#ffffff", 4.54),
    ],
)
def test_contrast_ratio_follows_wcag(tool, foreground, background, ratio):
    assert tool.contrast_ratio(foreground, background) == pytest.approx(ratio, abs=0.01)


def test_contrast_records_text_and_control_border_pairs_of_each_appearance(tool):
    from peakfit.gui.theme import DARK, LIGHT  # noqa: PLC0415

    record = tool.theme_contrast(LIGHT)
    assert record["text/panel"] == pytest.approx(tool.contrast_ratio(LIGHT.text, LIGHT.panel))
    assert record["border/panel"] == pytest.approx(tool.contrast_ratio(LIGHT.border, LIGHT.panel))
    assert {"text/chrome", "secondary/panel", "faint/panel", "border/chrome"} <= set(record)
    assert tool.theme_contrast(DARK)["text/panel"] > 10


@pytest.mark.gui
def test_window_record_names_logical_size_and_device_pixel_ratio(qapp, tool):
    widget = QtWidgets.QWidget()
    widget.resize(640, 480)
    record = tool.window_record(widget)
    dpr = widget.devicePixelRatioF()
    assert record["logical"] == [640, 480]
    assert record["dpr"] == dpr
    assert record["label"] == f"640×480 logical @ DPR {dpr:.1f}"
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
    w.fitting.configuration = config.model_copy(deep=True)
    w.fitting.setup_seen = True
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
        if label.text().startswith("Amplitude plane")
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
