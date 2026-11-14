"""Plot, readout and tile text take the sizes derived from the system font (spec §12)."""

from copy import deepcopy
from pathlib import Path

import pytest
from builders import two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtCore, QtGui

from peakfit.gui import theme as themes
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.profiles_view import ProfilesView, cluster_tiles, paint_tile
from peakfit.gui.editor.review import HighlightRules, review_clusters
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.spectrum import Dataset

pytestmark = pytest.mark.gui

# A 17 px system font: 16 px secondary text and plot labels, 14 px captions.
LARGE = themes.TypeScale.for_body(17)


@pytest.fixture(scope="module")
def scientific():
    spectra, peaks, config, prepared = two_cluster_dataset()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return spectra, peaks, config, result


@pytest.fixture
def large_font(qapp):
    """Sizes as derived at start-up from a 17 px system font; restored afterwards."""
    font = qapp.font()
    themes.use_type_scale(LARGE)
    yield LARGE
    themes.use_type_scale(None)
    qapp.setFont(font)


@pytest.fixture
def fitted(qapp, scientific, large_font):
    assert themes.type_scale() is large_font  # before any widget is built
    spectra, peaks, config, result = scientific
    widget = EditorWorkspace(Dataset(Path("synthetic.ft2"), spectra, peaks, (), True, 0.01))
    widget.fitting.configuration = config.model_copy(deep=True)
    widget.fitting.setup_seen = True
    themes.apply(widget, themes.LIGHT)
    widget.apply_theme(themes.LIGHT)
    widget.resize(1280, 750)
    widget.show()
    qapp.processEvents()
    fitting = widget.fitting
    fitting._submitted_revision = widget.edits.revision
    fitting.process.state = "completed"
    fitting._finished(0, result)
    wait_until(lambda: not fitting.busy)
    widget.select_peak(widget.peaks[0].id)
    qapp.processEvents()
    yield widget
    widget.stop()
    widget.close()
    widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    qapp.processEvents()


def test_spectrum_axes_use_the_derived_sizes(fitted):
    canvas = fitted.canvas
    for axis in (canvas.bottom_axis, canvas.left_axis):
        assert axis.style["tickFont"].pixelSize() == LARGE.caption
        assert axis.labelStyle["font-size"] == f"{LARGE.small}px"
    assert canvas.hover_label.textItem.font().pixelSize() == LARGE.small


def test_the_corner_readout_uses_the_caption_size(fitted):
    readout = fitted.canvas.readout
    readout.ensurePolished()
    assert readout.font().pixelSize() == LARGE.caption


def test_the_amplitude_profile_axes_use_the_derived_sizes(fitted):
    plot = fitted.panel.profile.plot
    left = plot.getAxis("left")
    assert left.style["tickFont"].pixelSize() == LARGE.caption
    assert plot.getAxis("bottom").style["tickFont"].pixelSize() == LARGE.caption
    assert left.labelStyle["font-size"] == f"{LARGE.caption}px"


class _Recording(QtGui.QPainter):
    """A painter that notes the pixel size of every font set on it."""

    def __init__(self, device) -> None:
        super().__init__(device)
        self.sizes: set[int] = set()

    def setFont(self, font) -> None:  # noqa: N802
        self.sizes.add(font.pixelSize())
        super().setFont(font)


@pytest.mark.usefixtures("large_font")
def test_profile_tiles_use_the_derived_sizes(scientific):
    result = scientific[-1]
    axes = list(scientific[0].spectral_params)
    reviews = review_clusters(result, axes, ["¹H", "¹⁵N"][: len(axes)], HighlightRules())
    tile = next(t for t in cluster_tiles(result, reviews, None).values() if t.profile is not None)
    view = ProfilesView()
    style = view.tile_style()
    image = QtGui.QImage(320, 230, QtGui.QImage.Format.Format_ARGB32)
    painter = _Recording(image)
    paint_tile(painter, QtCore.QRectF(0, 0, 320, 230), tile, style)
    painter.end()
    view.deleteLater()
    assert painter.sizes == {LARGE.small, LARGE.caption}
