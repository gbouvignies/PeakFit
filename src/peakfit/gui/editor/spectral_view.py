"""The spectral view: signed contours, aligned native slices and inspection gestures.

One graphics scene holds the 2D plot, its row and column slices, the optional
residual lanes and their dividers, placed by ``spectral_geometry.layout``. Slices show
experimental and model values at native samples; hover only updates their data.
"""

import html
from collections import OrderedDict
from concurrent.futures import CancelledError
from dataclasses import replace
from functools import partial
from threading import Event
from typing import TYPE_CHECKING, cast

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# isort: split
import pyqtgraph as pg

from peakfit.gui import formatting
from peakfit.gui.background import ViewerComputation
from peakfit.gui.editing import PeakMarkers
from peakfit.gui.editor import slices
from peakfit.gui.editor.spectral_geometry import (
    Extents,
    SliceSizes,
    drag,
    layout,
    minimum_size,
)
from peakfit.gui.fit_display import dataset_model_plane, normalized_residual
from peakfit.gui.spectrum import contour_geometry, display_position, paths_from_geometry
from peakfit.gui.theme import FONT_CAPTION, LIGHT, Theme, color

if TYPE_CHECKING:
    from peakfit.gui.dataset_fit import DatasetFitResult
    from peakfit.gui.fit_display import ModelPlanes
    from peakfit.gui.peaks import WorkingPeak
    from peakfit.gui.spectrum import ContourGeometry, Dataset

_PICK_RADIUS = 8.0  # Logical pixels, shared by hover/click/drag at every DPR.
_CACHE_BYTES = 32 * 1024**2
_CACHE_PLANES = 8
_RESIDUAL = 2
_RESIDUAL_LEVELS = np.asarray([3.0, 5.0, 8.0, 13.0, 21.0])
_FRAME_MARGIN = 0.35  # a framed extent gains this fraction on each side
_FRAME_POINTS = 6  # and at least this many points on each side
LANE_SIGMA = 8.0  # default residual lane range, ±σ_stat
_SCALE_STEP = 1.25  # one wheel step of a manual slice or lane scale
_COMMIT_MS = 200  # a zoom or pan is committed once the wheel or drag rests
_PENDING_SHOW_MS = 200  # a wait shorter than this needs no label; it would only flash
_READOUT_MIN = (40, 30)  # the corner readout needs at least this much room
REGION_ALPHA = 0.08  # every fitting region, behind the contours
SELECTED_REGION_ALPHA = 0.16  # the selected peak's cluster


def framed_ranges(
    rows: tuple[int, int],
    columns: tuple[int, int],
    ppm_y: np.ndarray,
    ppm_x: np.ndarray,
    points: int = _FRAME_POINTS,
) -> tuple[list[float], list[float]]:
    """Ranges in ppm (x, y) of an inclusive sample extent plus a margin on each side."""
    ranges = []
    for (first, last), ppm in ((columns, ppm_x), (rows, ppm_y)):
        pad = max(points, int((last - first) * _FRAME_MARGIN))
        low, high = max(0, first - pad), min(ppm.size - 1, last + pad)
        ranges.append(sorted((float(ppm[low]), float(ppm[high]))))
    return ranges[0], ranges[1]


def signed_geometry(data, axes, levels) -> list[tuple[np.ndarray, np.ndarray]]:
    """One geometry per level, positive levels then negative; no Qt work here."""
    return contour_geometry(data, axes, np.concatenate((levels, -levels)))


class ContourLayer(QtWidgets.QGraphicsItem):
    """One layer of signed contour levels painted as a single cached graphic.

    Positive levels come first, then the matching negative levels. Each level has
    its own pen so low levels can be lighter, without one scene item per level.
    """

    def __init__(self, role: str, paths: list[QtGui.QPainterPath], key: tuple = ()) -> None:
        super().__init__()
        self.role = role
        self.key = key  # what these contours are of: (run, mode, plane, levels…)
        self.levels = paths
        self._pens: list[QtGui.QPen] = []
        rect = QtCore.QRectF()
        for path in paths:
            rect = rect.united(path.boundingRect())
        margin = max(rect.width(), rect.height()) * 1e-3
        self._rect = rect.adjusted(-margin, -margin, margin, margin)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)

    def set_pens(self, pens: list[QtGui.QPen]) -> None:
        """Restyle retained geometry; never recomputes contours."""
        self._pens = pens
        self.update()

    def boundingRect(self) -> QtCore.QRectF:  # noqa: N802
        """Union of every level, with room for cosmetic strokes."""
        return self._rect

    def paint(self, painter, _option, widget=None) -> None:  # noqa: ARG002
        """Draw levels from lowest to highest so strong contours stay on top."""
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        for path, pen in zip(self.levels, self._pens, strict=False):
            painter.setPen(pen)
            painter.drawPath(path)


def _level_pens(colors: tuple[str, str], count: int, *, weight: float, fade: float, dashes):
    """Graded pens for `count` positive then `count` negative levels."""
    pens = []
    for sign_color, dash in zip(colors, dashes, strict=True):
        for index in range(count):
            position = index / (count - 1) if count > 1 else 1.0
            pen = QtGui.QPen(color(sign_color, fade * (0.62 + 0.38 * position)))
            pen.setWidthF(weight * (0.9 + 0.5 * position))
            pen.setCosmetic(True)
            pen.setStyle(dash)
            pens.append(pen)
    return pens


class InspectionMarkers(PeakMarkers):
    """Use the same logical-pixel tolerance for hover, click and drag acquisition."""

    def pointsAt(self, pos):  # noqa: N802
        """Pick in scene coordinates, independent of zoom and device pixel ratio."""
        center = self.mapToScene(pos)
        return [
            spot
            for spot in self.points()
            if (
                (self.mapToScene(spot.pos()).x() - center.x()) ** 2
                + (self.mapToScene(spot.pos()).y() - center.y()) ** 2
            )
            <= _PICK_RADIUS**2
        ]


class _SceneView(pg.GraphicsView):
    """The one graphics view; its scene coordinates are its pixels."""

    resized = QtCore.Signal()

    def resizeEvent(self, ev):  # noqa: N802
        """Lay the plots out again for the new size."""
        super().resizeEvent(ev)
        self.resized.emit()


class SliceBox(pg.ViewBox):
    """A slice or lane: no pan or zoom; the wheel rescales (the view handles double-clicks)."""

    def __init__(self, on_scale) -> None:
        super().__init__(enableMenu=False, defaultPadding=0)
        self.setMouseEnabled(x=False, y=False)
        self.on_scale = on_scale

    def wheelEvent(self, ev, axis=None) -> None:  # noqa: N802, ARG002
        """Wheel up narrows the shown range; down widens it."""
        self.on_scale(ev.delta())
        ev.accept()


class Divider(QtWidgets.QGraphicsRectItem):
    """A visible drag handle between the 2D plot and one slice."""

    def __init__(self, *, vertical: bool, on_drag, tip: str) -> None:
        super().__init__()
        self.vertical, self.on_drag = vertical, on_drag
        self.color, self.hot_color, self.hot = "#c9d0d8", "#2f5da8", False
        self.setAcceptHoverEvents(True)
        self.setCursor(
            QtCore.Qt.CursorShape.SplitHCursor if vertical else QtCore.Qt.CursorShape.SplitVCursor
        )
        self.setToolTip(tip)
        self.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self.setZValue(60)
        self._anchor = QtCore.QPointF()

    def paint(self, painter, _option, widget=None) -> None:  # noqa: ARG002
        """A line with three grip dots, stronger under the pointer."""
        rect = self.rect()
        pen = QtGui.QPen(QtGui.QColor(self.hot_color if self.hot else self.color), 1)
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor(self.hot_color if self.hot else self.color))
        center = rect.center()
        if self.vertical:
            painter.drawLine(
                QtCore.QPointF(center.x(), rect.top()), QtCore.QPointF(center.x(), rect.bottom())
            )
            dots = [QtCore.QPointF(center.x(), center.y() + k) for k in (-6, 0, 6)]
        else:
            painter.drawLine(
                QtCore.QPointF(rect.left(), center.y()), QtCore.QPointF(rect.right(), center.y())
            )
            dots = [QtCore.QPointF(center.x() + k, center.y()) for k in (-6, 0, 6)]
        for dot in dots:
            painter.drawEllipse(dot, 1.4, 1.4)

    def hoverEnterEvent(self, _event) -> None:  # noqa: N802
        """Show that the line can be dragged."""
        self.hot = True
        self.update()

    def hoverLeaveEvent(self, _event) -> None:  # noqa: N802
        """Return to the quiet line."""
        self.hot = False
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Anchor the drag: size = size at press + total displacement."""
        self._anchor = event.scenePos()
        self.on_drag(None, finished=False)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        """Follow the pointer."""
        delta = event.scenePos() - self._anchor
        self.on_drag(delta.x() if self.vertical else delta.y(), finished=False)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        """Commit the size."""
        delta = event.scenePos() - self._anchor
        self.on_drag(delta.x() if self.vertical else delta.y(), finished=True)
        event.accept()


class SpectralView(QtWidgets.QWidget):
    """Display observations only; scientific edits are proposals to the owning workspace."""

    selected = QtCore.Signal(str)
    sizes_changed = QtCore.Signal()
    scale_changed = QtCore.Signal()
    hovered = QtCore.Signal(object)
    moved = QtCore.Signal(str, object)
    placed = QtCore.Signal(object)
    edit_requested = QtCore.Signal()
    status_changed = QtCore.Signal(str)
    pin_changed = QtCore.Signal(bool)
    draft_changed = QtCore.Signal(bool)
    painted = QtCore.Signal()

    def __init__(self, dataset: Dataset, model_planes: ModelPlanes) -> None:  # noqa: PLR0915
        super().__init__()
        self.dataset = dataset
        self.spectra = dataset.spectra
        self.y_axis, self.x_axis = self.spectra.spectral_params
        self.ppm_x = self.x_axis.pts2ppm(np.arange(self.x_axis.size, dtype=float))
        self.ppm_y = self.y_axis.pts2ppm(np.arange(self.y_axis.size, dtype=float))
        self.comparison_result: DatasetFitResult | None = None
        self.comparison_mode = 0
        # The recorded run identity of a historical (Outdated) model; empty if current.
        self.comparison_label = ""
        self.added: frozenset[str] = frozenset()
        self.departed: tuple[WorkingPeak, ...] = ()
        self.residual_threshold = 3.0
        self.model_planes = model_planes
        self._model_cancel = Event()
        self._model = None
        # Slice display: user sizes, residual lanes and the intensity scale.
        self.sizes = SliceSizes()
        self.lane_sigma = LANE_SIGMA
        self.lanes_fit = True  # False while the window has no room for the lanes
        self.manual_scale: tuple[float, float] | None = None
        self.region_scale: tuple[float, float] = (-1.0, 1.0)
        self.clips: dict[str, tuple[int, int]] = {}
        self._drag_origin = 0
        self.editing_enabled = True
        self.plane = 0
        self.iy, self.ix = (size // 2 for size in self.spectra.data.shape[1:])
        self.pinned = False
        self.pin_peak_id: str | None = None
        self.selected_id: str | None = None
        self.hovered_id: str | None = None
        self.peaks: tuple[WorkingPeak, ...] = ()
        self._by_id: dict[str, WorkingPeak] = {}
        self.placing = False
        self.resolve_edit = lambda: True
        self._drag_peak: WorkingPeak | None = None
        self._drag_pin = None
        self._draft_position = None
        self.contour_threshold = dataset.noise * 5
        self.contour_factor = 1.6
        self.contour_count = 10
        self.heavy_strokes = False
        self.labels_visible = False
        self._labels: list[pg.TextItem] = []
        self._revision = 0
        self._last_plane = 0
        self._cache: OrderedDict[tuple, ContourGeometry] = OrderedDict()
        self.cache_bytes = 0
        self.contour_jobs = 0
        self.marker_rebuilds = 0
        self.preview_deliveries = 0
        self.preview_coalesced = 0
        self._displaying_experiment = True
        self.contour_pending = False
        self._paint_timer = QtCore.QTimer(self)
        self._paint_timer.setSingleShot(True)
        self._paint_timer.timeout.connect(self.painted)
        self._press_position = None
        self._pointer_dragged = False
        self._pointer: QtCore.QPointF | None = None
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(0)
        self._preview_timer.timeout.connect(self._deliver_pointer)
        self._worker = ViewerComputation(self)
        self._worker.completed.connect(self._contours_ready)
        self.theme: Theme = LIGHT
        self.decimals_x = formatting.ppm_decimals(self.x_axis)
        self.decimals_y = formatting.ppm_decimals(self.y_axis)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        # One scene: the 2D plot, its slices, lanes and dividers are placed by
        # spectral_geometry.layout, so shared extents are equal by construction.
        self.view = _SceneView()
        self.view.resized.connect(self._relayout)
        self.view.setViewportUpdateMode(
            QtWidgets.QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate
        )
        self.view.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.view.viewport().installEventFilter(self)
        self.view.installEventFilter(self)
        self.setFocusProxy(self.view)
        self.view.setAccessibleName(
            "2D spectrum with native slices; Space pins slices; A places peak; arrows step sample"
        )
        outer.addWidget(self.view)
        scene = cast("pg.GraphicsScene", self.view.scene())
        font = QtGui.QFont(QtWidgets.QApplication.font())
        font.setPixelSize(FONT_CAPTION)
        metrics = QtGui.QFontMetrics(font)
        self.extents = Extents(
            left=metrics.horizontalAdvance("000.0") + metrics.height() + 18,
            bottom=2 * metrics.height() + 14,
        )
        self.box = pg.ViewBox(enableMenu=False, defaultPadding=0)
        self.box.invertX(True)
        self.box.invertY(True)
        self.row_box = SliceBox(self.scale_intensity)
        self.row_lane = SliceBox(self.scale_lanes)
        self.column_box = SliceBox(self.scale_intensity)
        self.column_lane = SliceBox(self.scale_lanes)
        # Slices share the plot's ppm direction (high on the left and top) and its range.
        for box in (self.row_box, self.row_lane):
            box.invertX(True)
            box.setXLink(self.box)
        for box in (self.column_box, self.column_lane):
            box.invertY(True)
            box.setYLink(self.box)
        for box in (self.box, self.row_box, self.row_lane, self.column_box, self.column_lane):
            scene.addItem(box)
        self.bottom_axis = pg.AxisItem("bottom")
        self.left_axis = pg.AxisItem("left")
        for axis in (self.bottom_axis, self.left_axis):
            axis.linkToView(self.box)
            axis.setStyle(
                maxTickLevel=0,
                tickLength=-4,
                tickFont=font,
                tickTextOffset=4,
                autoExpandTextSpace=False,
                autoReduceTextSpace=False,
            )
            axis.enableAutoSIPrefix(False)
            scene.addItem(axis)
        self.left_axis.setWidth(self.extents.left)
        self.bottom_axis.setHeight(self.extents.bottom)
        x_name, y_name = (
            formatting.nucleus(a.nucleus, a.label) for a in (self.x_axis, self.y_axis)
        )
        self.row_divider = Divider(
            vertical=False,
            on_drag=partial(self._drag_divider, "row"),
            tip=f"Drag to resize the {x_name} slice; drag to the edge to collapse it. "
            "Double-click resets its size.",
        )
        self.column_divider = Divider(
            vertical=True,
            on_drag=partial(self._drag_divider, "column"),
            tip=f"Drag to resize the {y_name} slice; drag to the edge to collapse it. "
            "Double-click resets its size.",
        )
        for divider in (self.row_divider, self.column_divider):
            scene.addItem(divider)
        # Experimental and model values at native samples, joined by lines; no markers.
        self.curve_x, self.model_x = pg.PlotDataItem(), pg.PlotDataItem()
        self.curve_y, self.model_y = pg.PlotDataItem(), pg.PlotDataItem()
        self.lane_x, self.lane_y = pg.PlotDataItem(), pg.PlotDataItem()
        self.zero_x = pg.InfiniteLine(angle=0, pos=0)
        self.zero_y = pg.InfiniteLine(angle=90, pos=0)
        self.lane_zero_x = pg.InfiniteLine(angle=0, pos=0)
        self.lane_zero_y = pg.InfiniteLine(angle=90, pos=0)
        self.clip_x, self.model_clip_x, self.lane_clip_x = (pg.ScatterPlotItem() for _ in range(3))
        self.clip_y, self.model_clip_y, self.lane_clip_y = (pg.ScatterPlotItem() for _ in range(3))
        for box, items in (
            (
                self.row_box,
                (self.zero_x, self.curve_x, self.model_x, self.clip_x, self.model_clip_x),
            ),
            (
                self.column_box,
                (self.zero_y, self.curve_y, self.model_y, self.clip_y, self.model_clip_y),
            ),
            (self.row_lane, (self.lane_zero_x, self.lane_x, self.lane_clip_x)),
            (self.column_lane, (self.lane_zero_y, self.lane_y, self.lane_clip_y)),
        ):
            for item in items:
                box.addItem(item, ignoreBounds=True)
        # One compact intensity indication per slice: a bar from the lower to the upper
        # bound in the gutter, with a zero tick and its signs; the magnitude is in the caption.
        self.scale_bars = (QtWidgets.QGraphicsPathItem(), QtWidgets.QGraphicsPathItem())
        self.scale_signs = tuple(QtWidgets.QGraphicsSimpleTextItem() for _ in range(4))
        for item in (*self.scale_bars, *self.scale_signs):
            item.setZValue(50)
            scene.addItem(item)
        for sign in self.scale_signs:
            sign.setFont(font)
        self.caption = QtWidgets.QLabel(self.view)
        self.caption.setObjectName("caption")
        self.caption.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.caption.setAccessibleName("Slice scale")
        self.readout = QtWidgets.QLabel(self.view)
        self.readout.setObjectName("caption")
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.readout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self.readout.setWordWrap(True)
        self.readout.setAccessibleName("Values at the crosshair")
        self._commit = QtCore.QTimer(self)
        self._commit.setSingleShot(True)
        self._commit.setInterval(_COMMIT_MS)
        self._commit.timeout.connect(self._range_committed)
        self.box.sigRangeChangedManually.connect(lambda *_: self._commit.start())
        self.box.sigRangeChanged.connect(lambda *_: self._place_scale_bars())
        self.cross_x = pg.InfiniteLine(angle=90, movable=False)
        self.cross_y = pg.InfiniteLine(angle=0, movable=False)
        for line in (self.cross_x, self.cross_y):
            line.setZValue(10)
            self.box.addItem(line, ignoreBounds=True)
            line.sigDragged.connect(self._crosshair_dragged)
        # A plot-coloured cross under each marker keeps it legible over contours.
        self.marker_halo = self._scatter("+", 8, 19)
        self.markers = InspectionMarkers(
            symbol="+", size=7, brush=None, hoverable=True, tip=self._marker_tip
        )
        self.markers.setZValue(20)
        self.markers.sigClicked.connect(self._marker_clicked)
        self.markers.moved.connect(self._move_peak)
        self.box.addItem(self.markers)
        self.ring = self._scatter("o", 18, 30)
        # Other members of the selected peak's recorded cluster: lighter rings that
        # mark the unit of review while the selected peak stays the unit of editing.
        self.cluster_rings = self._scatter("o", 14, 29)
        # Recorded fitted positions the run did not adopt (working markers stay put).
        self.unadopted = self._scatter("s", 7, 18)
        self.cluster_ids: tuple[str, ...] = ()
        # Fitting regions on request: a faint tint behind the contours, one cell per
        # sampled point, with the selected peak's cluster slightly stronger.
        self.region_image = pg.ImageItem(axisOrder="row-major")
        self.region_image.setZValue(-1)
        self.region_image.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        cell = QtGui.QTransform()
        x0, y0 = (float(axis.pts2ppm(-0.5)) for axis in (self.x_axis, self.y_axis))
        cell.translate(x0, y0)
        cell.scale(float(self.x_axis.pts2ppm(0.5)) - x0, float(self.y_axis.pts2ppm(0.5)) - y0)
        self.region_image.setTransform(cell)
        self.box.addItem(self.region_image, ignoreBounds=True)
        self.region_image.setVisible(False)
        self.regions_visible = False
        self.region_key = None
        self.region_source = ""
        self._regions: tuple[tuple[str, ...], ...] = ()
        self._region_labels = np.zeros(0, dtype=np.int32)
        self._selected_region = -1
        # Against a historical model: a diamond marks a working peak that model does
        # not contain; × marks a model peak's recorded position, linked by a dotted
        # line to where it was moved.
        self.added_marks = self._scatter("d", 15, 21)
        self.recorded_marks = self._scatter("x", 9, 18)
        self.recorded_links = pg.PlotDataItem(connect="pairs")
        self.recorded_links.setZValue(18)
        self.box.addItem(self.recorded_links, ignoreBounds=True)
        self.halo = self._scatter("o", 15, 31)
        self.ghost = self._scatter("+", 10, 32)
        self.hover_label = self._label(35, (-0.08, 1.15))
        self.selection_label = self._label(34, (-0.12, 1.05))
        # Measured contours of one plane, and the model or residual contours drawn
        # with them; each kept until something of its own kind replaces it.
        self.experiment_layers: list[ContourLayer] = []
        self.model_layers: list[ContourLayer] = []
        self.pending_label = pg.TextItem("Preparing contours…", anchor=(0, 0))
        self.pending_label.setZValue(40)
        self.pending_label.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.box.addItem(self.pending_label, ignoreBounds=True)
        self._pending_show = QtCore.QTimer(self)
        self._pending_show.setSingleShot(True)
        self._pending_show.setInterval(_PENDING_SHOW_MS)
        self._pending_show.timeout.connect(
            lambda: self.pending_label.setVisible(bool(self.pending_label.toPlainText()))
        )
        self.legend = QtWidgets.QLabel(self.view)
        self.legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.legend.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.legend.setAccessibleName("Contour legend")
        scene.sigMouseMoved.connect(self._pointer_moved)
        scene.sigMouseClicked.connect(self._blank_click)
        self.apply_theme(LIGHT)
        self.full_spectrum()
        self.sample(float(self.ppm_x[self.ix]), float(self.ppm_y[self.iy]))
        self.request_contours()

    @staticmethod
    def axis_title(axis) -> str:
        """Use loaded names rather than assume a particular pair of nuclei."""
        return f"{formatting.nucleus(axis.nucleus, axis.label)} · {axis.label}"

    def _scatter(self, symbol, size, z):
        item = pg.ScatterPlotItem(symbol=symbol, size=size, brush=None)
        item.setZValue(z)
        item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.box.addItem(item)
        return item

    def _label(self, z: float, anchor: tuple[float, float]) -> pg.TextItem:
        label = pg.TextItem(anchor=anchor)
        label.setZValue(z)
        label.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.box.addItem(label, ignoreBounds=True)
        return label

    def _marker_tip(self, *, x: float, y: float, data: str) -> str:
        """Delayed native tooltip with exact unfolded and displayed coordinates."""
        peak = self._by_id.get(data)
        if peak is None:
            return ""
        text = f"{peak.name}\nUnfolded {self._coordinates(peak.positions[1], peak.positions[0])}"
        if (x, y) != tuple(reversed(peak.positions)):
            text += f"\nDisplayed {self._coordinates(x, y)}"
        if self.comparison_label and self.comparison_mode:
            if peak.id in self.added:
                text += f"\nNot in the model of {self.comparison_label}"
            elif any(p.id == peak.id for p in self.departed):
                text += f"\nMoved since {self.comparison_label}; × marks it in that model"
        return text

    def _coordinates(self, x: float, y: float) -> str:
        return (
            f"{formatting.number(x, self.decimals_x)}, {formatting.number(y, self.decimals_y)} ppm"
        )

    def _place_legend(self) -> None:
        rect = self.box.sceneBoundingRect()
        # One line when it fits; otherwise keys wrap within the plot, never past it.
        limit = max(120, int(rect.width()) - 16)
        self.legend.setWordWrap(False)
        self.legend.setMinimumWidth(0)
        self.legend.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX: undo a previous fixed width
        self.legend.adjustSize()
        if self.legend.width() > limit:
            self.legend.setWordWrap(True)
            self.legend.setFixedWidth(limit)
            self.legend.resize(limit, self.legend.heightForWidth(limit))
        corner = self.view.mapFromScene(rect.topRight())
        self.legend.move(max(0, corner.x() - self.legend.width() - 8), corner.y() + 6)

    def display(self, peak: WorkingPeak) -> tuple[float, float]:
        """Convert only indirect aliases; retain unfolded source coordinates."""
        return (
            display_position(self.x_axis, peak.positions[1]),
            display_position(self.y_axis, peak.positions[0]),
        )

    def set_peaks(self, peaks: tuple[WorkingPeak, ...]) -> None:
        """Rebuild markers only on committed collection changes."""
        self.peaks = peaks
        self._by_id = {peak.id: peak for peak in peaks}
        self.marker_rebuilds += 1
        self._place_cluster_rings()
        positions = [self.display(peak) for peak in peaks]
        xs, ys = [p[0] for p in positions], [p[1] for p in positions]
        self.marker_halo.setData(x=xs, y=ys)
        self.markers.setData(x=xs, y=ys, data=[p.id for p in peaks])
        self._place_changes()
        self._refresh_labels()
        self._selection_ring()
        self.hover(None)

    def select(self, peak_id: str | None) -> None:
        """Selection pins once; following hover never changes it."""
        self.selected_id = peak_id
        self.markers.selected_id = peak_id
        self._selection_ring()
        peak = self._by_id.get(peak_id or "")
        if peak is not None:
            x, y = self.display(peak)
            self.sample(x, y)
            self.set_pinned(True)
            self.pin_peak_id = peak.id
            view = self.box
            ranges = view.viewRange()
            shifted = []
            for value, (low, high) in zip((x, y), ranges, strict=True):
                margin = (high - low) * 0.05
                delta = min(value - low - margin, 0) + max(value - high + margin, 0)
                shifted.append((low + delta, high + delta))
            view.setRange(xRange=shifted[0], yRange=shifted[1], padding=0)

    def set_changes(self, added: frozenset[str], departed: tuple[WorkingPeak, ...]) -> None:
        """Mark working peaks added, and model peaks moved or deleted, since a historical run."""
        if (added, departed) != (self.added, self.departed):
            self.added, self.departed = added, departed
            self._place_changes()
            self._update_legend()

    def _place_changes(self) -> None:
        shown = bool(self.comparison_label and self.comparison_mode)
        added = (
            [self.display(self._by_id[i]) for i in self.added if i in self._by_id] if shown else []
        )
        self.added_marks.setData([p[0] for p in added], [p[1] for p in added])
        recorded = [self.display(p) for p in self.departed] if shown else []
        self.recorded_marks.setData([p[0] for p in recorded], [p[1] for p in recorded])
        links = [
            point
            for p in (self.departed if shown else ())
            if p.id in self._by_id
            for point in (self.display(p), self.display(self._by_id[p.id]))
        ]
        self.recorded_links.setData([p[0] for p in links], [p[1] for p in links])

    def set_unadopted(self, points: list[tuple[float, float]]) -> None:
        """Show recorded fitted positions that were not adopted as working positions."""
        self.unadopted.setData([p[0] for p in points], [p[1] for p in points])
        self._update_legend()

    def set_cluster(self, peak_ids: tuple[str, ...]) -> None:
        """Mark a recorded cluster's members; never changes view, mode or model scope."""
        if peak_ids != self.cluster_ids:
            self.cluster_ids = peak_ids
            self._place_cluster_rings()

    def _place_cluster_rings(self) -> None:
        positions = [
            self.display(self._by_id[i])
            for i in self.cluster_ids
            if i in self._by_id and i != self.selected_id
        ]
        self.cluster_rings.setData([p[0] for p in positions], [p[1] for p in positions])

    def set_regions(self, key, regions, source: str) -> None:
        """Show sampled fitting regions; never changes view, mode or model scope.

        ``regions`` pairs each cluster's sample indices with its working peak IDs;
        ``source`` says whether they were recorded by a run or merely prepared.
        """
        if key == self.region_key:
            return
        self.region_key = key
        self.region_source = source
        labels = np.zeros((self.y_axis.size, self.x_axis.size), dtype=np.int32)
        for number, (indices, _members) in enumerate(regions, 1):
            labels[tuple(np.asarray(i) for i in indices)] = number
        self._regions = tuple(members for _indices, members in regions)
        self._region_labels = labels
        self._selected_region = -1
        self._place_selected_region()
        self._update_legend()

    def set_regions_visible(self, visible: bool) -> None:
        """Explicit display choice; regions stay prepared while hidden."""
        self.regions_visible = visible
        self.region_image.setVisible(visible)
        self._update_legend()

    def _place_selected_region(self) -> None:
        number = next(
            (n for n, members in enumerate(self._regions, 1) if self.selected_id in members), 0
        )
        if number == self._selected_region:
            return
        self._selected_region = number
        self._paint_regions()

    def _paint_regions(self) -> None:
        """Tint exactly the sampled points: all clusters faintly, the selected one more."""
        labels = self._region_labels
        tint = np.zeros((*labels.shape, 4), dtype=np.uint8)
        layers = ((REGION_ALPHA, self.theme.secondary, labels > 0),)
        if self._selected_region:
            layers += ((SELECTED_REGION_ALPHA, self.theme.accent, labels == self._selected_region),)
        for alpha, value, mask in layers:
            rgb = QtGui.QColor(value)
            tint[mask] = (rgb.red(), rgb.green(), rgb.blue(), round(255 * alpha))
        self.region_image.setImage(tint, autoLevels=False, levels=(0, 255))

    def _selection_ring(self) -> None:
        peak = self._by_id.get(self.selected_id or "")
        if peak is None:
            self.ring.setData([], [])
            self.selection_label.setText("")
        else:
            x, y = self.display(peak)
            self.ring.setData([x], [y])
            self.selection_label.setText(peak.name)
            self.selection_label.setPos(x, y)
        self._place_cluster_rings()
        self._place_selected_region()

    def hover(self, peak_id: str | None, *, from_list: bool = False) -> None:
        """Change transient feedback only, preserving editor drafts and selection."""
        if peak_id != self.hovered_id:
            self.hovered_id = peak_id
            peak = self._by_id.get(peak_id or "")
            if peak:
                x, y = self.display(peak)
                self.halo.setData([x], [y])
                # A short on-spectrum label; exact coordinates live in the marker
                # tooltip, the list tooltip, the inspector and the status line.
                folded = (x, y) != tuple(reversed(peak.positions))
                self.hover_label.setText(peak.name + (" · alias" if folded else ""))
                self.hover_label.setPos(x, y)
                self.selection_label.setVisible(peak_id != self.selected_id)
            else:
                self.halo.setData([], [])
                self.hover_label.setText("")
                self.selection_label.setVisible(True)
            self.hovered.emit(peak_id)
        if from_list and not self.pinned and peak_id in self._by_id:
            self.sample(*self.display(self._by_id[peak_id]))

    def _hits(self, scene_position) -> list[WorkingPeak]:
        view = self.box
        candidates = []
        for peak in self.peaks:
            point = view.mapViewToScene(QtCore.QPointF(*self.display(peak)))
            distance = (point.x() - scene_position.x()) ** 2 + (point.y() - scene_position.y()) ** 2
            if distance <= _PICK_RADIUS**2:
                candidates.append((distance, peak))
        return [p for _, p in sorted(candidates, key=lambda item: item[0])]

    def _pointer_moved(self, point) -> None:
        if self._pointer is not None:
            self.preview_coalesced += 1
        self._pointer = point
        if not self._preview_timer.isActive():
            self._preview_timer.start()

    def _deliver_pointer(self) -> None:
        point, self._pointer = self._pointer, None
        if point is None or self._drag_peak is not None:
            return
        view = self.box
        if not view.sceneBoundingRect().contains(point):
            self.hover(None)
            return
        hits = self._hits(point)
        self.hover(hits[0].id if hits else None)
        if not self.pinned and not self.placing:
            position = view.mapSceneToView(point)
            self.sample(position.x(), position.y())
        self.preview_deliveries += 1

    def sample(self, x: float, y: float) -> None:
        """Extract exact native row/column, with no geometry work or autoscaling."""
        self.ix = int(np.clip(round(float(self.x_axis.ppm2pts(x))), 0, self.x_axis.size - 1))
        self.iy = int(np.clip(round(float(self.y_axis.ppm2pts(y))), 0, self.y_axis.size - 1))
        for line, value in (
            (self.cross_x, self.ppm_x[self.ix]),
            (self.cross_y, self.ppm_y[self.iy]),
        ):
            with QtCore.QSignalBlocker(line):
                line.setValue(float(value))
        self._draw_slices()

    def _emit_status(self, raw: float) -> None:
        text = (
            f"{'Draft' if self._drag_peak else 'Pinned' if self.pinned else 'Preview'} · "
            f"{self.axis_title(self.x_axis)} "
            f"{formatting.number(self.ppm_x[self.ix], self.decimals_x)}, "
            f"{self.axis_title(self.y_axis)} "
            f"{formatting.number(self.ppm_y[self.iy], self.decimals_y)} ppm · "
            + (
                f"{formatting.intensity(raw)} σ_stat residual"
                if self.comparison_mode == _RESIDUAL
                else f"{self._intensity(raw)} a.u."
            )
            + (
                f" · {'residual' if self.comparison_mode == _RESIDUAL else 'model'} of "
                f"{self.comparison_label}"
                if self.comparison_label and self.comparison_mode
                else ""
            )
        )
        # The corner readout shows the sample; the status bar only when it has no room.
        self.status_changed.emit("" if self.readout.isVisible() else text)
        self.view.setAccessibleDescription(text)

    def _intensity(self, value: float) -> str:
        """An intensity in the slice caption's power of ten, so both read alike."""
        low, high = self.slice_bounds()
        power = formatting.exponent(max(abs(low), abs(high)) or 1.0)
        number = f"{value * 10.0**-power:.6g}"
        return f"{number} ×10{formatting.superscript(str(power))}" if power else number

    def set_pinned(self, pinned: bool) -> None:
        """Pin the inspection sample independently of a selected assignment."""
        self.pinned = pinned
        self.pin_peak_id = None
        for line in (self.cross_x, self.cross_y):
            line.setMovable(pinned)
            line.setPen(
                pg.mkPen(
                    color(self.theme.pin)
                    if pinned
                    else color(self.theme.accent, 0.55 if self.theme.dark else 0.45),
                    width=1,
                    style=QtCore.Qt.PenStyle.DashLine if pinned else QtCore.Qt.PenStyle.DotLine,
                )
            )
        self.pin_changed.emit(pinned)
        self.sample(float(self.ppm_x[self.ix]), float(self.ppm_y[self.iy]))

    def _crosshair_dragged(self, _line) -> None:
        self.pin_peak_id = None
        self.sample(self.cross_x.value(), self.cross_y.value())

    def _blank_click(self, event) -> None:
        if self._pointer_dragged:
            event.accept()
            return
        view = self.box
        if (
            event.button() != QtCore.Qt.MouseButton.LeftButton
            or not view.sceneBoundingRect().contains(event.scenePos())
        ):
            return
        point = view.mapSceneToView(event.scenePos())
        if self.placing:
            self.placed.emit((point.y(), point.x()))
        else:
            hits = self._hits(event.scenePos())
            if hits:
                self._choose(hits, event.scenePos())
            else:
                self.sample(point.x(), point.y())
                self.set_pinned(True)
        event.accept()

    def _marker_clicked(self, _item, _spots, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._choose(self._hits(event.scenePos()), event.scenePos())
            event.accept()

    def _choose(self, hits: list[WorkingPeak], scene_position) -> None:
        if not hits:
            return
        if len(hits) == 1:
            self.selected.emit(hits[0].id)
            return
        menu = QtWidgets.QMenu(self)
        menu.setTitle("Choose coincident peak")
        for peak in hits:
            action = menu.addAction(
                f"{peak.name} · {self._coordinates(peak.positions[1], peak.positions[0])}"
                f" · {peak.id[:6]}"
            )
            action.triggered.connect(
                lambda _checked=False, identity=peak.id: self.selected.emit(identity)
            )
        menu.popup(self.view.viewport().mapToGlobal(self.view.mapFromScene(scene_position)))
        self._chooser = menu

    def _move_peak(self, dx: float, dy: float, finished: bool) -> None:
        peak = self._by_id.get(self.selected_id or "")
        if peak is None or not self.editing_enabled:
            return
        if self._drag_peak is None:
            if not self.resolve_edit():
                self.markers.cancel_drag()
                return
            assert self.selected_id is not None
            peak = self._by_id[self.selected_id]
            self._drag_peak = peak
            self._drag_pin = (self.ix, self.iy, self.pinned, self.pin_peak_id)
        original = self._drag_peak
        # Delta is continuous ppm in the unfolded coordinate system. No rounded
        # display delta or nearest-sample snapping ever enters a working peak.
        proposed = (original.positions[0] + dy, original.positions[1] + dx)
        self._draft_position = proposed
        x, y = self.display(replace(original, positions=proposed))
        self.ghost.setData([x], [y])
        self.draft_changed.emit(True)
        self.sample(x, y)
        if finished:
            self._drag_peak = None
            self._draft_position = None
            self.ghost.setData([], [])
            self.draft_changed.emit(False)
            self.moved.emit(original.id, proposed)
            self._drag_pin = None

    def set_placing(self, placing: bool) -> None:
        """One-shot placement owns the next click, including an existing marker."""
        self.placing = placing
        self.markers.placing = placing
        self.view.viewport().setCursor(
            QtCore.Qt.CursorShape.CrossCursor if placing else QtCore.Qt.CursorShape.ArrowCursor
        )

    def cancel_gesture(self) -> None:
        """Cancel placement/drag without changing history, selection or exact pin."""
        self.markers.cancel_drag()
        self._drag_peak = None
        self._draft_position = None
        self.ghost.setData([], [])
        self.draft_changed.emit(False)
        self.set_placing(False)
        if self._drag_pin:
            ix, iy, pinned, following = self._drag_pin
            self._drag_pin = None
            self.sample(float(self.ppm_x[ix]), float(self.ppm_y[iy]))
            self.set_pinned(pinned)
            self.pin_peak_id = following

    def full_spectrum(self) -> None:
        """Reset only frequency ranges, keeping plane, selection and sample."""
        self.box.setRange(
            xRange=(float(min(self.ppm_x)), float(max(self.ppm_x))),
            yRange=(float(min(self.ppm_y)), float(max(self.ppm_y))),
            padding=0,
        )
        self._range_committed()

    def frame(self, rows: tuple[int, int], columns: tuple[int, int], points: int) -> None:
        """Frame an inclusive sample extent with a margin, keeping plane and mode."""
        x_range, y_range = framed_ranges(rows, columns, self.ppm_y, self.ppm_x, points)
        self.set_ranges(x_range, y_range)

    def set_ranges(self, x_range, y_range) -> None:
        """Show these ppm ranges as a deliberate view change: the region scale follows."""
        self.box.setRange(xRange=x_range, yRange=y_range, padding=0)
        self._range_committed()

    def set_plane(self, plane: int) -> None:
        """Use original acquisition index, retaining view and sampled row/column."""
        self._model = None
        self.plane = int(np.clip(plane, 0, self.spectra.data.shape[0] - 1))
        self.request_contours()

    @property
    def historical(self) -> bool:
        """Whether the bound model is an Outdated run's, shown as historical evidence."""
        return bool(self.comparison_label)

    def set_comparison(
        self, result: DatasetFitResult | None, mode: int = 0, *, label: str = ""
    ) -> None:
        """Bind one immutable run and one mode across all three spectral views.

        ``label`` names a historical run (``Run 5d7b4768 · historical``) wherever its
        model or residual is shown; it is empty for the current run.
        """
        if result is not self.comparison_result:
            self.model_planes.keep_only(result.run_id if result is not None else None)
        self.comparison_result = result
        self.comparison_mode = mode if result is not None else 0
        self.comparison_label = label if result is not None else ""
        self._place_changes()
        # Residual mode hides the lanes (the slices show residuals) and keeps the preference.
        self._relayout()
        self._style_traces()
        self._update_legend()
        self.request_contours()

    @property
    def paths(self) -> list[ContourLayer]:
        """Every contour layer on show, measured first."""
        return [
            layer for layer in (*self.experiment_layers, *self.model_layers) if layer.isVisible()
        ]

    def _model_key(self) -> tuple:
        result, mode = self.comparison_result, self.comparison_mode
        return (
            result.run_id if result is not None and mode else None,
            mode,
            self.plane,
            self.contour_threshold,
            self.contour_factor,
            self.contour_count,
            self.residual_threshold,
        )

    def _levels(self) -> np.ndarray:
        return self.contour_threshold * self.contour_factor ** np.arange(self.contour_count)

    def request_contours(self) -> None:
        """Show this plane's measured contours at once when known; model contours follow.

        Measured contours are replaced only by those of the plane now shown, never
        carried to another plane; model and residual contours of an earlier request
        go at once and return only for this plane, run and mode. Hover only reads
        arrays.
        """
        self._revision += 1
        self._model_cancel.set()
        self._model_cancel = Event()
        cancel = self._model_cancel
        self._worker.cancel("contours")
        self._worker.cancel("model")
        self._set_layers("model", [], ())
        self._style_contours()  # measured contours alone again: their own pens
        result, mode, plane = self.comparison_result, self.comparison_mode, self.plane
        run = result.run_id if result is not None and mode else None
        grid = self.model_planes.get(run, plane) if run is not None else None
        # The slices show a model only together with its contours.
        self._model = None
        experiment_key = self._experiment_key()
        experiment = self._cache_get(experiment_key)
        if experiment is not None:
            self._show_experiment(experiment_key, experiment)
        elif not self.experiment_layers or self.experiment_layers[0].key != experiment_key:
            self._set_layers("experiment", [], ())
        model_key = self._model_key()
        model_geometry = self._cache_get(model_key) if mode and grid is not None else None
        if model_geometry is not None:
            self._model = grid
            self._show_model(model_key, model_geometry)
        self._update_pending()
        self.sample(float(self.ppm_x[self.ix]), float(self.ppm_y[self.iy]))
        observed = self.spectra.data[plane]
        axes = [axis.model_copy(deep=True) for axis in self.spectra.spectral_params]
        levels = self._levels()
        if experiment is None:
            self.contour_jobs += 1

            def prepare_experiment():
                return experiment_key, signed_geometry(observed, axes, levels)

            self._worker.submit("contours", self._revision, prepare_experiment)
        # Before the model: the next step's measured contours must not wait behind it.
        self._prefetch(axes, levels)
        if mode and model_geometry is None:
            self.contour_jobs += 1
            residual_levels = self.residual_threshold / 3 * _RESIDUAL_LEVELS

            def calculate():
                assert result is not None
                model = grid or dataset_model_plane(result, axes, plane, cancel=cancel)
                if cancel.is_set():
                    raise CancelledError
                if mode == _RESIDUAL:
                    residual = normalized_residual(observed, model.values, result.outcome.noise)
                    return model, signed_geometry(residual, axes, residual_levels)
                return model, signed_geometry(model.values, axes, levels)

            self._worker.submit("model", self._revision, calculate)

    def _prefetch(self, axes, levels) -> None:
        """Prepare the next plane's measured contours, in the direction of travel."""
        step = 1 if self.plane >= self._last_plane else -1
        self._last_plane = self.plane
        plane = self.plane + step
        key = self._experiment_key(plane)
        if not 0 <= plane < self.spectra.data.shape[0] or key in self._cache:
            return
        wanted = [(key, self.spectra.data[plane])]

        def prepare():
            return [(key, signed_geometry(data, axes, levels)) for key, data in wanted]

        self._worker.submit("prefetch", self._revision, prepare)

    def _experiment_key(self, plane: int | None = None) -> tuple:
        return (
            None,
            0,
            self.plane if plane is None else plane,
            self.contour_threshold,
            self.contour_factor,
            self.contour_count,
            0,
        )

    def _cache_get(self, key):
        geometry = self._cache.get(key)
        if geometry is not None:
            self._cache.move_to_end(key)
        return geometry

    def _contours_ready(self, kind, revision, outcome) -> None:
        if isinstance(outcome, Exception):
            if revision == self._revision and kind != "prefetch":
                self.pending_label.setText(f"Comparison unavailable: {outcome}")
                self.pending_label.show()
                self.contour_pending = False
            return
        if kind == "prefetch":
            for key, geometry in outcome:
                self._cache_geometry(key, geometry)
            return
        if kind == "contours":
            # Valid for its own plane whenever it arrives; shown only if still wanted.
            key, geometry = outcome
            self._cache_geometry(key, geometry)
            if key == self._experiment_key() and (
                not self.experiment_layers or self.experiment_layers[0].key != key
            ):
                self._show_experiment(key, geometry)
                self._update_pending()
            return
        if revision != self._revision:
            return
        model, geometry = outcome
        key = self._model_key()
        if key[0] is not None:
            self.model_planes.put(key[0], self.plane, model)
        self._model = model
        self._cache_geometry(key, geometry)
        self._show_model(key, geometry)
        self._update_pending()
        self.sample(float(self.ppm_x[self.ix]), float(self.ppm_y[self.iy]))

    def _cache_geometry(self, key, geometry) -> None:
        previous = self._cache.pop(key, ())
        self.cache_bytes -= sum(p.nbytes + c.nbytes for p, c in previous)
        size = sum(p.nbytes + c.nbytes for p, c in geometry)
        # At most _CACHE_PLANES planes: a plane's measured and model contours count once.
        while self._cache and (
            self.cache_bytes + size > _CACHE_BYTES
            or len({k[2] for k in self._cache} | {key[2]}) > _CACHE_PLANES
        ):
            _, evicted = self._cache.popitem(last=False)
            self.cache_bytes -= sum(p.nbytes + c.nbytes for p, c in evicted)
        if size <= _CACHE_BYTES:
            self._cache[key] = geometry
            self.cache_bytes += size

    def _set_layers(self, kind: str, layers: list[tuple[str, list]], key: tuple) -> None:
        """Replace one kind of contour layer in a single step, so nothing blinks."""
        current = self.experiment_layers if kind == "experiment" else self.model_layers
        for item in current:
            self.box.removeItem(item)
        current.clear()
        for role, paths in layers:
            item = ContourLayer(role, paths, key)
            self.box.addItem(item)
            current.append(item)

    def _show_experiment(self, key: tuple, geometry) -> None:
        if self.experiment_layers and self.experiment_layers[0].key == key:
            return
        self._set_layers("experiment", [("measured", paths_from_geometry(geometry))], key)
        self._style_contours()

    def _show_model(self, key: tuple, geometry) -> None:
        role = "residual" if self.comparison_mode == _RESIDUAL else "model"
        self._set_layers("model", [(role, paths_from_geometry(geometry))], key)
        self._style_contours()

    def _update_pending(self) -> None:
        """Say what is still coming for this plane; the residual hides measured contours."""
        mode = self.comparison_mode
        model = bool(self.model_layers)
        experiment = bool(self.experiment_layers)
        for layer in self.experiment_layers:
            # Residual mode shows measured contours only while its residual is pending.
            layer.setVisible(mode != _RESIDUAL or not model)
        self._displaying_experiment = experiment and not model
        needs_experiment = mode != _RESIDUAL or not model
        self.contour_pending = (needs_experiment and not experiment) or bool(mode and not model)
        historical = f" · {self.comparison_label}" if self.comparison_label else ""
        if needs_experiment and not experiment:
            text = "Preparing contours…"
        elif mode and not model:
            what = "residual" if mode == _RESIDUAL else "model"
            text = f"Experimental contours · {what} pending{historical}"
        else:
            text = ""
        self.pending_label.setPos(self.box.viewRange()[0][1], self.box.viewRange()[1][0])
        self.pending_label.setText(text)
        if not text:
            self._pending_show.stop()
            self.pending_label.hide()
        elif not self.pending_label.isVisible() and not self._pending_show.isActive():
            self._pending_show.start()  # shown only if the wait lasts

    def _style_contours(self) -> None:
        """Hue and stroke encode layer and sign; lightness grades the levels."""
        t = self.theme
        weight = 1.7 if self.heavy_strokes else 1.0
        solid, dash, dot = (
            QtCore.Qt.PenStyle.SolidLine,
            QtCore.Qt.PenStyle.DashLine,
            QtCore.Qt.PenStyle.DotLine,
        )
        overlay = any(layer.role == "model" for layer in self.model_layers)
        for layer in (*self.experiment_layers, *self.model_layers):
            count = len(layer.levels) // 2
            if layer.role == "residual":
                pens = _level_pens(
                    (t.residual_positive, t.residual_negative),
                    count,
                    weight=weight * 1.2,
                    fade=1.0,
                    dashes=(solid, dash),
                )
            elif layer.role == "model":
                pens = _level_pens(
                    (t.model, t.model), count, weight=weight * 0.85, fade=1.0, dashes=(solid, dot)
                )
            else:
                # Under a model the measured strokes widen so both stay visible where
                # they coincide; the thinner model contour is drawn on top.
                pens = _level_pens(
                    (t.measured, t.measured_negative),
                    count,
                    weight=weight * (1.5 if overlay else 1.0),
                    fade=0.75 if overlay else 1.0,
                    dashes=(solid, dash),
                )
            layer.set_pens(pens)

    def _style_traces(self) -> None:
        """Experimental solid, model dashed, residual in its own hue; clips by fill."""
        t = self.theme
        residual = self.comparison_mode == _RESIDUAL
        measured = pg.mkPen(color(t.text if residual else t.measured), width=1.3)
        model = pg.mkPen(color(t.model), width=1.3, style=QtCore.Qt.PenStyle.DashLine)
        for curve in (self.curve_x, self.curve_y):
            curve.setPen(measured)
        for curve in (self.model_x, self.model_y):
            curve.setPen(model)
        for curve in (self.lane_x, self.lane_y):
            curve.setPen(pg.mkPen(color(t.residual_positive), width=1.1))
        for clip in (self.clip_x, self.clip_y, self.lane_clip_x, self.lane_clip_y):
            clip.setPen(None)
            clip.setBrush(pg.mkBrush(t.error))
        for clip in (self.model_clip_x, self.model_clip_y):
            clip.setPen(pg.mkPen(t.model, width=1.4))
            clip.setBrush(None)

    def _update_legend(self) -> None:
        """Name each visible layer with its stroke, so no encoding needs memorizing."""
        t = self.theme

        def key(value: str, glyph: str, text: str) -> str:
            return (
                f'<span style="white-space:nowrap"><span style="color:{value}">{glyph}</span>'
                f"&nbsp;{text}</span>"
            )

        solid, dashed = "━━", "╍╍"
        run = html.escape(self.comparison_label)
        if self.comparison_mode == _RESIDUAL:
            first = self.residual_threshold
            items = [
                key(t.residual_positive, solid, f"residual &gt; +{first:g}σ"),
                key(
                    t.residual_negative, dashed, f"&lt; −{first:g}σ" + (f" · {run}" if run else "")
                ),
            ]
        else:
            items = [key(t.measured, solid, "measured +"), key(t.measured_negative, dashed, "−")]
            if self.comparison_mode == 1:
                items.append(key(t.model, solid, f"model · {run}" if run else "model"))
        if run and self.comparison_mode:
            if any(i in self._by_id for i in self.added):
                items.append(key(t.warning, "◇", "not in this model"))
            if self.departed:
                items.append(key(t.secondary, "×", "position in this model"))
        if self.regions_visible and self.region_source:
            items.append(key(t.secondary, "▭", f"fitting regions · {self.region_source}"))
        if len(self.unadopted.data):
            items.append(key(t.model, "□", "fitted position, not adopted"))
        self.legend.setText(
            f'<div style="color:{t.secondary};font-size:{FONT_CAPTION}px">'
            + "&nbsp;&nbsp; ".join(items)
            + "</div>"
        )
        self.legend.setStyleSheet(
            f"background:{color(t.plot, 0.85).name(QtGui.QColor.NameFormat.HexArgb)};"
            "border-radius:4px;padding:3px 7px;"
        )
        self._place_legend()

    def apply_theme(self, theme: Theme) -> None:
        """Restyle retained geometry; no scientific work on an appearance change."""
        self.theme = t = theme
        # The empty corner beside the traces belongs to the plot surface.
        surface = self.palette()
        surface.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(t.plot))
        self.setPalette(surface)
        self.setAutoFillBackground(True)
        self.view.setBackground(t.plot)
        for axis in (self.bottom_axis, self.left_axis):
            axis.setPen(pg.mkPen(t.divider))
            axis.setTextPen(pg.mkPen(t.secondary))
        style = {"color": t.secondary, "font-size": f"{FONT_CAPTION + 1}px"}
        for axis, spectral in ((self.bottom_axis, self.x_axis), (self.left_axis, self.y_axis)):
            axis.labelStyle = style
            axis.setLabel(f"{self.axis_title(spectral)}  (ppm)")
        for line in (self.zero_x, self.zero_y, self.lane_zero_x, self.lane_zero_y):
            line.setPen(pg.mkPen(t.divider))
        for divider in (self.row_divider, self.column_divider):
            divider.color, divider.hot_color = t.border, t.accent
            divider.update()
        bar = pg.mkPen(t.secondary, width=1)
        for item in self.scale_bars:
            item.setPen(bar)
        for sign in self.scale_signs:
            sign.setBrush(QtGui.QColor(t.secondary))
        self._style_traces()
        self.marker_halo.setPen(pg.mkPen(color(t.plot, 0.9), width=2.6))
        self.markers.setPen(pg.mkPen(color(t.text, 0.72), width=1.0))
        self.ring.setPen(pg.mkPen(t.accent, width=2))
        self.cluster_rings.setPen(pg.mkPen(color(t.accent, 0.6), width=1.2))
        self.unadopted.setPen(pg.mkPen(t.model, width=1.4))
        if self._region_labels.size:
            self._paint_regions()
        self.halo.setPen(pg.mkPen(color(t.accent, 0.8), width=1.2))
        self.ghost.setPen(pg.mkPen(t.pin, width=2))
        self.added_marks.setPen(pg.mkPen(t.warning, width=1.6))
        self.recorded_marks.setPen(pg.mkPen(color(t.secondary, 0.9), width=1.4))
        self.recorded_links.setPen(
            pg.mkPen(color(t.secondary, 0.8), width=1, style=QtCore.Qt.PenStyle.DotLine)
        )
        fill = pg.mkBrush(color(t.plot, 0.86))
        for label, value in ((self.hover_label, t.text), (self.selection_label, t.accent)):
            label.fill = fill
            label.setColor(value)
        font = QtGui.QFont(QtWidgets.QApplication.font())
        font.setPixelSize(FONT_CAPTION + 1)
        self.hover_label.setFont(font)
        bold = QtGui.QFont(font)
        bold.setWeight(QtGui.QFont.Weight.DemiBold)
        self.selection_label.setFont(bold)
        self.pending_label.setColor(t.faint)
        self._style_contours()
        self._update_legend()
        # Do not change whether this is a selected-peak pin.
        following = self.pin_peak_id
        self.set_pinned(self.pinned)
        self.pin_peak_id = following
        for label in self._labels:
            label.setColor(t.text)

    def _refresh_labels(self) -> None:
        for label in self._labels:
            self.box.removeItem(label)
        self._labels.clear()
        if self.labels_visible:
            for peak in self.peaks:
                label = pg.TextItem(peak.name, anchor=(-0.1, 1.1), color=self.theme.text)
                label.setPos(*self.display(peak))
                label.setZValue(25)
                label.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
                self.box.addItem(label, ignoreBounds=True)
                self._labels.append(label)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        """Canvas-only navigation never consumes a text field's Space/Delete keys."""
        if watched is self.view.viewport() and isinstance(event, QtGui.QMouseEvent):
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                self._press_position = event.position()
                self._pointer_dragged = False
            elif (
                event.type() == QtCore.QEvent.Type.MouseMove
                and event.buttons() & QtCore.Qt.MouseButton.LeftButton
                and self._press_position is not None
                and (event.position() - self._press_position).manhattanLength()
                >= QtWidgets.QApplication.startDragDistance()
            ):
                # A throttled scene move must never turn a physical pan into a pin.
                self._pointer_dragged = True
        if event.type() == QtCore.QEvent.Type.Paint and watched is self.view.viewport():
            # Paint-completion measurements use a queued callback after paintEvent.
            self._paint_timer.start(0)
        if event.type() == QtCore.QEvent.Type.Leave:
            self.hover(None)
        if event.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            return self._double_click(self.view.mapToScene(event.position().toPoint()))
        if event.type() == QtCore.QEvent.Type.KeyPress:
            key = event.key()
            if key == QtCore.Qt.Key.Key_Space:
                self.set_pinned(not self.pinned)
                return True
            steps = {
                QtCore.Qt.Key.Key_Left: (-1, 0),
                QtCore.Qt.Key.Key_Right: (1, 0),
                QtCore.Qt.Key.Key_Up: (0, -1),
                QtCore.Qt.Key.Key_Down: (0, 1),
            }
            if key in steps:
                dx, dy = steps[key]
                ix = int(np.clip(self.ix + dx, 0, self.x_axis.size - 1))
                iy = int(np.clip(self.iy + dy, 0, self.y_axis.size - 1))
                self.sample(float(self.ppm_x[ix]), float(self.ppm_y[iy]))
                self.set_pinned(True)
                return True
        return super().eventFilter(watched, event)

    def _double_click(self, point: QtCore.QPointF) -> bool:
        """The 2D plot edits; a slice, lane or divider returns to its default."""
        for item, action in (
            (self.row_divider, lambda: self.reset_size("row")),
            (self.column_divider, lambda: self.reset_size("column")),
            (self.box, self.edit_requested.emit),
            (self.row_box, lambda: self.scale_intensity(None)),
            (self.column_box, lambda: self.scale_intensity(None)),
            (self.row_lane, lambda: self.scale_lanes(None)),
            (self.column_lane, lambda: self.scale_lanes(None)),
        ):
            if item.isVisible() and item.sceneBoundingRect().contains(point):
                action()
                return True
        return False

    # ----------------------------------------------------------------- slices
    def _effective_sizes(self) -> SliceSizes:
        """Residual mode, or a window without room, hides the lanes; the preference stays."""
        shown = self.sizes.lanes and self.comparison_mode != _RESIDUAL and self.lanes_fit
        return replace(self.sizes, lanes=shown)

    def minimum(self, sizes: SliceSizes | None = None) -> tuple[int, int]:
        """The view's minimum: the 2D plot's, with visible slices and lanes at theirs."""
        if sizes is None:
            sizes = self._effective_sizes()
        return minimum_size(sizes, self.extents)

    def _relayout(self) -> None:
        """Place every plot from the pure layout; shared extents align by construction."""
        sizes = self._effective_sizes()
        self.setMinimumSize(*self.minimum())
        geometry = layout(self.view.width(), self.view.height(), sizes, self.extents)
        self.arrangement = geometry
        for item, rect in (
            (self.box, geometry.main),
            (self.row_box, geometry.row),
            (self.row_lane, geometry.row_lane),
            (self.column_box, geometry.column),
            (self.column_lane, geometry.column_lane),
        ):
            item.setVisible(rect is not None)
            if rect is not None:
                item.setGeometry(QtCore.QRectF(*rect))
        x, y, w, h = geometry.main
        self.left_axis.setGeometry(QtCore.QRectF(0, y, self.extents.left, h))
        self.bottom_axis.setGeometry(QtCore.QRectF(x, y + h, w, self.extents.bottom))
        self.row_divider.setRect(QtCore.QRectF(*geometry.row_divider))
        self.column_divider.setRect(QtCore.QRectF(*geometry.column_divider))
        hx, hy, hw, hh = geometry.header
        self.caption.setGeometry(int(hx), int(hy) + 3, int(hw), int(hh))
        cx, cy, cw, ch = geometry.corner
        self.readout.setGeometry(
            int(cx) + 6, int(cy) + 4, max(0, int(cw) - 10), max(0, int(ch) - 6)
        )
        self.readout.setVisible(cw > _READOUT_MIN[0] and ch > _READOUT_MIN[1])
        self._place_legend()
        self._apply_bounds()

    def slice_bounds(self) -> tuple[float, float]:
        """The shown intensity range: manual when set, otherwise the region rule."""
        return self.manual_scale or self.region_scale

    def _shown_bounds(self) -> tuple[float, float]:
        if self.comparison_mode == _RESIDUAL:
            return -self.lane_sigma, self.lane_sigma
        return self.slice_bounds()

    def _apply_bounds(self) -> None:
        """Set slice and lane ranges; never called on hover."""
        low, high = self._shown_bounds()
        self.row_box.setYRange(low, high, padding=0)
        self.column_box.setXRange(low, high, padding=0)
        self.row_lane.setYRange(-self.lane_sigma, self.lane_sigma, padding=0)
        self.column_lane.setXRange(-self.lane_sigma, self.lane_sigma, padding=0)
        self._place_scale_bars()
        self._draw_slices()

    def _range_committed(self) -> None:
        """A deliberate view change: the region scale follows; a manual one stays."""
        rows = slices.index_range(self.y_axis, self.box.viewRange()[1])
        columns = slices.index_range(self.x_axis, self.box.viewRange()[0])
        self.region_scale = slices.intensity_range(self.spectra.data, rows, columns)
        self._apply_bounds()

    def scale_intensity(self, delta: float | None) -> None:
        """Wheel over a slice: a manual range about zero; None returns to the region rule."""
        if delta is None:
            self.manual_scale = None
        else:
            factor = 1 / _SCALE_STEP if delta > 0 else _SCALE_STEP
            low, high = self.slice_bounds()
            self.manual_scale = (low * factor, high * factor)
        self._apply_bounds()
        self.scale_changed.emit()

    def scale_lanes(self, delta: float | None) -> None:
        """Wheel over a lane: the shared ±N σ_stat range; None restores the default."""
        if delta is None:
            self.lane_sigma = LANE_SIGMA
        else:
            self.lane_sigma *= 1 / _SCALE_STEP if delta > 0 else _SCALE_STEP
        self._apply_bounds()
        self.scale_changed.emit()

    def set_lanes(self, visible: bool) -> None:
        """Show or hide the residual lanes (a preference kept across modes)."""
        self.sizes = replace(self.sizes, lanes=visible)
        self._relayout()
        self.sizes_changed.emit()

    def set_sizes(self, sizes: SliceSizes) -> None:
        """Adopt saved or framing-specific slice sizes."""
        self.sizes = sizes
        self._relayout()

    def resize_slice(self, which: str, step: int) -> None:
        """Keyboard and menu resizing, with the divider's limits."""
        current = getattr(self.arrangement.sizes, which)
        self.sizes = drag(
            self.sizes, which, current + step, self.view.width(), self.view.height(), self.extents
        )
        self._relayout()
        self.sizes_changed.emit()

    def reset_size(self, which: str) -> None:
        """Restore one slice's default size, within the divider's limits."""
        default = getattr(SliceSizes(), which)
        self.sizes = drag(
            self.sizes, which, default, self.view.width(), self.view.height(), self.extents
        )
        self._relayout()
        self.sizes_changed.emit()

    def reset_sizes(self) -> None:
        """Restore both slices' default sizes."""
        for which in ("row", "column"):
            self.reset_size(which)

    def _drag_divider(self, which: str, delta: float | None, *, finished: bool) -> None:
        if delta is None:
            self._drag_origin = getattr(self.arrangement.sizes, which)
            return
        value = self._drag_origin + (delta if which == "row" else -delta)
        self.sizes = drag(
            self.sizes, which, value, self.view.width(), self.view.height(), self.extents
        )
        self._relayout()
        if finished:
            self.sizes_changed.emit()

    def _draw_slices(self) -> None:
        """Native row and column at the crosshair; only data changes on hover."""
        data = self.spectra.data[self.plane]
        row, column = data[self.iy, :], data[:, self.ix]
        model = self._model.values if self._model is not None and self.comparison_mode else None
        noise = self.comparison_result.outcome.noise if self.comparison_result else 1.0
        model_row = model[self.iy, :] if model is not None else None
        model_column = model[:, self.ix] if model is not None else None
        lanes = self.comparison_mode == 1 and model is not None and self._effective_sizes().lanes
        if self.comparison_mode == _RESIDUAL:
            if model_row is None or model_column is None:
                for item in (self.curve_x, self.curve_y, self.model_x, self.model_y):
                    item.setData([], [])
                for item in (self.clip_x, self.clip_y, self.model_clip_x, self.model_clip_y):
                    item.setData([], [])
                self.status_changed.emit(
                    "Residual pending · calculating total model"
                    + (f" of {self.comparison_label}" if self.comparison_label else "")
                    + " for this plane"
                )
                self._update_readout(None, None)
                return
            row = slices.residual(row, model_row, noise)
            column = slices.residual(column, model_column, noise)
            model_row = model_column = None
        bounds = self._shown_bounds()
        self.clips = {
            "row": self._set_slice(
                self.curve_x,
                self.model_x,
                self.clip_x,
                self.model_clip_x,
                row,
                model_row,
                bounds,
                vertical=False,
            ),
            "column": self._set_slice(
                self.curve_y,
                self.model_y,
                self.clip_y,
                self.model_clip_y,
                column,
                model_column,
                bounds,
                vertical=True,
            ),
        }
        lane = (-self.lane_sigma, self.lane_sigma)
        if lanes and model is not None:
            self._set_lane(
                self.lane_x,
                self.lane_clip_x,
                slices.residual(data[self.iy, :], model[self.iy, :], noise),
                lane,
                vertical=False,
            )
            self._set_lane(
                self.lane_y,
                self.lane_clip_y,
                slices.residual(data[:, self.ix], model[:, self.ix], noise),
                lane,
                vertical=True,
            )
        else:
            for item in (self.lane_x, self.lane_y, self.lane_clip_x, self.lane_clip_y):
                item.setData([], [])
        raw = float(row[self.ix])
        self._emit_status(raw)
        value = float(data[self.iy, self.ix])
        fitted = float(model[self.iy, self.ix]) if model is not None else None
        self._update_readout(value, fitted)
        self._update_caption()

    def _set_slice(
        self, curve, model_curve, clip, model_clip, values, model_values, bounds, *, vertical
    ):
        """One slice: lines through native samples, clipped samples marked at the edge."""
        axis = self.ppm_y if vertical else self.ppm_x
        low, high = bounds
        inset = 0.015 * (high - low)

        def edge(points):
            return np.where(points > high, high - inset, low + inset)

        def place(item, points):
            return item.setData(points, axis) if vertical else item.setData(axis, points)

        place(curve, values)
        out = slices.clipped(values, low, high)
        marks = (edge(values[out]), axis[out])
        clip.setData(
            *(marks if vertical else marks[::-1]), symbol="t" if not vertical else "t2", size=8
        )
        if model_values is None:
            model_curve.setData([], [])
            model_clip.setData([], [])
            return int(out.size), 0
        place(model_curve, model_values)
        missed = slices.clipped(model_values, low, high)
        marks = (edge(model_values[missed]), axis[missed])
        model_clip.setData(
            *(marks if vertical else marks[::-1]), symbol="t" if not vertical else "t2", size=8
        )
        return int(out.size), int(missed.size)

    def _set_lane(self, curve, clip, values, bounds, *, vertical) -> None:
        axis = self.ppm_y if vertical else self.ppm_x
        low, high = bounds
        if vertical:
            curve.setData(values, axis)
        else:
            curve.setData(axis, values)
        out = slices.clipped(values, low, high)
        edge = np.where(values[out] > high, high * 0.95, low * 0.95)
        marks = (edge, axis[out]) if vertical else (axis[out], edge)
        clip.setData(*marks, symbol="t" if not vertical else "t2", size=7)

    def _place_scale_bars(self) -> None:
        """Bars from the lower to the upper bound, with a zero tick and their signs."""
        geometry = getattr(self, "arrangement", None)
        if geometry is None:
            return
        low, high = self._shown_bounds()
        row_bar, column_bar = self.scale_bars
        plus_row, minus_row, minus_column, plus_column = self.scale_signs
        path = QtGui.QPainterPath()
        visible = geometry.row is not None
        if visible:
            x = geometry.row[0] - 9
            top = self.row_box.mapViewToScene(QtCore.QPointF(0, high)).y()
            bottom = self.row_box.mapViewToScene(QtCore.QPointF(0, low)).y()
            zero = self.row_box.mapViewToScene(QtCore.QPointF(0, 0)).y()
            path.moveTo(x, top)
            path.lineTo(x, bottom)
            path.moveTo(x - 4, zero)
            path.lineTo(x + 4, zero)
            plus_row.setText("+")
            plus_row.setPos(x - 16, top - 2)
            minus_row.setText("−" if low < 0 else "")
            minus_row.setPos(x - 16, bottom - 14)
        row_bar.setPath(path)
        for item in (plus_row, minus_row):
            item.setVisible(visible)
        path = QtGui.QPainterPath()
        visible = geometry.column is not None
        if visible:
            _cx, cy, _cw, ch = geometry.column
            y = cy + ch + 9
            left = self.column_box.mapViewToScene(QtCore.QPointF(low, 0)).x()
            right = self.column_box.mapViewToScene(QtCore.QPointF(high, 0)).x()
            zero = self.column_box.mapViewToScene(QtCore.QPointF(0, 0)).x()
            path.moveTo(left, y)
            path.lineTo(right, y)
            path.moveTo(zero, y - 4)
            path.lineTo(zero, y + 4)
            minus_column.setText("−" if low < 0 else "")
            minus_column.setPos(left - 2, y + 2)
            plus_column.setText("+")
            plus_column.setPos(right - 8, y + 2)
        column_bar.setPath(path)
        for item in (plus_column, minus_column):
            item.setVisible(visible)

    def _update_caption(self) -> None:
        """One compact scale statement for both slices, the lanes and any clipping."""
        t = self.theme
        if self.comparison_mode == _RESIDUAL:
            text = f"Slices: residual ±{self.lane_sigma:.3g} σ_stat"
        else:
            low, high = self.slice_bounds()
            power = formatting.exponent(max(abs(low), abs(high)))
            factor = 10.0**-power
            unit = f"×10{formatting.superscript(str(power))} a.u." if power else "a.u."
            source = "manual" if self.manual_scale else "view, all planes"
            text = (
                f"Slices {formatting.number(low * factor, 2)} … "
                f"{formatting.number(high * factor, 2)} {unit} ({source})"
            )
            if self._effective_sizes().lanes:
                text += f" · lanes ±{self.lane_sigma:.3g} σ"
        data = sum(d for d, _m in self.clips.values())
        model = sum(m for _d, m in self.clips.values())
        if data:
            text += f' · <span style="color:{t.error}">▲ {data} clipped</span>'
        if model:
            text += f' · <span style="color:{t.model}">△ {model} model clipped</span>'
        self.caption.setText(text)
        self.caption.setToolTip(
            "Both slices share this range: the lowest and highest experimental intensity "
            "in the visible region over all planes, with zero included"
            if not self.manual_scale and self.comparison_mode != _RESIDUAL
            else "Wheel over a slice to change its range; double-click to reset it"
        )

    def _update_readout(self, value: float | None, fitted: float | None) -> None:
        """Exact values at the crosshair; the slices themselves carry no numbers."""
        lines = [
            f"<b>{formatting.number(self.ppm_x[self.ix], self.decimals_x)} · "
            f"{formatting.number(self.ppm_y[self.iy], self.decimals_y)} ppm</b>"
        ]
        if value is not None:
            lines.append(f"I {self._intensity(value)} a.u.")
        if self.comparison_mode and fitted is None:
            lines.append("model pending")
        elif fitted is not None and value is not None:
            noise = self.comparison_result.outcome.noise if self.comparison_result else 1.0
            lines.append(f"model {self._intensity(fitted)}")
            lines.append(f"residual {(value - fitted) / noise:+.1f} σ")
        self.readout.setText("<br>".join(lines))

    def slice_state(self) -> dict[str, object]:
        """Saved view state of the slices."""
        return {
            "row": self.sizes.row,
            "column": self.sizes.column,
            "lanes": self.sizes.lanes,
            "lane_sigma": self.lane_sigma,
            "scale": list(self.manual_scale) if self.manual_scale else None,
        }

    def restore_slice_state(self, state: dict) -> None:
        """Adopt saved slice sizes, lanes and scale."""
        self.sizes = replace(
            self.sizes,
            row=int(state.get("row", self.sizes.row)),
            column=int(state.get("column", self.sizes.column)),
            lanes=bool(state.get("lanes", self.sizes.lanes)),
        )
        self.lane_sigma = float(state.get("lane_sigma", LANE_SIGMA))
        scale = state.get("scale")
        self.manual_scale = (float(scale[0]), float(scale[1])) if scale else None
        self._relayout()

    def stop(self) -> None:
        """Discard stale deliveries without waiting on a worker in the UI thread."""
        self._commit.stop()
        self._pending_show.stop()
        self._revision += 1
        self._model_cancel.set()
        self._preview_timer.stop()
        self._paint_timer.stop()
        self._worker.close()
