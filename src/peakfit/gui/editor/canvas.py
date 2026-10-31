"""Retained signed contours, aligned native slices and independent inspection gestures."""

from collections import OrderedDict
from concurrent.futures import CancelledError
from dataclasses import replace
from threading import Event
from typing import TYPE_CHECKING, cast

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# isort: split
import pyqtgraph as pg

from peakfit.gui import formatting
from peakfit.gui.background import ViewerComputation
from peakfit.gui.editing import PeakMarkers
from peakfit.gui.fit_display import dataset_model_plane
from peakfit.gui.spectrum import contour_geometry, display_position, paths_from_geometry
from peakfit.gui.theme import FONT_CAPTION, LIGHT, Theme, color

if TYPE_CHECKING:
    from peakfit.gui.dataset_fit import DatasetFitResult
    from peakfit.gui.peaks import WorkingPeak
    from peakfit.gui.spectrum import ContourGeometry, Dataset

_PICK_RADIUS = 8.0  # Logical pixels, shared by hover/click/drag at every DPR.
_CACHE_BYTES = 32 * 1024**2
_CACHE_PLANES = 8
_RESIDUAL = 2
_RESIDUAL_LEVELS = np.asarray([3.0, 5.0, 8.0, 13.0, 21.0])


def signed_geometry(data, axes, levels) -> list[tuple[np.ndarray, np.ndarray]]:
    """One geometry per level, positive levels then negative; no Qt work here."""
    return contour_geometry(data, axes, np.concatenate((levels, -levels)))


class ContourLayer(QtWidgets.QGraphicsItem):
    """One layer of signed contour levels painted as a single cached graphic.

    Positive levels come first, then the matching negative levels. Each level has
    its own pen so low levels can be lighter, without one scene item per level.
    """

    def __init__(self, role: str, paths: list[QtGui.QPainterPath]) -> None:
        super().__init__()
        self.role = role
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


class SpectrumCanvas(QtWidgets.QWidget):
    """Display observations only; scientific edits are proposals to the owning workspace."""

    selected = QtCore.Signal(str)
    hovered = QtCore.Signal(object)
    moved = QtCore.Signal(str, object)
    placed = QtCore.Signal(object)
    edit_requested = QtCore.Signal()
    status_changed = QtCore.Signal(str)
    pin_changed = QtCore.Signal(bool)
    draft_changed = QtCore.Signal(bool)
    painted = QtCore.Signal()

    def __init__(self, dataset: Dataset) -> None:  # noqa: PLR0915
        super().__init__()
        self.dataset = dataset
        self.spectra = dataset.spectra
        self.y_axis, self.x_axis = self.spectra.spectral_params
        self.ppm_x = self.x_axis.pts2ppm(np.arange(self.x_axis.size, dtype=float))
        self.ppm_y = self.y_axis.pts2ppm(np.arange(self.y_axis.size, dtype=float))
        self.comparison_result: DatasetFitResult | None = None
        self.comparison_mode = 0
        self.residual_threshold = 3.0
        self._model_cache = OrderedDict()
        self._model_cancel = Event()
        self._model = None
        self._expanded_model = False
        self._scale_note = ""
        self._trace_limits = {0: dataset.trace_limit, 2: 8.0}
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
        self.trace_limit = dataset.trace_limit
        self.contour_threshold = dataset.noise * 5
        self.contour_factor = 1.6
        self.contour_count = 10
        self.heavy_strokes = False
        self.labels_visible = False
        self._labels: list[pg.TextItem] = []
        self._revision = 0
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
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        self.top = self._plot()
        self.main = self._plot()
        self.right = self._plot()
        self.main.setViewportUpdateMode(
            QtWidgets.QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate
        )
        self.main.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.main.viewport().installEventFilter(self)
        self.main.installEventFilter(self)
        self.setFocusProxy(self.main)
        self.main.setAccessibleName(
            "2D spectrum; Space pins slices; A places peak; arrows step sample"
        )
        self.top.setAccessibleName("Horizontal native cross-section; raw intensity")
        self.right.setAccessibleName("Vertical native cross-section; raw intensity")
        self.top.setFixedHeight(96)
        self.right.setFixedWidth(104)
        grid.addWidget(self.top, 0, 0)
        grid.addWidget(self.main, 1, 0)
        grid.addWidget(self.right, 1, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        self.main.setMinimumSize(400, 300)
        for plot in (self.main, self.top):
            plot.getAxis("left").setWidth(58)
            plot.getViewBox().invertX(True)
        for plot in (self.main, self.right):
            plot.getAxis("bottom").setHeight(38)
            plot.getViewBox().invertY(True)
        self.top.hideAxis("bottom")
        self.right.hideAxis("left")
        self.top.setXLink(self.main)
        self.right.setYLink(self.main)
        self.top.setMouseEnabled(x=False, y=False)
        self.right.setMouseEnabled(x=False, y=False)
        self.curve_x = self.top.plot(antialias=True)
        self.curve_y = self.right.plot(antialias=True)
        self.model_x = self.top.plot(antialias=True)
        self.model_y = self.right.plot(antialias=True)
        self.zero_x = pg.InfiniteLine(angle=0, pos=0)
        self.zero_y = pg.InfiniteLine(angle=90, pos=0)
        self.top.addItem(self.zero_x)
        self.right.addItem(self.zero_y)
        self.cross_x = pg.InfiniteLine(angle=90, movable=False)
        self.cross_y = pg.InfiniteLine(angle=0, movable=False)
        for line in (self.cross_x, self.cross_y):
            line.setZValue(10)
            self.main.getPlotItem().addItem(line, ignoreBounds=True)
            line.sigDragged.connect(self._crosshair_dragged)
        # A plot-coloured cross under each marker keeps it legible over contours.
        self.marker_halo = self._scatter("+", 8, 19)
        self.markers = InspectionMarkers(
            symbol="+", size=7, brush=None, hoverable=True, tip=self._marker_tip
        )
        self.markers.setZValue(20)
        self.markers.sigClicked.connect(self._marker_clicked)
        self.markers.moved.connect(self._move_peak)
        self.main.addItem(self.markers)
        self.ring = self._scatter("o", 18, 30)
        self.halo = self._scatter("o", 15, 31)
        self.ghost = self._scatter("+", 10, 32)
        self.hover_label = self._label(35, (-0.08, 1.15))
        self.selection_label = self._label(34, (-0.12, 1.05))
        self.paths: list[ContourLayer] = []
        self.pending_label = pg.TextItem("Preparing contours…", anchor=(0, 0))
        self.pending_label.setZValue(40)
        self.pending_label.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.main.getPlotItem().addItem(self.pending_label, ignoreBounds=True)
        self.legend = QtWidgets.QLabel(self.main)
        self.legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.legend.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.legend.setAccessibleName("Contour legend")
        scene = cast("pg.GraphicsScene", self.main.scene())
        scene.sigMouseMoved.connect(self._pointer_moved)
        scene.sigMouseClicked.connect(self._blank_click)
        self._set_trace_limit(self.trace_limit)
        self.apply_theme(LIGHT)
        self.full_spectrum()
        self.sample(float(self.ppm_x[self.ix]), float(self.ppm_y[self.iy]))
        self.request_contours()

    @staticmethod
    def axis_title(axis) -> str:
        """Use loaded names rather than assume a particular pair of nuclei."""
        return f"{formatting.nucleus(axis.nucleus, axis.label)} · {axis.label}"

    @staticmethod
    def _plot() -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        plot.hideButtons()
        plot.setMenuEnabled(False)
        plot.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        plot.getViewBox().setDefaultPadding(0)
        plot.getPlotItem().layout.setContentsMargins(0, 0, 0, 0)
        plot.getPlotItem().layout.setSpacing(0)
        font = QtGui.QFont(QtWidgets.QApplication.font())
        font.setPixelSize(FONT_CAPTION)
        for name in ("left", "bottom"):
            axis = plot.getAxis(name)
            axis.setStyle(maxTickLevel=0, tickLength=-4, tickFont=font, tickTextOffset=4)
            axis.enableAutoSIPrefix(False)
        return plot

    def _scatter(self, symbol, size, z):
        item = pg.ScatterPlotItem(symbol=symbol, size=size, brush=None)
        item.setZValue(z)
        item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.main.addItem(item)
        return item

    def _label(self, z: float, anchor: tuple[float, float]) -> pg.TextItem:
        label = pg.TextItem(anchor=anchor)
        label.setZValue(z)
        label.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.main.getPlotItem().addItem(label, ignoreBounds=True)
        return label

    def _marker_tip(self, *, x: float, y: float, data: str) -> str:
        """Delayed native tooltip with exact unfolded and displayed coordinates."""
        peak = self._by_id.get(data)
        if peak is None:
            return ""
        text = f"{peak.name}\nUnfolded {self._coordinates(peak.positions[1], peak.positions[0])}"
        if (x, y) != tuple(reversed(peak.positions)):
            text += f"\nDisplayed {self._coordinates(x, y)}"
        return text

    def _coordinates(self, x: float, y: float) -> str:
        return (
            f"{formatting.number(x, self.decimals_x)}, {formatting.number(y, self.decimals_y)} ppm"
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Keep the legend in the plot's top-right corner."""
        super().resizeEvent(event)
        self._place_legend()

    def _place_legend(self) -> None:
        self.legend.adjustSize()
        rect = self.main.getViewBox().sceneBoundingRect()
        corner = self.main.mapFromScene(rect.topRight())
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
        positions = [self.display(peak) for peak in peaks]
        xs, ys = [p[0] for p in positions], [p[1] for p in positions]
        self.marker_halo.setData(x=xs, y=ys)
        self.markers.setData(x=xs, y=ys, data=[p.id for p in peaks])
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
            view = self.main.getViewBox()
            ranges = view.viewRange()
            shifted = []
            for value, (low, high) in zip((x, y), ranges, strict=True):
                margin = (high - low) * 0.05
                delta = min(value - low - margin, 0) + max(value - high + margin, 0)
                shifted.append((low + delta, high + delta))
            view.setRange(xRange=shifted[0], yRange=shifted[1], padding=0)

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
        view = self.main.getViewBox()
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
        view = self.main.getViewBox()
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
        data = self.spectra.data[self.plane]
        row, column = data[self.iy, :], data[:, self.ix]
        self.model_x.clear()
        self.model_y.clear()
        if self.comparison_mode == _RESIDUAL:
            if self._model is None:
                self.curve_x.clear()
                self.curve_y.clear()
                self.status_changed.emit(
                    "Residual pending · calculating total model for this plane"
                )
                return
            assert self.comparison_result is not None
            noise = self.comparison_result.outcome.noise
            row = (row - self._model.values[self.iy, :]) / noise
            column = (column - self._model.values[:, self.ix]) / noise
        elif self.comparison_mode == 1 and self._model is not None:
            self.model_x.setData(self.ppm_x, self._model.values[self.iy, :])
            self.model_y.setData(self._model.values[:, self.ix], self.ppm_y)
        self.curve_x.setData(self.ppm_x, row)
        self.curve_y.setData(column, self.ppm_y)
        raw = float(row[self.ix])
        clip = max(float(np.max(np.abs(row))), float(np.max(np.abs(column))))
        if self.comparison_mode == 1 and self._model is not None:
            clip = max(
                clip,
                float(np.max(np.abs(self._model.values[self.iy, :]))),
                float(np.max(np.abs(self._model.values[:, self.ix]))),
            )
        self._emit_status(raw, clip > self.trace_limit)

    def _emit_status(self, raw: float, clipped: bool) -> None:
        text = (
            f"{'Draft' if self._drag_peak else 'Pinned' if self.pinned else 'Preview'} · "
            f"{self.axis_title(self.x_axis)} "
            f"{formatting.number(self.ppm_x[self.ix], self.decimals_x)}, "
            f"{self.axis_title(self.y_axis)} "
            f"{formatting.number(self.ppm_y[self.iy], self.decimals_y)} ppm · "
            f"{formatting.intensity(raw)} "
            f"{'σ_stat residual' if self.comparison_mode == _RESIDUAL else 'a.u.'}"
            + (" · ▲ Trace clipped: use Display → Fit current traces" if clipped else "")
            + self._scale_note
        )
        self.status_changed.emit(text)
        self.main.setAccessibleDescription(text)

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
        view = self.main.getViewBox()
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
        menu.popup(self.main.viewport().mapToGlobal(self.main.mapFromScene(scene_position)))
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
        self.main.viewport().setCursor(
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
        self.main.getViewBox().setRange(
            xRange=(float(min(self.ppm_x)), float(max(self.ppm_x))),
            yRange=(float(min(self.ppm_y)), float(max(self.ppm_y))),
            padding=0,
        )

    def set_plane(self, plane: int) -> None:
        """Use original acquisition index, retaining view and sampled row/column."""
        self._model = None
        self.plane = int(np.clip(plane, 0, self.spectra.data.shape[0] - 1))
        self.request_contours()

    def _set_trace_limit(self, value: float) -> None:
        self.trace_limit = value
        self.top.getViewBox().setYRange(-value, value, padding=0)
        self.right.getViewBox().setXRange(-value, value, padding=0)
        self._label_traces()

    def _label_traces(self) -> None:
        """Signed trace axes share one locked range, labelled with a common power of ten."""
        power = formatting.exponent(self.trace_limit)
        unit = "σ" if self.comparison_mode == _RESIDUAL else "a.u."
        title = f"×10{formatting.superscript(str(power))} {unit}" if power else unit
        style = {"color": getattr(self, "theme", LIGHT).faint, "font-size": f"{FONT_CAPTION}px"}
        for axis in (self.top.getAxis("left"), self.right.getAxis("bottom")):
            axis.setScale(10.0**-power)
            axis.setLabel(title, **style)

    def fit_current_traces(self) -> None:
        """Explicit gain change, then keep a common signed scale locked again."""
        data = self.spectra.data[self.plane]
        if self.comparison_mode == _RESIDUAL:
            if self._model is None:
                return
            assert self.comparison_result is not None
            data = (data - self._model.values) / self.comparison_result.outcome.noise
        value = max(
            float(np.max(np.abs(data[self.iy, :]))), float(np.max(np.abs(data[:, self.ix])))
        )
        if self.comparison_mode == 1 and self._model is not None:
            value = max(
                value,
                float(np.max(np.abs(self._model.values[self.iy, :]))),
                float(np.max(np.abs(self._model.values[:, self.ix]))),
            )
        self._trace_limits[2 if self.comparison_mode == _RESIDUAL else 0] = max(
            value * 1.05, np.finfo(float).tiny
        )
        self._set_trace_limit(self._trace_limits[2 if self.comparison_mode == _RESIDUAL else 0])
        self.sample(float(self.ppm_x[self.ix]), float(self.ppm_y[self.iy]))

    def set_comparison(self, result: DatasetFitResult | None, mode: int = 0) -> None:
        """Bind a current immutable run and one mode across all three spectral views."""
        if result is not self.comparison_result:
            self._model_cache.clear()
            self._expanded_model = False
            self._scale_note = ""
        self.comparison_result = result
        self.comparison_mode = mode if result is not None else 0
        self._set_trace_limit(self._trace_limits[2 if self.comparison_mode == _RESIDUAL else 0])
        self._style_traces()
        self._update_legend()
        self.request_contours()

    def request_contours(self) -> None:  # noqa: PLR0915
        """Coalesce complete-plane projection and geometry; hover only reads arrays."""
        self._revision += 1
        self._model_cancel.set()
        self._model_cancel = Event()
        cancel = self._model_cancel
        for item in self.paths:
            self.main.removeItem(item)
        self.paths.clear()
        self._model = None
        result, mode, plane = self.comparison_result, self.comparison_mode, self.plane
        run = result.run_id if result is not None and mode else None
        key = (
            run,
            mode,
            plane,
            self.contour_threshold,
            self.contour_factor,
            self.contour_count,
            self.residual_threshold,
        )
        if not mode:
            key = self._experiment_key()
        model_key = (run, plane)
        self.contour_pending = True
        self.pending_label.setPos(
            self.main.getViewBox().viewRange()[0][1], self.main.getViewBox().viewRange()[1][0]
        )
        mode_name = "Residual" if mode == _RESIDUAL else "Overlay"
        self.pending_label.setText(
            f"Experimental spectrum · {mode_name} pending"
            if mode
            else "Preparing experimental contours…"
        )
        self.pending_label.show()
        self._worker.cancel("contours")
        self._worker.cancel("model")
        grid = self._model_cache.get(model_key) if mode else None
        self._model = grid
        self.sample(float(self.ppm_x[self.ix]), float(self.ppm_y[self.iy]))
        if key in self._cache and (not mode or grid is not None):
            geometry = self._cache.pop(key)
            self._cache[key] = geometry
            self._install_contours(geometry, experimental=not mode)
            return
        observed = self.spectra.data[plane]
        axes = [axis.model_copy(deep=True) for axis in self.spectra.spectral_params]
        levels = self.contour_threshold * self.contour_factor ** np.arange(self.contour_count)
        self.contour_jobs += 1
        experiment_key = self._experiment_key()
        experiment = [self._cache.get(experiment_key)]
        if experiment[0] is not None:
            self._cache.move_to_end(experiment_key)
            self._install_contours(experiment[0], pending=bool(mode), experimental=True)
        else:

            def prepare_experiment():
                if cancel.is_set():
                    raise CancelledError
                experiment[0] = signed_geometry(observed, axes, levels)
                return experiment[0]

            self._worker.submit("contours", self._revision, prepare_experiment)
        if not mode:
            return
        residual_levels = self.residual_threshold / 3 * _RESIDUAL_LEVELS

        def calculate():
            model = grid
            if mode:
                assert result is not None
                model = model or dataset_model_plane(result, axes, plane, cancel=cancel)
            if cancel.is_set():
                raise CancelledError
            if mode == _RESIDUAL:
                assert model is not None
                assert result is not None
                residual = (observed - model.values) / result.outcome.noise
                geometry = signed_geometry(residual, axes, residual_levels)
            else:
                assert model is not None
                assert experiment[0] is not None
                geometry = experiment[0] + signed_geometry(model.values, axes, levels)
            return model, geometry

        self._worker.submit("model", self._revision, calculate)

    def _experiment_key(self) -> tuple:
        return (
            None,
            0,
            self.plane,
            self.contour_threshold,
            self.contour_factor,
            self.contour_count,
            0,
        )

    def _contours_ready(self, kind, revision, outcome) -> None:
        if revision != self._revision:
            return
        if isinstance(outcome, Exception):
            self.pending_label.setText(f"Comparison unavailable: {outcome}")
            self.contour_pending = False
            return
        if kind == "contours":
            self._cache_geometry(self._experiment_key(), outcome)
            self._install_contours(outcome, pending=bool(self.comparison_mode), experimental=True)
            return
        model, geometry = outcome
        run = (
            self.comparison_result.run_id
            if self.comparison_result and self.comparison_mode
            else None
        )
        key = (
            run,
            self.comparison_mode,
            self.plane,
            self.contour_threshold,
            self.contour_factor,
            self.contour_count,
            self.residual_threshold,
        )
        if model is not None:
            model_key = (run, self.plane)
            self._model_cache[model_key] = model
            self._model_cache.move_to_end(model_key)
            while (
                len(self._model_cache) > _CACHE_PLANES
                or sum(m.nbytes for m in self._model_cache.values()) > 128 * 1024**2
            ):
                self._model_cache.popitem(last=False)
        self._model = model
        if model is not None and not self._expanded_model:
            maximum = float(np.max(np.abs(model.values))) * 1.05
            if maximum > self._trace_limits[0]:
                self._trace_limits[0] = maximum
                self._expanded_model = True
                self._scale_note = " · Trace scale expanded for accepted model"
                if self.comparison_mode != _RESIDUAL:
                    self._set_trace_limit(maximum)
        self._cache_geometry(key, geometry)
        self._install_contours(geometry)
        self.sample(float(self.ppm_x[self.ix]), float(self.ppm_y[self.iy]))

    def _cache_geometry(self, key, geometry) -> None:
        previous = self._cache.pop(key, ())
        self.cache_bytes -= sum(p.nbytes + c.nbytes for p, c in previous)
        size = sum(p.nbytes + c.nbytes for p, c in geometry)
        while self._cache and (
            self.cache_bytes + size > _CACHE_BYTES or len(self._cache) >= _CACHE_PLANES
        ):
            _, evicted = self._cache.popitem(last=False)
            self.cache_bytes -= sum(p.nbytes + c.nbytes for p, c in evicted)
        if size <= _CACHE_BYTES:
            self._cache[key] = geometry
            self.cache_bytes += size

    def _install_contours(self, geometry, *, pending=False, experimental=False) -> None:
        for item in self.paths:
            self.main.removeItem(item)
        self.paths.clear()
        self._displaying_experiment = experimental
        paths = paths_from_geometry(geometry)
        if self.comparison_mode == _RESIDUAL and not experimental:
            layers = [("residual", paths)]
        elif len(paths) == 4 * self.contour_count:
            half = 2 * self.contour_count
            layers = [("measured", paths[:half]), ("model", paths[half:])]
        else:
            layers = [("measured", paths)]
        for role, levels in layers:
            item = ContourLayer(role, levels)
            self.main.addItem(item)
            self.paths.append(item)
        self._style_contours()
        self.pending_label.setVisible(pending)
        self.contour_pending = pending

    def _style_contours(self) -> None:
        """Hue and stroke encode layer and sign; lightness grades the levels."""
        t = self.theme
        weight = 1.7 if self.heavy_strokes else 1.0
        solid, dash, dot = (
            QtCore.Qt.PenStyle.SolidLine,
            QtCore.Qt.PenStyle.DashLine,
            QtCore.Qt.PenStyle.DotLine,
        )
        overlay = any(layer.role == "model" for layer in self.paths)
        for layer in self.paths:
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
        t = self.theme
        residual = self.comparison_mode == _RESIDUAL
        measured = pg.mkPen(color(t.text if residual else t.measured), width=1.3)
        model = pg.mkPen(color(t.model), width=1.3, style=QtCore.Qt.PenStyle.DashLine)
        for curve in (self.curve_x, self.curve_y):
            curve.setPen(measured)
        for curve in (self.model_x, self.model_y):
            curve.setPen(model)

    def _update_legend(self) -> None:
        """Name each visible layer with its stroke, so no encoding needs memorizing."""
        t = self.theme

        def key(value: str, glyph: str, text: str) -> str:
            return f'<span style="color:{value}">{glyph}</span>&nbsp;{text}'

        solid, dashed = "━━", "╍╍"
        if self.comparison_mode == _RESIDUAL:
            first = self.residual_threshold
            items = [
                key(t.residual_positive, solid, f"residual &gt; +{first:g}σ"),
                key(t.residual_negative, dashed, f"&lt; −{first:g}σ"),
            ]
        else:
            items = [key(t.measured, solid, "measured +"), key(t.measured_negative, dashed, "−")]
            if self.comparison_mode == 1:
                items.append(key(t.model, solid, "model"))
        self.legend.setText(
            f'<div style="color:{t.secondary};font-size:{FONT_CAPTION}px">'
            + "&nbsp;&nbsp;&nbsp;".join(items)
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
        for plot in (self.top, self.main, self.right):
            plot.setBackground(t.plot)
            for name in ("left", "bottom"):
                axis = plot.getAxis(name)
                axis.setPen(pg.mkPen(t.divider))
                axis.setTextPen(pg.mkPen(t.secondary))
        style = {"color": t.secondary, "font-size": f"{FONT_CAPTION + 1}px"}
        self.main.setLabel("bottom", f"{self.axis_title(self.x_axis)}  (ppm)", **style)
        self.main.setLabel("left", f"{self.axis_title(self.y_axis)}  (ppm)", **style)
        self._label_traces()
        for line in (self.zero_x, self.zero_y):
            line.setPen(pg.mkPen(t.divider))
        self._style_traces()
        self.marker_halo.setPen(pg.mkPen(color(t.plot, 0.9), width=2.6))
        self.markers.setPen(pg.mkPen(color(t.text, 0.72), width=1.0))
        self.ring.setPen(pg.mkPen(t.accent, width=2))
        self.halo.setPen(pg.mkPen(color(t.accent, 0.8), width=1.2))
        self.ghost.setPen(pg.mkPen(t.pin, width=2))
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
            self.main.removeItem(label)
        self._labels.clear()
        if self.labels_visible:
            for peak in self.peaks:
                label = pg.TextItem(peak.name, anchor=(-0.1, 1.1), color=self.theme.text)
                label.setPos(*self.display(peak))
                label.setZValue(25)
                label.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
                self.main.getPlotItem().addItem(label, ignoreBounds=True)
                self._labels.append(label)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        """Canvas-only navigation never consumes a text field's Space/Delete keys."""
        if watched is self.main.viewport() and isinstance(event, QtGui.QMouseEvent):
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
        if event.type() == QtCore.QEvent.Type.Paint and watched is self.main.viewport():
            # Paint-completion measurements use a queued callback after paintEvent.
            self._paint_timer.start(0)
        if event.type() == QtCore.QEvent.Type.Leave:
            self.hover(None)
        if event.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            self.edit_requested.emit()
            return True
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

    def stop(self) -> None:
        """Discard stale deliveries without waiting on a worker in the UI thread."""
        self._revision += 1
        self._model_cancel.set()
        self._preview_timer.stop()
        self._paint_timer.stop()
        self._worker.close()
