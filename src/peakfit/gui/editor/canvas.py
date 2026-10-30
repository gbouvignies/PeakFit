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

from peakfit.gui.background import ViewerComputation
from peakfit.gui.editing import PeakMarkers
from peakfit.gui.fit_display import dataset_model_plane
from peakfit.gui.spectrum import contour_geometry, display_position, paths_from_geometry

if TYPE_CHECKING:
    from peakfit.gui.dataset_fit import DatasetFitResult
    from peakfit.gui.peaks import WorkingPeak
    from peakfit.gui.spectrum import ContourGeometry, Dataset

_PICK_RADIUS = 8.0  # Logical pixels, shared by hover/click/drag at every DPR.
_CACHE_BYTES = 32 * 1024**2
_CACHE_PLANES = 8
_RESIDUAL = 2
_MODEL_POSITIVE = 2
_MODEL_NEGATIVE = 3


def signed_geometry(data, axes, levels) -> list[tuple[np.ndarray, np.ndarray]]:
    """Combine levels into two sign paths; no Qt work on the numerical thread."""
    geometry = contour_geometry(data, axes, np.concatenate((levels, -levels)))
    count = len(levels)
    return [
        (np.concatenate([p for p, _ in group]), np.concatenate([c for _, c in group]))
        for group in (geometry[:count], geometry[count:])
    ]


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
        self._colors = {}
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
        self.top.setFixedHeight(112)
        self.right.setFixedWidth(116)
        grid.addWidget(self.top, 0, 0)
        grid.addWidget(self.main, 1, 0)
        grid.addWidget(self.right, 1, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        self.main.setMinimumSize(565, 330)
        self.main.setLabel("bottom", self.axis_title(self.x_axis), units="ppm")
        self.main.setLabel("left", self.axis_title(self.y_axis), units="ppm")
        for plot in (self.main, self.top):
            plot.getAxis("left").setWidth(64)
            plot.getViewBox().invertX(True)
        for plot in (self.main, self.right):
            plot.getAxis("bottom").setHeight(44)
            plot.getViewBox().invertY(True)
        self.top.hideAxis("bottom")
        self.right.hideAxis("left")
        self.top.getAxis("left").setLabel("a.u.")
        self.right.getAxis("bottom").setLabel("a.u.")
        self.top.setXLink(self.main)
        self.right.setYLink(self.main)
        self.top.setMouseEnabled(x=False, y=False)
        self.right.setMouseEnabled(x=False, y=False)
        self.curve_x = self.top.plot(antialias=True)
        self.curve_y = self.right.plot(antialias=True)
        self.model_x = self.top.plot(pen=pg.mkPen("#b47a32", width=1.5))
        self.model_y = self.right.plot(pen=pg.mkPen("#b47a32", width=1.5))
        self.top.addItem(pg.InfiniteLine(angle=0, pos=0, pen="#8493a0"))
        self.right.addItem(pg.InfiniteLine(angle=90, pos=0, pen="#8493a0"))
        self.cross_x = pg.InfiniteLine(angle=90, movable=False)
        self.cross_y = pg.InfiniteLine(angle=0, movable=False)
        for line in (self.cross_x, self.cross_y):
            line.setZValue(10)
            self.main.getPlotItem().addItem(line, ignoreBounds=True)
            line.sigDragged.connect(self._crosshair_dragged)
        self.markers = InspectionMarkers(symbol="+", size=6, brush=None, hoverable=True, tip=None)
        self.markers.setZValue(20)
        self.markers.sigClicked.connect(self._marker_clicked)
        self.markers.moved.connect(self._move_peak)
        self.main.addItem(self.markers)
        self.ring = self._scatter("o", 18, 30)
        self.halo = self._scatter("o", 15, 31)
        self.ghost = self._scatter("+", 10, 32)
        self.hover_label = pg.TextItem(anchor=(-0.1, 1.2))
        self.hover_label.setZValue(35)
        self.hover_label.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.main.getPlotItem().addItem(self.hover_label, ignoreBounds=True)
        self.paths: list[QtWidgets.QGraphicsPathItem] = []
        self.pending_label = pg.TextItem("Preparing contours…", anchor=(0, 0))
        self.pending_label.setZValue(40)
        self.pending_label.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.main.getPlotItem().addItem(self.pending_label, ignoreBounds=True)
        scene = cast("pg.GraphicsScene", self.main.scene())
        scene.sigMouseMoved.connect(self._pointer_moved)
        scene.sigMouseClicked.connect(self._blank_click)
        self._set_trace_limit(self.trace_limit)
        self.apply_colors(False)
        self.full_spectrum()
        self.sample(float(self.ppm_x[self.ix]), float(self.ppm_y[self.iy]))
        self.request_contours()

    @staticmethod
    def axis_title(axis) -> str:
        """Use loaded names rather than assume a particular pair of nuclei."""
        return f"{axis.nucleus or axis.label} · {axis.label}"

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
        for name in ("left", "bottom"):
            plot.getAxis(name).setStyle(maxTickLevel=0, tickLength=-4)
        return plot

    def _scatter(self, symbol, size, z):
        item = pg.ScatterPlotItem(symbol=symbol, size=size, brush=None)
        item.setZValue(z)
        item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.main.addItem(item)
        return item

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
        self.markers.setData(
            x=[p[0] for p in positions], y=[p[1] for p in positions], data=[p.id for p in peaks]
        )
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
        else:
            x, y = self.display(peak)
            self.ring.setData([x], [y])

    def hover(self, peak_id: str | None, *, from_list: bool = False) -> None:
        """Change transient feedback only, preserving editor drafts and selection."""
        if peak_id != self.hovered_id:
            self.hovered_id = peak_id
            peak = self._by_id.get(peak_id or "")
            if peak:
                x, y = self.display(peak)
                self.halo.setData([x], [y])
                self.hover_label.setText(
                    f"{peak.name} · {peak.positions[1]:.8f}, {peak.positions[0]:.8f} ppm"
                    + (
                        f"\nDisplayed {x:.8f}, {y:.8f}"
                        if (x, y) != tuple(reversed(peak.positions))
                        else ""
                    )
                )
                self.hover_label.setPos(x, y)
            else:
                self.halo.setData([], [])
                self.hover_label.setText("")
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
            f"{self.axis_title(self.x_axis)} {self.ppm_x[self.ix]:.8f}, "
            f"{self.axis_title(self.y_axis)} {self.ppm_y[self.iy]:.8f} ppm · "
            f"{raw:.9g} {'σ_stat residual' if self.comparison_mode == _RESIDUAL else 'a.u.'}"
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
                    self._colors["pin"] if pinned else self._colors["preview"],
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
                f"{peak.name} · {peak.positions[1]:.8f}, "
                f"{peak.positions[0]:.8f} ppm · {peak.id[:6]}"
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
        unit = "σ_stat" if self.comparison_mode == _RESIDUAL else "a.u."
        self.top.setLabel("left", unit)
        self.right.setLabel("bottom", unit)
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
        residual_levels = self.residual_threshold / 3 * np.asarray([3.0, 5.0, 8.0, 13.0, 21.0])

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
        for path in paths_from_geometry(geometry):
            item = QtWidgets.QGraphicsPathItem(path)
            item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            item.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)
            self.main.addItem(item)
            self.paths.append(item)
        self._style_contours()
        self.pending_label.setVisible(pending)
        self.contour_pending = pending

    def _style_contours(self) -> None:
        for index, item in enumerate(self.paths):
            pen = pg.mkPen(
                (
                    "#b47a32"
                    if index >= _MODEL_POSITIVE
                    or (
                        self.comparison_mode == _RESIDUAL
                        and not self._displaying_experiment
                        and index == 0
                    )
                    else self._colors["measured"]
                ),
                width=1.8 if self.heavy_strokes else 1,
                style=(
                    QtCore.Qt.PenStyle.DotLine
                    if index == _MODEL_NEGATIVE
                    else QtCore.Qt.PenStyle.DashLine
                    if index in (1, 2)
                    else QtCore.Qt.PenStyle.SolidLine
                ),
            )
            pen.setCosmetic(True)
            item.setPen(pen)

    def apply_colors(self, dark: bool) -> None:
        """Restyle retained geometry; no scientific work on an appearance change."""
        self._colors = {
            "bg": "#171d24" if dark else "#fbfcfd",
            "fg": "#e1e7ee" if dark else "#25313d",
            "muted": "#aab7c4" if dark else "#586875",
            "measured": "#73b5d5" if dark else "#23688a",
            "pin": "#e2ad65" if dark else "#966021",
            "preview": "#78accc" if dark else "#357da7",
        }
        colors = self._colors
        for plot in (self.top, self.main, self.right):
            plot.setBackground(colors["bg"])
            for name in ("left", "bottom"):
                axis = plot.getAxis(name)
                axis.setPen(colors["muted"])
                axis.setTextPen(colors["muted"])
                axis.setLabel(axis.labelText, units=axis.labelUnits, color=colors["muted"])
        for curve in (self.curve_x, self.curve_y):
            curve.setPen(pg.mkPen(colors["measured"], width=1.5))
        self.markers.setPen(pg.mkPen(colors["muted"], width=1))
        self.ring.setPen(pg.mkPen(colors["measured"], width=2))
        self.halo.setPen(pg.mkPen(colors["measured"], width=1))
        self.ghost.setPen(pg.mkPen(colors["pin"], width=2))
        self.hover_label.fill = pg.mkBrush(colors["bg"])
        self.hover_label.setColor(colors["fg"])
        self.pending_label.setColor(colors["muted"])
        self._style_contours()
        # Do not change whether this is a selected-peak pin.
        following = self.pin_peak_id
        self.set_pinned(self.pinned)
        self.pin_peak_id = following
        for label in self._labels:
            label.setColor(colors["fg"])

    def _refresh_labels(self) -> None:
        for label in self._labels:
            self.main.removeItem(label)
        self._labels.clear()
        if self.labels_visible:
            for peak in self.peaks:
                label = pg.TextItem(peak.name, anchor=(-0.1, 1.1), color=self._colors["fg"])
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
