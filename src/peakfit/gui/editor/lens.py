"""Cluster lens: one cluster judged as Experimental, Overlay and whole-spectrum residual.

The lens replaces the spectral area while open. Its three panels share zoom,
plane, markers and the selected member's crosshair; below them, the row through
the selected member shows measured data with the total model, and the
whole-spectrum residual in σ_stat with a ±3 σ band. Contours, model and residual
are the same quantities as in the overview (§8), computed off Qt.
"""

from threading import Event
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui import formatting
from peakfit.gui import theme as themes
from peakfit.gui.background import ViewerComputation
from peakfit.gui.editor.canvas import (
    _RESIDUAL_LEVELS,
    ContourLayer,
    SpectrumCanvas,
    _level_pens,
    signed_geometry,
)
from peakfit.gui.fit_display import dataset_model_plane
from peakfit.gui.spectrum import display_position, paths_from_geometry
from peakfit.gui.theme import FONT_CAPTION, color

if TYPE_CHECKING:
    from peakfit.gui.dataset_fit import DatasetFitResult
    from peakfit.gui.peaks import WorkingPeak
    from peakfit.gui.spectrum import Dataset

_MARGIN = 0.35  # default region: fitting-sample extent plus this fraction on each side
_MINIMUM_POINTS = 6  # and at least this many points on each side
_BAND = 3.0  # σ band around the residual trace
_TITLES = ("Experimental", "Overlay", "Whole-spectrum residual")


class ClusterLens(QtWidgets.QWidget):
    """Three linked views of one recorded cluster and the row through one member."""

    close_requested = QtCore.Signal()
    selected = QtCore.Signal(str)

    def __init__(self, dataset: Dataset) -> None:  # noqa: PLR0915
        super().__init__()
        self.setObjectName("lens")
        # Match the overview's minimum: a stacked spectral area takes its largest page's.
        self.setMinimumSize(400, 300)
        self.spectra = dataset.spectra
        self.y_axis, self.x_axis = self.spectra.spectral_params
        self.ppm_x = self.x_axis.pts2ppm(np.arange(self.x_axis.size, dtype=float))
        self.ppm_y = self.y_axis.pts2ppm(np.arange(self.y_axis.size, dtype=float))
        self.theme = themes.LIGHT
        self.levels = (1.0, 1.4, 6, 3.0)  # threshold, factor, count, residual threshold
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(themes.SPACE_M, themes.SPACE_S, themes.SPACE_M, themes.SPACE_S)
        layout.setSpacing(themes.SPACE_S)
        header = QtWidgets.QHBoxLayout()
        self.back = QtWidgets.QPushButton("‹ Spectrum")
        self.back.setObjectName("chip")
        self.back.setToolTip("Return to the full spectrum (Escape)")
        self.back.clicked.connect(self.close_requested)
        self.title = QtWidgets.QLabel()
        self.title.setObjectName("value")
        self.note = QtWidgets.QLabel()
        self.note.setObjectName("muted")
        header.addWidget(self.back)
        header.addSpacing(themes.SPACE_S)
        header.addWidget(self.title)
        header.addSpacing(themes.SPACE_S)
        header.addWidget(self.note, 1)
        layout.addLayout(header)
        panels = QtWidgets.QHBoxLayout()
        panels.setSpacing(themes.SPACE_S)
        self.panels: list[pg.PlotWidget] = []
        self.captions: list[QtWidgets.QLabel] = []
        self.layers: list[list[ContourLayer]] = [[], [], []]
        self.markers: list[pg.ScatterPlotItem] = []
        self.rings: list[pg.ScatterPlotItem] = []
        self.crosses: list[tuple[pg.InfiniteLine, pg.InfiniteLine]] = []
        self.unavailable: list[pg.TextItem] = []
        for caption in _TITLES:
            column = QtWidgets.QVBoxLayout()
            column.setSpacing(2)
            label = QtWidgets.QLabel(caption)
            label.setObjectName("section")
            column.addWidget(label)
            plot = SpectrumCanvas._plot()
            plot.getViewBox().invertX(True)
            plot.getViewBox().invertY(True)
            plot.getAxis("left").setWidth(48)
            plot.getAxis("bottom").setHeight(30)
            plot.setAccessibleName(f"Cluster lens · {caption}")
            if self.panels:
                plot.setXLink(self.panels[0])
                plot.setYLink(self.panels[0])
            markers = pg.ScatterPlotItem(symbol="+", size=8, brush=None, hoverable=True)
            markers.setZValue(20)
            markers.sigClicked.connect(self._marker_clicked)
            ring = pg.ScatterPlotItem(symbol="o", size=16, brush=None)
            ring.setZValue(21)
            ring.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            cross = (
                pg.InfiniteLine(angle=90, movable=False),
                pg.InfiniteLine(angle=0, movable=False),
            )
            for item in (markers, ring, *cross):
                plot.getPlotItem().addItem(item, ignoreBounds=True)
            text = pg.TextItem("", anchor=(0.5, 0.5))
            text.setZValue(30)
            plot.getPlotItem().addItem(text, ignoreBounds=True)
            column.addWidget(plot, 1)
            panels.addLayout(column, 1)
            self.panels.append(plot)
            self.captions.append(label)
            self.markers.append(markers)
            self.rings.append(ring)
            self.crosses.append(cross)
            self.unavailable.append(text)
        layout.addLayout(panels, 1)
        self.trace = SpectrumCanvas._plot()
        self.residual_trace = SpectrumCanvas._plot()
        for plot, height in ((self.trace, 92), (self.residual_trace, 72)):
            plot.setFixedHeight(height)
            plot.getViewBox().invertX(True)
            plot.setMouseEnabled(x=False, y=False)
            plot.getAxis("left").setWidth(48)
        # The traces span the full width, so follow the panels' ppm range explicitly:
        # pyqtgraph links views of different widths by screen position, not by value.
        self.panels[0].getViewBox().sigXRangeChanged.connect(self._follow_range)
        self.trace.hideAxis("bottom")
        self.trace_caption = QtWidgets.QLabel()
        self.trace_caption.setObjectName("caption")
        layout.addWidget(self.trace_caption)
        layout.addWidget(self.trace)
        layout.addWidget(self.residual_trace)
        self.measured = self.trace.plot(antialias=True)
        self.model = self.trace.plot(antialias=True)
        self.band = pg.LinearRegionItem(
            values=(-_BAND, _BAND), orientation="horizontal", movable=False
        )
        self.band.setZValue(-5)
        self.residual_trace.getPlotItem().addItem(self.band, ignoreBounds=True)
        self.residual_curve = self.residual_trace.plot(antialias=True)
        self.residual_trace.getViewBox().setYRange(-3 * _BAND, 3 * _BAND, padding=0)
        # Escape anywhere in the lens returns; drafts live in the inspector, not here.
        escape = QtGui.QShortcut(QtGui.QKeySequence("Escape"), self)
        escape.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        escape.activated.connect(self.close_requested)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._worker = ViewerComputation(self)
        self._worker.completed.connect(self._ready)
        self._cancel = Event()
        self._revision = 0
        self._key = None
        self._frame_key = None
        self._data: dict | None = None
        self.result: DatasetFitResult | None = None
        self.current = False
        self.ids: tuple[str, ...] = ()
        self.selected_id: str | None = None
        self.peaks: tuple[WorkingPeak, ...] = ()
        self.plane = 0
        self.pending = False
        self.apply_theme(self.theme)

    # Public state -------------------------------------------------------------

    def show_cluster(
        self,
        result: DatasetFitResult,
        *,
        current: bool,
        ids: tuple[str, ...],
        names: tuple[str, ...],
        selected: str | None,
        peaks: tuple[WorkingPeak, ...],
        plane: int,
        levels: tuple[float, float, int, float],
        grid_indices: tuple[np.ndarray, ...],
        model_cache=None,
    ) -> None:
        """Show a recorded cluster; reframe only when the cluster itself changes."""
        self.result, self.current, self.ids, self.peaks = result, current, ids, peaks
        self.selected_id = selected
        self.levels = levels
        label = names[0] + (f" +{len(names) - 1}" if len(names) > 1 else "") if names else ""
        self.title.setText(f"Lens · {label}")
        self.note.setText(
            f"run {result.run_id[:8]} · {'Current' if current else 'Outdated'}"
            " · comparison selector not used in the lens"
        )
        frame = (result.run_id, ids)
        if frame != self._frame_key:
            self._frame_key = frame
            self.frame(grid_indices)
        self.plane = plane
        key = (result.run_id, current, plane, levels)
        if key != self._key:
            self._key = key
            self._request(model_cache)
        self._place_markers()

    def frame(self, grid_indices: tuple[np.ndarray, ...]) -> None:
        """Default region: the recorded fitting-sample extent plus a margin."""
        rows, columns = (np.asarray(i) for i in grid_indices)
        ranges = []
        for indices, ppm in ((columns, self.ppm_x), (rows, self.ppm_y)):
            low, high = int(indices.min()), int(indices.max())
            pad = max(_MINIMUM_POINTS, int((high - low) * _MARGIN))
            low, high = max(0, low - pad), min(ppm.size - 1, high + pad)
            ranges.append(sorted((float(ppm[low]), float(ppm[high]))))
        self.panels[0].getViewBox().setRange(xRange=ranges[0], yRange=ranges[1], padding=0)

    def apply_theme(self, theme: themes.Theme) -> None:
        """Restyle retained geometry; no recomputation."""
        self.theme = t = theme
        style = {"color": t.secondary, "font-size": f"{FONT_CAPTION}px"}
        for plot in (*self.panels, self.trace, self.residual_trace):
            plot.setBackground(t.plot)
            for name in ("left", "bottom"):
                axis = plot.getAxis(name)
                axis.setPen(pg.mkPen(t.divider))
                axis.setTextPen(pg.mkPen(t.secondary))
        for plot in self.panels:
            plot.setLabel("bottom", f"{SpectrumCanvas.axis_title(self.x_axis)} (ppm)", **style)
        self.panels[0].setLabel("left", f"{SpectrumCanvas.axis_title(self.y_axis)} (ppm)", **style)
        self.residual_trace.setLabel("left", "σ", **style)
        for markers, ring, cross, text in zip(
            self.markers, self.rings, self.crosses, self.unavailable, strict=True
        ):
            markers.setPen(pg.mkPen(color(t.text, 0.7), width=1.0))
            ring.setPen(pg.mkPen(t.accent, width=2))
            for line in cross:
                line.setPen(pg.mkPen(color(t.pin, 0.7), width=1, style=QtCore.Qt.PenStyle.DashLine))
            text.setColor(t.secondary)
        self.measured.setPen(pg.mkPen(color(t.measured), width=1.3))
        self.model.setPen(pg.mkPen(color(t.model), width=1.3, style=QtCore.Qt.PenStyle.DashLine))
        self.residual_curve.setPen(pg.mkPen(color(t.text), width=1.2))
        self.band.setBrush(pg.mkBrush(color(t.secondary, 0.12)))
        for line in self.band.lines:
            line.setPen(pg.mkPen(color(t.secondary, 0.4)))
        self._style_layers()

    def stop(self) -> None:
        """Invalidate pending work without waiting for it."""
        self._cancel.set()
        self._worker.close()

    # Computation ----------------------------------------------------------------

    def _request(self, model_cache) -> None:
        self._revision += 1
        self._cancel.set()
        self._cancel = Event()
        cancel = self._cancel
        for index, layers in enumerate(self.layers):
            for layer in layers:
                self.panels[index].removeItem(layer)
            layers.clear()
        self._data = None
        self.pending = True
        for text in self.unavailable:
            text.setText("Preparing…")
        result, plane, current = self.result, self.plane, self.current
        assert result is not None
        threshold, factor, count, residual_threshold = self.levels
        levels = threshold * factor ** np.arange(count)
        residual_levels = residual_threshold / 3 * _RESIDUAL_LEVELS
        observed = self.spectra.data[plane]
        axes = [axis.model_copy(deep=True) for axis in self.spectra.spectral_params]
        cached = model_cache.get((result.run_id, plane)) if model_cache is not None else None

        def compute():
            experiment = signed_geometry(observed, axes, levels)
            if not current:
                return {"experiment": experiment, "model": None}
            model = cached or dataset_model_plane(result, axes, plane, cancel=cancel)
            if cancel.is_set():
                raise RuntimeError("superseded")
            residual = (observed - model.values) / result.outcome.noise
            return {
                "experiment": experiment,
                "model": model.values,
                "overlay": signed_geometry(model.values, axes, levels),
                "residual": residual,
                "residual_geometry": signed_geometry(residual, axes, residual_levels),
            }

        self._worker.submit("lens", self._revision, compute)

    def _ready(self, _kind: str, revision: int, outcome: object) -> None:
        if revision != self._revision:
            return
        self.pending = False
        if isinstance(outcome, Exception):
            for text in self.unavailable:
                text.setText(f"Unavailable: {outcome}")
            return
        data = cast("dict[str, Any]", outcome)
        self._data = data
        experiment = paths_from_geometry(data["experiment"])
        self._add(0, "measured", experiment)
        if data["model"] is None:
            for index in (1, 2):
                self.unavailable[index].setText("Outdated · fit to compare")
        else:
            self._add(1, "measured", experiment)
            self._add(1, "model", paths_from_geometry(data["overlay"]))
            self._add(2, "residual", paths_from_geometry(data["residual_geometry"]))
            for index in (1, 2):
                self.unavailable[index].setText("")
        self.unavailable[0].setText("")
        self._style_layers()
        self._place_markers()

    def _add(self, index: int, role: str, paths) -> None:
        layer = ContourLayer(role, paths)
        self.panels[index].addItem(layer)
        self.layers[index].append(layer)

    def _style_layers(self) -> None:
        t = self.theme
        solid, dash, dot = (
            QtCore.Qt.PenStyle.SolidLine,
            QtCore.Qt.PenStyle.DashLine,
            QtCore.Qt.PenStyle.DotLine,
        )
        for index, layers in enumerate(self.layers):
            overlay = index == 1
            for layer in layers:
                count = len(layer.levels) // 2
                if layer.role == "residual":
                    pens = _level_pens(
                        (t.residual_positive, t.residual_negative),
                        count,
                        weight=1.2,
                        fade=1.0,
                        dashes=(solid, dash),
                    )
                elif layer.role == "model":
                    pens = _level_pens(
                        (t.model, t.model), count, weight=0.85, fade=1.0, dashes=(solid, dot)
                    )
                else:
                    pens = _level_pens(
                        (t.measured, t.measured_negative),
                        count,
                        weight=1.5 if overlay else 1.0,
                        fade=0.75 if overlay else 1.0,
                        dashes=(solid, dash),
                    )
                layer.set_pens(pens)

    # Markers and the selected member's row ----------------------------------------

    def _place_markers(self) -> None:
        positions = {
            peak.id: (
                display_position(self.x_axis, peak.positions[1]),
                display_position(self.y_axis, peak.positions[0]),
            )
            for peak in self.peaks
        }
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        for markers in self.markers:
            markers.setData(x=xs, y=ys, data=list(positions))
        selected = positions.get(self.selected_id or "")
        for ring, (vertical, horizontal) in zip(self.rings, self.crosses, strict=True):
            ring.setData([selected[0]] if selected else [], [selected[1]] if selected else [])
            for line, value in (
                (vertical, selected and selected[0]),
                (horizontal, selected and selected[1]),
            ):
                line.setVisible(selected is not None)
                if selected is not None:
                    line.setValue(value)
        for text in self.unavailable:
            x_range, y_range = self.panels[0].getViewBox().viewRange()
            text.setPos(sum(x_range) / 2, sum(y_range) / 2)
        self._draw_trace(selected)

    def _draw_trace(self, selected) -> None:
        if selected is None:
            for curve in (self.measured, self.model, self.residual_curve):
                curve.setData([], [])
            self.trace_caption.setText("")
            return
        row = int(np.clip(round(float(self.y_axis.ppm2pts(selected[1]))), 0, self.y_axis.size - 1))
        measured = self.spectra.data[self.plane][row]
        self.measured.setData(self.ppm_x, measured)
        power = formatting.exponent(float(np.max(np.abs(measured))) or 1.0)
        axis = self.trace.getAxis("left")
        axis.setScale(10.0**-power)
        style = {"color": self.theme.faint, "font-size": f"{FONT_CAPTION}px"}
        axis.setLabel(f"×10{formatting.superscript(str(power))}" if power else "", **style)
        data = self._data or {}
        model, residual = data.get("model"), data.get("residual")
        self.model.setData(self.ppm_x, model[row] if model is not None else [])
        self.residual_curve.setData(self.ppm_x, residual[row] if residual is not None else [])
        name = next((p.name for p in self.peaks if p.id == self.selected_id), "")
        decimals = formatting.ppm_decimals(self.y_axis)
        self.trace_caption.setText(
            f"Row through {name} at {formatting.number(float(self.ppm_y[row]), decimals)} ppm · "
            f"measured, total model (dashed) · residual in σ, ±{_BAND:g} σ band"
            + ("" if model is not None else " · model unavailable")
        )

    def _follow_range(self, _view, x_range) -> None:
        for plot in (self.trace, self.residual_trace):
            plot.getViewBox().setXRange(*x_range, padding=0)

    def _marker_clicked(self, _item, points, _event) -> None:
        if points:
            self.selected.emit(points[0].data())
