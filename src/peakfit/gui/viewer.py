"""Retained spectrum contours, synchronized inspection and manual peak editing."""

from typing import TYPE_CHECKING, cast

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# isort: split
# PySide6 must be imported before PyQtGraph selects a binding.
import pyqtgraph as pg

from peakfit.engine.algorithms.noise import estimate_noise
from peakfit.gui.editing import PeakInspector, PeakMarkers
from peakfit.gui.peaks import PeakEdits
from peakfit.gui.spectrum import contour_paths, display_position, validate_plane_values

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
    from peakfit.gui.peaks import WorkingPeak


class SpectrumViewer(QtWidgets.QWidget):
    """Inspect unchanged spectrum samples and edit independent working peaks."""

    edits_changed = QtCore.Signal()

    def __init__(self, spectra: Spectra, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.spectra = spectra
        self.plane = 0
        self.iy, self.ix = (size // 2 for size in spectra.data.shape[1:])
        self.y_axis, self.x_axis = spectra.spectral_params
        self.ppm_x = self.x_axis.pts2ppm(np.arange(self.x_axis.size, dtype=float))
        self.ppm_y = self.y_axis.pts2ppm(np.arange(self.y_axis.size, dtype=float))
        self.noise = estimate_noise(spectra.data[0])
        self.edits = PeakEdits()
        self.selected_id: str | None = None
        self.adding_peak = False
        self._drag_peak: WorkingPeak | None = None
        self.contours: list[QtWidgets.QGraphicsPathItem] = []
        self.peak_labels: list[pg.TextItem] = []
        self.labels_visible = False
        self._colors = self._palette_colors()
        self._build_ui()
        self._build_edit_actions()
        self._apply_palette()
        self._contour_timer = QtCore.QTimer(self)
        self._contour_timer.setSingleShot(True)
        self._contour_timer.setInterval(30)
        self._contour_timer.timeout.connect(self._update_contours)
        for control in (self.threshold, self.factor, self.levels):
            control.valueChanged.connect(self._schedule_contours)
        self._update_contours()
        self.reset_view()
        self._update_slices()

    def _build_edit_actions(self) -> None:
        self.undo_action = QtGui.QAction("Undo peak edit", self)
        self.redo_action = QtGui.QAction("Redo peak edit", self)
        self.undo_action.setShortcuts(QtGui.QKeySequence.StandardKey.Undo)
        self.redo_action.setShortcuts(QtGui.QKeySequence.StandardKey.Redo)
        self.undo_action.triggered.connect(self.undo)
        self.redo_action.triggered.connect(self.redo)
        self.add_action = QtGui.QAction("Add peak…", self)
        self.add_action.triggered.connect(self.start_adding_peak)
        self.edit_actions = [self.undo_action, self.redo_action, self.add_action]
        for action in self.edit_actions:
            self.addAction(action)
        delete = QtGui.QAction("Delete peak", self.spectrum_plot)
        delete.setShortcuts([QtGui.QKeySequence("Delete"), QtGui.QKeySequence("Backspace")])
        delete.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        delete.triggered.connect(self.delete_peak)
        self.spectrum_plot.addAction(delete)
        for key, callback in (("A", self.start_adding_peak), ("Escape", self.cancel_interaction)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self.spectrum_plot)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
        self.edits_changed.connect(self._update_edit_actions)
        self._update_edit_actions()

    def _update_edit_actions(self) -> None:
        pending = self.inspector.pending
        self.undo_action.setEnabled(self.edits.can_undo and not pending)
        self.redo_action.setEnabled(self.edits.can_redo and not pending)
        self.inspector.delete.setEnabled(self.selected_id is not None and not pending)
        hint = "Apply inspector edits or press Escape before Undo/Redo/Delete"
        if pending:
            self.inspector.details.setText(hint)
        elif self.inspector.details.text() == hint:
            self.inspector.details.clear()

    def _plot(self, title: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(title=title, background=self._colors["background"])
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setMenuEnabled(False)
        return plot

    def _palette_colors(self) -> dict[str, str]:
        palette = self.palette()
        background = palette.color(QtGui.QPalette.ColorRole.Base)
        foreground = palette.color(QtGui.QPalette.ColorRole.Text)
        dark = background.lightness() < foreground.lightness()
        return {
            "background": background.name(),
            "foreground": foreground.name(),
            "positive": "#64b5f6" if dark else "#1265a8",
            "negative": "#ff8a80" if dark else "#b34244",
            "crosshair": "#ffcc80" if dark else "#9a5500",
        }

    def _apply_palette(self) -> None:
        """Use Qt's appearance without recalculating any scientific geometry."""
        self._colors = self._palette_colors()
        foreground = self._colors["foreground"]
        for plot in (self.spectrum_plot, self.trace_x, self.trace_y, self.profile):
            plot.setBackground(self._colors["background"])
            plot.setTitle(plot.getPlotItem().titleLabel.text, color=foreground)
            for name in ("left", "bottom"):
                axis = plot.getAxis(name)
                axis.setPen(foreground)
                axis.setTextPen(foreground)
                axis.setLabel(axis.labelText, units=axis.labelUnits, color=foreground)
        for line in (self.cross_x, self.cross_y):
            line.setPen(pg.mkPen(self._colors["crosshair"]))
            line.setHoverPen(pg.mkPen(self._colors["crosshair"], width=2))
        for line in self.zero_lines:
            line.setPen(pg.mkPen(foreground, style=QtCore.Qt.PenStyle.DotLine))
        for curve in (self.curve_x, self.curve_y):
            curve.setPen(pg.mkPen(self._colors["positive"], width=1.5))
        self.peak_scatter.setPen(pg.mkPen(foreground, width=1.5))
        for label in self.peak_labels:
            label.setColor(foreground)
        self.curve_profile.setSymbolBrush(self._colors["crosshair"])
        self.profile_marker.setBrush(self._colors["crosshair"])
        self.profile_marker.setPen(foreground)
        for item in self.contours:
            pen = item.pen()
            negative = pen.style() == QtCore.Qt.PenStyle.DashLine
            pen.setColor(QtGui.QColor(self._colors["negative" if negative else "positive"]))
            item.setPen(pen)

    def changeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        """Follow application palette changes, including system appearance switches."""
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.PaletteChange and hasattr(self, "curve_profile"):
            self._apply_palette()

    @staticmethod
    def _axis_title(axis: SpectralParameters) -> str:
        return f"{axis.nucleus or axis.label} ({axis.label})"

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        splitter = QtWidgets.QSplitter()
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.spectrum_plot = self._plot("Spectrum")
        self.spectrum_plot.setLabel("bottom", self._axis_title(self.x_axis), units="ppm")
        self.spectrum_plot.setLabel("left", self._axis_title(self.y_axis), units="ppm")
        view = self.spectrum_plot.getViewBox()
        view.invertX(True)
        view.invertY(True)
        view.setDefaultPadding(0.02)
        self.spectrum_plot.setToolTip(
            "Drag to pan; scroll to zoom. Click to position the crosshair; "
            "drag either crosshair line."
        )
        left_layout.addWidget(self.spectrum_plot, 1)
        self.cross_x = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("#a65e00"))
        self.cross_y = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen("#a65e00"))
        for line in (self.cross_x, self.cross_y):
            self.spectrum_plot.getPlotItem().addItem(line, ignoreBounds=True)
            line.sigPositionChanged.connect(self._crosshair_dragged)
        cast("pg.GraphicsScene", self.spectrum_plot.scene()).sigMouseClicked.connect(self._clicked)
        self.peak_scatter = PeakMarkers(
            symbol="+",
            size=10,
            pen=pg.mkPen("#202020", width=1.5),
            hoverable=True,
            tip=lambda x, y, data: self._peak_tip(data, x, y),
        )
        self.peak_scatter.setZValue(20)
        self.peak_scatter.sigClicked.connect(self._peak_clicked)
        self.peak_scatter.moved.connect(self._preview_peak_move)
        self.spectrum_plot.addItem(self.peak_scatter)
        self.selection_marker = pg.ScatterPlotItem(symbol="o", size=18, brush=None)
        self.selection_marker.setZValue(21)
        self.spectrum_plot.addItem(self.selection_marker)
        self._build_plane_controls(left_layout)
        self._build_contour_controls(left_layout)
        splitter.addWidget(left)

        right = self._build_inspection_panels()
        splitter.addWidget(right)
        splitter.setSizes([850, 450])
        self.position_label = QtWidgets.QLabel()
        layout.addWidget(self.position_label)

    def _build_inspection_panels(self) -> QtWidgets.QWidget:
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.inspector = PeakInspector(self._axis_title(self.x_axis), self._axis_title(self.y_axis))
        self.inspector.applied.connect(self._apply_peak_fields)
        self.inspector.fields_changed.connect(self.edits_changed)
        self.inspector.deleted.connect(self.delete_peak)
        add = QtWidgets.QPushButton("Add peak…")
        add.setToolTip("Place one peak at the next click (A in the spectrum); Escape cancels")
        add.clicked.connect(self.start_adding_peak)
        cast("QtWidgets.QGridLayout", self.inspector.layout()).addWidget(add, 2, 2)
        right_layout.addWidget(self.inspector)
        self.trace_x = self._plot(self._axis_title(self.x_axis))
        self.trace_y = self._plot(self._axis_title(self.y_axis))
        self.profile = self._plot("Raw intensity across planes")
        self.zero_lines: list[pg.InfiniteLine] = []
        for plot, axis in ((self.trace_x, self.x_axis), (self.trace_y, self.y_axis)):
            plot.setLabel("bottom", self._axis_title(axis), units="ppm")
            plot.setLabel("left", "Intensity")
            trace_view = plot.getViewBox()
            trace_view.invertX(True)
            trace_view.setMouseEnabled(x=False, y=False)
            trace_view.setAutoVisible(y=True)
            self.zero_lines.append(plot.addLine(y=0))
            plot.setToolTip(
                "Follows the spectrum's visible range; intensity scales to visible data."
            )
        self.spectrum_plot.getViewBox().sigRangeChanged.connect(self._sync_trace_ranges)
        self.curve_x = self.trace_x.plot(pen=pg.mkPen("#1265a8", width=1.5))
        self.curve_y = self.trace_y.plot(pen=pg.mkPen("#1265a8", width=1.5))
        self.profile.setLabel("bottom", "Plane value")
        self.profile.setLabel("left", "Raw intensity")
        # Points do not imply interpolation across unordered/repeated experimental values.
        self.curve_profile = self.profile.plot(
            pen=None, symbol="o", symbolSize=5, symbolBrush="#a65e00", symbolPen=None
        )
        self.profile_marker = pg.ScatterPlotItem(size=11, brush="#a65e00", pen="#202020")
        self.profile.addItem(self.profile_marker)
        for plot in (self.trace_x, self.trace_y, self.profile):
            right_layout.addWidget(plot, 1)
        return right

    def _build_plane_controls(self, layout: QtWidgets.QVBoxLayout) -> None:
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Plane"))
        self.plane_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.plane_spin = QtWidgets.QSpinBox()
        self.plane_spin.setKeyboardTracking(False)
        self.plane_spin.setAccessibleName("Plane number")
        self.plane_slider.setAccessibleName("Plane")
        self.plane_slider.setToolTip(
            "Select a plane; Ctrl+Page Up / Ctrl+Page Down also step between planes."
        )
        n_planes = self.spectra.data.shape[0]
        self.plane_slider.setRange(1, n_planes)
        self.plane_spin.setRange(1, n_planes)
        for control in (self.plane_slider, self.plane_spin):
            control.setValue(1)
            control.setEnabled(n_planes > 1)
            control.valueChanged.connect(self.set_plane)
        row.addWidget(self.plane_slider, 1)
        row.addWidget(self.plane_spin)
        self.plane_label = QtWidgets.QLabel()
        row.addWidget(self.plane_label)
        layout.addLayout(row)

    def _build_contour_controls(self, layout: QtWidgets.QVBoxLayout) -> None:
        grid = QtWidgets.QGridLayout()
        grid.setVerticalSpacing(2)
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setRange(0.1, 1000)
        self.threshold.setValue(5)
        self.threshold.setSingleStep(0.5)
        self.threshold.setSuffix(" × noise")
        self.factor = QtWidgets.QDoubleSpinBox()
        self.factor.setRange(1.05, 3)
        self.factor.setValue(1.3)
        self.factor.setSingleStep(0.05)
        self.levels = QtWidgets.QSpinBox()
        self.levels.setRange(1, 30)
        self.levels.setValue(10)
        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 400)
        self.threshold_slider.setPageStep(25)
        self.factor_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.factor_slider.setRange(105, 300)
        self.factor_slider.setSingleStep(5)
        self.factor_slider.setPageStep(10)
        self.levels_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.levels_slider.setRange(1, 30)
        self.levels_slider.setPageStep(5)
        for row, (title, control, slider, tooltip) in enumerate(
            (
                (
                    "&Threshold",
                    self.threshold,
                    self.threshold_slider,
                    "First contour in multiples of noise. Slider uses logarithmic spacing.",
                ),
                (
                    "&Spacing",
                    self.factor,
                    self.factor_slider,
                    "Multiplier between successive contour levels.",
                ),
                ("&Levels", self.levels, self.levels_slider, "Number of contours for each sign."),
            )
        ):
            label = QtWidgets.QLabel(title)
            label.setBuddy(slider)
            slider.setAccessibleName(label.text().replace("&", "") + " slider")
            control.setAccessibleName(label.text().replace("&", ""))
            control.setKeyboardTracking(False)
            slider.setToolTip(tooltip)
            control.setToolTip(tooltip)
            slider.valueChanged.connect(self._contour_slider_changed)
            control.valueChanged.connect(self._sync_contour_sliders)
            grid.addWidget(label, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(control, row, 2)
        grid.setColumnStretch(1, 1)
        self._sync_contour_sliders()
        reset = QtWidgets.QPushButton("Reset view")
        reset.clicked.connect(self.reset_view)
        grid.addWidget(reset, 0, 3)
        layout.addLayout(grid)

    def _sync_contour_sliders(self) -> None:
        for slider, value in (
            (self.threshold_slider, round(100 * np.log10(self.threshold.value() / 0.1))),
            (self.factor_slider, round(100 * self.factor.value())),
            (self.levels_slider, self.levels.value()),
        ):
            with QtCore.QSignalBlocker(slider):
                slider.setValue(value)

    def _contour_slider_changed(self) -> None:
        slider = self.sender()
        if slider is self.threshold_slider:
            self.threshold.setValue(0.1 * 10 ** (self.threshold_slider.value() / 100))
        elif slider is self.factor_slider:
            self.factor.setValue(self.factor_slider.value() / 100)
        else:
            self.levels.setValue(self.levels_slider.value())

    def _sync_trace_ranges(self) -> None:
        """The spectrum owns navigation; trace X ranges follow its two spectral axes."""
        x_range, y_range = self.spectrum_plot.getViewBox().viewRange()
        for plot, limits in ((self.trace_x, x_range), (self.trace_y, y_range)):
            plot.getViewBox().setXRange(*limits, padding=0)

    def reset_view(self) -> None:
        """Show the entire sampled spectrum without recalculating contours."""
        self.spectrum_plot.getViewBox().setRange(
            xRange=(float(self.ppm_x[-1]), float(self.ppm_x[0])),
            yRange=(float(self.ppm_y[-1]), float(self.ppm_y[0])),
        )

    def set_plane(self, number: int) -> None:
        """Select a one-based plane while retaining the current viewing range."""
        plane = int(np.clip(number - 1, 0, self.spectra.data.shape[0] - 1))
        if plane == self.plane:
            return
        self.plane = plane
        for control in (self.plane_slider, self.plane_spin):
            with QtCore.QSignalBlocker(control):
                control.setValue(plane + 1)
        self._schedule_contours()
        self._update_slices()

    def _schedule_contours(self) -> None:
        # Throttle rather than debounce: scrubbing still displays intermediate planes.
        if not self._contour_timer.isActive():
            self._contour_timer.start()

    def _update_contours(self) -> None:
        n_levels = self.levels.value()
        while len(self.contours) < 2 * n_levels:
            item = QtWidgets.QGraphicsPathItem()
            item.setZValue(-10)
            self.spectrum_plot.addItem(item)
            self.contours.append(item)
        base = max(self.noise * self.threshold.value(), np.finfo(float).tiny)
        positive = base * self.factor.value() ** np.arange(n_levels)
        levels = np.concatenate((positive, -positive))
        paths = contour_paths(self.spectra.data[self.plane], self.spectra.spectral_params, levels)
        for index, item in enumerate(self.contours):
            item.setVisible(index < len(paths))
            if index < len(paths):
                negative = index >= n_levels
                item.setPen(
                    pg.mkPen(
                        self._colors["negative" if negative else "positive"],
                        width=1,
                        style=QtCore.Qt.PenStyle.DashLine
                        if negative
                        else QtCore.Qt.PenStyle.SolidLine,
                    )
                )
                item.setPath(paths[index])

    def _clicked(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if event.isAccepted() and not self.adding_peak:
            return
        view = self.spectrum_plot.getViewBox()
        if view.sceneBoundingRect().contains(event.scenePos()):
            point = view.mapSceneToView(event.scenePos())
            if self.adding_peak:
                peak = self.edits.add((point.y(), point.x()))
                self.cancel_interaction()
                self.selected_id = peak.id
                self._refresh_peaks()
                self.edits_changed.emit()
            else:
                self.select_peak(None)
            self._select_position(point.x(), point.y())

    def _crosshair_dragged(self) -> None:
        self._select_position(self.cross_x.value(), self.cross_y.value())

    def _select_position(self, x: float, y: float) -> None:
        self.ix = int(np.clip(round(self.x_axis.ppm2pts(x)), 0, self.x_axis.size - 1))
        self.iy = int(np.clip(round(self.y_axis.ppm2pts(y)), 0, self.y_axis.size - 1))
        self._update_slices()

    def _update_slices(self) -> None:
        plane = self.spectra.data[self.plane]
        for line, value in (
            (self.cross_x, self.ppm_x[self.ix]),
            (self.cross_y, self.ppm_y[self.iy]),
        ):
            with QtCore.QSignalBlocker(line):
                line.setValue(float(value))
        self.curve_x.setData(self.ppm_x, plane[self.iy, :])
        self.curve_y.setData(self.ppm_y, plane[:, self.ix])
        profile = self.spectra.data[:, self.iy, self.ix]
        values = self.spectra.z_values
        self.curve_profile.setData(values, profile)
        self.profile_marker.setData([values[self.plane]], [profile[self.plane]])
        self.plane_label.setText(f"/ {len(values)}   value: {values[self.plane]:g}")
        self.position_label.setText(
            f"{self._axis_title(self.x_axis)}: {self.ppm_x[self.ix]:.4f} ppm    "
            f"{self._axis_title(self.y_axis)}: {self.ppm_y[self.iy]:.4f} ppm    "
            f"Raw intensity: {plane[self.iy, self.ix]:.5g}"
        )

    def set_plane_values(self, values: np.ndarray) -> None:
        """Replace experimental plane coordinates without changing spectral samples."""
        validate_plane_values(values, self.spectra.data.shape[0])
        self.spectra.z_values = np.array(values, dtype=float, copy=True)
        self.profile.enableAutoRange()
        self._update_slices()

    def set_peaks(self, peaks: Sequence[WorkingPeak]) -> None:
        """Display imported peaks, folding only indirect coordinates."""
        self.edits = PeakEdits(tuple(peaks))
        self.selected_id = None
        self._refresh_peaks()
        self.edits_changed.emit()

    @property
    def peaks(self) -> tuple[WorkingPeak, ...]:
        """Expose committed working records without fitting shapes."""
        return self.edits.peaks

    def _refresh_peaks(self) -> None:
        peaks = self.peaks
        for label in self.peak_labels:
            self.spectrum_plot.removeItem(label)
        self.peak_labels.clear()
        positions = [
            (
                display_position(self.x_axis, peak.positions[1]),
                display_position(self.y_axis, peak.positions[0]),
            )
            for peak in peaks
        ]
        self.peak_scatter.setData(
            x=[position[0] for position in positions],
            y=[position[1] for position in positions],
            data=[peak.id for peak in peaks],
        )
        for peak, (x, y) in zip(peaks, positions, strict=True):
            label = pg.TextItem(peak.name, color=self._colors["foreground"], anchor=(0, 1))
            label.setPos(x, y)
            label.setVisible(self.labels_visible)
            self.spectrum_plot.getPlotItem().addItem(label, ignoreBounds=True)
            self.peak_labels.append(label)
        self.select_peak(self.selected_id)

    def show_peak_labels(self, visible: bool) -> None:
        """Keep crowded survey views readable while allowing explicit assignments."""
        self.labels_visible = visible
        for peak, label in zip(self.peaks, self.peak_labels, strict=True):
            label.setVisible(visible or peak.id == self.selected_id)

    def _peak_tip(self, peak_id: str, x: float, y: float) -> str:
        peak = next(p for p in self.peaks if p.id == peak_id)
        return f"{peak.name}\n{x:.6f}, {y:.6f} ppm displayed\nClick to select"

    def _peak_clicked(self, _item, points, event) -> None:
        self.spectrum_plot.setFocus()
        candidates = [p.data() for p in points]
        if len(candidates) == 1:
            self.select_peak(candidates[0], inspect=True)
            return
        menu = QtWidgets.QMenu(self)
        for peak in self.peaks:
            if peak.id in candidates:
                action = menu.addAction(
                    f"{peak.name}  ({peak.positions[1]:.8g}, {peak.positions[0]:.8g} ppm)"
                )
                action.setData(peak.id)
        menu.triggered.connect(lambda action: self.select_peak(action.data(), inspect=True))
        menu.aboutToHide.connect(menu.deleteLater)
        menu.popup(event.screenPos().toPoint())

    def select_peak(self, peak_id: str | None, *, inspect: bool = False) -> None:
        """Select by identity; inspection alone snaps to measured sample indices."""
        if peak_id != self.selected_id and not self.apply_pending_fields():
            return
        peak = next((p for p in self.peaks if p.id == peak_id), None)
        self.selected_id = peak.id if peak else None
        self.peak_scatter.selected_id = self.selected_id
        position = (
            (
                display_position(self.x_axis, peak.positions[1]),
                display_position(self.y_axis, peak.positions[0]),
            )
            if peak
            else (0.0, 0.0)
        )
        self.selection_marker.setData(
            x=[position[0]] if peak else [],
            y=[position[1]] if peak else [],
            pen=pg.mkPen(self._colors["crosshair"], width=2),
        )
        self.inspector.show_peak(peak, position)
        self.show_peak_labels(self.labels_visible)
        if peak and inspect:
            self._select_position(*position)

    def _apply_peak_fields(self, name: str, positions: tuple[float, float]) -> bool:
        if self.selected_id is None:
            return True
        try:
            self.edits.update(self.selected_id, name, positions)
        except ValueError as exc:
            self.inspector.details.setText(str(exc))
            return False
        self._refresh_peaks()
        self.edits_changed.emit()
        return True

    def apply_pending_fields(self) -> bool:
        """Commit valid typed edits before selection changes or saving work."""
        if not self.inspector.pending:
            return True
        try:
            name, positions = self.inspector.values()
        except ValueError as exc:
            self.inspector.details.setText(str(exc))
            return False
        return self._apply_peak_fields(name, positions)

    def delete_peak(self) -> None:
        """Delete the selected assignment as one reversible edit."""
        if self.selected_id is not None and not self.inspector.pending:
            self.peak_scatter.cancel_drag()
            self._drag_peak = None
            self.edits.delete(self.selected_id)
            self.selected_id = None
            self._refresh_peaks()
            self.edits_changed.emit()

    def undo(self) -> None:
        """Undo a committed peak edit only after inspector text is resolved."""
        if self.inspector.pending:
            return
        self.cancel_interaction()
        self.edits.undo()
        self._refresh_peaks()
        self.edits_changed.emit()

    def redo(self) -> None:
        """Redo a committed peak edit only after inspector text is resolved."""
        if self.inspector.pending:
            return
        self.cancel_interaction()
        self.edits.redo()
        self._refresh_peaks()
        self.edits_changed.emit()

    def start_adding_peak(self) -> None:
        """Arm a single placement without a persistent editing mode."""
        if not self.apply_pending_fields():
            return
        self.cancel_interaction()
        self.adding_peak = True
        self.peak_scatter.placing = True
        for line in (self.cross_x, self.cross_y):
            line.setMovable(False)
        self.spectrum_plot.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.spectrum_plot.setFocus()
        self.inspector.details.setText("Click to add one peak · Escape cancels · drag to pan")

    def cancel_interaction(self) -> None:
        """Discard a drag preview or uncommitted placement."""
        self.adding_peak = False
        self.peak_scatter.placing = False
        self.peak_scatter.cancel_drag()
        self._drag_peak = None
        for line in (self.cross_x, self.cross_y):
            line.setMovable(True)
        self.spectrum_plot.unsetCursor()
        self.peak_scatter.unsetCursor()
        self.select_peak(self.selected_id)
        for peak, label in zip(self.peaks, self.peak_labels, strict=True):
            if peak.id == self.selected_id:
                label.setPos(
                    display_position(self.x_axis, peak.positions[1]),
                    display_position(self.y_axis, peak.positions[0]),
                )

    def _preview_peak_move(self, dx: float, dy: float, finished: bool) -> None:
        peak = self._drag_peak
        if peak is None:
            if not self.apply_pending_fields():
                self.peak_scatter.cancel_drag()
                return
            peak = next((p for p in self.peaks if p.id == self.selected_id), None)
            self._drag_peak = peak
        if peak is None:
            return
        # Apply display deltas to the original UNFOLDED coordinates. Do not replace
        # them with the folded marker coordinates or the snapped crosshair position.
        positions = (peak.positions[0] + dy, peak.positions[1] + dx)
        x = display_position(self.x_axis, positions[1])
        y = display_position(self.y_axis, positions[0])
        self.selection_marker.setData([x], [y])
        for original, label in zip(self.peaks, self.peak_labels, strict=True):
            if original.id == peak.id:
                label.setPos(x, y)
        self.inspector.details.setText(
            f"Moving: {positions[1]:.8f}, {positions[0]:.8f} ppm · Escape cancels"
        )
        if finished:
            self.edits.update(peak.id, peak.name, positions)
            self._drag_peak = None
            self._refresh_peaks()
            self.edits_changed.emit()
