"""Retained spectrum contours, synchronized inspection and manual peak editing."""

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, cast

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# isort: split
# PySide6 must be imported before PyQtGraph selects a binding.
import pyqtgraph as pg

from peakfit.engine.algorithms.noise import estimate_noise
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.gui.background import ViewerComputation
from peakfit.gui.comparison import FitComparison
from peakfit.gui.editing import PeakInspector, PeakMarkers
from peakfit.gui.export_dialog import choose_and_export
from peakfit.gui.fit_positions import fitted_position_updates
from peakfit.gui.fit_preview import FitPreview
from peakfit.gui.fit_process import FitProcessResult, LocalFitProcess
from peakfit.gui.fitting import PreparedLocalFit, prepare_local_fit
from peakfit.gui.local_fit import execution_unavailable
from peakfit.gui.peaks import PeakEdits
from peakfit.gui.spectrum import (
    contour_geometry,
    display_position,
    paths_from_geometry,
    validate_plane_values,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
    from peakfit.gui.peaks import WorkingPeak
    from peakfit.gui.spectrum import ContourGeometry


_POSITION_PREVIEW_PEAKS = 4


class SpectrumViewer(QtWidgets.QWidget):
    """Inspect unchanged spectrum samples and edit independent working peaks."""

    edits_changed = QtCore.Signal()
    fit_state_changed = QtCore.Signal()

    def __init__(self, spectra: Spectra, parent: QtWidgets.QWidget | None = None) -> None:  # noqa: PLR0915
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
        self.fit_preparation: PreparedLocalFit | None = None
        self._fit_noise: float | None = None
        self._fit_revision = 0
        self._contour_revision = 0
        self._closing = False
        self.source_files: tuple[tuple[str, str], ...] = ()
        self.exported_result_id: str | None = None
        self._exporting = False
        self.fit_result: FitProcessResult | None = None
        self._scientific_revision = 0
        self._submitted_revision = 0
        self._result_revision = 0
        self._submitted: PreparedLocalFit | None = None
        self._showing_result = False
        self._fit_process = LocalFitProcess(self)
        self._fit_process.changed.connect(self._fit_state_changed)
        self._fit_process.finished.connect(self._fit_finished)
        self._computation = ViewerComputation(self)
        self._computation.completed.connect(self._computation_finished)
        self._colors = self._palette_colors()
        self._build_ui()
        self.comparison = FitComparison(self)
        self._build_edit_actions()
        self.edits_changed.connect(self._scientific_inputs_changed)
        self._fit_status_timer = QtCore.QTimer(self)
        self._fit_status_timer.setInterval(250)
        self._fit_status_timer.timeout.connect(self._fit_state_changed)
        self._apply_palette()
        self._contour_timer = QtCore.QTimer(self)
        self._contour_timer.setSingleShot(True)
        self._contour_timer.setInterval(30)
        self._contour_timer.timeout.connect(self._update_contours)
        self._fit_timer = QtCore.QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.setInterval(150)
        self._fit_timer.timeout.connect(self.preview_fit_group)
        self.threshold.valueChanged.connect(self._schedule_fit_preview)
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
        self.add_action.setCheckable(True)
        self.add_action.triggered.connect(self._toggle_peak_placement)
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
        self._update_interaction_hint()

    def _update_edit_actions(self) -> None:
        pending = self.inspector.pending
        self.undo_action.setEnabled(self.edits.can_undo and not pending and not self.fit_busy)
        self.redo_action.setEnabled(self.edits.can_redo and not pending and not self.fit_busy)
        self.inspector.delete.setEnabled(
            self.selected_id is not None and not pending and not self.fit_busy
        )
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
        if hasattr(self, "comparison"):
            self.comparison.palette_changed()

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
        self._build_fit_overlay()
        self._build_plane_controls(left_layout)
        self._build_contour_controls()
        splitter.addWidget(left)

        right = self._build_workspaces()
        splitter.addWidget(right)
        splitter.setSizes([850, 450])
        self.position_label = QtWidgets.QLabel()
        self.position_label.setFixedHeight(self.position_label.fontMetrics().lineSpacing() + 4)
        self.position_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
        )
        footer = QtWidgets.QHBoxLayout()
        self.footer_status = QtWidgets.QStackedWidget()
        self.footer_status.setFixedHeight(self.position_label.height())
        self.footer_status.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.footer_status.addWidget(self.position_label)
        self.footer_status.addWidget(self.contour_status)
        footer.addWidget(self.footer_status, 1)
        self.result_notice = QtWidgets.QLabel()
        footer.addWidget(self.result_notice)
        layout.addLayout(footer)

    def _build_fit_overlay(self) -> None:
        self.fit_mask = pg.ImageItem(axisOrder="row-major")
        self.fit_mask.setZValue(-20)
        self.fit_mask.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.spectrum_plot.getPlotItem().addItem(self.fit_mask, ignoreBounds=True)
        self.fit_members = pg.ScatterPlotItem(
            symbol="o", size=24, brush=None, pen=pg.mkPen("#b87b00", width=2)
        )
        self.fit_members.setZValue(19)
        self.fit_members.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.spectrum_plot.getPlotItem().addItem(self.fit_members, ignoreBounds=True)

    def _build_workspaces(self) -> QtWidgets.QWidget:
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.workspace_tabs = QtWidgets.QTabBar()
        self.workspace_tabs.addTab("Inspect")
        self.workspace_tabs.addTab("Fit")
        self.workspace_tabs.setTabEnabled(1, False)
        right_layout.addWidget(self.workspace_tabs)
        self.workspaces = QtWidgets.QStackedWidget()
        plots = self._build_inspection_panels()
        self.workspaces.addWidget(self.inspector)
        self.fit_preview = FitPreview()
        self.fit_preview.run_requested.connect(self.run_local_fit)
        self.fit_preview.cancel_requested.connect(self._fit_process.cancel)
        self.fit_preview.prepare_requested.connect(self.preview_fit_group)
        self.fit_preview.result_requested.connect(self.show_fit_result)
        self.fit_preview.apply_positions_requested.connect(self.apply_fitted_positions)
        self.workspaces.addWidget(self.fit_preview)
        self.workspace_tabs.currentChanged.connect(self._workspace_changed)
        # Reserve identical header space in every state so the plots never jump.
        self.workspaces.setFixedHeight(
            max(self.inspector.sizeHint().height(), self.fit_preview.sizeHint().height())
        )
        right_layout.addWidget(self.workspaces)
        self.inspection_content = QtWidgets.QStackedWidget()
        self.inspection_content.addWidget(plots)
        details_scroll = QtWidgets.QScrollArea()
        details_scroll.setWidgetResizable(True)
        details_scroll.setWidget(self.fit_preview.detail_content)
        self.inspection_content.addWidget(details_scroll)
        self.fit_preview.details_requested.connect(
            lambda checked: self.inspection_content.setCurrentIndex(int(checked))
        )
        right_layout.addWidget(self.inspection_content, 1)
        right.setMinimumWidth(350)
        return right

    def _workspace_changed(self, index: int) -> None:
        self.fit_preview.details.setChecked(False)
        if index == 1:
            if self.fit_busy:
                self.workspaces.setCurrentIndex(1)
            elif self.fit_result is not None:
                self.show_fit_result()
            else:
                self.preview_fit_group()
        else:
            self.dismiss_fit_preview()

    def preview_fit_group(self) -> None:
        """Snapshot committed peaks and show a local scope without fitting."""
        if self.fit_busy:
            return
        self.comparison.clear()
        self._showing_result = False
        if self.selected_id is None or not self.apply_pending_fields():
            self.dismiss_fit_preview()
            return
        self._fit_timer.stop()
        selected_id = self.selected_id
        self.cancel_interaction()
        self._fit_revision += 1
        revision = self._fit_revision
        self.fit_preparation = None
        self.fit_mask.setOpacity(0.35)
        self.fit_members.setOpacity(0.35)
        self.fit_preview.show_status(
            "Updating scope · previous scope shown until ready"
            if self.fit_mask.isVisible()
            else "Preparing fitting scope…"
        )
        with QtCore.QSignalBlocker(self.workspace_tabs):
            self.workspace_tabs.setCurrentIndex(1)
        self.workspaces.setCurrentIndex(1)
        self._update_interaction_hint()
        # Capture immutable working records and private metadata before leaving Qt.
        spectra = self.spectra.model_copy(
            update={
                "params": [axis.model_copy(deep=True) for axis in self.spectra.params],
                "z_values": self.spectra.z_values.copy(),
            }
        )
        peaks, source_id = self.peaks, id(self.spectra)
        source_files = self.source_files
        config = PeakFitConfig(noise_level=self._fit_noise)
        threshold = self.lowest_contour_level

        def prepare() -> PreparedLocalFit:
            result = prepare_local_fit(spectra, peaks, selected_id, config, contour_level=threshold)
            return replace(
                result,
                snapshot=replace(result.snapshot, spectrum_id=source_id, source_files=source_files),
            )

        self._computation.submit("fit", revision, prepare)

    def _apply_fit_preparation(self, prepared: PreparedLocalFit) -> None:
        """Install only the current scientific result, using Qt on the GUI thread."""
        self._fit_noise = prepared.noise
        self.fit_preparation = prepared
        self.fit_preview.show_preparation(prepared)
        self.fit_preview.retained.setVisible(self.fit_result is not None and not self.fit_busy)
        self.fit_mask.setOpacity(1)
        self.fit_members.setOpacity(1)
        rgba = np.zeros((*self.spectra.data.shape[1:], 4), dtype=np.uint8)
        rgba[prepared.scope.grid_indices] = (230, 160, 20, 85)
        self.fit_mask.setImage(rgba, autoLevels=False)
        self.fit_mask.setTransform(
            QtGui.QTransform(
                self.x_axis.delta,
                0,
                0,
                self.y_axis.delta,
                self.ppm_x[0] - self.x_axis.delta / 2,
                self.ppm_y[0] - self.y_axis.delta / 2,
            )
        )
        self.fit_mask.show()
        members = {item.working_id for item in prepared.identities}
        self.fit_members.setData(
            x=[
                display_position(self.x_axis, p.positions[1]) for p in self.peaks if p.id in members
            ],
            y=[
                display_position(self.y_axis, p.positions[0]) for p in self.peaks if p.id in members
            ],
        )
        with QtCore.QSignalBlocker(self.workspace_tabs):
            self.workspace_tabs.setCurrentIndex(1)
        self.workspaces.setCurrentIndex(1)
        self._update_interaction_hint()

    def _schedule_fit_preview(self) -> None:
        """Debounce threshold drags; an outdated region is not a reviewed input."""
        if self.fit_busy or self._showing_result:
            return
        if self.workspaces.currentIndex() != 1:
            return
        if (
            self.fit_preparation is not None
            and self.fit_preparation.contour_level == self.lowest_contour_level
        ):
            return
        self._fit_revision += 1
        self._computation.cancel("fit")
        self.fit_preparation = None
        self.fit_mask.setOpacity(0.35)
        self.fit_members.setOpacity(0.35)
        self.fit_preview.show_status(
            "Updating scope · previous scope shown until ready"
            if self.fit_mask.isVisible()
            else "Preparing fitting scope…"
        )
        self._fit_timer.start()

    def dismiss_fit_preview(self) -> None:
        """Drop a preparation when returning to inspection or changing inputs."""
        self.fit_preview.details.setChecked(False)
        if self.fit_busy or self._showing_result:
            with QtCore.QSignalBlocker(self.workspace_tabs):
                self.workspace_tabs.setCurrentIndex(0)
            self.workspaces.setCurrentIndex(0)
            return
        self._fit_timer.stop()
        self._fit_revision += 1
        self._computation.cancel("fit")
        self.fit_preparation = None
        self.fit_preview.setEnabled(True)
        self.fit_mask.hide()
        self.fit_members.clear()
        with QtCore.QSignalBlocker(self.workspace_tabs):
            self.workspace_tabs.setCurrentIndex(0)
        self.workspaces.setCurrentIndex(0)
        self.workspace_tabs.setTabEnabled(1, self.selected_id is not None)
        self._update_interaction_hint()

    @property
    def fit_busy(self) -> bool:
        """Whether scientific edits must wait for the one active fit process."""
        return self._fit_process.busy

    def _scientific_inputs_changed(self) -> None:
        self.comparison.clear()
        self._scientific_revision += 1
        self._showing_result = False
        self.dismiss_fit_preview()

    def run_local_fit(self) -> None:
        """Submit the reviewed preparation, keeping any previous result until success."""
        prepared = self.fit_preparation
        if self.fit_busy or self._exporting or prepared is None:
            return
        reason = execution_unavailable(prepared)
        if reason:
            self.fit_preview.show_status(reason)
            return
        if self.fit_result is not None and not self.confirm_result_loss("replace"):
            return
        self.cancel_interaction()
        self._fit_timer.stop()
        self._fit_revision += 1
        self._computation.cancel("fit")
        self._submitted = prepared
        self._submitted_revision = self._scientific_revision
        self._showing_result = False
        self._fit_process.start(prepared)
        self._fit_status_timer.start()

    def _fit_state_changed(self) -> None:
        from time import perf_counter  # noqa: PLC0415

        if self._closing:
            return
        busy = self.fit_busy
        if busy:
            elapsed = perf_counter() - self._fit_process.started_at
            state = {
                "starting": "Starting fit process",
                "running": "Fitting reviewed group",
                "canceling": "Canceling fit · stopping process",
            }[self._fit_process.state]
            self.fit_preview.show_running(f"{state} · {elapsed:.1f} s")
            self.fit_preview.cancel.setEnabled(self._fit_process.state != "canceling")
        else:
            self._fit_status_timer.stop()
            self.fit_preview.cancel.hide()
        self.peak_scatter.editable = not busy
        self._update_interaction_hint()
        self._update_edit_actions()
        self.fit_state_changed.emit()

    def _fit_finished(self, generation: int, outcome: object) -> None:
        if self._closing or generation != self._fit_process.generation:
            return
        prepared = self._submitted
        self._submitted = None
        if prepared is None:
            return
        if self._submitted_revision != self._scientific_revision:
            self.fit_preview.show_status("Fit outdated · inputs changed; result was rejected")
        elif isinstance(outcome, FitProcessResult):
            self.fit_result = outcome
            self.exported_result_id = None
            self.result_notice.setText("Fit result in memory · Not saved")
            self.result_notice.setToolTip("")
            self._result_revision = self._submitted_revision
            self.show_fit_result(activate=False)
        else:
            self.fit_preview.show_preparation(prepared)
            self.fit_preview.status.setText(
                "Fit canceled · process stopped" if outcome is None else "Fit failed · Details…"
            )
            self.fit_preview._details = str(outcome or "Canceled; no partial result retained.")
            self.fit_preview.status.setToolTip(self.fit_preview._details)
        self.fit_preview.retained.setVisible(
            self.fit_result is not None and not self._showing_result
        )
        self._update_interaction_hint()
        self.fit_state_changed.emit()

    def show_fit_result(self, *, activate: bool = True) -> None:
        """Restore retained numerical inspection without changing viewport or peaks."""
        if self.fit_result is None or self.fit_busy:
            return
        self._fit_timer.stop()
        self._fit_revision += 1
        self._computation.cancel("fit")
        previous = self.workspaces.currentIndex()
        outdated = self._result_revision != self._scientific_revision
        self._apply_fit_preparation(self.fit_result.scientific.inputs)
        self._showing_result = True
        if outdated:
            self.fit_preparation = None
            self.fit_mask.hide()
            self.fit_members.clear()
        self.fit_preview.show_result(self.fit_result, self.plane, outdated=outdated)
        self.comparison.show_result(self.fit_result, outdated=outdated)
        self.fit_preview.retained.hide()
        self.fit_preview.prepare.setEnabled(self.selected_id is not None)
        try:
            self._fitted_position_updates()
        except ValueError as error:
            self.fit_preview.apply_positions.setEnabled(False)
            self.fit_preview.apply_positions.setToolTip(str(error))
        else:
            self.fit_preview.apply_positions.setEnabled(True)
            self.fit_preview.apply_positions.setToolTip("Update the group in one undoable edit")
        if not activate:
            with QtCore.QSignalBlocker(self.workspace_tabs):
                self.workspace_tabs.setCurrentIndex(previous)
            self.workspaces.setCurrentIndex(previous)
        self._update_interaction_hint()

    def _fitted_position_updates(self) -> dict[str, tuple[float, float]]:
        if self.fit_busy or self._exporting or self.inspector.pending:
            raise ValueError("Finish the current fit, export or inspector edit first.")
        if self.fit_result is None or self._result_revision != self._scientific_revision:
            raise ValueError("This result is outdated. Prepare and run a new fit.")
        result = self.fit_result.scientific
        snapshot = result.inputs.snapshot
        if (
            snapshot.spectrum_id != id(self.spectra)
            or snapshot.data_shape != tuple(self.spectra.data.shape)
            or snapshot.spectral_parameters
            != tuple(a.model_dump_json() for a in self.spectra.params)
            or snapshot.plane_values != tuple(float(v) for v in self.spectra.z_values)
            or snapshot.source_files != self.source_files
        ):
            raise ValueError("Scientific inputs changed since this fit. Run a new fit.")
        return fitted_position_updates(result, self.peaks)

    def apply_fitted_positions(self) -> None:
        """Confirm and commit the complete fitted group, retaining its original result."""
        try:
            updates = self._fitted_position_updates()
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Cannot apply fitted positions", str(error))
            return
        if not updates:
            QtWidgets.QMessageBox.information(
                self,
                "Fitted positions unchanged",
                "All fitted positions already match the working peaks. No edit was made.",
            )
            return
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("Apply fitted positions")
        dialog.setText(f"Update chemical shifts for {len(updates)} peaks in the fitted group?")
        changes = []
        for number, peak in enumerate(self.peaks, 1):
            if peak.id in updates:
                coordinates = ", ".join(
                    f"{'Indirect' if dim == 0 else 'Direct'}: {before:.9g} → {after:.9g} ppm"
                    for dim, (before, after) in enumerate(
                        zip(peak.positions, updates[peak.id], strict=True)
                    )
                    if before != after
                )
                changes.append(f"{number}. {peak.name} · {coordinates}")
        dialog.setInformativeText(
            "Only chemical-shift coordinates change. Undo restores the whole group in one step. "
            "The original fit remains available for export; comparison graphics will be hidden "
            "because the working inputs change.\n\n"
            + "\n".join(changes[:_POSITION_PREVIEW_PEAKS])
            + (
                f"\n… and {len(changes) - _POSITION_PREVIEW_PEAKS} more peaks (Details)."
                if len(changes) > _POSITION_PREVIEW_PEAKS
                else ""
            )
        )
        dialog.setDetailedText("\n".join(changes))
        dialog.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Apply | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        if dialog.exec() != QtWidgets.QMessageBox.StandardButton.Apply:
            return
        # Modal dialogs process events: validate the complete live state again.
        try:
            current = self._fitted_position_updates()
            if current != updates:
                raise ValueError(
                    "Fitted positions changed during confirmation. Review the result again."
                )
            self.edits.update_positions(updates)
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Cannot apply fitted positions", str(error))
            return
        self._refresh_peaks()
        self.edits_changed.emit()
        self.show_fit_result()

    @property
    def result_exported(self) -> bool:
        """Whether the exact currently retained scientific result was exported."""
        return (
            self.fit_result is not None
            and self.exported_result_id == self.fit_result.scientific.run_id
        )

    def export_fit_results(self) -> bool:
        """Export original completed inputs, leaving working edits and comparison intact."""
        if self.fit_result is None or self._exporting:
            return False
        result = self.fit_result.scientific
        self._exporting = True
        try:
            destination = choose_and_export(self, result)
        finally:
            self._exporting = False
        if destination is None:
            return False
        # A running fit can complete while a close/export dialog processes Qt events.
        if self.fit_result is not None and self.fit_result.scientific.run_id == result.run_id:
            self.exported_result_id = result.run_id
            self.result_notice.setText("Fit results exported")
            self.result_notice.setToolTip(str(destination))
        QtWidgets.QMessageBox.information(
            self,
            "Fit results exported",
            f"Exported completed fit to:\n{destination}\n\n"
            + (
                "This outdated result belongs to its original scientific inputs."
                if self._result_revision != self._scientific_revision
                else "Scientific results exported; working peaks are unchanged."
            ),
        )
        return True

    def _result_loss_choice(self, action: str) -> QtWidgets.QMessageBox.StandardButton:
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("Unexported fitted result")
        outdated = self._result_revision != self._scientific_revision
        dialog.setText(
            f"Export the completed fit before you {action} it?"
            + (
                " This outdated result belongs to its original scientific inputs."
                if outdated
                else ""
            )
            + " Export saves scientific results. Working peaks and GUI sessions are separate."
        )
        dialog.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        dialog.button(QtWidgets.QMessageBox.StandardButton.Save).setText("Export…")
        dialog.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        return QtWidgets.QMessageBox.StandardButton(dialog.exec())

    def confirm_result_loss(self, action: str) -> bool:
        """Resolve each unexported result without discarding it during confirmation."""
        while self.fit_result is not None and not self.result_exported:
            result = self.fit_result
            choice = self._result_loss_choice(action)
            if choice == QtWidgets.QMessageBox.StandardButton.Cancel:
                return False
            if choice == QtWidgets.QMessageBox.StandardButton.Save:
                if not self.export_fit_results():
                    return False
            elif choice != QtWidgets.QMessageBox.StandardButton.Discard:
                return False
            if self.fit_result is result:
                return True
            # Completion during a dialog must protect the new result independently.
        return True

    def _build_inspection_panels(self) -> QtWidgets.QWidget:
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.inspector = PeakInspector(self._axis_title(self.x_axis), self._axis_title(self.y_axis))
        self.inspector.applied.connect(self._apply_peak_fields)
        self.inspector.fields_changed.connect(self.edits_changed)
        self.inspector.deleted.connect(self.delete_peak)
        self.add_peak_button = QtWidgets.QPushButton("Add peak…")
        self.add_peak_button.setCheckable(True)
        self.add_peak_button.clicked.connect(self._toggle_peak_placement)
        cast("QtWidgets.QGridLayout", self.inspector.layout()).addWidget(self.add_peak_button, 2, 2)
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

    def _build_contour_controls(self) -> None:  # noqa: PLR0915
        self.contour_button = QtWidgets.QToolButton(self.spectrum_plot.viewport())
        self.contour_button.setAutoRaise(True)
        self.contour_button.setFixedWidth(
            self.contour_button.fontMetrics().horizontalAdvance("Contours · 1000.00× ▾") + 20
        )
        self.contour_popup = QtWidgets.QDialog(self, QtCore.Qt.WindowType.Popup)
        self.contour_popup.setWindowTitle("Contour controls")
        self.contour_button.clicked.connect(self._show_contour_controls)
        panel = self.contour_popup
        panel.setMinimumWidth(340)
        grid = QtWidgets.QGridLayout(panel)
        grid.setVerticalSpacing(6)
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
            label.setBuddy(control)
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
        # A native widget in the existing title band stays accessible to Qt/macOS.
        # It never covers data, and its reserved width is independent of the value.
        item = self.spectrum_plot.getPlotItem()
        item.titleLabel.setMinimumHeight(self.contour_button.sizeHint().height() + 4)
        self.contour_button.setFixedHeight(self.contour_button.sizeHint().height())
        item.vb.sigResized.connect(self._position_contour_control)
        self._position_contour_control()
        self.contour_status = QtWidgets.QLabel()
        self.contour_status.setFixedHeight(self.contour_status.fontMetrics().lineSpacing() + 4)
        self.contour_status.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
        )

    def _position_contour_control(self) -> None:
        """Keep the control within the title band, outside the spectral ViewBox."""
        title = self.spectrum_plot.getPlotItem().titleLabel
        rect = self.spectrum_plot.mapFromScene(title.sceneBoundingRect()).boundingRect()
        self.contour_button.move(
            rect.right() - self.contour_button.width() - 4,
            rect.top() + (rect.height() - self.contour_button.height()) // 2,
        )

    def _show_contour_controls(self) -> None:
        """Anchor transient controls without changing the spectrum layout."""
        popup = self.contour_popup
        popup.adjustSize()
        anchor = self.contour_button.mapToGlobal(self.contour_button.rect().bottomRight())
        screen = self.contour_button.screen().availableGeometry()
        x = min(anchor.x() - popup.width() + 1, screen.right() - popup.width() + 1)
        y = anchor.y()
        if y + popup.height() > screen.bottom():
            y = self.contour_button.mapToGlobal(QtCore.QPoint(0, 0)).y() - popup.height()
        popup.move(max(screen.left(), x), max(screen.top(), y))
        popup.show()
        self.threshold.setFocus()
        self.threshold.selectAll()

    def _sync_contour_sliders(self) -> None:
        value = f"{self.threshold.value():.2f}".rstrip("0")
        if value.endswith("."):
            value += "0"
        self.contour_button.setText(f"Contours · {value}× ▾")
        self.contour_button.setAccessibleName(f"Contours, threshold {value} times noise")
        self.contour_button.setToolTip(
            f"Lowest contour: {self.lowest_contour_level:.9g} ({value} × first-plane noise). "
            "Click to edit threshold, spacing and levels. During preparation, "
            "threshold changes update the fitting group; submitted fits stay fixed."
        )
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
        self.fit_preview.set_result_plane(plane)
        for control in (self.plane_slider, self.plane_spin):
            with QtCore.QSignalBlocker(control):
                control.setValue(plane + 1)
        self._schedule_contours()
        self._update_slices()

    def _schedule_contours(self) -> None:
        self._contour_revision += 1
        self._computation.cancel("contours")
        self.comparison.contours_changed()
        self.contour_status.setText("Updating contours…")
        self.footer_status.setCurrentIndex(1)
        self.contour_status.setToolTip(
            "Previous contours remain visible until the new set is ready."
        )
        # Coalesce redraw requests; the worker keeps only the latest pending work.
        if not self._contour_timer.isActive():
            self._contour_timer.start()

    @property
    def lowest_contour_level(self) -> float:
        """Absolute base intensity shared by displayed contours and local clustering."""
        return max(self.noise * self.threshold.value(), np.finfo(float).tiny)

    def _update_contours(self) -> None:
        """Queue numerical contour extraction; no Qt paths are made in the worker."""
        n_levels = self.levels.value()
        positive = self.lowest_contour_level * self.factor.value() ** np.arange(n_levels)
        levels = np.concatenate((positive, -positive))
        axes = [axis.model_copy(deep=True) for axis in self.spectra.spectral_params]
        operation = partial(
            contour_geometry,
            self.spectra.data[self.plane],
            axes,
            levels,
        )
        self._computation.submit("contours", self._contour_revision, operation)

    def _apply_contour_geometry(self, geometry: ContourGeometry) -> None:
        paths = paths_from_geometry(geometry)
        while len(self.contours) < len(paths):
            item = QtWidgets.QGraphicsPathItem()
            item.setZValue(-10)
            self.spectrum_plot.addItem(item)
            self.contours.append(item)
        n_levels = len(paths) // 2
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
        self._contour_plane = self.plane
        self._contour_applied_revision = self._contour_revision
        self.contour_status.clear()
        self.footer_status.setCurrentIndex(0)
        self.contour_status.setToolTip("")

    @QtCore.Slot(str, int, object)
    def _computation_finished(self, kind: str, revision: int, outcome: object) -> None:
        if self._closing:
            return
        if kind == "fit" and revision == self._fit_revision:
            if isinstance(outcome, Exception):
                self.fit_mask.hide()
                self.fit_members.clear()
                self.fit_preview.show_status(str(outcome), clear=True)
            else:
                self._apply_fit_preparation(cast("PreparedLocalFit", outcome))
        elif kind == "contours" and revision == self._contour_revision:
            if isinstance(outcome, Exception):
                self.footer_status.setCurrentIndex(1)
                self.contour_status.setText("Contour update failed · hover for details")
                self.contour_status.setToolTip(str(outcome))
            else:
                self._apply_contour_geometry(cast("ContourGeometry", outcome))

    def stop_preparation(self) -> None:
        """Invalidate work without waiting for background computation at shutdown."""
        self.comparison.clear()
        self._closing = True
        self._fit_process.close()
        self._fit_status_timer.stop()
        self._fit_timer.stop()
        self._contour_timer.stop()
        self._computation.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        """Close without waiting for obsolete numerical work."""
        self.stop_preparation()
        super().closeEvent(event)

    def _clicked(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if event.isAccepted() and not self.adding_peak:
            return
        view = self.spectrum_plot.getViewBox()
        if view.sceneBoundingRect().contains(event.scenePos()):
            point = view.mapSceneToView(event.scenePos())
            if self.adding_peak:
                # End placement while working records and graphics still agree.
                # Only then commit, rebuild graphics, and select the added peak.
                self.cancel_interaction()
                peak = self.edits.add((point.y(), point.x()))
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
        if hasattr(self, "comparison"):
            self.comparison.update_slices()

    def set_plane_values(self, values: np.ndarray) -> None:
        """Replace experimental plane coordinates without changing spectral samples."""
        if self.fit_busy:
            raise ValueError("Cancel the running fit before changing plane values.")
        validate_plane_values(values, self.spectra.data.shape[0])
        self._scientific_inputs_changed()
        self.spectra.z_values = np.array(values, dtype=float, copy=True)
        self.profile.enableAutoRange()
        self._update_slices()

    def set_peaks(self, peaks: Sequence[WorkingPeak]) -> None:
        """Display imported peaks, folding only indirect coordinates."""
        if self.fit_busy:
            raise ValueError("Cancel the running fit before replacing working peaks.")
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
        if (
            peak_id != self.selected_id
            and self.fit_preparation is None
            and not self.fit_busy
            and not self._showing_result
            and self.workspaces.currentIndex() == 1
        ):
            self.dismiss_fit_preview()
        peak = next((p for p in self.peaks if p.id == peak_id), None)
        self.selected_id = peak.id if peak else None
        if (
            not self.fit_busy
            and not self._showing_result
            and self.fit_preparation is not None
            and not any(
                member.working_id == self.selected_id for member in self.fit_preparation.identities
            )
        ):
            self.dismiss_fit_preview()
        self.workspace_tabs.setTabEnabled(
            1,
            peak is not None
            or self.fit_preparation is not None
            or self.fit_busy
            or self.fit_result is not None,
        )
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
        # Selection populates fields, but does not edit scientific working state.
        # Do not let those text changes invalidate an in-group preparation.
        with QtCore.QSignalBlocker(self.inspector):
            self.inspector.show_peak(peak, position)
        self._update_edit_actions()
        self.show_peak_labels(self.labels_visible)
        self._update_interaction_hint()
        if peak and inspect:
            self._select_position(*position)

    def _apply_peak_fields(self, name: str, positions: tuple[float, float]) -> bool:
        """Commit working values only when no scientific operation owns the inputs."""
        if self.fit_busy:
            return False
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
        if not self.fit_busy and self.selected_id is not None and not self.inspector.pending:
            self.peak_scatter.cancel_drag()
            self._drag_peak = None
            self.edits.delete(self.selected_id)
            self.selected_id = None
            self._refresh_peaks()
            self.edits_changed.emit()

    def undo(self) -> None:
        """Undo a committed peak edit only after inspector text is resolved."""
        if self.fit_busy or self.inspector.pending:
            return
        self.cancel_interaction()
        self.edits.undo()
        self._refresh_peaks()
        self.edits_changed.emit()

    def redo(self) -> None:
        """Redo a committed peak edit only after inspector text is resolved."""
        if self.fit_busy or self.inspector.pending:
            return
        self.cancel_interaction()
        self.edits.redo()
        self._refresh_peaks()
        self.edits_changed.emit()

    def _toggle_peak_placement(self) -> None:
        if self.adding_peak:
            self.cancel_interaction()
        else:
            self.start_adding_peak()

    def _update_interaction_hint(self) -> None:
        if self.fit_busy:
            text = (
                "LOCAL FIT RUNNING · Navigation available · Scientific edits pause until fit stops"
            )
        elif self._showing_result:
            text = (
                "Outdated local result · Numerical values use original inputs"
                if self._result_revision != self._scientific_revision
                else "Local result · Convergence does not establish scientific fit quality"
            )
        elif self.fit_preparation is not None:
            text = "Amber cells: exact fitting samples · Rings: group members"
        elif self.workspaces.currentIndex() == 1:
            text = "LOCAL FIT PREVIEW · Use Threshold to review scope · Back to Inspect to edit"
        elif self.adding_peak:
            text = "PLACING PEAK · Click in spectrum · Cancel placement or Esc to stop"
        elif self.selected_id is not None:
            text = "Drag selected peak to move · Edit fields at right · Fit previews fitting scope"
        else:
            text = "Click a peak to select · Drag background to pan · Scroll to zoom"
        self.spectrum_plot.setToolTip(text)
        # Resolve placement before editing another assignment; Cancel stays available.
        for field in (
            self.inspector.name,
            self.inspector.x_field,
            self.inspector.y_field,
            self.inspector.apply,
            self.inspector.delete,
        ):
            field.setEnabled(
                self.selected_id is not None and not self.adding_peak and not self.fit_busy
            )
        self.inspector.delete.setEnabled(
            self.selected_id is not None
            and not self.adding_peak
            and not self.inspector.pending
            and not self.fit_busy
        )
        self.add_peak_button.setChecked(self.adding_peak)
        self.add_peak_button.setText("Cancel placement" if self.adding_peak else "Add peak…")
        self.add_peak_button.setToolTip(
            "Cancel placement (Escape)"
            if self.adding_peak
            else "Click here, then click the spectrum to place one peak (A with spectrum focus)"
        )
        self.add_action.setChecked(self.adding_peak)
        self.add_action.setText("Cancel peak placement" if self.adding_peak else "Add peak…")
        self.add_action.setEnabled(not self.fit_busy)
        self.add_peak_button.setEnabled(not self.fit_busy)

    def start_adding_peak(self) -> None:
        """Arm a single placement without a persistent editing mode."""
        if self.fit_busy or not self.apply_pending_fields():
            self._update_interaction_hint()
            return
        self.dismiss_fit_preview()
        self.cancel_interaction()
        self.adding_peak = True
        self.peak_scatter.placing = True
        for line in (self.cross_x, self.cross_y):
            line.setMovable(False)
        self.spectrum_plot.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.spectrum_plot.setFocus()
        self.inspector.details.setText(
            "Click in the spectrum to place one peak; panning and zoom still work."
        )
        self._update_interaction_hint()

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
        """Preview a reversible working move while scientific editing is available."""
        if self.fit_busy:
            return
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
        if self.fit_busy:
            raise ValueError("Cancel the running fit before replacing working peaks.")
