"""One explicit inspection/edit workflow composed from the rail and spectrum canvas."""

from dataclasses import replace
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui.editor.canvas import SpectrumCanvas
from peakfit.gui.editor.rail import PeakRail
from peakfit.gui.peaks import PeakEdits

if TYPE_CHECKING:
    from peakfit.gui.editor.loading import Dataset
    from peakfit.gui.peaks import WorkingPeak


class EditorWorkspace(QtWidgets.QWidget):
    """Own committed scientific edits; navigation and drafts never change revision."""

    changed = QtCore.Signal()

    def __init__(self, dataset: Dataset) -> None:  # noqa: PLR0915
        super().__init__()
        self.dataset = dataset
        self.edits = PeakEdits(dataset.peaks)
        peak_sources = tuple(source for source in dataset.sources if source[0] == "peak_list")
        # Keep origins for undoable IDs, but expose only sources of current peaks.
        self._peak_sources = {peak.id: peak_sources for peak in dataset.peaks}
        self.plane_values_modified = False
        self.selected_id: str | None = None
        self.canvas = SpectrumCanvas(dataset)
        self.rail = PeakRail(
            self.canvas.axis_title(self.canvas.x_axis), self.canvas.axis_title(self.canvas.y_axis)
        )
        self.inspector = self.rail.inspector
        self.canvas.resolve_edit = self.commit_fields
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(8)
        self.splitter.addWidget(self.rail)
        self.splitter.addWidget(self.canvas)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([320, 1000])
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 4)
        layout.setSpacing(8)
        header = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel(dataset.path.stem)
        self.title.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.title.setObjectName("title")
        header.addWidget(self.title)
        self.modified = QtWidgets.QLabel()
        self.modified.setObjectName("muted")
        header.addWidget(self.modified)
        header.addStretch()
        self.peaks_button = QtWidgets.QPushButton("Peaks")
        self.peaks_button.setCheckable(True)
        self.peaks_button.setChecked(bool(dataset.peaks))
        self.peaks_button.clicked.connect(self.toggle_rail)
        self.add_button = QtWidgets.QPushButton("Add peak")
        self.add_button.setCheckable(True)
        self.add_button.clicked.connect(self.start_placing)
        for button in (self.peaks_button, self.add_button):
            header.addWidget(button)
        layout.addLayout(header)
        strip = QtWidgets.QHBoxLayout()
        strip.setSpacing(0)
        self.mode_buttons = []
        for index, name in enumerate(("Experimental", "Overlay", "Residual")):
            button = QtWidgets.QPushButton(name)
            button.setObjectName("mode")
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.setEnabled(index == 0)
            if index:
                button.setToolTip("Available after whole-dataset fitting in the next milestone")
            self.mode_buttons.append(button)
            strip.addWidget(button)
        strip.addSpacing(16)
        display = QtWidgets.QPushButton("Display…")
        display.clicked.connect(self.display_dialog)
        full = QtWidgets.QPushButton("Full spectrum")
        full.clicked.connect(self.canvas.full_spectrum)
        self.pin_button = QtWidgets.QPushButton("Pin slices")
        self.pin_button.setCheckable(True)
        self.pin_button.setMinimumWidth(110)
        self.pin_button.toggled.connect(self.canvas.set_pinned)
        self.canvas.pin_changed.connect(self._pin_changed)
        for button in (display, full, self.pin_button):
            strip.addWidget(button)
            strip.addSpacing(8)
        strip.addStretch()
        self.legend = QtWidgets.QLabel("Measured: + solid / − dashed · a.u.")
        self.legend.setObjectName("muted")
        strip.addWidget(self.legend)
        layout.addLayout(strip)
        self.summary = QtWidgets.QLabel()
        self.summary.setObjectName("muted")
        layout.addWidget(self.summary)
        layout.addWidget(self.splitter, 1)
        planes = QtWidgets.QHBoxLayout()
        self.previous = QtWidgets.QPushButton("‹")
        self.previous.setAccessibleName("Previous plane")
        self.next = QtWidgets.QPushButton("›")
        self.next.setAccessibleName("Next plane")
        for button in (self.previous, self.next):
            button.setFixedWidth(32)
        self.previous.clicked.connect(lambda: self.set_plane(self.canvas.plane))
        self.next.clicked.connect(lambda: self.set_plane(self.canvas.plane + 2))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setAccessibleName("Plane navigation")
        self.spin = QtWidgets.QSpinBox()
        self.spin.setAccessibleName("One-based plane number")
        self.spin.setKeyboardTracking(False)
        n_planes = dataset.spectra.data.shape[0]
        for control in (self.slider, self.spin):
            control.setRange(1, n_planes)
            control.setValue(1)
            control.valueChanged.connect(self.set_plane)
        self.plane_label = QtWidgets.QLabel()
        planes.addWidget(QtWidgets.QLabel("Plane"))
        for widget in (self.previous, self.next, self.slider, self.spin, self.plane_label):
            planes.addWidget(widget, 1 if widget is self.slider else 0)
        if n_planes == 1:
            for widget in (self.previous, self.next, self.slider, self.spin):
                widget.hide()
        layout.addLayout(planes)
        self.status = QtWidgets.QLabel()
        self.status.setObjectName("muted")
        self.status.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.status.setAccessibleName("Sampled spectrum values")
        layout.addWidget(self.status)
        self.error = QtWidgets.QLabel()
        self.error.setObjectName("error")
        self.error.setWordWrap(True)
        self.error.hide()
        layout.addWidget(self.error)
        self.canvas.status_changed.connect(self.status.setText)
        self.canvas.hovered.connect(self.rail.highlight)
        self.rail.hovered.connect(lambda identity: self.canvas.hover(identity, from_list=True))
        self.rail.selected.connect(self.select_peak)
        self.canvas.selected.connect(self.select_peak)
        self.canvas.placed.connect(self.place_peak)
        self.canvas.moved.connect(self.move_peak)
        self.canvas.draft_changed.connect(self._drag_draft)
        self.rail.edit_requested.connect(self.focus_editor)
        self.canvas.edit_requested.connect(self.focus_editor)
        self.inspector.commit_requested.connect(self.commit_fields)
        self.inspector.deleted.connect(self.delete_peaks)
        self.inspector.fields_changed.connect(self._fields_changed)
        self._build_actions()
        self._refresh()
        self.rail.setVisible(bool(dataset.peaks))
        self.set_plane(1)

    @property
    def peaks(self):
        """Committed immutable input records, independent of the display."""
        return self.edits.peaks

    @property
    def pending(self) -> bool:
        """Uncommitted editor or pointer draft."""
        return self.inspector.pending or self.canvas._drag_peak is not None

    @property
    def has_changes(self) -> bool:
        """Whether closing would discard a draft, unexported peaks or session plane metadata."""
        return self.edits.dirty or self.pending or self.plane_values_modified

    def import_peaks(
        self,
        peaks: tuple[WorkingPeak, ...],
        sources: tuple[tuple[str, str], ...],
        *,
        append: bool,
    ) -> None:
        """Bind fresh imported IDs to their origins in the same undoable handoff."""
        if any(peak.id in self._peak_sources for peak in peaks):
            raise ValueError("Imported peaks require new stable IDs.")
        self.edits.replace_all(self.peaks + peaks if append else peaks)
        self._peak_sources.update((peak.id, sources) for peak in peaks)
        self._refresh()

    def _pin_changed(self, pinned: bool) -> None:
        with QtCore.QSignalBlocker(self.pin_button):
            self.pin_button.setChecked(pinned)
        self.pin_button.setText("Slices pinned" if pinned else "Pin slices")

    def _build_actions(self) -> None:
        self.undo_action = QtGui.QAction("Undo peak edit", self)
        self.redo_action = QtGui.QAction("Redo peak edit", self)
        self.undo_action.setShortcuts(QtGui.QKeySequence.StandardKey.Undo)
        self.redo_action.setShortcuts(QtGui.QKeySequence.StandardKey.Redo)
        self.undo_action.triggered.connect(lambda: self.history(False))
        self.redo_action.triggered.connect(lambda: self.history(True))
        self.add_action = QtGui.QAction("Add peak", self)
        self.add_action.triggered.connect(self.start_placing)
        self.delete_action = QtGui.QAction("Delete selected peaks", self)
        self.delete_action.triggered.connect(self.delete_peaks)
        self.actions_for_menu = [
            self.undo_action,
            self.redo_action,
            self.add_action,
            self.delete_action,
        ]
        for action in self.actions_for_menu:
            self.addAction(action)
        for widget in (self.canvas.main, self.rail.list):
            for key, callback in (("Delete", self.delete_peaks), ("Backspace", self.delete_peaks)):
                shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), widget)
                shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
                shortcut.activated.connect(callback)
        for key, callback in (("A", self.start_placing), ("Escape", self.cancel_gesture)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self.canvas.main)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)

    def _fields_changed(self) -> None:
        if hasattr(self, "undo_action"):
            self._update_actions()
            self.changed.emit()
        if not self.inspector.pending:
            self.error.hide()

    def _drag_draft(self, _draft: bool) -> None:
        self._update_actions()
        self.changed.emit()

    def commit_fields(self) -> bool:
        """Commit a whole valid group or retain its visible invalid text."""
        if not self.inspector.pending:
            return True
        try:
            name, positions = self.inspector.values()
            if self.selected_id is None:
                return True
            self.edits.update(self.selected_id, name, positions)
        except ValueError as exc:
            self.inspector.details.setText(str(exc))
            self.error.setText(str(exc))
            self.error.show()
            return False
        self.error.hide()
        self._refresh()
        return True

    def select_peak(self, identity: str) -> None:
        """Resolve drafts before changing the one stable inspector identity."""
        if not self.commit_fields():
            # Restore list identity without overwriting the invalid visible draft.
            with QtCore.QSignalBlocker(self.rail.list):
                item = self.rail._items.get(self.selected_id or "")
                if item:
                    self.rail.list.setCurrentItem(item)
            return
        self.selected_id = identity
        self.canvas.select(identity)
        self._sync_inspector()
        self._update_actions()

    def _sync_inspector(self) -> None:
        peak = next((p for p in self.peaks if p.id == self.selected_id), None)
        self.rail.select(peak, self.canvas.display(peak) if peak else None)

    def _refresh(self) -> None:
        peak_sources = tuple(
            dict.fromkeys(
                source for peak in self.peaks for source in self._peak_sources.get(peak.id, ())
            )
        )
        sources = tuple(source for source in self.dataset.sources if source[0] != "peak_list")
        sources += peak_sources
        if sources != self.dataset.sources:
            self.dataset = replace(self.dataset, sources=sources)
        self.canvas.dataset = self.dataset
        if self.selected_id not in {p.id for p in self.peaks}:
            self.selected_id = None
        self.canvas.selected_id = self.selected_id
        self.canvas.markers.selected_id = self.selected_id
        self.canvas.set_peaks(self.peaks)
        self.rail.refresh(self.peaks, self.selected_id)
        self._sync_inspector()
        self._update_actions()
        self.summary.setText(
            f"Not fitted · {len(self.peaks)} peaks · {self.dataset.spectra.data.shape[0]} planes"
        )
        self.changed.emit()

    def _update_actions(self) -> None:
        self.undo_action.setText("Undo " + self.edits.undo_label)
        self.redo_action.setText("Redo " + self.edits.redo_label)
        self.undo_action.setEnabled(self.edits.can_undo and not self.pending)
        self.redo_action.setEnabled(self.edits.can_redo and not self.pending)
        self.delete_action.setEnabled(self.selected_id is not None)
        self.modified.setText(
            "Modified inputs · plane values session-only"
            if self.plane_values_modified
            else "Modified peaks"
            if self.edits.dirty or self.pending
            else ""
        )

    def history(self, redo: bool) -> None:
        """Undo exact records; monotonic scientific revision never recertifies a run."""
        if self.pending:
            return
        self.canvas.cancel_gesture()
        if redo:
            self.edits.redo()
        else:
            self.edits.undo()
        self._refresh()

    def start_placing(self) -> None:
        """Arm one precise placement after resolving any complete text group."""
        if not self.commit_fields():
            self.add_button.setChecked(False)
            return
        self.canvas.set_placing(True)
        self.add_button.setChecked(True)
        self.canvas.main.setFocus()
        self.status.setText("Place peak · click exact ppm position · Escape cancels")

    def place_peak(self, positions) -> None:
        """Store continuous unfolded ppm from a single click, without snapping."""
        if not self.commit_fields():
            return
        peak = self.edits.add(positions)
        self.cancel_gesture()
        self.selected_id = peak.id
        self._refresh()
        self.canvas.select(peak.id)

    def move_peak(self, identity, positions) -> None:
        """One drag release is one complete precision-preserving input change."""
        if not self.commit_fields():
            self.canvas.cancel_gesture()
            return
        peak = next(p for p in self.peaks if p.id == identity)
        self.edits.update(identity, peak.name, positions)
        self._refresh()
        self.canvas.select(identity)

    def delete_peaks(self) -> None:
        """Deletion owns the selected draft; multi-delete is list-only."""
        identities = {self.selected_id} if self.selected_id else set()
        if self.rail.list.hasFocus():
            identities = {item.data(256) for item in self.rail.list.selectedItems()}
        if identities:
            self.cancel_gesture()
            self.edits.delete_many(identities)
            self.selected_id = None
            self._refresh()

    def cancel_gesture(self) -> None:
        """Escape on the canvas cancels its current gesture, never numerical work."""
        self.canvas.cancel_gesture()
        self.add_button.setChecked(False)

    def toggle_rail(self) -> None:
        """Resolve a draft on explicit hide, preserving the user's later resize choice."""
        visible = self.peaks_button.isChecked()
        if not visible and not self.commit_fields():
            self.peaks_button.setChecked(True)
            return
        self.rail.setVisible(visible)
        if visible:
            self.splitter.setSizes([320, max(520, self.width() - 320)])

    def focus_editor(self) -> None:
        """Explicit editing opens the rail without changing selection."""
        self.peaks_button.setChecked(True)
        self.toggle_rail()
        if self.selected_id:
            self.inspector.x_field.setFocus()
            self.inspector.x_field.selectAll()

    def set_plane(self, one_based: int) -> None:
        """Plane navigation stays available even with an invalid coordinate draft."""
        plane = max(1, min(one_based, self.dataset.spectra.data.shape[0]))
        for control in (self.slider, self.spin):
            with QtCore.QSignalBlocker(control):
                control.setValue(plane)
        self.canvas.set_plane(plane - 1)
        count = self.dataset.spectra.data.shape[0]
        self.previous.setEnabled(plane > 1)
        self.next.setEnabled(plane < count)
        value = (
            f"value = {self.dataset.spectra.z_values[plane - 1]:.9g} (file units)"
            if self.dataset.has_plane_values
            else "no plane values"
        )
        self.plane_label.setText("Single plane" if count == 1 else f"/ {count} · {value}")

    def display_dialog(self) -> None:
        """Display choices do not change scientific inputs or edit history."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Spectrum display")
        form = QtWidgets.QFormLayout(dialog)
        threshold = QtWidgets.QDoubleSpinBox()
        threshold.setRange(0.01, 1e12)
        threshold.setDecimals(4)
        threshold.setValue(self.canvas.contour_threshold)
        threshold.setKeyboardTracking(False)
        factor = QtWidgets.QDoubleSpinBox()
        factor.setRange(1.1, 5)
        factor.setValue(self.canvas.contour_factor)
        count = QtWidgets.QSpinBox()
        count.setRange(1, 20)
        count.setValue(self.canvas.contour_count)
        labels = QtWidgets.QCheckBox("Show peak labels")
        labels.setChecked(self.canvas.labels_visible)
        heavy = QtWidgets.QCheckBox("Heavier contours")
        heavy.setChecked(self.canvas.heavy_strokes)
        fit = QtWidgets.QPushButton("Fit current traces")
        fit.clicked.connect(self.canvas.fit_current_traces)
        for title, field in (
            ("First contour / a.u.", threshold),
            ("Level factor", factor),
            ("Levels per sign", count),
            ("", labels),
            ("", heavy),
            ("Signed trace scale", fit),
        ):
            form.addRow(title, field)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.canvas.contour_threshold = threshold.value()
            self.canvas.contour_factor = factor.value()
            self.canvas.contour_count = count.value()
            self.canvas.labels_visible = labels.isChecked()
            self.canvas.heavy_strokes = heavy.isChecked()
            self.canvas._refresh_labels()
            self.canvas.request_contours()

    def stop(self) -> None:
        """Stop display computation without blocking window close."""
        self.canvas.stop()
