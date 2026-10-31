"""Three columns: navigator, spectrum and inspector, around one editing workflow."""

from dataclasses import replace
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.engine.domain.spectrum import get_shape_names
from peakfit.gui import formatting
from peakfit.gui import theme as themes
from peakfit.gui.dataset_fit import dataset_summary
from peakfit.gui.editor.canvas import SpectrumCanvas
from peakfit.gui.editor.fitting import DatasetFitting
from peakfit.gui.editor.inspector import Inspector, find_cluster
from peakfit.gui.editor.labels import ElidedLabel
from peakfit.gui.editor.navigator import Navigator
from peakfit.gui.icons import icon
from peakfit.gui.peaks import PeakEdits

if TYPE_CHECKING:
    from peakfit.gui.peaks import WorkingPeak
    from peakfit.gui.spectrum import Dataset


_RESIDUAL = 2
_WIDE_TOOLBAR = 1200
_SIDES = (280, 320)  # default navigator and inspector widths at 1280 px


class EditorWorkspace(QtWidgets.QWidget):
    """Own committed scientific edits; navigation and drafts never change revision."""

    changed = QtCore.Signal()

    def __init__(self, dataset: Dataset) -> None:  # noqa: PLR0915
        super().__init__()
        self.inputs_loading = False
        self.dataset = dataset
        self.edits = PeakEdits(dataset.peaks)
        peak_sources = tuple(source for source in dataset.sources if source[0] == "peak_list")
        # Keep origins for undoable IDs, but expose only sources of current peaks.
        self._peak_sources = {peak.id: peak_sources for peak in dataset.peaks}
        self.plane_values_modified = False
        self.selected_id: str | None = None
        self.canvas = SpectrumCanvas(dataset)
        decimals = (self.canvas.decimals_x, self.canvas.decimals_y)
        self.navigator = Navigator(
            self.canvas.axis_title(self.canvas.x_axis),
            self.canvas.axis_title(self.canvas.y_axis),
            decimals,
        )
        self.panel = Inspector(self.canvas.x_axis, self.canvas.y_axis, decimals)
        self.inspector = self.panel.editor
        self.canvas.resolve_edit = self.commit_fields
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(1)
        self.splitter.addWidget(self.navigator)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.panel)
        for index, stretch in enumerate((0, 1, 0)):
            self.splitter.setStretchFactor(index, stretch)
        self.splitter.setSizes([_SIDES[0], 1280 - sum(_SIDES), _SIDES[1]])
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_toolbar(dataset))
        layout.addWidget(self.splitter, 1)
        layout.addWidget(self._build_plane_strip(dataset))
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
        self.error.setContentsMargins(
            themes.SPACE_M, themes.SPACE_XS, themes.SPACE_M, themes.SPACE_XS
        )
        self.error.hide()
        layout.addWidget(self.error)
        self.canvas.status_changed.connect(self.status.setText)
        self.canvas.hovered.connect(self.navigator.highlight)
        self.navigator.hovered.connect(lambda identity: self.canvas.hover(identity, from_list=True))
        self.navigator.selected.connect(self.select_peak)
        self.navigator.next_warning.connect(lambda: self.fitting.next_warning(None))
        self.panel.next_warning.connect(lambda: self.fitting.next_warning(None))
        self.panel.settings_requested.connect(lambda: self.fitting.fit(settings=True))
        self.panel.plane_requested.connect(self.set_plane)
        self.canvas.selected.connect(self.select_peak)
        self.canvas.placed.connect(self.place_peak)
        self.canvas.moved.connect(self.move_peak)
        self.canvas.draft_changed.connect(self._drag_draft)
        self.navigator.edit_requested.connect(self.focus_editor)
        self.canvas.edit_requested.connect(self.focus_editor)
        self.inspector.commit_requested.connect(self.commit_fields)
        self.inspector.deleted.connect(self.delete_peaks)
        self.inspector.fields_changed.connect(self._fields_changed)
        self._build_actions()
        self.fitting = DatasetFitting(self)
        self._refresh()
        self.navigator.setVisible(bool(dataset.peaks))
        self.set_plane(1)

    def _tool(self, name: str, tip: str, *, checkable: bool = False) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton()
        button.setCheckable(checkable)
        button.setToolTip(tip)
        button.setAccessibleName(tip.split(" (")[0])
        button.setIconSize(QtCore.QSize(18, 18))
        button.setFixedSize(28, 28)
        self._icons.append((button, name))
        return button

    def _build_toolbar(self, dataset: Dataset) -> QtWidgets.QWidget:  # noqa: PLR0915
        """One row: identity and run, comparison, canvas tools, settings, Fit."""
        self._icons: list[tuple[QtWidgets.QAbstractButton, str]] = []
        toolbar = QtWidgets.QWidget()
        toolbar.setObjectName("toolbar")
        toolbar.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        bar = QtWidgets.QHBoxLayout(toolbar)
        bar.setContentsMargins(themes.SPACE_M, 7, themes.SPACE_M, 7)
        bar.setSpacing(6)
        self.peaks_button = self._tool(
            "panel-left", "Navigator: peaks and clusters", checkable=True
        )
        self.peaks_button.setChecked(bool(dataset.peaks))
        self.peaks_button.clicked.connect(self.toggle_navigator)
        bar.addWidget(self.peaks_button)
        self.title = ElidedLabel(dataset.path.stem)
        self.title.setObjectName("title")
        self.title.setMaximumWidth(160)
        bar.addWidget(self.title)
        self.modified = QtWidgets.QLabel()
        self.modified.setObjectName("muted")
        bar.addWidget(self.modified)
        # The run line: currentness and counts first, so a narrow window elides the tail.
        self.summary = ElidedLabel()
        self.summary.setObjectName("muted")
        bar.addWidget(self.summary, 1)
        self.warnings = QtWidgets.QLabel()
        self.warnings.setObjectName("warning")
        self.warnings.hide()
        bar.addWidget(self.warnings)
        segment = QtWidgets.QHBoxLayout()
        segment.setSpacing(0)
        self.mode_buttons = []
        for index, name in enumerate(("Experimental", "Overlay", "Residual")):
            button = QtWidgets.QPushButton(name)
            button.setObjectName("segment")
            button.setProperty("edge", ("first", "", "last")[index])
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.setEnabled(index == 0)
            button.clicked.connect(lambda _checked=False, mode=index: self.set_mode(mode))
            self.mode_buttons.append(button)
            segment.addWidget(button)
        bar.addLayout(segment)
        bar.addSpacing(themes.SPACE_XS)
        self.add_button = self._tool("add-peak", "Add peak (A)", checkable=True)
        self.add_button.clicked.connect(self.start_placing)
        self.pin_button = self._tool(
            "pin", "Pin traces at the sampled position (Space)", checkable=True
        )
        self.pin_button.toggled.connect(self.canvas.set_pinned)
        self.canvas.pin_changed.connect(self._pin_changed)
        full = self._tool("full", "Full spectrum (Ctrl+0)")
        full.clicked.connect(self.canvas.full_spectrum)
        display = self._tool("sliders", "Display: contours, labels, trace scale")
        display.clicked.connect(self.display_dialog)
        for button in (self.add_button, self.pin_button, full, display):
            bar.addWidget(button)
        bar.addSpacing(themes.SPACE_XS)
        self.settings_button = QtWidgets.QPushButton()
        self.settings_button.setObjectName("chip")
        self.settings_button.setToolTip("Fit settings: lineshapes, clustering and passes")
        self.settings_button.clicked.connect(lambda: self.fitting.fit(settings=True))
        bar.addWidget(self.settings_button)
        self.cancel_fit = QtWidgets.QPushButton("Cancel")
        self.cancel_fit.clicked.connect(lambda: self.fitting.cancel())
        bar.addWidget(self.cancel_fit)
        self.fit_button = QtWidgets.QPushButton("Fit")
        self.fit_button.setObjectName("primary")
        self.fit_button.setToolTip("Fit the complete dataset (Ctrl+Return)")
        self.fit_button.clicked.connect(lambda: self.fitting.fit())
        bar.addWidget(self.fit_button)
        self.inspector_button = self._tool("panel-right", "Inspector", checkable=True)
        self.inspector_button.setChecked(True)
        self.inspector_button.clicked.connect(self.toggle_inspector)
        bar.addWidget(self.inspector_button)
        return toolbar

    def _build_plane_strip(self, dataset: Dataset) -> QtWidgets.QWidget:
        """Plain plane navigation: index, count, slider and the plane value."""
        strip = QtWidgets.QWidget()
        strip.setObjectName("planeStrip")
        strip.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        planes = QtWidgets.QHBoxLayout(strip)
        planes.setContentsMargins(themes.SPACE_M, 6, themes.SPACE_M, 6)
        planes.setSpacing(themes.SPACE_S)
        self.previous = self._tool("chevron-left", "Previous plane")
        self.next = self._tool("chevron-right", "Next plane")
        self.previous.clicked.connect(lambda: self.set_plane(self.canvas.plane))
        self.next.clicked.connect(lambda: self.set_plane(self.canvas.plane + 2))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setAccessibleName("Plane navigation")
        self.spin = QtWidgets.QSpinBox()
        self.spin.setAccessibleName("One-based plane number")
        self.spin.setKeyboardTracking(False)
        self.spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        n_planes = dataset.spectra.data.shape[0]
        self.spin.setFixedWidth(16 + self.spin.fontMetrics().horizontalAdvance(str(n_planes)) * 2)
        for control in (self.slider, self.spin):
            control.setRange(1, n_planes)
            control.setValue(1)
            control.valueChanged.connect(self.set_plane)
        caption = QtWidgets.QLabel("Plane")
        caption.setObjectName("muted")
        count = QtWidgets.QLabel(f"of {n_planes}")
        count.setObjectName("muted")
        self.plane_caption = QtWidgets.QLabel("Plane value")
        self.plane_caption.setObjectName("muted")
        self.plane_label = QtWidgets.QLabel()
        self.plane_label.setObjectName("value")
        self.plane_label.setMinimumWidth(64)
        for widget in (self.previous, self.next, caption, self.spin, count):
            planes.addWidget(widget)
        planes.addSpacing(themes.SPACE_S)
        planes.addWidget(self.slider, 1)
        planes.addSpacing(themes.SPACE_S)
        planes.addWidget(self.plane_caption)
        planes.addWidget(self.plane_label)
        if n_planes == 1:
            for widget in (self.previous, self.next, caption, self.spin, count, self.slider):
                widget.hide()
        return strip

    def apply_theme(self, theme: themes.Theme) -> None:
        """Retint icons and plots; no scientific state or geometry changes."""
        self.theme = theme
        for button, name in self._icons:
            button.setIcon(icon(name, theme))
        self.canvas.apply_theme(theme)
        self.navigator.apply_theme(theme)
        self.panel.apply_theme(theme)
        self._update_inspector()

    def update_settings_summary(self) -> None:
        """The working fit settings, always visible beside the Fit button."""
        config = self.fitting.configuration
        try:
            shapes = " × ".join(
                name.upper() for name in get_shape_names(config.fitting, self.dataset.spectra)
            )
        except (ValueError, KeyError, AttributeError):
            shapes = config.fitting.lineshape
        level = config.clustering.contour_level
        threshold, clustering = (
            (f"{level:.3g}", f"clustering threshold {level:.6g} (explicit)")
            if level is not None
            else (
                f"{config.clustering.contour_factor:g}σ",
                f"clustering threshold {config.clustering.contour_factor:g} × noise",
            )
        )
        steps = config.fitting.steps
        passes = f"{len(steps)} steps" if steps else f"{config.fitting.refine_iterations} passes"
        self._settings_text = f"{shapes} · {threshold} · {passes}"
        self._settings_rows = [
            ("Lineshapes", shapes),
            ("Clustering", clustering.removeprefix("clustering ").capitalize()),
            ("Optimizer", passes + (", cross-talk updated between passes" if not steps else "")),
        ]
        self._adapt_toolbar()
        self.settings_button.setToolTip(
            f"Fit settings: lineshapes {shapes} · {clustering} · {passes}. Click to review."
        )

    def _adapt_toolbar(self) -> None:
        """Below 1200 px the run line needs the room; tooltips keep the full text."""
        wide = self.width() >= _WIDE_TOOLBAR
        self.title.setMaximumWidth(160 if wide else 120)
        self.settings_button.setText(
            getattr(self, "_settings_text", "") if wide else "Fit settings"
        )
        count = self.warnings.property("count") or 0
        self.warnings.setText(
            f"▲ {count} uncertainty warning{'s' if count != 1 else ''}" if wide else f"▲ {count}"
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Adapt the toolbar to the window width without changing panel choices."""
        super().resizeEvent(event)
        if hasattr(self, "settings_button"):
            self._adapt_toolbar()

    @property
    def spectra(self):
        """Expose the loaded scientific source to the shared settings dialog."""
        return self.dataset.spectra

    @property
    def source_files(self):
        """Verified effective sources of the current working dataset."""
        return self.dataset.sources

    @property
    def fit_busy(self) -> bool:
        """Include input preparation and completion publication in the editing lock."""
        return hasattr(self, "fitting") and self.fitting.busy

    def set_mode(self, mode: int) -> None:
        """One visible choice controls the plane and both aligned slices."""
        result = self.fitting.result if self.fitting.current else None
        if result is None or not any(c.usable for c in result.outcome.clusters):
            mode = 0
        for index, button in enumerate(self.mode_buttons):
            button.setChecked(index == mode)
        self.canvas.set_comparison(result, mode)

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
        self.pin_button.setToolTip(
            "Traces pinned · click to follow the pointer (Space)"
            if pinned
            else "Pin traces at the sampled position (Space)"
        )

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
        for widget in (self.canvas.main, self.navigator.list):
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
            with QtCore.QSignalBlocker(self.navigator.list):
                item = self.navigator._items.get(self.selected_id or "")
                if item:
                    self.navigator.list.setCurrentItem(item)
            return
        self.selected_id = identity
        self.canvas.select(identity)
        self._sync_inspector()
        self._update_actions()

    def _sync_inspector(self) -> None:
        peak = next((p for p in self.peaks if p.id == self.selected_id), None)
        self.navigator.select(peak)
        self.panel.set_peak(peak, self.canvas.display(peak) if peak else None)
        self._update_inspector()

    def _cluster_members(self, result) -> tuple[str, ...]:
        """Working IDs that the active run fitted together with the selected peak."""
        found = find_cluster(result, self.selected_id) if result and self.selected_id else None
        if found is None:
            return ()
        ids = {i.engine_name: i.working_id for i in result.inputs.identities}
        return tuple(ids[name] for name in found[1].peak_names if name in ids)

    def _update_inspector(self) -> None:
        """Follow the accepted selection and active run; never resolves drafts."""
        fitting = getattr(self, "fitting", None)
        if fitting is None:
            return
        spectra = self.dataset.spectra
        result = fitting.result
        self.navigator.set_result(result, current=bool(fitting.current))
        self.canvas.set_cluster(self._cluster_members(result))
        self.panel.show_result(
            result,
            current=bool(fitting.current),
            busy=fitting.busy,
            plane_values=tuple(float(v) for v in spectra.z_values)
            if self.dataset.has_plane_values
            else None,
            plane=self.canvas.plane,
        )
        x, y = (
            formatting.nucleus(a.nucleus, a.label) for a in (self.canvas.x_axis, self.canvas.y_axis)
        )
        planes = spectra.data.shape[0]
        if fitting.busy:
            run = self.summary.text()
        elif result is not None:
            flagged = self.warnings.property("count") or 0
            run = (
                f"Run {result.run_id[:8]} · {'Current' if fitting.current else 'Outdated'}\n"
                f"{dataset_summary(result, warnings=False)}\n"
                f"Fitted in {result.execution_seconds:.1f} s · {result.completed_at[:19]} UTC"
                + (f"\n▲ {flagged} clusters with an uncertainty warning" if flagged else "")
            )
        else:
            run = "Not fitted yet. Fit uses every working peak across all planes."
        if fitting.failure:
            run += f"\n{fitting.attempt}: {fitting.failure}"
        self.panel.show_analysis(
            self.dataset.path.stem,
            f"{len(self.peaks)} working peaks · {planes} planes · {x} × {y}",
            getattr(self, "_settings_rows", []),
            run,
            "Select a peak or cluster to see its position, profile, fitted values and "
            "review. Save analysis keeps working state and the latest ten runs; "
            "spectra stay external.",
        )

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
        self.navigator.refresh(self.peaks, self.selected_id)
        self._sync_inspector()
        self._update_actions()
        self.fitting.update()
        self.changed.emit()

    def _update_actions(self) -> None:
        self.undo_action.setText("Undo " + self.edits.undo_label)
        self.redo_action.setText("Redo " + self.edits.redo_label)
        self.undo_action.setEnabled(self.edits.can_undo and not self.pending and not self.fit_busy)
        self.redo_action.setEnabled(self.edits.can_redo and not self.pending and not self.fit_busy)
        self.delete_action.setEnabled(self.selected_id is not None and not self.fit_busy)
        detail = (
            "Modified inputs · plane values session-only"
            if self.plane_values_modified
            else "Modified peaks"
            if self.edits.dirty or self.pending
            else ""
        )
        self.modified.setText("Modified" if detail else "")
        self.modified.setToolTip(detail)

    def history(self, redo: bool) -> None:
        """Undo exact records; monotonic scientific revision never recertifies a run."""
        if self.pending or self.fit_busy:
            return
        self.canvas.cancel_gesture()
        if redo:
            self.edits.redo()
        else:
            self.edits.undo()
        self._refresh()

    def start_placing(self) -> None:
        """Arm one precise placement after resolving any complete text group."""
        if self.fit_busy:
            return
        if not self.commit_fields():
            self.add_button.setChecked(False)
            return
        self.canvas.set_placing(True)
        self.add_button.setChecked(True)
        self.canvas.main.setFocus()
        self.status.setText("Place peak · click exact ppm position · Escape cancels")

    def place_peak(self, positions) -> None:
        """Store continuous unfolded ppm from a single click, without snapping."""
        if self.fit_busy:
            return
        if not self.commit_fields():
            return
        peak = self.edits.add(positions)
        self.cancel_gesture()
        self.selected_id = peak.id
        self._refresh()
        self.canvas.select(peak.id)

    def move_peak(self, identity, positions) -> None:
        """One drag release is one complete precision-preserving input change."""
        if self.fit_busy:
            return
        if not self.commit_fields():
            self.canvas.cancel_gesture()
            return
        peak = next(p for p in self.peaks if p.id == identity)
        self.edits.update(identity, peak.name, positions)
        self._refresh()
        self.canvas.select(identity)

    def delete_peaks(self) -> None:
        """Deletion owns the selected draft; multi-delete is list-only."""
        if self.fit_busy:
            return
        identities = {self.selected_id} if self.selected_id else set()
        if self.navigator.list.hasFocus():
            identities = {item.data(256) for item in self.navigator.list.selectedItems()}
        if identities:
            self.cancel_gesture()
            self.edits.delete_many(identities)
            self.selected_id = None
            self._refresh()

    def cancel_gesture(self) -> None:
        """Escape on the canvas cancels its current gesture, never numerical work."""
        self.canvas.cancel_gesture()
        self.add_button.setChecked(False)

    def toggle_navigator(self) -> None:
        """Hide or show the navigator; panel choices never change on resize."""
        self.navigator.setVisible(self.peaks_button.isChecked())

    def toggle_inspector(self) -> None:
        """Resolve a draft on explicit hide, since the editor lives in the inspector."""
        visible = self.inspector_button.isChecked()
        if not visible and not self.commit_fields():
            self.inspector_button.setChecked(True)
            return
        self.panel.setVisible(visible)

    def focus_editor(self) -> None:
        """Explicit editing opens the inspector without changing selection."""
        self.inspector_button.setChecked(True)
        self.toggle_inspector()
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
        values = self.dataset.has_plane_values and count > 1
        self.plane_caption.setVisible(values)
        self.plane_label.setText(
            "Single plane"
            if count == 1
            else formatting.intensity(float(self.dataset.spectra.z_values[plane - 1]))
            if values
            else ""
        )
        self.panel.set_plane(plane - 1)

    def display_dialog(self) -> None:
        """Display choices do not change scientific inputs or edit history."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Spectrum display")
        form = QtWidgets.QFormLayout(dialog)
        threshold = QtWidgets.QDoubleSpinBox()
        threshold.setRange(0.01, 1e12)
        threshold.setDecimals(4)
        residual = self.canvas.comparison_mode == _RESIDUAL
        threshold.setValue(
            self.canvas.residual_threshold if residual else self.canvas.contour_threshold
        )
        threshold.setKeyboardTracking(False)
        factor = QtWidgets.QDoubleSpinBox()
        factor.setRange(1.1, 5)
        factor.setValue(self.canvas.contour_factor)
        count = QtWidgets.QSpinBox()
        count.setRange(1, 20)
        count.setValue(self.canvas.contour_count)
        factor.setEnabled(not residual)
        count.setEnabled(not residual)
        labels = QtWidgets.QCheckBox("Show peak labels")
        labels.setChecked(self.canvas.labels_visible)
        heavy = QtWidgets.QCheckBox("Heavier contours")
        heavy.setChecked(self.canvas.heavy_strokes)
        fit = QtWidgets.QPushButton("Fit current traces")
        fit.clicked.connect(self.canvas.fit_current_traces)
        for title, field in (
            ("First residual contour / σ_stat" if residual else "First contour / a.u.", threshold),
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
            if residual:
                self.canvas.residual_threshold = threshold.value()
            else:
                self.canvas.contour_threshold = threshold.value()
            self.canvas.contour_factor = factor.value()
            self.canvas.contour_count = count.value()
            self.canvas.labels_visible = labels.isChecked()
            self.canvas.heavy_strokes = heavy.isChecked()
            self.canvas._refresh_labels()
            self.set_mode(self.canvas.comparison_mode)

    def stop(self) -> None:
        """Stop display computation without blocking window close."""
        self.fitting.stop()
        self.canvas.stop()
