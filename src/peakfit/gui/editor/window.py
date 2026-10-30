"""Native application shell and validated, asynchronous input replacement."""

import argparse
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui.background import ViewerComputation
from peakfit.gui.editor.loading import Dataset, read_assignments, read_dataset
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.peaks import export_peaks
from peakfit.gui.spectrum import validate_plane_values, verified_source_files

_DARK_LIGHTNESS = 0.5


class OpenDatasetDialog(QtWidgets.QDialog):
    """Choose optional companions and explicitly confirm the two peak axes."""

    def __init__(self, parent=None, *, assignments_only: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import peaks" if assignments_only else "Open dataset")
        self.resize(620, 240)
        form = QtWidgets.QFormLayout(self)
        form.setSpacing(12)
        self.paths = {}
        roles = ("Peak list",) if assignments_only else ("Spectrum", "Peak list", "Plane values")
        for role in roles:
            row = QtWidgets.QHBoxLayout()
            field = QtWidgets.QLineEdit()
            field.setAccessibleName(role + " file")
            if role != "Spectrum":
                field.setPlaceholderText(
                    "Optional" if not assignments_only else "Choose assignments"
                )
            self.paths[role] = field
            button = QtWidgets.QPushButton("Browse…")
            button.clicked.connect(partial(self._browse, role))
            row.addWidget(field, 1)
            row.addWidget(button)
            label = QtWidgets.QLabel(role)
            label.setBuddy(field)
            form.addRow(label, row)
        self.mapping = QtWidgets.QComboBox()
        self.mapping.setAccessibleName("Peak column to spectral axis mapping")
        self.mapping.addItems(
            [
                "Choose peak-axis mapping…",
                "First coordinate → vertical / indirect; second → horizontal / direct",
                "First coordinate → horizontal / direct; second → vertical / indirect",
            ]
        )
        form.addRow("Peak axes", self.mapping)
        self.message = QtWidgets.QLabel(
            "NMRPipe 2D / pseudo-3D · existing PeakFit peak formats\n"
            "Confirm axis order from your peak-list metadata. Values retain acquisition order."
        )
        self.message.setWordWrap(True)
        form.addRow(self.message)
        self.error = QtWidgets.QLabel()
        self.error.setObjectName("error")
        form.addRow(self.error)
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Open
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self._validate)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)

    def _browse(self, role) -> None:
        filters = {
            "Spectrum": "NMRPipe (*.ft2 *.ft3 *.ft *.dat);;All files (*)",
            "Peak list": "Peak lists (*.list *.csv *.json *.xlsx *.xls);;All files (*)",
            "Plane values": "Text values (*.txt *.dat);;All files (*)",
        }
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, role, "", filters[role])
        if filename:
            self.paths[role].setText(filename)

    def _validate(self) -> None:
        required = "Spectrum" if "Spectrum" in self.paths else "Peak list"
        if not self.paths[required].text().strip():
            self.error.setText(f"Choose a {required.lower()} file.")
        elif self.paths["Peak list"].text().strip() and self.mapping.currentIndex() == 0:
            self.error.setText("Choose the peak-axis mapping before loading. No axes are guessed.")
        else:
            self.accept()

    def path(self, role: str) -> Path | None:
        """Return a chosen optional path without inventing missing companions."""
        text = self.paths[role].text().strip() if role in self.paths else ""
        return Path(text).expanduser() if text else None


class EditorWindow(QtWidgets.QMainWindow):
    """Keep the existing application working while v2 gains complete vertical slices."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PeakFit")
        self.resize(1280, 800)
        self.setMinimumSize(1040, 720)
        self.workspace: EditorWorkspace | None = None
        self._revision = 0
        self._close_timer = QtCore.QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.timeout.connect(self.close)
        self._busy = False
        self._job_kind = "dataset"
        self._append = False
        self._export_path: Path | None = None
        self._appearance = "System"
        self._system_palette = QtGui.QPalette(QtWidgets.QApplication.palette())
        self._worker = ViewerComputation(self)
        self._worker.completed.connect(self._loaded)
        self._build_welcome()
        self._build_menus()
        self.loading_status = QtWidgets.QLabel()
        self.loading_status.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.cancel_load = QtWidgets.QPushButton("Cancel loading")
        self.cancel_load.clicked.connect(self.cancel_loading)
        self.statusBar().addWidget(self.loading_status, 1)
        self.statusBar().addPermanentWidget(self.cancel_load)
        self.cancel_load.hide()
        self.apply_appearance("System")
        QtGui.QGuiApplication.styleHints().colorSchemeChanged.connect(self._system_changed)

    def _build_welcome(self) -> None:
        welcome = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(welcome)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title = QtWidgets.QLabel("PeakFit")
        title.setObjectName("welcome")
        layout.addWidget(title)
        subtitle = QtWidgets.QLabel(
            "Open a spectrum. Inspect native samples. Edit exact peak positions."
        )
        subtitle.setObjectName("muted")
        layout.addWidget(subtitle)
        button = QtWidgets.QPushButton("Open dataset…")
        button.setObjectName("primary")
        button.clicked.connect(self.open_dialog)
        layout.addWidget(button)
        note = QtWidgets.QLabel(
            "Fit the complete dataset · Compare native model and residual · Export results"
        )
        note.setObjectName("muted")
        layout.addWidget(note)
        self.setCentralWidget(welcome)

    def _action(self, menu, title, callback, key=None):
        action = menu.addAction(title)
        action.triggered.connect(callback)
        if key is not None:
            action.setShortcuts(key) if isinstance(
                key, QtGui.QKeySequence.StandardKey
            ) else action.setShortcut(key)
        return action

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self.open_action = self._action(
            file_menu, "&Open dataset…", self.open_dialog, QtGui.QKeySequence.StandardKey.Open
        )
        self.import_action = self._action(file_menu, "Import &peaks…", self.import_dialog)
        self.values_action = self._action(file_menu, "Import plane &values…", self.values_dialog)
        self.export_action = self._action(
            file_menu, "&Export working peaks…", self.export_dialog, "Ctrl+Shift+E"
        )
        file_menu.addSeparator()
        self._action(file_menu, "&Close", self.close, QtGui.QKeySequence.StandardKey.Close)
        self.fit_export_action = self._action(
            file_menu, "Export fit results…", self.export_fit_dialog
        )
        self.fit_export_action.setEnabled(False)
        self.edit_menu = self.menuBar().addMenu("&Edit")
        view = self.menuBar().addMenu("&View")
        self._action(
            view,
            "Full spectrum",
            lambda: self.workspace and self.workspace.canvas.full_spectrum(),
            "Ctrl+0",
        )
        self._action(
            view, "Find assignment", self.find_assignment, QtGui.QKeySequence.StandardKey.Find
        )
        self._action(
            view,
            "Previous plane",
            lambda: self._step_plane(-1),
            "Meta+PgUp" if sys.platform == "darwin" else "Ctrl+PgUp",
        )
        self._action(
            view,
            "Next plane",
            lambda: self._step_plane(1),
            "Meta+PgDown" if sys.platform == "darwin" else "Ctrl+PgDown",
        )
        self._action(view, "Go to sampled coordinates…", self.sample_dialog)
        self._action(view, "Spectrum ranges…", self.ranges_dialog)
        appearance = view.addMenu("Appearance")
        self.appearance_group = QtGui.QActionGroup(self)
        for name in ("System", "Light", "Dark"):
            action = self._action(appearance, name, partial(self.apply_appearance, name))
            action.setCheckable(True)
            self.appearance_group.addAction(action)
            action.setChecked(name == "System")
        self.analysis_menu = self.menuBar().addMenu("&Analysis")
        self.fit_action = self._action(
            self.analysis_menu, "Fit dataset", self.fit_dataset, "Ctrl+Return"
        )
        self.settings_action = self._action(self.analysis_menu, "Fit settings…", self.fit_settings)
        self.fit_action.setEnabled(False)
        self.settings_action.setEnabled(False)
        help_menu = self.menuBar().addMenu("&Help")
        self._action(help_menu, "Keyboard &shortcuts", self.shortcuts)
        for action in (self.import_action, self.values_action, self.export_action):
            action.setEnabled(False)

    def fit_dataset(self) -> None:
        """Run all working peaks and original planes in the current workspace."""
        if self.workspace and not self._busy:
            self.workspace.fitting.fit()

    def fit_settings(self) -> None:
        """Explicitly review settings; Cancel preserves accepted comparison."""
        if self.workspace and not self._busy:
            self.workspace.fitting.fit(settings=True)

    def export_fit_dialog(self) -> bool:
        """Export the retained scientific run using the canonical writers."""
        return bool(self.workspace and self.workspace.fitting.export())

    def open_dialog(self) -> None:
        """Native file selection plus optional companions and explicit mapping."""
        dialog = OpenDatasetDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            path = dialog.path("Spectrum")
            assert path is not None
            self.load(
                path,
                dialog.path("Peak list"),
                dialog.path("Plane values"),
                swap=dialog.mapping.currentText().startswith("First coordinate → horizontal"),
            )

    def load(
        self,
        path: Path,
        peaks: Path | None = None,
        values: Path | None = None,
        *,
        swap: bool = False,
    ) -> None:
        """Start cancellable off-thread validation; preserve the current workspace."""
        self._job_kind = "dataset"
        self._start_loading(partial(read_dataset, path, peaks, values, swap_peak_axes=swap))

    def _start_loading(self, operation) -> None:
        if self.workspace and self.workspace.fit_busy:
            self.loading_status.setText("Cancel the fit before replacing scientific inputs")
            return
        if self.workspace:
            self.workspace.inputs_loading = True
        self._revision += 1
        self._busy = True
        self.loading_status.setText(
            "Loading and validating inputs… · Existing workspace remains available"
        )
        self.cancel_load.show()
        for action in (self.open_action, self.import_action, self.values_action):
            action.setEnabled(False)
        self._worker.submit("load", self._revision, operation)

    def cancel_loading(self) -> None:
        """Cancel publication immediately; an in-progress file read may finish privately."""
        self._revision += 1
        self._worker.cancel("load")
        self._end_loading()
        self.loading_status.setText("Loading canceled · Previous dataset retained")

    def _end_loading(self) -> None:
        if self.workspace:
            self.workspace.inputs_loading = False
        self._busy = False
        self.cancel_load.hide()
        self.open_action.setEnabled(True)
        for action in (self.import_action, self.values_action, self.export_action):
            action.setEnabled(self.workspace is not None)

    def _loaded(self, _kind, revision, outcome) -> None:
        if revision != self._revision:
            return
        self._end_loading()
        if isinstance(outcome, Exception):
            self.loading_status.setText(f"Load failed: {outcome} · Current dataset retained")
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot load inputs",
                f"{outcome}\n\nCheck the file, format, axis mapping and plane-value count. "
                "The current dataset was retained.",
            )
            return
        if self._job_kind == "peaks":
            self._accept_peaks(outcome)
            return
        if self._job_kind == "values":
            self._accept_values(outcome)
            return
        assert isinstance(outcome, Dataset)
        if not self.confirm_dataset(outcome) or not self.protect_edits():
            self.loading_status.setText("Dataset replacement canceled")
            return
        self.install_dataset(outcome)

    def confirm_dataset(self, dataset: Dataset) -> bool:
        """Review real dimensions/axis metadata/assignments before replacement."""
        axes = "\n".join(
            f"{axis.nucleus or axis.label} ({axis.label}): {axis.size} points, "
            f"{axis.pts2ppm(0):.8f} → {axis.pts2ppm(axis.size - 1):.8f} ppm"
            for axis in dataset.spectra.spectral_params
        )
        names = (
            ", ".join(p.name for p in dataset.peaks[:6])
            or "No peak list; inspect or add peaks manually"
        )
        return (
            QtWidgets.QMessageBox.question(
                self,
                "Open validated dataset",
                f"{dataset.path.name}\n"
                f"{dataset.spectra.data.shape[0]} planes in acquisition order\n{axes}\n\n"
                f"{len(dataset.peaks)} peaks · {names}\n\n"
                + (
                    "Plane values: file order and file units"
                    if dataset.has_plane_values
                    else "No plane values"
                ),
                QtWidgets.QMessageBox.StandardButton.Open
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Open,
            )
            == QtWidgets.QMessageBox.StandardButton.Open
        )

    def install_dataset(self, dataset: Dataset) -> None:
        """Publish one complete handoff; no half-loaded arrays or assignments."""
        previous = self.workspace
        workspace = EditorWorkspace(dataset)
        if previous:
            previous.stop()
        self.setCentralWidget(workspace)
        self.workspace = workspace
        self._export_path = None
        self.edit_menu.clear()
        self.edit_menu.addActions(workspace.actions_for_menu)
        workspace.changed.connect(self._work_changed)
        workspace.fitting.state_changed.connect(self._work_changed)
        self.setWindowTitle(f"{dataset.path.name}[*] — PeakFit")
        self.apply_appearance(self._appearance)
        self._end_loading()
        self._work_changed()
        self.loading_status.setText(
            "Click to pin · Select then drag to move · Space resumes preview"
        )
        duplicates = len({p.name for p in dataset.peaks}) != len(dataset.peaks)
        if duplicates:
            workspace.error.setText(
                "Duplicate assignments retained as separate peak IDs; "
                "coordinates disambiguate them."
            )
            workspace.error.show()

    def _work_changed(self) -> None:
        if self.workspace:
            w = self.workspace
            self.setWindowModified(
                w.has_changes
                or (
                    w.fitting.result is not None
                    and w.fitting.result.run_id not in w.fitting.exported
                )
            )
            for action in (self.open_action, self.import_action, self.values_action):
                action.setEnabled(not w.fit_busy and not self._busy)
            self.fit_action.setEnabled(bool(w.peaks) and not w.fit_busy)
            self.settings_action.setEnabled(bool(w.peaks) and not w.fit_busy)
            self.fit_export_action.setEnabled(w.fitting.result is not None and not w.fit_busy)

    def import_dialog(self) -> None:
        """Import uses the same readers and requires an explicit axis mapping."""
        if not self.workspace:
            return
        dialog = OpenDatasetDialog(self, assignments_only=True)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        path = dialog.path("Peak list")
        assert path is not None
        spectra = self.workspace.dataset.spectra
        swap = dialog.mapping.currentText().startswith("First coordinate → horizontal")

        def read():
            with verified_source_files({"peak_list": path}) as records:
                peaks = read_assignments(path, spectra, swap=swap)
            return peaks, records

        self._job_kind = "peaks"
        self._start_loading(read)

    def _accept_peaks(self, outcome) -> None:
        workspace = self.workspace
        if workspace is None or not workspace.commit_fields():
            return
        peaks, records = outcome
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("Import validated peaks")
        dialog.setText(
            f"{len(peaks)} peaks read. Replace {len(workspace.peaks)} working peaks "
            "or append with new identities?"
        )
        dialog.setInformativeText(", ".join(p.name for p in peaks[:6]))
        replace_button = dialog.addButton("Replace", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        append_button = dialog.addButton("Append", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        dialog.exec()
        chosen = dialog.clickedButton()
        if chosen not in (replace_button, append_button):
            return
        workspace.import_peaks(peaks, records, append=chosen is append_button)
        self.loading_status.setText(
            f"Imported {len(peaks)} peaks · Undo restores the previous list"
        )

    def values_dialog(self) -> None:
        """Read all values off Qt, preserving duplicate/unordered original indices."""
        if not self.workspace:
            return
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import plane values")
        if not filename:
            return
        count = self.workspace.dataset.spectra.data.shape[0]

        def read():
            with verified_source_files({"plane_values": Path(filename)}) as records:
                values = np.atleast_1d(np.loadtxt(filename, dtype=float))
                validate_plane_values(values, count)
                values.flags.writeable = False
            return values, records

        self._job_kind = "values"
        self._start_loading(read)

    def _accept_values(self, outcome) -> None:
        workspace = self.workspace
        if workspace is None:
            return
        values, records = outcome
        spectra = workspace.dataset.spectra.model_copy(update={"z_values": values})
        dataset = replace(
            workspace.dataset,
            spectra=spectra,
            sources=tuple(
                (role, record)
                for role, record in workspace.dataset.sources
                if role != "plane_values"
            )
            + records,
            has_plane_values=True,
        )
        workspace.dataset = dataset
        workspace.canvas.dataset = dataset
        workspace.canvas.spectra = spectra
        # A plane mapping replacement is a scientific edit, never navigation.
        workspace.edits.revision += 1
        workspace.plane_values_modified = True
        workspace._update_actions()
        workspace.changed.emit()
        workspace.set_plane(workspace.canvas.plane + 1)
        self.loading_status.setText(
            "Plane values imported in original acquisition order · file units"
        )

    def export_dialog(self) -> bool:
        """Atomically export full precision working inputs through the existing writer."""
        workspace = self.workspace
        if not workspace or not workspace.commit_fields():
            return False
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export working peaks",
            str(self._export_path or workspace.dataset.path.with_suffix(".csv")),
            "PeakFit CSV (*.csv)",
        )
        if not filename:
            return False
        path = Path(filename)
        try:
            export_peaks(path, workspace.peaks)
        except (OSError, UnicodeError) as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot export peaks", str(exc))
            return False
        self._export_path = path
        workspace.edits.mark_saved()
        workspace._update_actions()
        self._work_changed()
        self.loading_status.setText(
            f"Exported {len(workspace.peaks)} working peaks to {path}; "
            + (
                "plane values remain session-only and modified"
                if workspace.plane_values_modified
                else "analysis persistence is not available yet"
            )
        )
        return True

    def protect_edits(self) -> bool:
        """Require an explicit loss decision for every unpreserved scientific input."""
        workspace = self.workspace
        if workspace is not None and workspace.fitting.result is not None:
            fitting = workspace.fitting
            assert fitting.result is not None
            if fitting.result.run_id not in fitting.exported:
                decision = QtWidgets.QMessageBox.question(
                    self,
                    "Unexported fit result",
                    "Export the completed scientific result before closing "
                    "or replacing this dataset?",
                    QtWidgets.QMessageBox.StandardButton.Save
                    | QtWidgets.QMessageBox.StandardButton.Discard
                    | QtWidgets.QMessageBox.StandardButton.Cancel,
                    QtWidgets.QMessageBox.StandardButton.Cancel,
                )
                if decision == QtWidgets.QMessageBox.StandardButton.Cancel:
                    return False
                if decision == QtWidgets.QMessageBox.StandardButton.Save and not fitting.export():
                    return False
        if workspace is None or not workspace.has_changes:
            return True
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("Unpreserved session changes")
        dialog.setText("Close or replace this dataset and discard session changes?")
        dialog.setInformativeText(
            "Imported/replaced plane values are session-only. Peak CSV export does not "
            "preserve plane metadata or save an analysis; exporting peaks keeps this session "
            "modified. Keep the values file and select it again when reopening."
            if workspace.plane_values_modified
            else "Export working peaks to preserve names and exact unfolded coordinates. "
            "Analysis persistence is a later milestone."
        )
        export_button = None
        if workspace.edits.dirty or workspace.pending:
            export_button = dialog.addButton(
                "Export peaks…", QtWidgets.QMessageBox.ButtonRole.AcceptRole
            )
        discard_button = dialog.addButton(
            "Discard session changes", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
        )
        dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        dialog.exec()
        if export_button is not None and dialog.clickedButton() is export_button:
            exported = self.export_dialog()
            return exported and not workspace.plane_values_modified
        return dialog.clickedButton() is discard_button

    def find_assignment(self) -> None:
        """Search explicitly opens the rail; selection alone does not."""
        if self.workspace:
            self.workspace.peaks_button.setChecked(True)
            self.workspace.toggle_rail()
            self.workspace.rail.search.setFocus()

    def _step_plane(self, offset) -> None:
        if self.workspace:
            self.workspace.set_plane(self.workspace.canvas.plane + 1 + offset)

    def sample_dialog(self) -> None:
        """Keyboard access to a sampled coordinate without moving a working peak."""
        if not self.workspace:
            return
        canvas = self.workspace.canvas
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Go to sampled coordinates")
        form = QtWidgets.QFormLayout(dialog)
        fields = []
        for title, value in (
            (canvas.axis_title(canvas.x_axis), canvas.ppm_x[canvas.ix]),
            (canvas.axis_title(canvas.y_axis), canvas.ppm_y[canvas.iy]),
        ):
            field = QtWidgets.QLineEdit(repr(float(value)))
            form.addRow(title + " / ppm", field)
            fields.append(field)
        error = QtWidgets.QLabel()
        form.addRow(error)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )

        def accept():
            try:
                coordinates = tuple(float(f.text()) for f in fields)
                if not all(np.isfinite(v) for v in coordinates):
                    raise ValueError
            except ValueError:
                error.setText("Enter two finite ppm numbers.")
                return
            canvas.sample(*coordinates)
            canvas.set_pinned(True)
            dialog.accept()

        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        dialog.exec()

    def ranges_dialog(self) -> None:
        """Expose viewport ranges for keyboard-only zoom and pan."""
        if not self.workspace:
            return
        view = self.workspace.canvas.main.getViewBox()
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Spectrum ranges")
        form = QtWidgets.QFormLayout(dialog)
        fields = []
        for axis, values in zip(
            (self.workspace.canvas.x_axis, self.workspace.canvas.y_axis),
            view.viewRange(),
            strict=True,
        ):
            for bound, value in zip(("minimum", "maximum"), values, strict=True):
                field = QtWidgets.QLineEdit(repr(float(value)))
                form.addRow(f"{axis.nucleus or axis.label} {bound} / ppm", field)
                fields.append(field)
        error = QtWidgets.QLabel()
        form.addRow(error)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )

        def accept():
            try:
                values = [float(field.text()) for field in fields]
                if (
                    not all(np.isfinite(v) for v in values)
                    or values[0] >= values[1]
                    or values[2] >= values[3]
                ):
                    raise ValueError
            except ValueError:
                error.setText("Enter finite minimum < maximum for each axis.")
                return
            view.setRange(xRange=values[:2], yRange=values[2:], padding=0)
            dialog.accept()

        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        dialog.exec()

    def shortcuts(self) -> None:
        """Keep essential gestures discoverable through a normal Help command."""
        QtWidgets.QMessageBox.information(
            self,
            "Keyboard shortcuts",
            "Open: Cmd/Ctrl+O · Search: Cmd/Ctrl+F\nUndo: Cmd/Ctrl+Z · Redo: Cmd/Ctrl+Shift+Z\n"
            "Full spectrum: Cmd/Ctrl+0 · Plane: Ctrl+Page Up/Down\n\n"
            "Canvas: Space pin/unpin · A add peak · arrows step native sample\n"
            "Canvas/list: Delete selected peak(s) · Enter/double-click edits\n"
            "Fields: Enter commits whole group · Escape discards · Tab retains draft\n\n"
            "Select then drag a marker to move. Drag blank space to pan; wheel zooms.\n"
            "View provides exact sample and frequency-range fields. File exports working peaks.",
        )

    def _system_changed(self, _scheme) -> None:
        if self._appearance == "System":
            self.apply_appearance("System")

    def apply_appearance(self, name: str) -> None:
        """Use specified palettes without touching scientific state or contour geometry."""
        self._appearance = name
        scheme = QtGui.QGuiApplication.styleHints().colorScheme()
        system_dark = (
            scheme == QtCore.Qt.ColorScheme.Dark
            if scheme != QtCore.Qt.ColorScheme.Unknown
            else self._system_palette.color(QtGui.QPalette.ColorRole.Window).lightnessF()
            < _DARK_LIGHTNESS
        )
        dark = name == "Dark" or (name == "System" and system_dark)
        bg, surface, fg, muted, border, accent, wash = (
            ("#171d24", "#202833", "#e1e7ee", "#aab7c4", "#4b5b6c", "#73b5d5", "#304959")
            if dark
            else ("#fbfcfd", "#f0f3f6", "#25313d", "#586875", "#bac5d0", "#23688a", "#dceaf2")
        )
        palette = QtGui.QPalette()
        for role, value in {
            QtGui.QPalette.ColorRole.Window: bg,
            QtGui.QPalette.ColorRole.Base: bg,
            QtGui.QPalette.ColorRole.WindowText: fg,
            QtGui.QPalette.ColorRole.Text: fg,
            QtGui.QPalette.ColorRole.Button: surface,
            QtGui.QPalette.ColorRole.ButtonText: fg,
            QtGui.QPalette.ColorRole.AlternateBase: surface,
            QtGui.QPalette.ColorRole.Highlight: wash,
            QtGui.QPalette.ColorRole.HighlightedText: fg,
            QtGui.QPalette.ColorRole.PlaceholderText: muted,
            QtGui.QPalette.ColorRole.ToolTipBase: surface,
            QtGui.QPalette.ColorRole.ToolTipText: fg,
            QtGui.QPalette.ColorRole.Link: accent,
            QtGui.QPalette.ColorRole.LinkVisited: accent,
        }.items():
            palette.setColor(role, QtGui.QColor(value))
        for role in (
            QtGui.QPalette.ColorRole.Text,
            QtGui.QPalette.ColorRole.ButtonText,
            QtGui.QPalette.ColorRole.WindowText,
        ):
            palette.setColor(QtGui.QPalette.ColorGroup.Disabled, role, QtGui.QColor(muted))
        self.setStyleSheet(f"""
            QWidget {{font-size:13px;}}
            QWidget#rail {{background:{surface};border-radius:4px;}}
            QLabel#title {{font-size:18px;font-weight:600;}}
            QLabel#welcome {{font-size:30px;font-weight:600;}}
            QLabel#section {{font-weight:600;}}
            QLabel#muted {{color:{muted};font-size:12px;}}
            QLabel#error {{color:{"#ffa09c" if dark else "#9a2924"};}}
            QLabel#warning {{color:{"#e2ad65" if dark else "#875619"};font-size:12px;}}
            QPushButton {{padding:6px 10px;border:1px solid {border};
                border-radius:4px;background:{bg};}}
            QPushButton:hover:enabled {{background:{surface};border-color:{accent};}}
            QPushButton:checked {{background:{wash};border-color:{accent};}}
            QPushButton:disabled {{color:{muted};}}
            QPushButton#primary {{background:{accent};
                color:{"#171d24" if dark else "#ffffff"};font-weight:600;}}
            QPushButton#mode {{border-radius:0;}}
            QLineEdit, QAbstractSpinBox, QComboBox {{padding:5px 6px;
                border:1px solid {muted};border-radius:4px;background:{bg};}}
            QLineEdit:focus, QAbstractSpinBox:focus, QPushButton:focus,
            QListWidget:focus, QComboBox:focus {{border:2px solid {accent};}}
            QListWidget {{border:0;background:{surface};outline:0;}}
            QListWidget::item {{padding:6px 2px;}}
            QListWidget::item:selected {{background:{wash};color:{fg};}}
            QHeaderView::section {{background:{surface};color:{muted};
                border:0;padding:4px 2px;font-size:11px;}}
            QGroupBox {{border:0;border-top:1px solid {border};
                margin-top:12px;padding-top:16px;font-weight:600;}}
            QGroupBox::title {{subcontrol-origin:margin;left:0;}}
            QScrollArea {{border:0;background:{surface};}}
            QSplitter::handle {{background:{bg};}}
        """)
        self.setPalette(palette)
        for widget in self.findChildren(QtWidgets.QWidget):
            widget.setPalette(palette)
        if self.workspace:
            self.workspace.canvas.apply_colors(dark)
            self.workspace.rail._wash = QtGui.QColor(wash)
        if hasattr(self, "appearance_group"):
            for action in self.appearance_group.actions():
                action.setChecked(action.text() == name)

    def changeEvent(self, event) -> None:  # noqa: N802
        """Deactivate cancels a drag exactly, preserving typed drafts."""
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.WindowDeactivate and self.workspace:
            self.workspace.cancel_gesture()

    def closeEvent(self, event) -> None:  # noqa: N802
        """Protect the usable input editor, then cancel private display deliveries."""
        if self.workspace and self.workspace.fit_busy:
            if self.workspace.fitting.process.state != "canceling":
                decision = QtWidgets.QMessageBox.question(
                    self,
                    "Fit running",
                    "Stop the running fit and close PeakFit?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.Cancel,
                    QtWidgets.QMessageBox.StandardButton.Cancel,
                )
                if decision != QtWidgets.QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
                self.workspace.fitting.cancel()
            # Reaping and any preceding publication finish outside Qt. Only then
            # resolve unexported prior results and peak inputs in the usual guard.
            if self.workspace.fit_busy:
                self._close_timer.start(50)
                event.ignore()
                return
        if not self.protect_edits():
            event.ignore()
            return
        self._close_timer.stop()
        self._revision += 1
        self._worker.close()
        if self.workspace:
            self.workspace.stop()
        event.accept()


def main() -> None:
    """Launch the v2 editor separately; peakfit gui keeps the fitting application."""
    parser = argparse.ArgumentParser(
        description="PeakFit v2: inspect, edit, fit, compare and export NMR spectra"
    )
    parser.add_argument("spectrum", nargs="?", type=Path)
    parser.add_argument("--peaks", type=Path)
    parser.add_argument("--plane-values", type=Path)
    parser.add_argument("--peak-axis-order", choices=("indirect-direct", "direct-indirect"))
    args = parser.parse_args()
    if args.peaks and not args.peak_axis_order:
        parser.error("--peaks requires --peak-axis-order; confirm your list's coordinate mapping")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("PeakFit")
    app.setOrganizationName("PeakFit")
    assert isinstance(app, QtWidgets.QApplication)
    app.setStyle("Fusion")
    window = EditorWindow()
    window.show()
    if args.spectrum:
        window.load(
            args.spectrum,
            args.peaks,
            args.plane_values,
            swap=args.peak_axis_order == "direct-indirect",
        )
    app.exec()
