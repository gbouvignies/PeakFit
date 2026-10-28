"""Desktop window and file-opening actions for PeakFit."""

import sys
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui.peaks import export_peaks
from peakfit.gui.spectrum import load_peaks, load_spectrum, verified_source_files
from peakfit.gui.viewer import SpectrumViewer


class PeakFitWindow(QtWidgets.QMainWindow):
    """Application shell; scientific operations remain outside presentation code."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PeakFit")
        self.resize(1360, 900)
        self.viewer: SpectrumViewer | None = None
        self._export_path: Path | None = None
        self._spectrum_name = "PeakFit"
        open_key = QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Open).toString(
            QtGui.QKeySequence.SequenceFormat.NativeText
        )
        welcome = QtWidgets.QLabel(
            f"Open an NMRPipe spectrum to begin.\nFile → Open spectrum ({open_key})"
        )
        welcome.setMargin(40)
        self.setCentralWidget(welcome)
        menu = self.menuBar().addMenu("&File")
        self.open_action = self._action(
            menu, "&Open spectrum…", QtGui.QKeySequence.StandardKey.Open, self._open_spectrum
        )
        self.peaks_action = self._action(menu, "Load &peak list…", "", self._open_peaks)
        self.values_action = self._action(menu, "Load plane &values…", "", self._open_values)
        self.export_action = self._action(
            menu, "Export peaks…", QtGui.QKeySequence.StandardKey.Save, self._export_peaks
        )
        self.fit_export_action = self._action(
            menu, "Export fit results…", "", self._export_fit_results
        )
        self.fit_export_action.setEnabled(False)
        self.export_action.setEnabled(False)
        self.peaks_action.setEnabled(False)
        self.values_action.setEnabled(False)
        menu.addSeparator()
        self._action(menu, "&Quit", QtGui.QKeySequence.StandardKey.Quit, self.close)
        self.edit_menu = self.menuBar().addMenu("&Edit")
        view = self.menuBar().addMenu("&View")
        self.view_menu = view
        self._action(view, "Reset view", "Ctrl+0", self._reset_view)
        self._action(view, "Previous plane", "Ctrl+PgUp", lambda: self._step_plane(-1))
        self._action(view, "Next plane", "Ctrl+PgDown", lambda: self._step_plane(1))
        self.labels_action = view.addAction("Peak labels")
        self.labels_action.setCheckable(True)
        self.labels_action.toggled.connect(self._show_peak_labels)
        self.statusBar().showMessage("Open a spectrum to inspect and edit peak coordinates")

    def _action(
        self, menu, title: str, shortcut: str | QtGui.QKeySequence.StandardKey, callback
    ) -> QtGui.QAction:
        action = menu.addAction(title)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(callback)
        return action

    def open_spectrum(
        self, path: Path, peaks: Path | None = None, plane_values: Path | None = None
    ) -> None:
        """Prepare a replacement before discarding the currently displayed spectrum."""
        if self.viewer is not None and self.viewer.fit_busy:
            raise ValueError("Cancel the running fit before replacing the spectrum.")
        sources = {"spectrum": path}
        if peaks is not None:
            sources["peak_list"] = peaks
        if plane_values is not None:
            sources["plane_values"] = plane_values
        with verified_source_files(sources) as source_files:
            spectra = load_spectrum(path, plane_values)
            imported_peaks = load_peaks(peaks, spectra) if peaks is not None else []
        if not self._protect_edits() or (
            self.viewer is not None and not self.viewer.confirm_result_loss("discard")
        ):
            return
        viewer = SpectrumViewer(spectra)
        viewer.source_files = source_files
        viewer.set_peaks(imported_peaks)
        viewer.show_peak_labels(self.labels_action.isChecked())
        if self.viewer is not None:
            self.viewer.stop_preparation()
            self.view_menu.removeAction(self.viewer.workspace.slices_action)
        self.setCentralWidget(viewer)
        self.viewer = viewer
        self.view_menu.addAction(viewer.workspace.slices_action)
        self._spectrum_name = path.name
        self._export_path = None
        viewer.edits_changed.connect(self._update_work_state)
        viewer.pending_edits_changed.connect(self._update_work_state)
        viewer.fit_state_changed.connect(self._update_fit_state)
        self.edit_menu.clear()
        self.edit_menu.addActions(viewer.edit_actions)
        self.peaks_action.setEnabled(True)
        self.values_action.setEnabled(True)
        self.export_action.setEnabled(True)
        self.setWindowTitle(f"{path.name}[*] — PeakFit")
        self._update_work_state()
        self.fit_export_action.setEnabled(False)
        self.statusBar().showMessage(
            f"{spectra.data.shape[0]} planes · {len(imported_peaks)} peaks · "
            "Blue: positive; dashed red: negative · Click to inspect, drag to pan, scroll to zoom"
        )

    def _open_spectrum(self) -> None:
        if self.viewer is not None and self.viewer.fit_busy:
            return
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open NMRPipe spectrum",
            "",
            "NMRPipe spectra (*.ft2 *.ft3 *.ft *.dat);;All files (*)",
        )
        if filename:
            try:
                self.open_spectrum(Path(filename))
            except (OSError, ValueError, KeyError) as exc:
                self._show_error(exc)

    def _open_peaks(self) -> None:
        if self.viewer is None or self.viewer.fit_busy:
            return
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load peak list",
            "",
            "Peak lists (*.list *.csv *.json *.xlsx *.xls);;All files (*)",
        )
        if filename:
            try:
                with verified_source_files({"peak_list": Path(filename)}) as source_files:
                    peaks = load_peaks(Path(filename), self.viewer.spectra)
                if not self._protect_edits():
                    return
                self.viewer.set_peaks(peaks)
                self.viewer.source_files = (
                    tuple(
                        (role, record)
                        for role, record in self.viewer.source_files
                        if role != "peak_list"
                    )
                    + source_files
                )
                self._export_path = None
                self.statusBar().showMessage(f"Displaying {len(peaks)} peaks from {filename}")
            except (OSError, ValueError, KeyError, ImportError) as exc:
                self._show_error(exc)

    def _open_values(self) -> None:
        if self.viewer is None or self.viewer.fit_busy:
            return
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load plane values")
        if filename:
            try:
                with verified_source_files({"plane_values": Path(filename)}) as source_files:
                    values = np.atleast_1d(np.loadtxt(filename, dtype=float))
                self.viewer.set_plane_values(values)
                self.viewer.source_files = (
                    tuple(
                        (role, record)
                        for role, record in self.viewer.source_files
                        if role != "plane_values"
                    )
                    + source_files
                )
            except (OSError, ValueError) as exc:
                self._show_error(exc)

    def _show_error(self, error: Exception) -> None:
        QtWidgets.QMessageBox.warning(self, "PeakFit", str(error))

    def _update_fit_state(self) -> None:
        busy = self.viewer is not None and self.viewer.fit_busy
        self.fit_export_action.setEnabled(
            self.viewer is not None and self.viewer.fit_result is not None and not busy
        )
        self.open_action.setEnabled(not busy)
        self.peaks_action.setEnabled(self.viewer is not None and not busy)
        self.values_action.setEnabled(self.viewer is not None and not busy)
        self.statusBar().showMessage(
            "Local fit running · Cancel beside the fit status to edit peaks or replace the dataset"
            if busy
            else "Local fit stopped · Working peaks unchanged"
        )

    def _update_work_state(self) -> None:
        if self.viewer is not None:
            dirty = self.viewer.edits.dirty or self.viewer.inspector.pending
            self.setWindowModified(dirty)
            self.statusBar().showMessage(
                f"{len(self.viewer.peaks)} working peaks · "
                + ("Unsaved peak edits · " if dirty else "")
                + "Click to inspect/select · selected peak: drag to move · A: add · Esc: cancel"
            )

    def _export_fit_results(self) -> bool:
        if self.viewer is None:
            return False
        return self.viewer.export_fit_results()

    def _export_peaks(self) -> bool:
        if self.viewer is None or not self.viewer.commit_pending_fields():
            return False
        dialog = QtWidgets.QFileDialog(self, "Export working peaks")
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter("PeakFit CSV (*.csv)")
        dialog.setDefaultSuffix("csv")
        dialog.selectFile(str(self._export_path or Path(self._spectrum_name).with_suffix(".csv")))
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False
        path = Path(dialog.selectedFiles()[0])
        try:
            export_peaks(path, self.viewer.peaks)
        except (OSError, UnicodeError) as exc:
            self._show_error(exc)
            return False
        self._export_path = path
        self.viewer.edits.mark_saved()
        self._update_work_state()
        self.statusBar().showMessage(f"Exported {len(self.viewer.peaks)} peaks to {path}")
        return True

    def _protect_edits(self) -> bool:
        if self.viewer is None or not (self.viewer.edits.dirty or self.viewer.inspector.pending):
            return True
        answer = QtWidgets.QMessageBox.warning(
            self,
            "Unsaved peak edits",
            "Export edited peaks before discarding this working list?",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Save:
            return self._export_peaks()
        return answer == QtWidgets.QMessageBox.StandardButton.Discard

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        """Protect coordinates and typed fields before closing the application."""
        if self.viewer is not None and self.viewer._exporting:
            event.ignore()
            return
        if (
            self.viewer is not None
            and self.viewer.fit_busy
            and QtWidgets.QMessageBox.question(
                self,
                "Running local fit",
                "Cancel the running fit and close PeakFit?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            != QtWidgets.QMessageBox.StandardButton.Yes
        ):
            event.ignore()
            return
        if not self._protect_edits():
            event.ignore()
            return
        # Ask about results last: a fit may complete during either preceding dialog.
        if self.viewer is not None:
            if not self.viewer.confirm_result_loss("discard"):
                event.ignore()
                return
            # No cancellation or shutdown until every close decision is accepted.
            self.viewer.stop_preparation()
        event.accept()

    def changeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        """Cancel active gestures when focus leaves the application."""
        super().changeEvent(event)
        if (
            event.type() == QtCore.QEvent.Type.WindowDeactivate
            and self.viewer is not None
            and (self.viewer._drag_peak is not None or self.viewer.adding_peak)
        ):
            self.viewer.cancel_interaction()

    def _reset_view(self) -> None:
        if self.viewer is not None:
            self.viewer.reset_view()

    def _step_plane(self, offset: int) -> None:
        if self.viewer is not None:
            self.viewer.set_plane(self.viewer.plane + 1 + offset)

    def _show_peak_labels(self, visible: bool) -> None:
        if self.viewer is not None:
            self.viewer.show_peak_labels(visible)


def launch(
    spectrum: Path | None = None, peaks: Path | None = None, plane_values: Path | None = None
) -> None:
    """Start the desktop application, optionally opening an initial spectrum."""
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication([sys.argv[0]])
        app.setApplicationName("PeakFit")
        app.setOrganizationName("PeakFit")
    window = PeakFitWindow()
    if spectrum is not None:
        window.open_spectrum(spectrum, peaks, plane_values)
    window.show()
    if owns_app:
        app.exec()
