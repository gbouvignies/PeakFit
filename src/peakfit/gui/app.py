"""Desktop window and file-opening actions for PeakFit."""

import sys
from pathlib import Path

import numpy as np
from PySide6 import QtGui, QtWidgets

from peakfit.gui.spectrum import load_peaks, load_spectrum
from peakfit.gui.viewer import SpectrumViewer


class PeakFitWindow(QtWidgets.QMainWindow):
    """Application shell; scientific operations remain outside presentation code."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PeakFit")
        self.resize(1360, 900)
        self.viewer: SpectrumViewer | None = None
        welcome = QtWidgets.QLabel(
            "Open an NMRPipe spectrum to begin.\nFile → Open spectrum (Ctrl+O)"
        )
        welcome.setMargin(40)
        self.setCentralWidget(welcome)
        menu = self.menuBar().addMenu("&File")
        self._action(menu, "&Open spectrum…", "Ctrl+O", self._open_spectrum)
        self.peaks_action = self._action(menu, "Load &peak list…", "", self._open_peaks)
        self.values_action = self._action(menu, "Load plane &values…", "", self._open_values)
        self.peaks_action.setEnabled(False)
        self.values_action.setEnabled(False)
        menu.addSeparator()
        self._action(menu, "&Quit", "Ctrl+Q", self.close)
        view = self.menuBar().addMenu("&View")
        self._action(view, "Reset view", "Ctrl+0", self._reset_view)
        self._action(view, "Previous plane", "Ctrl+PgUp", lambda: self._step_plane(-1))
        self._action(view, "Next plane", "Ctrl+PgDown", lambda: self._step_plane(1))
        self.labels_action = view.addAction("Peak labels")
        self.labels_action.setCheckable(True)
        self.labels_action.toggled.connect(self._show_peak_labels)
        self.statusBar().showMessage("Read-only spectrum exploration")

    def _action(self, menu, title: str, shortcut: str, callback) -> QtGui.QAction:
        action = menu.addAction(title)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(callback)
        return action

    def open_spectrum(
        self, path: Path, peaks: Path | None = None, plane_values: Path | None = None
    ) -> None:
        """Prepare a replacement before discarding the currently displayed spectrum."""
        spectra = load_spectrum(path, plane_values)
        imported_peaks = load_peaks(peaks, spectra) if peaks is not None else []
        viewer = SpectrumViewer(spectra)
        viewer.set_peaks(imported_peaks)
        viewer.show_peak_labels(self.labels_action.isChecked())
        self.setCentralWidget(viewer)
        self.viewer = viewer
        self.peaks_action.setEnabled(True)
        self.values_action.setEnabled(True)
        self.setWindowTitle(f"{path.name} — PeakFit")
        self.statusBar().showMessage(
            f"{spectra.data.shape[0]} planes · {len(imported_peaks)} peaks · "
            "Blue: positive; dashed red: negative · Click to inspect, drag to pan, scroll to zoom"
        )

    def _open_spectrum(self) -> None:
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
        if self.viewer is None:
            return
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load peak list",
            "",
            "Peak lists (*.list *.csv *.json *.xlsx *.xls);;All files (*)",
        )
        if filename:
            try:
                peaks = load_peaks(Path(filename), self.viewer.spectra)
                self.viewer.set_peaks(peaks)
                self.statusBar().showMessage(f"Displaying {len(peaks)} peaks from {filename}")
            except (OSError, ValueError, KeyError, ImportError) as exc:
                self._show_error(exc)

    def _open_values(self) -> None:
        if self.viewer is None:
            return
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load plane values")
        if filename:
            try:
                values = np.atleast_1d(np.loadtxt(filename, dtype=float))
                self.viewer.set_plane_values(values)
            except (OSError, ValueError) as exc:
                self._show_error(exc)

    def _show_error(self, error: Exception) -> None:
        QtWidgets.QMessageBox.warning(self, "Could not load input", str(error))

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
