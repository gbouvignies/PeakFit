"""Small modal export interaction; disk work leaves the Qt event loop responsive."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui.dataset_fit import DatasetFitResult
from peakfit.gui.fit_export import export_fit_result

if TYPE_CHECKING:
    from peakfit.gui.fit_positions import DatasetCompletion
    from peakfit.gui.local_fit import LocalFitResult


class _ExportProgress(QtWidgets.QProgressDialog):
    """Keep export modal through finalization, including Escape/window-close keys."""

    def reject(self) -> None:
        """An active multi-file write cannot be dismissed halfway through."""

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        """Leave the worker and its progress visible until it has completed."""
        event.ignore()


def choose_and_export(
    parent: QtWidgets.QWidget,
    result: LocalFitResult | DatasetFitResult,
    *,
    completion: DatasetCompletion | None = None,
    current: bool | None = None,
) -> Path | None:
    """Choose a parent directory and create a new result-specific child directory."""
    directory = QtWidgets.QFileDialog.getExistingDirectory(
        parent, "Export fit results — choose parent directory"
    )
    if not directory:
        return None
    prefix = "dataset-fit" if isinstance(result, DatasetFitResult) else "local-fit"
    destination = Path(directory) / f"{prefix}-{result.run_id}"
    progress = _ExportProgress("Exporting completed fit results…", "", 0, 0, parent)
    progress.setCancelButton(None)
    progress.setWindowTitle("PeakFit export")
    progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
    progress.setMinimumDuration(0)
    # A local worker and modal event loop suffice: no reusable job machinery.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="peakfit-export") as executor:
        future = executor.submit(
            export_fit_result, result, destination, completion=completion, current=current
        )
        timer = QtCore.QTimer(progress)
        timer.setInterval(30)
        timer.timeout.connect(lambda: progress.accept() if future.done() else None)
        timer.start()
        progress.exec()
        timer.stop()
        try:
            return future.result()
        except Exception as error:
            QtWidgets.QMessageBox.warning(parent, "Fit export failed", str(error))
            return None
