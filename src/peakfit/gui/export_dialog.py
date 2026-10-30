"""Small modal export interaction; disk work leaves the Qt event loop responsive."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.engine.scientific.completed import FinalClusterSamples
from peakfit.gui.dataset_fit import DatasetFitResult
from peakfit.gui.fit_export import export_fit_result

if TYPE_CHECKING:
    from concurrent.futures import Future

    from peakfit.gui.fit_positions import DatasetCompletion
    from peakfit.gui.local_fit import LocalFitResult


class _ExportProgress(QtWidgets.QProgressDialog):
    """Keep export modal through finalization, including Escape/window-close keys."""

    def reject(self) -> None:
        """An active multi-file write cannot be dismissed halfway through."""

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        """Leave the worker and its progress visible until it has completed."""
        event.ignore()


def _choose_export_directory(parent, include_evidence=False, *, evidence_available=True):
    """One optional evidence checkbox alongside the destination picker."""
    dialog = QtWidgets.QFileDialog(parent, "Export fit results — choose parent directory")
    dialog.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
    dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
    detailed = QtWidgets.QCheckBox(
        "Include detailed evidence for portable sample review and continuation"
    )
    detailed.setChecked(include_evidence and evidence_available)
    detailed.setEnabled(evidence_available)
    if not evidence_available:
        detailed.setToolTip("Relink the verified spectrum before exporting detailed evidence.")
    layout = dialog.layout()
    assert layout is not None
    layout.addWidget(detailed)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None, False
    return dialog.selectedFiles()[0], detailed.isChecked()


def choose_and_export(
    parent: QtWidgets.QWidget,
    result: LocalFitResult | DatasetFitResult,
    *,
    completion: DatasetCompletion | None = None,
    current: bool | None = None,
    dataset=None,
) -> Path | None:
    """Choose a parent directory and create a new result-specific child directory."""
    setting = json.loads(result.inputs.snapshot.configuration).get("output", {})
    available = (
        dataset is not None
        or not isinstance(result, DatasetFitResult)
        or all(isinstance(s, FinalClusterSamples) for s in result.completed_fit.samples)
    )
    directory, detailed = _choose_export_directory(
        parent, setting.get("detailed_evidence", False), evidence_available=available
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
        future: Future[Path] | None = None

        def begin() -> None:
            nonlocal future
            future = executor.submit(
                export_fit_result,
                result,
                destination,
                completion=completion,
                current=current,
                detailed_evidence=detailed,
                dataset=dataset,
            )

        # Let the modal transition show progress and enter Qt before competing
        # for the GIL with serialization. No numerical work runs in this callback.
        start = QtCore.QTimer(progress)
        start.setSingleShot(True)
        start.timeout.connect(begin)
        start.start(0)
        timer = QtCore.QTimer(progress)
        timer.setInterval(30)
        timer.timeout.connect(
            lambda: progress.accept() if future is not None and future.done() else None
        )
        timer.start()
        progress.exec()
        timer.stop()
        progress.deleteLater()
        try:
            assert future is not None
            return future.result()
        except Exception as error:
            QtWidgets.QMessageBox.warning(parent, "Fit export failed", str(error))
            return None
