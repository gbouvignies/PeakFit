"""Compact selected-peak fields and marker interaction feedback."""

import math
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

# isort: split
import pyqtgraph as pg

if TYPE_CHECKING:
    from peakfit.gui.peaks import WorkingPeak


class PeakInspector(QtWidgets.QGroupBox):
    """Edit one complete assignment without rounding unchanged coordinates."""

    applied = QtCore.Signal(str, tuple)
    deleted = QtCore.Signal()

    def __init__(self, x_title: str, y_title: str) -> None:
        super().__init__("Peak — click a marker to select")
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        self.name = QtWidgets.QLineEdit()
        self.x_field = QtWidgets.QLineEdit()
        self.y_field = QtWidgets.QLineEdit()
        self.details = QtWidgets.QLabel("Coordinates are unfolded ppm")
        self.details.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.details.setWordWrap(True)
        self.apply = QtWidgets.QPushButton("Apply")
        self.delete = QtWidgets.QPushButton("Delete")
        for row, (title, field) in enumerate(
            (("Name", self.name), (x_title, self.x_field), (y_title, self.y_field))
        ):
            label = QtWidgets.QLabel(title)
            label.setBuddy(field)
            field.setAccessibleName(title)
            layout.addWidget(label, row, 0)
            layout.addWidget(field, row, 1)
            field.returnPressed.connect(self._apply)
            field.installEventFilter(self)
        layout.addWidget(self.apply, 0, 2)
        layout.addWidget(self.delete, 1, 2)
        layout.addWidget(self.details, 3, 0, 1, 3)
        self.apply.clicked.connect(self._apply)
        self.delete.clicked.connect(self.deleted)
        self._peak: WorkingPeak | None = None
        self.setEnabled(False)

    def show_peak(
        self, peak: WorkingPeak | None, displayed: tuple[float, float] | None = None
    ) -> None:
        """Populate precise fields; the caller retains selection and identity."""
        self._peak = peak
        self.setEnabled(peak is not None)
        self.setTitle("Selected peak" if peak else "Peak — click a marker to select")
        for field, value in zip(
            (self.name, self.x_field, self.y_field),
            (peak.name, repr(peak.positions[1]), repr(peak.positions[0])) if peak else ("", "", ""),
            strict=True,
        ):
            field.setText(value)
        self.details.setText(
            f"Displayed: {displayed[0]:.6f}, {displayed[1]:.6f} ppm · unfolded fields above"
            if peak
            and displayed
            and any(
                not math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
                for a, b in zip(displayed, reversed(peak.positions), strict=True)
            )
            else "Unfolded ppm · select first, then drag to move"
        )

    def _apply(self) -> None:
        try:
            positions = (float(self.y_field.text()), float(self.x_field.text()))
        except ValueError:
            self.details.setText("Enter a finite number for each ppm coordinate.")
            return
        self.applied.emit(self.name.text(), positions)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        """Escape restores committed fields; other editing keys remain native."""
        if (
            event.type() == QtCore.QEvent.Type.KeyPress
            and isinstance(event, QtGui.QKeyEvent)
            and event.key() == QtCore.Qt.Key.Key_Escape
        ):
            self.show_peak(self._peak)
            return True
        return super().eventFilter(watched, event)


class PeakMarkers(pg.ScatterPlotItem):
    """Claim marker clicks before crosshairs while leaving background drags alone."""

    def hoverEvent(self, ev) -> None:  # noqa: N802
        """Advertise selectable markers without claiming pan gestures."""
        super().hoverEvent(ev)
        hits = [] if ev.isExit() else self.pointsAt(ev.pos())
        if len(hits):
            ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton)
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()
