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

    commit_requested = QtCore.Signal()
    deleted = QtCore.Signal()
    fields_changed = QtCore.Signal()

    def __init__(self, x_title: str, y_title: str) -> None:
        super().__init__("Peak editing — select a marker")
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        self.name = QtWidgets.QLineEdit()
        self.x_field = QtWidgets.QLineEdit()
        self.y_field = QtWidgets.QLineEdit()
        self.details = QtWidgets.QLabel("Coordinates are unfolded ppm")
        self.details.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.details.setWordWrap(True)
        self.delete = QtWidgets.QPushButton("Delete")
        for row, (title, field) in enumerate(
            (("Name", self.name), (x_title, self.x_field), (y_title, self.y_field))
        ):
            label = QtWidgets.QLabel(title)
            label.setBuddy(field)
            field.setAccessibleName(title)
            layout.addWidget(label, row, 0)
            layout.addWidget(field, row, 1)
            field.setToolTip("Enter or leave the fields to commit; Escape reverts the whole peak")
            field.returnPressed.connect(self.commit_requested)
            field.textChanged.connect(self._fields_changed)
            field.installEventFilter(self)
        layout.addWidget(self.delete, 0, 2)
        layout.addWidget(self.details, 3, 0, 1, 3)
        self.delete.clicked.connect(self.deleted)
        self._peak: WorkingPeak | None = None
        app = QtWidgets.QApplication.instance()
        assert isinstance(app, QtWidgets.QApplication)
        app.focusChanged.connect(self._focus_changed)
        self.show_peak(None)

    def show_peak(
        self, peak: WorkingPeak | None, displayed: tuple[float, float] | None = None
    ) -> None:
        """Populate precise fields; the caller retains selection and identity."""
        self._peak = peak
        controls = (self.name, self.x_field, self.y_field, self.delete)
        for control in controls:
            control.setEnabled(peak is not None)
        self.setTitle("Selected peak" if peak else "Peak editing — select a marker")
        for field, value in zip(
            (self.name, self.x_field, self.y_field),
            (peak.name, repr(peak.positions[1]), repr(peak.positions[0])) if peak else ("", "", ""),
            strict=True,
        ):
            with QtCore.QSignalBlocker(field):
                field.setText(value)
        self.details.setText(
            f"Displayed: {displayed[0]:.6f}, {displayed[1]:.6f} ppm · unfolded fields above"
            if peak
            and displayed
            and any(
                not math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
                for a, b in zip(displayed, reversed(peak.positions), strict=True)
            )
            else "Unfolded ppm · Enter or leave fields to commit · Escape reverts"
        )
        self.fields_changed.emit()

    def _fields_changed(self) -> None:
        self.details.setText("Enter or leave fields to commit · Escape reverts")
        self.fields_changed.emit()

    def _focus_changed(self, old: QtWidgets.QWidget, new: QtWidgets.QWidget) -> None:
        fields = (self.name, self.x_field, self.y_field)
        # Tab/clicks between fields and native text context menus keep one draft.
        if (
            old in fields
            and new is not None
            and new not in fields
            and QtWidgets.QApplication.activePopupWidget() is None
        ):
            self.commit_requested.emit()

    @property
    def pending(self) -> bool:
        """Whether typed fields differ from the displayed committed record."""
        return self._peak is not None and (
            self.name.text(),
            self.x_field.text(),
            self.y_field.text(),
        ) != (self._peak.name, repr(self._peak.positions[1]), repr(self._peak.positions[0]))

    def values(self) -> tuple[str, tuple[float, float]]:
        """Validate the complete draft, focusing a field that needs correction."""
        if not self.name.text().strip():
            self.name.setFocus()
            raise ValueError(
                "Give the peak a non-empty name. Correct it or press Escape to revert."
            )
        coordinates = []
        for field in (self.y_field, self.x_field):
            try:
                value = float(field.text())
                if not math.isfinite(value):
                    raise ValueError
            except ValueError as exc:
                field.setFocus()
                raise ValueError(
                    f"{field.accessibleName()}: enter a finite ppm number. "
                    "Correct it or press Escape to revert."
                ) from exc
            coordinates.append(value)
        return self.name.text(), (coordinates[0], coordinates[1])

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
    """Select on click; route drags to peak movement or ordinary spectrum pan."""

    moved = QtCore.Signal(float, float, bool)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.selected_id: str | None = None
        self.placing = False
        self.editable = True
        self._cancelled = False
        self._owns_drag = False

    def mouseClickEvent(self, ev) -> None:  # noqa: N802
        """Placement owns the next click, including clicks over existing markers."""
        if self.placing:
            ev.ignore()
        else:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev) -> None:  # noqa: N802
        """Preview deltas from the press point; the viewer commits only on release."""
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        if ev.isStart():
            self._cancelled = False
            self._owns_drag = (
                self.editable
                and not self.placing
                and any(p.data() == self.selected_id for p in self.pointsAt(ev.buttonDownPos()))
            )
        if not self._owns_drag:
            # A marker click selects; an unselected marker drag is ordinary pan.
            # Forward with ViewBox-local coordinates, including when a cached hover
            # claim predates a selection change.
            view = self.getViewBox()
            ev.currentItem = view
            view.mouseDragEvent(ev)
            return
        ev.accept()
        if not self._cancelled:
            delta = ev.pos() - ev.buttonDownPos()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            self.moved.emit(delta.x(), delta.y(), ev.isFinish())
        if ev.isFinish():
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)

    def cancel_drag(self) -> None:
        """Ignore the remainder of a gesture after Escape or window deactivation."""
        self._cancelled = True

    def hoverEvent(self, ev) -> None:  # noqa: N802
        """Advertise selectable markers without claiming pan gestures."""
        if self.placing:
            self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            return
        super().hoverEvent(ev)
        hits = [] if ev.isExit() else self.pointsAt(ev.pos())
        if len(hits):
            ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton)
            selected = any(p.data() == self.selected_id for p in hits)
            ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)
            self.setCursor(
                QtCore.Qt.CursorShape.OpenHandCursor
                if selected
                else QtCore.Qt.CursorShape.PointingHandCursor
            )
        else:
            self.unsetCursor()
