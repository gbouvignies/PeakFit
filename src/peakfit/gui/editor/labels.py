"""Plain-text labels: selectable styled text, and one that elides without losing its text."""

from PySide6 import QtCore, QtGui, QtWidgets


class ElidedLabel(QtWidgets.QLabel):
    """Keep ``text()``, tooltip and accessibility complete; elide only the painting."""

    def __init__(self, text: str = "") -> None:
        super().__init__()
        self.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802
        """Store the complete text; the tooltip shows it when the label is elided."""
        super().setText(text)
        self.setToolTip(text)

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        """Allow the layout to shrink the label to nothing rather than overlap."""
        return QtCore.QSize(0, super().minimumSizeHint().height())

    def displayed_text(self) -> str:
        """Return the text as painted at the current width."""
        return self.fontMetrics().elidedText(
            self.text(), QtCore.Qt.TextElideMode.ElideRight, self.contentsRect().width()
        )

    def paintEvent(self, _event) -> None:  # noqa: N802
        """Draw elided text with the label's own palette and style-sheet colors."""
        painter = QtGui.QPainter(self)
        self.style().drawItemText(
            painter,
            self.contentsRect(),
            int(self.alignment() | QtCore.Qt.AlignmentFlag.AlignVCenter),
            self.palette(),
            self.isEnabled(),
            self.displayed_text(),
            self.foregroundRole(),
        )


def text_label(text: str = "", name: str = "", *, wrap: bool = False) -> QtWidgets.QLabel:
    """Selectable label styled by its object name (section, muted, caption, warning)."""
    label = QtWidgets.QLabel(text)
    if name:
        label.setObjectName(name)
    label.setWordWrap(wrap)
    label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    return label
