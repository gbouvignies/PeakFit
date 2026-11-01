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


def review_item(
    layout: QtWidgets.QVBoxLayout,
    family: str,
    glyph: str,
    text: str,
    detail: str = "",
    *,
    warn: bool = False,
) -> None:
    """One review line: an optional family heading, a glyph, text and a caption."""
    if family:
        layout.addSpacing(4)
        layout.addWidget(text_label(family, "caption"))
    row = QtWidgets.QHBoxLayout()
    row.setSpacing(6)
    mark = text_label(glyph, "warning" if warn else "caption")
    mark.setFixedWidth(12)
    row.addWidget(mark, 0, QtCore.Qt.AlignmentFlag.AlignTop)
    body = QtWidgets.QVBoxLayout()
    body.setSpacing(0)
    body.addWidget(text_label(text, "warning" if warn else "", wrap=True))
    if detail:
        body.addWidget(text_label(detail, "caption", wrap=True))
    row.addLayout(body, 1)
    layout.addLayout(row)


def clear_layout(layout: QtWidgets.QLayout) -> None:
    """Remove every widget and nested row layout from a section."""
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            continue
        widget, nested = item.widget(), item.layout()
        if widget is not None:
            # Detach now: deferred deletion alone would leave it painted until the
            # event loop runs.
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
        elif nested is not None:
            clear_layout(nested)
            nested.deleteLater()
