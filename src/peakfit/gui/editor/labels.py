"""Plain-text labels: selectable styled text, and one that elides without losing its text."""

from html import escape

from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui.theme import SPACE_M

# The inspector's label column: forms, tables and review rows share this left edge
# for their values.
LABEL_WIDTH = 104

# One review row: muted label, value, label style of the value, caption under the
# value, tooltip on the value.
type Row = tuple[str, str, str, str, str]


def label_rows(grid: QtWidgets.QGridLayout, rows: list[Row]) -> None:
    """Muted labels in the shared column, values beside them, captions underneath."""
    grid.setColumnMinimumWidth(0, LABEL_WIDTH)
    grid.setColumnStretch(1, 1)
    line = 0
    for label, value, style, detail, tip in rows:
        title = text_label(label, "muted")
        grid.addWidget(title, line, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        shown = text_label(value, style, wrap=True)
        shown.setToolTip(tip)
        grid.addWidget(shown, line, 1)
        line += 1
        if detail:
            grid.addWidget(text_label(detail, "caption", wrap=True), line, 1)
            line += 1


def section(layout: QtWidgets.QBoxLayout, text: str) -> QtWidgets.QLabel:
    """A section title; every section starts with the same gap above it."""
    label = text_label(text.upper(), "section")
    layout.addSpacing(SPACE_M)
    layout.addWidget(label)
    return label


def phrases(*lines: tuple[str, ...]) -> str:
    """Rich text that wraps only between phrases: each line's parts joined by " · "."""
    return "<br>".join(
        " · ".join(
            f'<span style="white-space:nowrap">{escape(part)}</span>' for part in parts if part
        )
        for parts in lines
        if any(parts)
    )


def show_text(label: QtWidgets.QLabel, text: str) -> None:
    """Set a label's text and hide it when empty, so it leaves no gap."""
    label.setText(text)
    label.setVisible(bool(text))


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


def about(
    layout: QtWidgets.QBoxLayout, title: str
) -> tuple[QtWidgets.QToolButton, QtWidgets.QLabel]:
    """An "About …" disclosure and its explanation, added to ``layout``, collapsed at first.

    Both widgets live as long as the page, so an opened explanation stays open across
    selections. The toggle takes keyboard focus (Tab, then Space).
    """
    button = disclosure(title)
    button.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
    button.setAccessibleName(title)
    body = text_label("", "caption", wrap=True)
    body.hide()
    button.toggled.connect(body.setVisible)
    layout.addWidget(button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
    layout.addWidget(body)
    return button, body


def disclosure(text: str) -> QtWidgets.QToolButton:
    """A collapsed-by-default section toggle: a turning triangle, then the title.

    The triangle is text, not a style-drawn arrow, so the toggle starts on the
    content's left edge like every other line.
    """
    button = QtWidgets.QToolButton()
    button.setObjectName("text")
    button.setCheckable(True)
    button.setAccessibleName(text)
    button.setText(f"▸  {text}")
    button.toggled.connect(lambda shown: button.setText(f"{'▾' if shown else '▸'}  {text}"))
    return button
