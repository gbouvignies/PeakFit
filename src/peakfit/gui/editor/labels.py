"""Plain-text labels: selectable styled text, and one that elides without losing its text."""

from dataclasses import dataclass
from html import escape

from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui.theme import SPACE_M

POPOVER_WIDTH = 280  # logical px: one comfortable measure of explanation

# The inspector's label column: forms, tables and review rows share this left edge
# for their values.
LABEL_WIDTH = 104


@dataclass(frozen=True)
class Row:
    """One review row: a muted label, its value, and the value's style, caption and tooltip.

    ``info`` is a longer explanation behind an ⓘ button beside the value.
    """

    label: str
    value: str
    style: str = ""
    detail: str = ""
    tip: str = ""
    info: str = ""


def label_rows(grid: QtWidgets.QGridLayout, rows: list[Row]) -> None:
    """Muted labels in the shared column, values beside them, captions underneath."""
    grid.setColumnMinimumWidth(0, LABEL_WIDTH)
    grid.setColumnStretch(1, 1)
    line = 0
    for row in rows:
        title = text_label(row.label, "muted")
        grid.addWidget(title, line, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        shown = text_label(row.value, row.style, wrap=True)
        shown.setToolTip(row.tip)
        if row.info:
            cell = QtWidgets.QHBoxLayout()
            cell.setSpacing(4)
            cell.addWidget(shown, 1)
            cell.addWidget(InfoButton(row.label, row.info), 0, QtCore.Qt.AlignmentFlag.AlignTop)
            grid.addLayout(cell, line, 1)
        else:
            grid.addWidget(shown, line, 1)
        line += 1
        if row.detail:
            grid.addWidget(text_label(row.detail, "caption", wrap=True), line, 1)
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


def fit_phrases(parts: list[tuple[str, str]], metrics: QtGui.QFontMetrics, width: int) -> str:
    """Rich text of ``(plain, rich)`` phrases joined by " · " within ``width``.

    Phrases that do not fit are dropped whole from the end and "…" marks the cut,
    so no phrase is ever shown cut in the middle.
    """
    for count in range(len(parts), -1, -1):
        shown = [plain for plain, _rich in parts[:count]]
        if count < len(parts):
            shown.append("…")
        if count == 0 or metrics.horizontalAdvance(" · ".join(shown)) <= width:
            rich = [r for _plain, r in parts[:count]]
            return " · ".join(rich + (["…"] if count < len(parts) else []))
    return ""


def show_text(label: QtWidgets.QLabel, text: str) -> None:
    """Set a label's text and hide it when empty, so it leaves no gap."""
    label.setText(text)
    label.setVisible(bool(text))


class PhraseLabel(QtWidgets.QLabel):
    """Rich-text phrases on one line that drop whole phrases, marked "…", to fit.

    The tooltip holds every phrase and a hint; the label refits when resized.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._phrases: list[tuple[str, str]] = []

    def set_phrases(self, phrases: list[tuple[str, str]], hint: str) -> None:
        """Show ``(plain, rich)`` phrases; the tooltip lists them all, then ``hint``."""
        self._phrases = phrases
        full = " · ".join(plain for plain, _rich in phrases)
        self.setToolTip(f"{full}\n{hint}" if full else hint)
        self._fit()

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Refit the phrases to the new width."""
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        self.setText(fit_phrases(self._phrases, self.fontMetrics(), self.width()))


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
        """Return the text as painted at the current width.

        Text of " · "-separated phrases drops whole trailing phrases, marked "…", so a
        word is never cut; a first phrase that does not fit is elided.
        """
        metrics, width = self.fontMetrics(), self.contentsRect().width()
        text = self.text()
        if metrics.horizontalAdvance(text) <= width:
            return text
        phrases = text.split(" · ")
        for count in range(len(phrases) - 1, 0, -1):
            shown = " · ".join([*phrases[:count], "…"])
            if metrics.horizontalAdvance(shown) <= width:
                return shown
        return metrics.elidedText(text, QtCore.Qt.TextElideMode.ElideRight, width)

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


class InfoButton(QtWidgets.QToolButton):
    """ⓘ: a longer explanation in a popover, opened by click, Space or Return.

    The button is a Tab stop named "About <topic>"; the popover closes on Escape or a
    click elsewhere and gives the keyboard back to the button. Screen readers get the
    text as the button's description.
    """

    def __init__(self, topic: str, text: str = "") -> None:
        super().__init__()
        self.setObjectName("info")
        self.setText("ⓘ")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(f"About {topic.lower()}")
        self.setToolTip(f"About {topic.lower()}")
        self._popover = QtWidgets.QFrame(self, QtCore.Qt.WindowType.Popup)
        self._popover.setObjectName("popover")
        layout = QtWidgets.QVBoxLayout(self._popover)
        layout.setContentsMargins(SPACE_M, SPACE_M, SPACE_M, SPACE_M)
        self.body = text_label("", "", wrap=True)
        self.body.setFixedWidth(POPOVER_WIDTH)
        layout.addWidget(self.body)
        self.set_text(text)
        self.clicked.connect(self.open_popover)

    def set_text(self, text: str) -> None:
        """Replace the explanation; an empty one hides the button."""
        self.body.setText(text)
        self.setAccessibleDescription(text)
        self.setVisible(bool(text))

    def open_popover(self) -> None:
        """Show the explanation below the button, kept on the button's screen."""
        popover = self._popover
        popover.adjustSize()
        anchor = self.mapToGlobal(QtCore.QPoint(0, self.height() + 2))
        screen = (self.screen() or QtGui.QGuiApplication.primaryScreen()).availableGeometry()
        x = max(screen.left(), min(anchor.x(), screen.right() - popover.width()))
        popover.move(x, anchor.y())
        popover.show()
        popover.setFocus()

    def popover_visible(self) -> bool:
        """Whether the explanation is on screen."""
        return self._popover.isVisible()
