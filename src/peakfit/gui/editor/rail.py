"""One rail for browsing assignments and committing precise coordinate groups."""

import math
from typing import TYPE_CHECKING, Any, cast

from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui import theme as themes
from peakfit.gui.editing import PeakInspector

if TYPE_CHECKING:
    from peakfit.gui.peaks import WorkingPeak


class PeakRowDelegate(QtWidgets.QStyledItemDelegate):
    """Draw one flat native list row with aligned coordinate columns."""

    secondary = QtGui.QColor(themes.LIGHT.secondary)  # coordinate text

    def paint(self, painter, option, index) -> None:
        """Keep native focus/selection and expose full row text to accessibility."""
        # PySide exposes these public Qt data members dynamically; its stub omits them.
        styled = cast("Any", QtWidgets.QStyleOptionViewItem(option))
        self.initStyleOption(styled, index)
        text = styled.text
        styled.text = ""
        style = styled.widget.style() if styled.widget else QtWidgets.QApplication.style()
        style.drawControl(
            QtWidgets.QStyle.ControlElement.CE_ItemViewItem, styled, painter, styled.widget
        )
        name, x, y = text.rsplit("\t", 2)
        rect = styled.rect.adjusted(themes.SPACE_M, 0, -themes.SPACE_S, 0)
        width = column_width(styled.fontMetrics)
        palette = styled.palette
        painter.save()
        painter.setFont(styled.font)
        painter.setPen(palette.color(QtGui.QPalette.ColorRole.Text))
        label = QtCore.QRect(
            rect.left(), rect.top(), max(0, rect.width() - width * 2), rect.height()
        )
        painter.drawText(
            label,
            QtCore.Qt.AlignmentFlag.AlignVCenter,
            styled.fontMetrics.elidedText(name, QtCore.Qt.TextElideMode.ElideRight, label.width()),
        )
        painter.setPen(self.secondary)
        for offset, coordinate in ((2, x), (1, y)):
            cell = QtCore.QRect(rect.right() - width * offset, rect.top(), width, rect.height())
            painter.drawText(
                cell,
                QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignRight,
                coordinate,
            )
        painter.restore()

    def sizeHint(self, option, index) -> QtCore.QSize:  # noqa: N802
        """Uniform rows at the shared row height."""
        return QtCore.QSize(super().sizeHint(option, index).width(), themes.ROW_HEIGHT)


def column_width(metrics: QtGui.QFontMetrics) -> int:
    """Room for one coordinate column, shared by the rows and their headings."""
    return max(64, metrics.horizontalAdvance("−123.1234") + themes.SPACE_S)


class PeakRail(QtWidgets.QWidget):
    """Keep hover, keyboard selection and typed drafts independent."""

    selected = QtCore.Signal(str)
    hovered = QtCore.Signal(object)
    edit_requested = QtCore.Signal()

    def __init__(  # noqa: PLR0915
        self, x_title: str, y_title: str, decimals: tuple[int, int] = (4, 3)
    ) -> None:
        super().__init__()
        self.setObjectName("rail")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumWidth(280)
        self.decimals = decimals
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, themes.SPACE_M, 0, themes.SPACE_M)
        layout.setSpacing(themes.SPACE_S)
        inset = (themes.SPACE_M, 0, themes.SPACE_M, 0)
        title = QtWidgets.QLabel("PEAKS")
        title.setObjectName("section")
        title.setContentsMargins(*inset)
        layout.addWidget(title)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Find assignment or ID")
        self.search.setAccessibleName("Find assignment or peak identity")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        search_row = QtWidgets.QHBoxLayout()
        search_row.setContentsMargins(*inset)
        search_row.addWidget(self.search)
        layout.addLayout(search_row)
        headings = QtWidgets.QHBoxLayout()
        headings.setContentsMargins(themes.SPACE_M, 0, themes.SPACE_S + 10, 0)
        headings.setSpacing(0)
        assignment = QtWidgets.QLabel("Assignment")
        assignment.setObjectName("caption")
        headings.addWidget(assignment, 1)
        for title in (x_title, y_title):
            heading = QtWidgets.QLabel(title.split(" · ")[0] + " ppm")
            heading.setObjectName("caption")
            heading.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            heading.setFixedWidth(column_width(heading.fontMetrics()))
            headings.addWidget(heading)
        layout.addLayout(headings)
        self.list = QtWidgets.QListWidget()
        self.list.setItemDelegate(PeakRowDelegate(self.list))
        self.list.setUniformItemSizes(True)
        self.list.setMouseTracking(True)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setAccessibleName("Working peaks; assignment, horizontal ppm, vertical ppm")
        self.list.setMinimumHeight(150)
        self.list.currentItemChanged.connect(self._selection_changed)
        self.list.itemSelectionChanged.connect(lambda: self._filter(self.search.text()))
        self.list.itemEntered.connect(lambda item: self.hovered.emit(item.data(256)))
        self.list.itemDoubleClicked.connect(lambda *_: self.edit_requested.emit())
        self.list.viewport().installEventFilter(self)
        self.list.installEventFilter(self)
        layout.addWidget(self.list, 1)
        self.count = QtWidgets.QLabel()
        self.count.setObjectName("caption")
        self.count.setContentsMargins(*inset)
        layout.addWidget(self.count)
        self.reveal = QtWidgets.QPushButton("Reveal selected peak in list")
        self.reveal.clicked.connect(self._reveal)
        self.reveal.hide()
        layout.addWidget(self.reveal)
        self.inspector = PreciseInspector(x_title + " ppm", y_title + " ppm")
        self.inspector.decimals = decimals
        self.inspector.details.setObjectName("caption")
        grid = self.inspector.layout()
        assert isinstance(grid, QtWidgets.QGridLayout)
        # Same validated group-commit component, composed vertically in the rail.
        grid.removeWidget(self.inspector.delete)
        grid.addWidget(self.inspector.delete, 4, 0, 1, 2)
        self.inspector.details.setMinimumWidth(0)
        self.inspector.details.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred
        )
        self.draft = QtWidgets.QLabel()
        self.draft.setObjectName("warning")
        grid.addWidget(self.draft, 5, 0, 1, 2)
        self.inspector.fields_changed.connect(self._draft_changed)
        scroll = QtWidgets.QScrollArea()
        self.inspector_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(self.inspector)
        scroll.setMinimumHeight(250)
        scroll.setMaximumHeight(310)
        scroll.setContentsMargins(*inset)
        self.inspector.setContentsMargins(themes.SPACE_M, 0, themes.SPACE_M, 0)
        layout.addWidget(scroll)
        self._items: dict[str, QtWidgets.QListWidgetItem] = {}
        self._selected: str | None = None
        self._hovered: str | None = None
        self._wash = QtGui.QColor(themes.LIGHT.hover)
        self._draft_changed()

    def apply_theme(self, theme: themes.Theme) -> None:
        """Hovered-marker rows use the theme's hover wash."""
        self._wash = QtGui.QColor(theme.hover)
        delegate = self.list.itemDelegate()
        if isinstance(delegate, PeakRowDelegate):
            delegate.secondary = QtGui.QColor(theme.secondary)
        if self._hovered in self._items:
            self._items[self._hovered].setBackground(QtGui.QBrush(self._wash))

    def refresh(self, peaks: tuple[WorkingPeak, ...], selected: str | None) -> None:
        """Update rows after a commit, never during hover or a drag draft."""
        self._selected = selected
        with QtCore.QSignalBlocker(self.list):
            live_ids = {peak.id for peak in peaks}
            for identity in list(self._items):
                if identity not in live_ids:
                    self.list.takeItem(self.list.row(self._items.pop(identity)))
            for row, peak in enumerate(peaks):
                item = self._items.get(peak.id)
                if item is None:
                    item = QtWidgets.QListWidgetItem()
                    item.setData(256, peak.id)
                    self._items[peak.id] = item
                    self.list.insertItem(row, item)
                elif self.list.row(item) != row:
                    self.list.takeItem(self.list.row(item))
                    self.list.insertItem(row, item)
                dx, dy = self.decimals
                item.setText(f"{peak.name}\t{peak.positions[1]:.{dx}f}\t{peak.positions[0]:.{dy}f}")
                item.setToolTip(
                    f"{peak.name}\nID {peak.id}\nUnfolded ppm: "
                    f"{peak.positions[0]!r}, {peak.positions[1]!r}"
                )
            if selected in self._items:
                self.list.setCurrentItem(self._items[selected])
        self._filter(self.search.text())

    def select(self, peak: WorkingPeak | None, displayed: tuple[float, float] | None) -> None:
        """Synchronize a committed selection without opening or clearing the rail."""
        self._selected = peak.id if peak else None
        with QtCore.QSignalBlocker(self.list):
            item = self._items.get(self._selected or "")
            if item is not None:
                if self.list.currentItem() is not item:
                    self.list.setCurrentItem(item)
                if not item.isHidden():
                    self.list.scrollToItem(item)
            else:
                self.list.clearSelection()
        self.inspector.show_peak(peak, displayed)
        self._filter(self.search.text())

    def highlight(self, peak_id: str | None) -> None:
        """Wash just two rows, with no scroll, selection or editor updates."""
        for identity, color in ((self._hovered, QtGui.QColor()), (peak_id, self._wash)):
            if identity in self._items:
                self._items[identity].setBackground(QtGui.QBrush(color))
        self._hovered = peak_id

    def _filter(self, query: str) -> None:
        query = query.casefold()
        for identity, item in self._items.items():
            item.setHidden(query not in (item.text() + " " + identity).casefold())
        visible = sum(not item.isHidden() for item in self._items.values())
        multiple = len(self.list.selectedItems())
        self.count.setText(
            f"{visible} of {len(self._items)} peaks"
            + (f" · {multiple} selected" if multiple > 1 else "")
        )
        item = self._items.get(self._selected or "")
        self.reveal.setVisible(item is not None and item.isHidden())

    def _reveal(self) -> None:
        self.search.clear()
        if self._selected in self._items:
            self.list.scrollToItem(self._items[self._selected])

    def _selection_changed(self, item, _previous) -> None:
        if item is not None:
            self.selected.emit(item.data(256))
        self._filter(self.search.text())

    def _draft_changed(self) -> None:
        self.draft.setText(
            "Draft · Enter commits · Escape discards" if self.inspector.pending else ""
        )

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        """Clear row hover on leave; Enter explicitly opens precise editing."""
        if event.type() == QtCore.QEvent.Type.Leave:
            self.hovered.emit(None)
        if (
            watched is self.list
            and event.type() == QtCore.QEvent.Type.KeyPress
            and event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter)
        ):
            self.edit_requested.emit()
            return True
        return super().eventFilter(watched, event)


class PreciseInspector(PeakInspector):
    """Display-precision labels with exact unchanged values and round-trip focus/copy."""

    decimals: tuple[int, int] = (4, 3)  # (horizontal, vertical) display decimals

    def show_peak(self, peak, displayed=None) -> None:
        """Populate a new committed group without feeding formatted values into science."""
        self._peak = peak
        self._baseline = {}
        self.setTitle(peak.name if peak else "Select a peak to edit")
        for field in (self.name, self.x_field, self.y_field, self.delete):
            field.setEnabled(peak is not None)
        dx, dy = self.decimals
        values = (
            (peak.name, f"{peak.positions[1]:.{dx}f}", f"{peak.positions[0]:.{dy}f}")
            if peak
            else ("", "", "")
        )
        for field, initial_text in zip(
            (self.name, self.x_field, self.y_field), values, strict=True
        ):
            text = initial_text
            if peak and field.hasFocus() and field in (self.x_field, self.y_field):
                text = repr(peak.positions[1 if field is self.x_field else 0])
            with QtCore.QSignalBlocker(field):
                field.setText(text)
            self._baseline[field] = text
        alias = (
            f"Displayed alias: {displayed[0]:.{dx}f}, {displayed[1]:.{dy}f} ppm\n"
            if peak
            and displayed
            and any(
                not math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
                for a, b in zip(displayed, reversed(peak.positions), strict=True)
            )
            else ""
        )
        grid = self.layout()
        assert isinstance(grid, QtWidgets.QGridLayout)
        for row in range(3):
            for column in range(2):
                item = grid.itemAtPosition(row, column)
                widget = item.widget() if item else None
                if widget is not None:
                    widget.setVisible(peak is not None)
        self.delete.setVisible(peak is not None)
        self.details.setText(
            alias + "Unfolded ppm · Enter commits · Escape discards"
            if peak
            else "Choose a peak in the list or spectrum to edit its exact coordinates."
        )
        self.fields_changed.emit()

    @property
    def pending(self) -> bool:
        """Formatting/focus alone is never a scientific draft."""
        return self._peak is not None and any(
            field.text() != value for field, value in self._baseline.items()
        )

    def values(self):
        """Keep untouched float64 values exact; accept the local decimal separator."""
        if not self.name.text().strip():
            self.name.setFocus()
            raise ValueError("Give the peak a name. Correct it or press Escape.")
        assert self._peak is not None
        positions = []
        for field, original in (
            (self.y_field, self._peak.positions[0]),
            (self.x_field, self._peak.positions[1]),
        ):
            if field.text() == self._baseline[field]:
                positions.append(original)
                continue
            text = field.text().strip()
            if QtCore.QLocale().decimalPoint() == "," and "." not in text:
                text = text.replace(",", ".")
            try:
                value = float(text)

                if not math.isfinite(value) or "_" in text:
                    raise ValueError
            except ValueError as exc:
                field.setFocus()
                raise ValueError(
                    f"{field.accessibleName()}: enter finite ppm; correct it or press Escape."
                ) from exc
            positions.append(value)
        return self.name.text(), (positions[0], positions[1])

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        """Reveal round-trip coordinates on focus without creating a draft."""
        if (
            event.type() == QtCore.QEvent.Type.ShortcutOverride
            and isinstance(event, QtGui.QKeyEvent)
            and not self.pending
            and (
                event.matches(QtGui.QKeySequence.StandardKey.Undo)
                or event.matches(QtGui.QKeySequence.StandardKey.Redo)
            )
        ):
            event.ignore()
            return True  # Let the workspace's named peak action handle this key.
        peak = getattr(self, "_peak", None)
        if (
            peak is not None
            and watched in (self.x_field, self.y_field)
            and event.type() in (QtCore.QEvent.Type.FocusIn, QtCore.QEvent.Type.FocusOut)
            and watched.text() == self._baseline.get(watched)
        ):
            index = 1 if watched is self.x_field else 0
            value = peak.positions[index]
            decimals = self.decimals[0 if index == 1 else 1]
            text = (
                repr(value)
                if event.type() == QtCore.QEvent.Type.FocusIn
                else f"{value:.{decimals}f}"
            )
            with QtCore.QSignalBlocker(watched):
                watched.setText(text)
            self._baseline[watched] = text
        return super().eventFilter(watched, event)
