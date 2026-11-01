"""Left navigator: browse working peaks, or the clusters of the active run."""

import math
from typing import TYPE_CHECKING, Any, cast

from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui import theme as themes
from peakfit.gui.editor.review import HIGHLIGHT_KINDS, WARNING_KINDS, HighlightRules

if TYPE_CHECKING:
    from peakfit.gui.dataset_fit import DatasetFitResult
    from peakfit.gui.editor.review import ClusterReview
    from peakfit.gui.peaks import WorkingPeak

_ID = QtCore.Qt.ItemDataRole.UserRole
_SORT = QtCore.Qt.ItemDataRole.UserRole + 1
_PEAKS, _CLUSTERS = 0, 1


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
        painter.save()
        painter.setFont(styled.font)
        painter.setPen(styled.palette.color(QtGui.QPalette.ColorRole.Text))
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


class _ClusterItem(QtWidgets.QTreeWidgetItem):
    """Sort numeric columns by value, with unavailable values last."""

    def __lt__(self, other: QtWidgets.QTreeWidgetItem) -> bool:
        column = self.treeWidget().sortColumn() if self.treeWidget() else 0
        mine, theirs = self.data(column, _SORT), other.data(column, _SORT)
        if isinstance(mine, float) and isinstance(theirs, float):
            return mine < theirs
        return str(mine) < str(theirs)


class Navigator(QtWidgets.QWidget):
    """Peaks for editing, clusters for review; hover, selection and drafts stay independent."""

    selected = QtCore.Signal(str)
    hovered = QtCore.Signal(object)
    edit_requested = QtCore.Signal()
    next_warning = QtCore.Signal()

    def __init__(self, x_title: str, y_title: str, decimals: tuple[int, int] = (4, 3)) -> None:
        super().__init__()
        self.setObjectName("navigator")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumWidth(260)
        self.decimals = decimals
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, themes.SPACE_S, 0, themes.SPACE_S)
        layout.setSpacing(themes.SPACE_S)
        inset = (themes.SPACE_M, 0, themes.SPACE_M, 0)
        tabs = QtWidgets.QHBoxLayout()
        tabs.setContentsMargins(*inset)
        tabs.setSpacing(themes.SPACE_L)
        self.tab_buttons = []
        for index, title in enumerate(("Peaks", "Clusters")):
            button = QtWidgets.QPushButton(title)
            button.setObjectName("tab")
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.clicked.connect(lambda _checked=False, page=index: self.show_page(page))
            self.tab_buttons.append(button)
            tabs.addWidget(button)
        tabs.addStretch()
        layout.addLayout(tabs)
        self.pages = QtWidgets.QStackedWidget()
        layout.addWidget(self.pages, 1)
        self.pages.addWidget(self._build_peaks(x_title, y_title, inset))
        self.pages.addWidget(self._build_clusters(inset))
        self._items: dict[str, QtWidgets.QListWidgetItem] = {}
        self._clusters: dict[str, _ClusterItem] = {}
        self._run: tuple | None = None  # run, currentness and review shown
        self._selected: str | None = None
        self._hovered: str | None = None
        self._wash = QtGui.QColor(themes.LIGHT.hover)
        self._warning = QtGui.QColor(themes.LIGHT.warning)
        self._subdued = QtGui.QColor(themes.LIGHT.secondary)
        self._reviews: list[ClusterReview] = []
        self.rules = HighlightRules()  # display thresholds, named in the filter
        self.tab_buttons[_CLUSTERS].setEnabled(False)
        self.show_page(_PEAKS)

    def _build_peaks(self, x_title: str, y_title: str, inset) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(themes.SPACE_S)
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
        return page

    def _build_clusters(self, inset) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(themes.SPACE_S)
        self.run_label = QtWidgets.QLabel()
        self.run_label.setObjectName("caption")
        self.run_label.setContentsMargins(*inset)
        layout.addWidget(self.run_label)
        self.filter = QtWidgets.QComboBox()
        self.filter.setAccessibleName("Show clusters")
        self.filter.currentIndexChanged.connect(self._filter_clusters)
        filter_row = QtWidgets.QHBoxLayout()
        filter_row.setContentsMargins(*inset)
        filter_row.addWidget(self.filter)
        layout.addLayout(filter_row)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["", "Cluster", "χ²ᵣ", "|r|max", "|ρ|"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setSortingEnabled(True)
        self.tree.setMouseTracking(True)
        self.tree.setAccessibleName(
            "Clusters of the active run; warning or highlight, members, reduced chi-squared, "
            "largest recorded residual, largest amplitude correlation"
        )
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column, width in ((0, 24), (2, 48), (3, 52), (4, 42)):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.tree.setColumnWidth(column, width)
        for column in (2, 3, 4):
            self.tree.headerItem().setTextAlignment(
                column, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
        self.tree.headerItem().setToolTip(0, "▲ numerical warning · ◇ highlight")
        self.tree.headerItem().setToolTip(2, "Reduced χ² on corrected fitting samples")
        self.tree.headerItem().setToolTip(
            3, "Largest |recorded residual| on the cluster's fitting samples, in σ"
        )
        self.tree.headerItem().setToolTip(
            4, "Largest |amplitude correlation| between members, conditional on lineshapes"
        )
        self.tree.currentItemChanged.connect(self._cluster_changed)
        self.tree.itemEntered.connect(lambda item, _column: self.hovered.emit(item.data(1, _ID)))
        self.tree.viewport().installEventFilter(self)
        layout.addWidget(self.tree, 1)
        self.cluster_count = QtWidgets.QLabel()
        self.cluster_count.setObjectName("caption")
        self.cluster_count.setContentsMargins(*inset)
        layout.addWidget(self.cluster_count)
        self.tree.installEventFilter(self)
        return page

    def show_page(self, page: int) -> None:
        """Switch between peak and cluster lists without changing the selection."""
        self.tab_buttons[page].setChecked(True)
        self.pages.setCurrentIndex(page)

    def apply_theme(self, theme: themes.Theme) -> None:
        """Hovered-marker rows use the theme's hover wash; warnings use its amber."""
        self._wash = QtGui.QColor(theme.hover)
        self._warning = QtGui.QColor(theme.warning)
        self._subdued = QtGui.QColor(theme.secondary)
        delegate = self.list.itemDelegate()
        if isinstance(delegate, PeakRowDelegate):
            delegate.secondary = QtGui.QColor(theme.secondary)
        for item in self._clusters.values():
            item.setForeground(
                0, QtGui.QBrush(self._warning if item.text(0) == "▲" else self._subdued)
            )
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
        self.tab_buttons[_PEAKS].setText(f"Peaks  {len(peaks)}")
        self._filter(self.search.text())

    def set_result(
        self,
        result: DatasetFitResult | None,
        reviews: list[ClusterReview],
        *,
        current: bool,
    ) -> None:
        """List the active run's clusters; the first result switches to this tab."""
        key = (result.run_id, current, tuple(reviews)) if result is not None else None
        if key == self._run:
            return
        first = self._run is None and key is not None
        self._run = key
        self._reviews = reviews
        self._clusters.clear()
        with QtCore.QSignalBlocker(self.tree):
            self.tree.clear()
            if result is not None:
                self.tree.setSortingEnabled(False)
                for review in reviews:
                    self._add_cluster(review)
                self.tree.setSortingEnabled(True)
                if first:
                    self.tree.sortByColumn(2, QtCore.Qt.SortOrder.DescendingOrder)
        self.tab_buttons[_CLUSTERS].setEnabled(result is not None)
        if result is None:
            self.tab_buttons[_CLUSTERS].setText("Clusters")
            self.show_page(_PEAKS)
            return
        self.tab_buttons[_CLUSTERS].setText(f"Clusters  {len(reviews)}")
        self.run_label.setText(
            f"Run {result.run_id[:8]} · {'Current' if current else 'Outdated: refit to update'}"
        )
        self._fill_filter()
        self._mark_selected()
        if first:
            self.show_page(_CLUSTERS)

    def _fill_filter(self) -> None:
        """One choice per warning type and highlight present, with counts."""
        reviews = self._reviews
        chosen = self.filter.currentData()
        options: list[tuple[str, object]] = [(f"All clusters ({len(reviews)})", None)]
        families = (
            ("Any warning", "warning", WARNING_KINDS, lambda r: [w.kind for w in r.warnings]),
            (
                "Any highlight",
                "highlight",
                HIGHLIGHT_KINDS,
                lambda r: [h.kind for h in r.highlights],
            ),
        )
        for title, family, kinds, found in families:
            count = sum(bool(found(r)) for r in reviews)
            options.append((f"{title} ({count})", (family, None)))
            for kind in kinds:
                count = sum(kind in found(r) for r in reviews)
                if count:
                    options.append((f"   {self._kind_title(kind)} ({count})", (family, kind)))
        with QtCore.QSignalBlocker(self.filter):
            self.filter.clear()
            for text, data in options:
                self.filter.addItem(text, data)
            index = self.filter.findData(chosen)
            self.filter.setCurrentIndex(max(index, 0))
        self._filter_clusters()

    def _kind_title(self, kind: str) -> str:
        """Name a highlight by its rule, so the threshold is always visible."""
        rules = self.rules
        return {
            "Amplitude correlation": f"|ρ| ≥ {rules.correlation:g}",
            "Position shift": f"Shift > {rules.shift:g} × LW",
            "Linewidth outlier": (
                f"LW outside {rules.linewidth_low:g}–{rules.linewidth_high:g} × median"
            ),
        }.get(kind, kind)

    def _filter_clusters(self) -> None:
        choice = self.filter.currentData()
        shown = 0
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item is None:
                continue
            review = item.data(0, _ID)
            if choice is None:
                visible = True
            else:
                family, kind = choice
                kinds = (
                    [w.kind for w in review.warnings]
                    if family == "warning"
                    else [h.kind for h in review.highlights]
                )
                visible = bool(kinds) if kind is None else kind in kinds
            item.setHidden(not visible)
            shown += visible
        total = self.tree.topLevelItemCount()
        warned = sum(bool(r.warnings) for r in self._reviews)
        prefix = f"{shown} of {total} clusters" if shown != total else f"{total} clusters"
        self.cluster_count.setText(f"{prefix} · ▲ {warned} · N: next ▲")
        self.cluster_count.setToolTip(
            f"{warned} clusters with a numerical warning; N in the list selects the next one"
        )

    def next_warned(self, selected: str | None) -> str | None:
        """First member of the next shown cluster with a numerical warning, in list order."""
        rows = [
            item
            for index in range(self.tree.topLevelItemCount())
            if (item := self.tree.topLevelItem(index)) is not None
            and not item.isHidden()
            and item.data(0, _ID).warnings
        ]
        if not rows:
            return None
        current = next((i for i, row in enumerate(rows) if selected in row.data(2, _ID)), -1)
        return rows[(current + 1) % len(rows)].data(1, _ID)

    def _add_cluster(self, review: ClusterReview) -> None:
        item = _ClusterItem()
        names = review.names
        label = names[0] + (f"  +{len(names) - 1}" if len(names) > 1 else "")
        warned = bool(review.warnings)
        item.setText(0, "▲" if warned else "◇" if review.highlights else "")
        tips = [f"▲ {w.kind}" for w in review.warnings] + [f"◇ {h.rule}" for h in review.highlights]
        item.setToolTip(0, "\n".join(tips) or "No numerical warning or highlight")
        item.setForeground(0, QtGui.QBrush(self._warning if warned else self._subdued))
        item.setText(1, label)
        item.setToolTip(1, " · ".join(names))
        chi2, largest, rho = review.chi2, review.largest_residual, review.correlation
        item.setText(2, f"{chi2:.2f}" if math.isfinite(chi2) else "–")
        item.setText(3, f"{largest:.1f}" if math.isfinite(largest) else "–")
        item.setText(4, "–" if review.single else f"{rho:.2f}" if math.isfinite(rho) else "n/a")
        if not review.single and not math.isfinite(rho):
            item.setToolTip(4, "Amplitude covariance unavailable for this cluster")
        for column in (2, 3, 4):
            item.setTextAlignment(
                column, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
        item.setData(0, _SORT, f"{2 if warned else 1 if review.highlights else 0}")
        item.setData(1, _SORT, " ".join(names))
        for column, value in ((2, chi2), (3, largest), (4, rho)):
            item.setData(column, _SORT, value if math.isfinite(value) else -1.0)
        item.setData(0, _ID, review)
        item.setData(1, _ID, review.ids[0] if review.ids else None)
        item.setData(2, _ID, list(review.ids))
        self.tree.addTopLevelItem(item)
        for identity in review.ids:
            self._clusters[identity] = item

    def select(self, peak: WorkingPeak | None) -> None:
        """Synchronize a committed selection without opening or clearing the navigator."""
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
        self._mark_selected()
        self._filter(self.search.text())

    def _mark_selected(self) -> None:
        with QtCore.QSignalBlocker(self.tree):
            item = self._clusters.get(self._selected or "")
            if item is None:
                self.tree.clearSelection()
                self.tree.setCurrentIndex(QtCore.QModelIndex())
            elif self.tree.currentItem() is not item:
                self.tree.setCurrentItem(item)
                self.tree.scrollToItem(item)

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
        self.show_page(_PEAKS)
        if self._selected in self._items:
            self.list.scrollToItem(self._items[self._selected])

    def _selection_changed(self, item, _previous) -> None:
        if item is not None:
            self.selected.emit(item.data(256))
        self._filter(self.search.text())

    def _cluster_changed(self, item, _previous) -> None:
        if item is None:
            return
        members = item.data(2, _ID) or []
        # Keep the selected peak when it belongs to the chosen cluster.
        target = self._selected if self._selected in members else item.data(1, _ID)
        if target:
            self.selected.emit(target)

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
        if (
            watched is getattr(self, "tree", None)  # filters run during construction
            and event.type() == QtCore.QEvent.Type.KeyPress
            and event.key() == QtCore.Qt.Key.Key_N
            and not event.modifiers()
        ):
            self.next_warning.emit()
            return True
        return super().eventFilter(watched, event)
