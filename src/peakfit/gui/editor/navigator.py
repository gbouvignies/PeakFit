"""Left navigator: browse working peaks, or the clusters of the active run."""

import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui import theme as themes

if TYPE_CHECKING:
    from peakfit.gui.dataset_fit import DatasetFitResult
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


def cluster_rows(result: DatasetFitResult) -> list[dict]:
    """One row per recorded cluster, with values read from the completed outcome."""
    identities = {i.engine_name: i for i in result.inputs.identities}
    rows = []
    for cluster in result.outcome.clusters:
        members = [identities[name] for name in cluster.peak_names if name in identities]
        evaluation = cluster.analytical_evaluation
        reasons = []
        if not cluster.usable:
            reasons.append(f"Unusable: {cluster.unusable_reason or 'no reason recorded'}")
        elif not cluster.optimizer_provenance.converged:
            reasons.append("Not converged")
        if evaluation is not None and evaluation.uncertainty_warning:
            reasons.append("Uncertainty unreliable")
        chi2 = float(evaluation.statistics.reduced_chi_squared) if evaluation else math.nan
        rho = math.nan
        if evaluation is not None and len(members) > 1:
            covariance = np.asarray(evaluation.amplitude_covariance, dtype=float)
            if np.all(np.isfinite(covariance)):
                scale = np.sqrt(np.diag(covariance))
                with np.errstate(divide="ignore", invalid="ignore"):
                    correlation = covariance / np.outer(scale, scale)
                np.fill_diagonal(correlation, 0.0)
                rho = float(np.nanmax(np.abs(correlation)))
        rows.append(
            {
                "ids": [m.working_id for m in members],
                "names": [m.assignment for m in members],
                "reasons": reasons,
                "chi2": chi2,
                "rho": rho,
                "single": len(members) == 1,
            }
        )
    return rows


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
        self._run: tuple[str, bool] | None = None
        self._selected: str | None = None
        self._hovered: str | None = None
        self._wash = QtGui.QColor(themes.LIGHT.hover)
        self._warning = QtGui.QColor(themes.LIGHT.warning)
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
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["", "Cluster", "χ²ᵣ", "|ρ|"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setSortingEnabled(True)
        self.tree.setMouseTracking(True)
        self.tree.setAccessibleName(
            "Clusters of the active run; engine warning, members, reduced chi-squared, "
            "largest amplitude correlation"
        )
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column, width in ((0, 26), (2, 56), (3, 50)):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.tree.setColumnWidth(column, width)
        for column in (2, 3):
            self.tree.headerItem().setTextAlignment(
                column, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
        self.tree.headerItem().setToolTip(2, "Reduced χ² on corrected fitting samples")
        self.tree.headerItem().setToolTip(
            3, "Largest |amplitude correlation| between members, conditional on lineshapes"
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
        delegate = self.list.itemDelegate()
        if isinstance(delegate, PeakRowDelegate):
            delegate.secondary = QtGui.QColor(theme.secondary)
        for item in self._clusters.values():
            item.setForeground(0, QtGui.QBrush(self._warning))
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

    def set_result(self, result: DatasetFitResult | None, *, current: bool) -> None:
        """List the active run's clusters; the first result switches to this tab."""
        key = (result.run_id, current) if result is not None else None
        if key == self._run:
            return
        first = self._run is None and key is not None
        self._run = key
        self._clusters.clear()
        with QtCore.QSignalBlocker(self.tree):
            self.tree.clear()
            if result is not None:
                self.tree.setSortingEnabled(False)
                for row in cluster_rows(result):
                    self._add_cluster(row)
                self.tree.setSortingEnabled(True)
                self.tree.sortByColumn(2, QtCore.Qt.SortOrder.DescendingOrder)
        self.tab_buttons[_CLUSTERS].setEnabled(result is not None)
        if result is None:
            self.tab_buttons[_CLUSTERS].setText("Clusters")
            self.show_page(_PEAKS)
            return
        count = self.tree.topLevelItemCount()
        # Several members map to one row; count rows, not peaks.
        rows = (self.tree.topLevelItem(i) for i in range(count))
        flagged = sum(bool(row.text(0)) for row in rows if row is not None)
        self.tab_buttons[_CLUSTERS].setText(f"Clusters  {count}")
        self.run_label.setText(
            f"Run {result.run_id[:8]} · {'Current' if current else 'Outdated: refit to update'}"
        )
        self.cluster_count.setText(
            f"{count} clusters · {flagged} with ▲ engine warnings · N: next warning"
        )
        self._mark_selected()
        if first:
            self.show_page(_CLUSTERS)

    def _add_cluster(self, row: dict) -> None:
        item = _ClusterItem()
        names = row["names"]
        label = names[0] + (f"  +{len(names) - 1}" if len(names) > 1 else "")
        item.setText(0, "▲" if row["reasons"] else "")
        item.setToolTip(0, "\n".join(row["reasons"]) or "No engine warning")
        item.setForeground(0, QtGui.QBrush(self._warning))
        item.setText(1, label)
        item.setToolTip(1, " · ".join(names))
        chi2, rho = row["chi2"], row["rho"]
        item.setText(2, f"{chi2:.2f}" if math.isfinite(chi2) else "–")
        item.setText(3, "–" if row["single"] else f"{rho:.2f}" if math.isfinite(rho) else "n/a")
        if not row["single"] and not math.isfinite(rho):
            item.setToolTip(3, "Amplitude covariance unavailable for this cluster")
        for column in (2, 3):
            item.setTextAlignment(
                column, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
        item.setData(0, _SORT, str(bool(row["reasons"])))
        item.setData(1, _SORT, " ".join(names))
        item.setData(2, _SORT, chi2 if math.isfinite(chi2) else -1.0)
        item.setData(3, _SORT, rho if math.isfinite(rho) else -1.0)
        item.setData(1, _ID, row["ids"][0] if row["ids"] else None)
        item.setData(2, _ID, row["ids"])
        self.tree.addTopLevelItem(item)
        for identity in row["ids"]:
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
