"""The Profiles view: every recorded cluster's amplitude profiles, one tile per cluster.

A tile overlays all members of a cluster on one vertical scale with zero visible, so
cross-contamination between correlated members, opposite signs and large/small
splits show at a glance. Tiles follow the navigator's order and filter; choosing one
selects its cluster, and activating it opens the cluster in focus. Values are drawn
exactly as the run recorded them, with no recomputation.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.engine.diagnostics.identifiability import Cause
from peakfit.gui.editor.profile import PeakProfile, framed_range, peak_profile
from peakfit.gui.theme import FONT_CAPTION, FONT_SMALL, LIGHT, SPACE_M, SPACE_S, Theme

if TYPE_CHECKING:
    from peakfit.gui.dataset_fit import DatasetFitResult
    from peakfit.gui.editor.review import ClusterReview

_TILE = QtCore.QSize(240, 184)  # preferred tile; columns stretch to fill the width
_MIN_WIDTH = 200
_PAD = 8
_ROLE = QtCore.Qt.ItemDataRole.UserRole


@dataclass(frozen=True, slots=True)
class Tile:
    """One recorded cluster: its review and member profiles, or why it has none."""

    review: ClusterReview
    profile: PeakProfile | None
    reason: str = ""


def cluster_tiles(
    result: DatasetFitResult,
    reviews: list[ClusterReview],
    plane_values: tuple[float, ...] | None,
) -> dict[int, Tile]:
    """Each cluster's tile by cluster ID, members in the cluster's own order."""
    tiles = {}
    for review in reviews:
        profile = peak_profile(result, review.ids[0], plane_values) if review.ids else None
        if isinstance(profile, str) or profile is None:
            tiles[review.cluster_id] = Tile(review, None, profile or "No members")
        else:
            tiles[review.cluster_id] = Tile(review, profile)
    return tiles


def withheld(profile: PeakProfile) -> list[str]:
    """Members whose individual amplitudes the tile does not draw, and why."""
    return [
        f"{s.assignment} not "
        + ("identifiable" if s.identifiability.cause is Cause.NULL_SPACE else "established")
        for s in profile.series
        if not s.identifiability.determined
    ]


def _measures(review: ClusterReview) -> str:
    parts = [f"χ²ᵣ {review.chi2:.2f}" if math.isfinite(review.chi2) else "χ²ᵣ –"]
    if math.isfinite(review.largest_residual):
        parts.append(f"|r|max {review.largest_residual:.1f}")
    if not review.single:
        rho = review.correlation
        parts.append(f"|ρ| {rho:.2f}" if math.isfinite(rho) else "|ρ| n/a")
    return " · ".join(parts)


def _describe(tile: Tile) -> str:
    """The tile in words: tooltip and accessible text."""
    review = tile.review
    lines = [" · ".join(review.names), _measures(review)]
    lines += [f"▲ {w.kind}: {w.detail}" if w.detail else f"▲ {w.kind}" for w in review.warnings]
    lines += [f"◇ {h.rule}" for h in review.highlights]
    if tile.profile is None:
        lines.append(tile.reason)
    else:
        lines += withheld(tile.profile)
        lines.append("Signed amplitudes, all members on one scale · double-click to focus")
    return "\n".join(lines)


class _Tiles(QtCore.QAbstractListModel):
    """Tiles in the navigator's order; the view only reads them."""

    def __init__(self) -> None:
        super().__init__()
        self.tiles: list[Tile] = []

    def rowCount(self, parent=None) -> int:  # noqa: N802
        return 0 if parent is not None and parent.isValid() else len(self.tiles)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        tile = self.tiles[index.row()]
        if role == _ROLE:
            return tile
        if role in (QtCore.Qt.ItemDataRole.ToolTipRole, QtCore.Qt.ItemDataRole.AccessibleTextRole):
            return _describe(tile)
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return tile.review.names[0] if tile.review.names else ""
        return None

    def set_tiles(self, tiles: list[Tile]) -> None:
        self.beginResetModel()
        self.tiles = tiles
        self.endResetModel()


class _TileDelegate(QtWidgets.QStyledItemDelegate):
    """Paints one cluster tile; the view holds the theme, plane and selection."""

    def __init__(self, view: ProfilesView) -> None:
        super().__init__(view)
        self.owner = view

    def sizeHint(self, _option, _index) -> QtCore.QSize:  # noqa: N802
        """Every tile has the view's current grid size."""
        return self.owner.tile_size

    def paint(self, painter: QtGui.QPainter, option, index) -> None:
        tile = index.data(_ROLE)
        if not isinstance(tile, Tile):
            return
        theme = self.owner.theme
        rect = QtCore.QRectF(option.rect).adjusted(3, 3, -3, -3)
        selected = bool(option.state & QtWidgets.QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QtWidgets.QStyle.StateFlag.State_MouseOver)
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(
            QtGui.QPen(
                QtGui.QColor(theme.accent if selected else theme.divider), 2 if selected else 1
            )
        )
        painter.setBrush(QtGui.QColor(theme.hover if hovered and not selected else theme.plot))
        painter.drawRoundedRect(rect, 5, 5)
        inner = rect.adjusted(_PAD, _PAD - 2, -_PAD, -_PAD + 2)
        top = self._header(painter, inner, tile, theme)
        bottom = self._legend(painter, inner, tile, theme)
        plot = QtCore.QRectF(inner.left(), top + 4, inner.width(), bottom - top - 8)
        if tile.profile is None:
            self._text(painter, plot, tile.reason, theme.secondary, FONT_CAPTION, center=True)
        else:
            self._plot(painter, plot, tile, theme)
        painter.restore()

    def _font(self, size: int, *, bold: bool = False) -> QtGui.QFont:
        font = QtGui.QFont(self.owner.font())
        font.setPixelSize(size)
        font.setBold(bold)
        return font

    def _text(self, painter, rect, text, color, size, *, center=False, bold=False) -> None:
        painter.setFont(self._font(size, bold=bold))
        painter.setPen(QtGui.QColor(color))
        metrics = QtGui.QFontMetrics(painter.font())
        flags = (
            QtCore.Qt.AlignmentFlag.AlignCenter
            if center
            else QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        if center:
            painter.drawText(rect, flags | QtCore.Qt.TextFlag.TextWordWrap, text)
        else:
            painter.drawText(
                rect,
                flags,
                metrics.elidedText(text, QtCore.Qt.TextElideMode.ElideRight, int(rect.width())),
            )

    def _header(self, painter, inner: QtCore.QRectF, tile: Tile, theme: Theme) -> float:
        """Mark, first member and count, then the cluster's measures; returns its bottom."""
        review = tile.review
        height = FONT_SMALL + 6
        row = QtCore.QRectF(inner.left(), inner.top(), inner.width(), height)
        x = row.left()
        if review.warnings or review.highlights:
            mark = "▲" if review.warnings else "◇"
            color = theme.warning if review.warnings else theme.secondary
            self._text(painter, row, mark, color, FONT_SMALL)
            x += QtGui.QFontMetrics(self._font(FONT_SMALL)).horizontalAdvance(mark) + 5
        count = len(review.names)
        name = (review.names[0] if review.names else "") + (f"  +{count - 1}" if count > 1 else "")
        mine = self.owner.selected in review.ids
        measures = _measures(review)
        width = QtGui.QFontMetrics(self._font(FONT_CAPTION)).horizontalAdvance(measures)
        right = QtCore.QRectF(row.right() - width, row.top(), width, height)
        self._text(painter, right, measures, theme.secondary, FONT_CAPTION)
        left = QtCore.QRectF(x, row.top(), max(0.0, right.left() - x - 6), height)
        self._text(painter, left, name, theme.text, FONT_SMALL, bold=mine)
        return row.bottom()

    def _legend(self, painter, inner: QtCore.QRectF, tile: Tile, theme: Theme) -> float:
        """Members in their profile colours, then any withheld ones; returns its top."""
        height = FONT_CAPTION + 5
        profile = tile.profile
        notes = withheld(profile) if profile is not None else []
        rows = 2 if notes else 1
        top = inner.bottom() - rows * height
        if profile is None:
            return inner.bottom()
        font = self._font(FONT_CAPTION)
        metrics = QtGui.QFontMetrics(font)
        painter.setFont(font)
        colors = theme.members
        x, limit = inner.left(), inner.right()
        for index, series in enumerate(profile.series):
            label = series.assignment
            need = 10 + metrics.horizontalAdvance(label) + 10
            rest = len(profile.series) - index
            if x + need > limit and rest > 0:
                painter.setPen(QtGui.QColor(theme.secondary))
                painter.drawText(QtCore.QPointF(x, top + height - 4), f"+{rest}")
                break
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(colors[index % len(colors)]))
            painter.drawEllipse(QtCore.QPointF(x + 3.5, top + height / 2), 3.5, 3.5)
            shown = series.identifiability.determined
            painter.setPen(QtGui.QColor(theme.text if shown else theme.faint))
            painter.drawText(QtCore.QPointF(x + 10, top + height - 4), label)
            x += need
        if notes:
            row = QtCore.QRectF(inner.left(), top + height, inner.width(), height)
            self._text(painter, row, "▲ " + " · ".join(notes), theme.warning, FONT_CAPTION)
        return top

    def _plot(self, painter, area: QtCore.QRectF, tile: Tile, theme: Theme) -> None:
        """All determined members on one vertical scale that always includes zero."""
        profile = tile.profile
        assert profile is not None
        x = profile.x
        frame = framed_range(x)
        visible = (
            np.ones(x.size, dtype=bool) if frame is None else (x >= frame[0]) & (x <= frame[1])
        )
        drawn = [
            (index, s.amplitudes)
            for index, s in enumerate(profile.series)
            if s.identifiability.determined
        ]
        if not drawn or not visible.any():
            note = "No member's amplitude is determined"
            self._text(painter, area, note, theme.secondary, FONT_CAPTION, center=True)
            return
        values = np.concatenate([a[visible] for _i, a in drawn])
        finite = values[np.isfinite(values)]
        low = min(0.0, float(finite.min())) if finite.size else 0.0
        high = max(0.0, float(finite.max())) if finite.size else 1.0
        span = (high - low) or 1.0
        low, high = low - 0.06 * span, high + 0.06 * span
        x0, x1 = float(x[visible].min()), float(x[visible].max())
        width = (x1 - x0) or 1.0

        def to_point(xv: float, yv: float) -> QtCore.QPointF:
            return QtCore.QPointF(
                area.left() + (xv - x0) / width * area.width(),
                area.bottom() - (yv - low) / (high - low) * area.height(),
            )

        zero = to_point(x0, 0.0).y()
        painter.setPen(QtGui.QPen(QtGui.QColor(theme.border), 1))
        painter.drawLine(QtCore.QPointF(area.left(), zero), QtCore.QPointF(area.right(), zero))
        column = np.flatnonzero(profile.source_planes == self.owner.plane)
        if column.size and visible[column[0]]:
            pen = QtGui.QPen(QtGui.QColor(theme.pin), 1, QtCore.Qt.PenStyle.DashLine)
            painter.setPen(pen)
            px = to_point(float(x[column[0]]), 0.0).x()
            painter.drawLine(QtCore.QPointF(px, area.top()), QtCore.QPointF(px, area.bottom()))
        colors = theme.members
        selected = self.owner.selected
        # Series follow the cluster's member order, as the review's IDs do.
        ids = dict(enumerate(tile.review.ids))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        # The selected peak is drawn last and slightly larger, so it stays on top.
        order = sorted(drawn, key=lambda item: ids.get(item[0]) == selected)
        for index, amplitudes in order:
            radius = 2.4 if ids.get(index) == selected else 1.8
            painter.setBrush(QtGui.QColor(colors[index % len(colors)]))
            for xv, yv in zip(x[visible], amplitudes[visible], strict=True):
                if math.isfinite(yv):
                    painter.drawEllipse(to_point(float(xv), float(yv)), radius, radius)


class ProfilesView(QtWidgets.QWidget):
    """Scan every cluster's profiles; choose one to review it, activate it to focus it."""

    cluster_chosen = QtCore.Signal(str)  # a member of the chosen cluster
    focus_requested = QtCore.Signal(str)  # a member of the cluster to focus

    def __init__(self) -> None:
        super().__init__()
        self.theme = LIGHT
        self.selected: str | None = None
        self.plane = 0
        self.tile_size = QtCore.QSize(_TILE)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(SPACE_M, SPACE_S, SPACE_M, 0)
        layout.setSpacing(SPACE_S)
        self.caption = QtWidgets.QLabel()
        self.caption.setObjectName("caption")
        layout.addWidget(self.caption)
        self.model = _Tiles()
        self.list = QtWidgets.QListView()
        self.list.setObjectName("profiles")
        self.list.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.list.setMovement(QtWidgets.QListView.Movement.Static)
        self.list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.list.setUniformItemSizes(True)
        self.list.setWrapping(True)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.list.setMouseTracking(True)
        self.list.setAccessibleName("Amplitude profiles of every cluster, in the navigator's order")
        self.list.setModel(self.model)
        self.list.viewport().installEventFilter(self)
        self.list.setItemDelegate(_TileDelegate(self))
        self.list.selectionModel().currentChanged.connect(self._current_changed)
        self.list.activated.connect(self._activated)
        self.list.doubleClicked.connect(self._activated)
        layout.addWidget(self.list, 1)
        self.empty = QtWidgets.QLabel()
        self.empty.setObjectName("muted")
        self.empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty.setWordWrap(True)
        self.empty.hide()
        layout.addWidget(self.empty, 1)

    def apply_theme(self, theme: Theme) -> None:
        """Repaint tiles in the new appearance."""
        self.theme = theme
        self.list.viewport().update()

    def set_tiles(self, tiles: list[Tile], caption: str, empty: str) -> None:
        """Show these tiles in this order; an empty list shows why there are none."""
        same = len(tiles) == len(self.model.tiles) and all(
            a is b for a, b in zip(tiles, self.model.tiles, strict=True)
        )
        if not same:  # a reset would lose the scroll position on every selection
            self.model.set_tiles(tiles)
        self.caption.setText(caption)
        self.empty.setText(empty)
        self.list.setVisible(bool(tiles))
        self.empty.setVisible(not tiles)
        self._fit_columns()
        self.select(self.selected)

    def select(self, identity: str | None) -> None:
        """Mark the tile of the selected peak's cluster, without emitting a choice."""
        self.selected = identity
        row = next((r for r, t in enumerate(self.model.tiles) if identity in t.review.ids), None)
        selection = self.list.selectionModel()
        with QtCore.QSignalBlocker(selection):
            if row is None:
                selection.clear()
            else:
                index = self.model.index(row)
                selection.setCurrentIndex(
                    index, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
                )
                self.list.scrollTo(index)
        self.list.viewport().update()

    def set_plane(self, plane: int) -> None:
        """Move the current-plane line on every tile (zero-based source plane)."""
        self.plane = plane
        self.list.viewport().update()

    def tile_ids(self) -> list[tuple[str, ...]]:
        """Member IDs of each tile, in display order."""
        return [tile.review.ids for tile in self.model.tiles]

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        """Refit the columns whenever the tiles' viewport changes width."""
        if watched is self.list.viewport() and event.type() == QtCore.QEvent.Type.Resize:
            self._fit_columns()
        return super().eventFilter(watched, event)

    def _fit_columns(self) -> None:
        """Columns fill the width: as many preferred-size tiles as fit, then stretched."""
        width = self.list.viewport().width() - 2
        columns = max(1, width // _TILE.width())
        size = QtCore.QSize(max(_MIN_WIDTH, width // columns), _TILE.height())
        if size != self.tile_size or self.list.gridSize() != size:
            self.tile_size = size
            self.list.setGridSize(size)

    def _target(self, row: int) -> str | None:
        ids = self.model.tiles[row].review.ids
        # Keep the selected peak when it belongs to the chosen cluster.
        return self.selected if self.selected in ids else (ids[0] if ids else None)

    def _current_changed(self, current, _previous) -> None:
        if current.isValid() and (target := self._target(current.row())):
            self.cluster_chosen.emit(target)

    def _activated(self, index) -> None:
        if index.isValid() and (target := self._target(index.row())):
            self.focus_requested.emit(target)
