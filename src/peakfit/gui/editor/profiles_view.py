"""The Profiles view: every recorded cluster's amplitude profiles, one tile per cluster.

A tile overlays all members of a cluster on one vertical scale with zero visible, so
cross-contamination between correlated members, opposite signs and large/small
splits show at a glance. Tiles follow the navigator's order and filter; choosing one
selects its cluster, and activating it opens the cluster in focus. Values are drawn
exactly as the run recorded them; a reference ratio is the shared display transform
of ``engine.scientific.profiles``, never a refit.
"""

import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.engine.diagnostics.identifiability import Cause
from peakfit.engine.scientific.profiles import Unavailable, normalize
from peakfit.gui.editor.profile import (
    PeakProfile,
    ask_reference,
    framed_range,
    peak_profile,
    quantity_menu,
    reference_text,
)
from peakfit.gui.theme import FONT_CAPTION, FONT_SMALL, LIGHT, SPACE_M, SPACE_S, Theme

if TYPE_CHECKING:
    from peakfit.gui.dataset_fit import DatasetFitResult
    from peakfit.gui.editor.review import ClusterReview

_TILE = QtCore.QSize(240, 184)  # preferred tile; columns stretch to fill the width
_MIN_WIDTH = 200
_PAD = 8
_ROLE = QtCore.Qt.ItemDataRole.UserRole
# PDF pages: A4 landscape at 96 dpi, four columns by three rows of tiles.
_PDF_DPI, _PDF_COLUMNS, _PDF_ROWS = 96, 4, 3


@dataclass(frozen=True, slots=True)
class Tile:
    """One recorded cluster: its review and member profiles, or why it has none."""

    review: ClusterReview
    profile: PeakProfile | None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Shown:
    """What a tile draws: member rows on one scale, and what it could not draw."""

    rows: list[tuple[int, np.ndarray]] = field(default_factory=list)  # (member, values)
    notes: list[str] = field(default_factory=list)
    ratio: bool = False  # values are A / A_ref, not raw amplitudes


@dataclass(frozen=True, slots=True)
class Style:
    """Everything a tile's appearance depends on besides the tile itself."""

    theme: Theme
    font: QtGui.QFont
    plane: int = -1  # zero-based source plane marked on every tile; -1 for none
    selected: str | None = None
    reference: tuple[int, ...] = ()
    normalized: bool = False


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


def shown_rows(profile: PeakProfile, reference: tuple[int, ...], normalized: bool) -> Shown:
    """Determined members, raw or as ratios to the reference; never both on one scale.

    A member whose ratio is unavailable is left out and named. When no member can be
    normalized the tile falls back to raw amplitudes and says so, as the inspector's
    profile does; a withheld member is never normalized.
    """
    raw = [(i, s.amplitudes) for i, s in enumerate(profile.series) if s.identifiability.determined]
    notes = withheld(profile)
    if not (normalized and reference):
        return Shown(raw, notes)
    columns = np.flatnonzero(np.isin(profile.source_planes, reference))
    if columns.size != len(reference):
        return Shown(raw, [*notes, "Raw: a reference plane is not in this run"])
    rows, unavailable = [], []
    for index, series in enumerate(profile.series):
        if not series.identifiability.determined:
            continue
        ratio = normalize(series.amplitudes, series.errors, columns)
        if isinstance(ratio, Unavailable):
            unavailable.append(f"{series.assignment}: {ratio.reason}")
        else:
            rows.append((index, ratio.values))
    if not rows:
        reason = unavailable[0].split(": ", 1)[-1] if unavailable else "no member to normalize"
        return Shown(raw, [*notes, f"Raw: {reason}"])
    return Shown(rows, [*notes, *(f"Not normalized: {u}" for u in unavailable)], ratio=True)


def _measures(review: ClusterReview) -> str:
    parts = [f"χ²ᵣ {review.chi2:.2f}" if math.isfinite(review.chi2) else "χ²ᵣ –"]
    if math.isfinite(review.largest_residual):
        parts.append(f"|r|max {review.largest_residual:.1f}")
    if not review.single:
        rho = review.correlation
        parts.append(f"|ρ| {rho:.2f}" if math.isfinite(rho) else "|ρ| n/a")
    return " · ".join(parts)


def _describe(tile: Tile, reference: tuple[int, ...], normalized: bool) -> str:
    """The tile in words: tooltip and accessible text."""
    review = tile.review
    lines = [" · ".join(review.names), _measures(review)]
    lines += [f"▲ {w.kind}: {w.detail}" if w.detail else f"▲ {w.kind}" for w in review.warnings]
    lines += [f"◇ {h.rule}" for h in review.highlights]
    if tile.profile is None:
        lines.append(tile.reason)
    else:
        shown = shown_rows(tile.profile, reference, normalized)
        lines += shown.notes
        quantity = (
            f"A / A_ref ({reference_text(reference)})" if shown.ratio else "Signed amplitudes"
        )
        lines.append(f"{quantity}, all members on one scale · double-click to focus")
    return "\n".join(lines)


def _font(base: QtGui.QFont, size: int, *, bold: bool = False) -> QtGui.QFont:
    font = QtGui.QFont(base)
    font.setPixelSize(size)
    font.setBold(bold)
    return font


def _text(painter, rect, text, color, font, *, center=False) -> None:
    painter.setFont(font)
    painter.setPen(QtGui.QColor(color))
    if center:
        painter.drawText(
            rect, QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap, text
        )
        return
    elided = QtGui.QFontMetrics(font).elidedText(
        text, QtCore.Qt.TextElideMode.ElideRight, int(rect.width())
    )
    painter.drawText(
        rect, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, elided
    )


def paint_tile(
    painter: QtGui.QPainter,
    rect: QtCore.QRectF,
    tile: Tile,
    style: Style,
    *,
    selected: bool = False,
    hovered: bool = False,
) -> None:
    """Draw one cluster tile into ``rect``: on screen and in exported figures alike."""
    theme = style.theme
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    edge = QtGui.QColor(theme.accent if selected else theme.divider)
    painter.setPen(QtGui.QPen(edge, 2 if selected else 1))
    painter.setBrush(QtGui.QColor(theme.hover if hovered and not selected else theme.plot))
    painter.drawRoundedRect(rect, 5, 5)
    inner = rect.adjusted(_PAD, _PAD - 2, -_PAD, -_PAD + 2)
    top = _header(painter, inner, tile, style)
    if tile.profile is None:
        area = QtCore.QRectF(inner.left(), top + 4, inner.width(), inner.bottom() - top - 4)
        caption = _font(style.font, FONT_CAPTION)
        _text(painter, area, tile.reason, theme.secondary, caption, center=True)
    else:
        shown = shown_rows(tile.profile, style.reference, style.normalized)
        bottom = _legend(painter, inner, tile.profile, shown, style)
        area = QtCore.QRectF(inner.left(), top + 4, inner.width(), bottom - top - 8)
        _plot(painter, area, tile, shown, style)
    painter.restore()


def _header(painter, inner: QtCore.QRectF, tile: Tile, style: Style) -> float:
    """Mark, first member and count, then the cluster's measures; returns its bottom."""
    theme, review = style.theme, tile.review
    small, caption = _font(style.font, FONT_SMALL), _font(style.font, FONT_CAPTION)
    height = FONT_SMALL + 6
    row = QtCore.QRectF(inner.left(), inner.top(), inner.width(), height)
    x = row.left()
    if review.warnings or review.highlights:
        mark = "▲" if review.warnings else "◇"
        _text(painter, row, mark, theme.warning if review.warnings else theme.secondary, small)
        x += QtGui.QFontMetrics(small).horizontalAdvance(mark) + 5
    count = len(review.names)
    name = (review.names[0] if review.names else "") + (f"  +{count - 1}" if count > 1 else "")
    measures = _measures(review)
    width = QtGui.QFontMetrics(caption).horizontalAdvance(measures)
    right = QtCore.QRectF(row.right() - width, row.top(), width, height)
    _text(painter, right, measures, theme.secondary, caption)
    left = QtCore.QRectF(x, row.top(), max(0.0, right.left() - x - 6), height)
    bold = _font(style.font, FONT_SMALL, bold=style.selected in review.ids)
    _text(painter, left, name, theme.text, bold)
    return row.bottom()


def _legend(painter, inner: QtCore.QRectF, profile: PeakProfile, shown: Shown, style) -> float:
    """Members in their profile colours, then what is not drawn; returns its top."""
    theme = style.theme
    height = FONT_CAPTION + 5
    top = inner.bottom() - (2 if shown.notes else 1) * height
    font = _font(style.font, FONT_CAPTION)
    metrics = QtGui.QFontMetrics(font)
    painter.setFont(font)
    colors = theme.members
    drawn = {index for index, _values in shown.rows}
    x, limit = inner.left(), inner.right()
    for index, series in enumerate(profile.series):
        label = series.assignment
        need = 10 + metrics.horizontalAdvance(label) + 10
        if x + need > limit:
            painter.setPen(QtGui.QColor(theme.secondary))
            painter.drawText(QtCore.QPointF(x, top + height - 4), f"+{len(profile.series) - index}")
            break
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(colors[index % len(colors)]))
        painter.drawEllipse(QtCore.QPointF(x + 3.5, top + height / 2), 3.5, 3.5)
        painter.setPen(QtGui.QColor(theme.text if index in drawn else theme.faint))
        painter.drawText(QtCore.QPointF(x + 10, top + height - 4), label)
        x += need
    if shown.notes:
        row = QtCore.QRectF(inner.left(), top + height, inner.width(), height)
        _text(painter, row, "▲ " + " · ".join(shown.notes), theme.warning, font)
    return top


def _plot(painter, area: QtCore.QRectF, tile: Tile, shown: Shown, style: Style) -> None:
    """All drawn members on one vertical scale that always includes zero (and 1 for ratios)."""
    theme, profile = style.theme, tile.profile
    assert profile is not None
    x = profile.x
    frame = framed_range(x)
    visible = np.ones(x.size, dtype=bool) if frame is None else (x >= frame[0]) & (x <= frame[1])
    if not shown.rows or not visible.any():
        note = "No member's amplitude is determined"
        _text(painter, area, note, theme.secondary, _font(style.font, FONT_CAPTION), center=True)
        return
    values = np.concatenate([v[visible] for _i, v in shown.rows])
    finite = values[np.isfinite(values)]
    low = min(0.0, float(finite.min())) if finite.size else 0.0
    high = max(1.0 if shown.ratio else 0.0, float(finite.max())) if finite.size else 1.0
    span = (high - low) or 1.0
    low, high = low - 0.06 * span, high + 0.06 * span
    x0, x1 = float(x[visible].min()), float(x[visible].max())
    width = (x1 - x0) or 1.0

    def to_point(xv: float, yv: float) -> QtCore.QPointF:
        return QtCore.QPointF(
            area.left() + (xv - x0) / width * area.width(),
            area.bottom() - (yv - low) / (high - low) * area.height(),
        )

    # Zero always; for ratios, a dotted line at 1 (the reference level).
    levels = [(0.0, QtCore.Qt.PenStyle.SolidLine)]
    if shown.ratio:
        levels.append((1.0, QtCore.Qt.PenStyle.DotLine))
    for level, pen_style in levels:
        y = to_point(x0, level).y()
        painter.setPen(QtGui.QPen(QtGui.QColor(theme.border), 1, pen_style))
        painter.drawLine(QtCore.QPointF(area.left(), y), QtCore.QPointF(area.right(), y))
    column = np.flatnonzero(profile.source_planes == style.plane)
    if column.size and visible[column[0]]:
        painter.setPen(QtGui.QPen(QtGui.QColor(theme.pin), 1, QtCore.Qt.PenStyle.DashLine))
        px = to_point(float(x[column[0]]), 0.0).x()
        painter.drawLine(QtCore.QPointF(px, area.top()), QtCore.QPointF(px, area.bottom()))
    colors = theme.members
    # Series follow the cluster's member order, as the review's IDs do.
    ids = dict(enumerate(tile.review.ids))
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    # The selected peak is drawn last and slightly larger, so it stays on top.
    for index, row in sorted(shown.rows, key=lambda item: ids.get(item[0]) == style.selected):
        radius = 2.4 if ids.get(index) == style.selected else 1.8
        painter.setBrush(QtGui.QColor(colors[index % len(colors)]))
        for xv, yv in zip(x[visible], row[visible], strict=True):
            if math.isfinite(yv):
                painter.drawEllipse(to_point(float(xv), float(yv)), radius, radius)


def export_pdf(path: Path, tiles: list[Tile], style: Style, title: str) -> int:
    """Write the tiles, in order, to a multi-page PDF; returns the page count.

    Pages are A4 landscape with a title line naming the run, quantity and filter.
    The file is written beside its destination and moved into place only when
    complete, so a failed export never leaves a partial figure.
    """
    path = Path(path)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".pdf", dir=path.parent)
    os.close(handle)
    try:
        writer = QtGui.QPdfWriter(temporary)
        writer.setResolution(_PDF_DPI)
        writer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.PageSizeId.A4))
        writer.setPageOrientation(QtGui.QPageLayout.Orientation.Landscape)
        writer.setPageMargins(QtCore.QMarginsF(12, 12, 12, 12), QtGui.QPageLayout.Unit.Millimeter)
        writer.setTitle(title)
        writer.setCreator("PeakFit")
        per_page = _PDF_COLUMNS * _PDF_ROWS
        pages = max(1, math.ceil(len(tiles) / per_page))
        painter = QtGui.QPainter(writer)
        try:
            page = QtCore.QRectF(painter.viewport())
            caption = _font(style.font, FONT_SMALL)
            heading = FONT_SMALL + 14
            width = page.width() / _PDF_COLUMNS
            height = (page.height() - heading) / _PDF_ROWS
            for number in range(pages):
                if number:
                    writer.newPage()
                line = QtCore.QRectF(page.left(), page.top(), page.width(), heading - 6)
                _text(painter, line, f"{title} · page {number + 1} of {pages}", "#1d232b", caption)
                for slot, tile in enumerate(tiles[number * per_page : (number + 1) * per_page]):
                    row, column = divmod(slot, _PDF_COLUMNS)
                    rect = QtCore.QRectF(
                        page.left() + column * width,
                        page.top() + heading + row * height,
                        width,
                        height,
                    ).adjusted(3, 3, -3, -3)
                    paint_tile(painter, rect, tile, style)
        finally:
            painter.end()
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return pages


class _Tiles(QtCore.QAbstractListModel):
    """Tiles in the navigator's order; the view only reads them."""

    def __init__(self, view: ProfilesView) -> None:
        super().__init__()
        self.view = view
        self.tiles: list[Tile] = []

    def rowCount(self, parent=None) -> int:  # noqa: N802
        """One row per tile."""
        return 0 if parent is not None and parent.isValid() else len(self.tiles)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        """The tile itself, its description, or its first member's name."""
        if not index.isValid():
            return None
        tile = self.tiles[index.row()]
        if role == _ROLE:
            return tile
        if role in (QtCore.Qt.ItemDataRole.ToolTipRole, QtCore.Qt.ItemDataRole.AccessibleTextRole):
            return _describe(tile, self.view.reference, self.view.normalized)
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return tile.review.names[0] if tile.review.names else ""
        return None

    def set_tiles(self, tiles: list[Tile]) -> None:
        """Replace the tiles; the view rebuilds its layout."""
        self.beginResetModel()
        self.tiles = tiles
        self.endResetModel()


class _TileDelegate(QtWidgets.QStyledItemDelegate):
    """Paints one cluster tile in the view's current style."""

    def __init__(self, view: ProfilesView) -> None:
        super().__init__(view)
        self.view = view

    def sizeHint(self, _option, _index) -> QtCore.QSize:  # noqa: N802
        """Every tile has the view's current grid size."""
        return self.view.tile_size

    def paint(self, painter: QtGui.QPainter, option, index) -> None:
        """Draw the tile, with the view's selection and hover state."""
        tile = index.data(_ROLE)
        if not isinstance(tile, Tile):
            return
        state = option.state
        paint_tile(
            painter,
            QtCore.QRectF(option.rect).adjusted(3, 3, -3, -3),
            tile,
            self.view.tile_style(),
            selected=bool(state & QtWidgets.QStyle.StateFlag.State_Selected),
            hovered=bool(state & QtWidgets.QStyle.StateFlag.State_MouseOver),
        )


class ProfilesView(QtWidgets.QWidget):
    """Scan every cluster's profiles; choose one to review it, activate it to focus it."""

    cluster_chosen = QtCore.Signal(str)  # a member of the chosen cluster
    focus_requested = QtCore.Signal(str)  # a member of the cluster to focus
    quantity_changed = QtCore.Signal(tuple, bool)  # reference planes, normalized

    def __init__(self) -> None:
        super().__init__()
        self.theme = LIGHT
        self.selected: str | None = None
        self.plane = 0
        self.reference: tuple[int, ...] = ()
        self.normalized = False
        self.title = "Profiles"  # names the dataset and run in exported figures
        self.tile_size = QtCore.QSize(_TILE)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(SPACE_M, SPACE_S, SPACE_M, 0)
        layout.setSpacing(SPACE_S)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(6)
        self.caption = QtWidgets.QLabel()
        self.caption.setObjectName("caption")
        top.addWidget(self.caption, 1)
        self.mode = QtWidgets.QPushButton("Raw amplitudes")
        self.mode.setObjectName("chip")
        self.mode.setToolTip(
            "Profile quantity for every tile and the inspector: raw amplitudes or "
            "normalized to reference planes"
        )
        self.mode.clicked.connect(self._mode_menu)
        top.addWidget(self.mode)
        self.export = QtWidgets.QPushButton("Export PDF…")
        self.export.setObjectName("chip")
        self.export.setToolTip("Write the tiles shown, in this order, to a PDF figure")
        self.export.clicked.connect(self.export_figure)
        top.addWidget(self.export)
        layout.addLayout(top)
        self.model = _Tiles(self)
        self.list = self._build_list()
        layout.addWidget(self.list, 1)
        self.empty = QtWidgets.QLabel()
        self.empty.setObjectName("muted")
        self.empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty.setWordWrap(True)
        self.empty.hide()
        layout.addWidget(self.empty, 1)

    def _build_list(self) -> QtWidgets.QListView:
        """A wrapping, virtualized grid of painted tiles."""
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
        return self.list

    def tile_style(self) -> Style:
        """The tiles' current appearance."""
        return Style(
            self.theme,
            QtGui.QFont(self.font()),
            self.plane,
            self.selected,
            self.reference,
            self.normalized,
        )

    def apply_theme(self, theme: Theme) -> None:
        """Repaint tiles in the new appearance."""
        self.theme = theme
        self.list.viewport().update()

    def set_quantity(self, reference: tuple[int, ...], normalized: bool) -> None:
        """Adopt the session's profile quantity; recorded amplitudes never change."""
        self.reference = reference
        self.normalized = normalized and bool(reference)
        self.mode.setText(f"÷ {reference_text(reference)}" if self.normalized else "Raw amplitudes")
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
        self.export.setEnabled(bool(tiles))
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

    def export_figure(self, path: str | None = None) -> None:
        """Ask for a PDF destination and write the tiles shown, in light appearance."""
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export profiles", "profiles.pdf", "PDF (*.pdf)"
            )
            if not path:
                return
        destination = Path(path).with_suffix(".pdf")
        quantity = (
            f"A / A_ref ({reference_text(self.reference)})"
            if self.normalized
            else "signed amplitudes"
        )
        # Printed figures carry no selection or current-plane mark.
        style = Style(
            LIGHT, QtGui.QFont(self.font()), reference=self.reference, normalized=self.normalized
        )
        try:
            pages = export_pdf(destination, self.model.tiles, style, f"{self.title} · {quantity}")
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Profiles not exported",
                f"Could not write {destination}: {exc.strerror or exc}. "
                "Choose a folder you can write to and export again.",
            )
            return
        self.caption.setText(
            f"Exported {len(self.model.tiles)} tiles on {pages} page{'s' if pages != 1 else ''} "
            f"to {destination.name}"
        )

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        """Refit the columns whenever the tiles' viewport changes width."""
        if watched is self.list.viewport() and event.type() == QtCore.QEvent.Type.Resize:
            self._fit_columns()
        return super().eventFilter(watched, event)

    def _mode_menu(self) -> None:
        first = next((t.profile for t in self.model.tiles if t.profile is not None), None)
        self._menu = quantity_menu(
            self,
            self.mode,
            self.reference,
            self.normalized,
            lambda on: self.quantity_changed.emit(self.reference, on),
            lambda: self._ask(first),
        )

    def _ask(self, profile: PeakProfile | None) -> None:
        self._dialog = ask_reference(
            self,
            profile,
            self.reference,
            lambda planes: self.quantity_changed.emit(tuple(sorted(set(planes))), True),
        )

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
