"""Where the 2D spectrum, its slices and residual lanes go: one pure layout function.

The row slice sits above the 2D plot and the column slice to its right. Each shares
its spectral extent with the plot by construction: the row slice has the plot's x
and width, the column slice its y and height. Slice sizes are shares of the view
(the row slice's of its height, the column slice's of its width), so they follow
the window; each layout resolves them to logical pixels. When space is short,
lanes compress first, then slices, so the 2D plot keeps its minimum. Rectangles
are (x, y, width, height) tuples; no Qt here.
"""

from dataclasses import dataclass, replace

MAIN_MIN = (360, 300)  # the 2D plot never shrinks below this when space allows
SLICE_MIN = 72  # a visible slice compresses no further
SLICE_COLLAPSE = 48  # a divider dragged below this collapses the slice
LANE_MIN = 24  # a visible residual lane compresses no further
SLICE_MAX = 0.45  # a slice takes at most this share of the view
# Default shares (row slice of the height, column slice of the width). Focus gives
# the slices more room while the 2D contours stay large enough to judge overlap.
OVERVIEW_ROW, OVERVIEW_COLUMN = 0.22, 0.18
FOCUS_ROW, FOCUS_COLUMN = 0.30, 0.26
GAP = 7  # divider thickness
HEADER = 22  # caption row above the row slice
PAD = 4  # right margin

Rect = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SliceSizes:
    """User-controlled shares of the view; 0 collapses a slice. Lanes follow their slice."""

    row: float = OVERVIEW_ROW  # row-slice height over the view's height
    column: float = OVERVIEW_COLUMN  # column-slice width over the view's width
    lanes: bool = False
    lane: int = 44  # residual lane thickness in pixels when shown


@dataclass(frozen=True, slots=True)
class SlicePixels:
    """Slice and lane sizes in logical pixels, as one layout uses them."""

    row: int
    column: int
    lanes: bool
    lane: int


def default_sizes(*, focus: bool) -> SliceSizes:
    """The default shares for the overview or for cluster focus (residual lanes on)."""
    if focus:
        return SliceSizes(row=FOCUS_ROW, column=FOCUS_COLUMN, lanes=True)
    return SliceSizes()


@dataclass(frozen=True, slots=True)
class Extents:
    """Axis gutters from font metrics: left of the 2D plot and below it."""

    left: int = 66
    bottom: int = 40


@dataclass(frozen=True, slots=True)
class Layout:
    """Every rectangle of one spectral view, and the sizes actually used."""

    main: Rect
    row: Rect | None
    row_lane: Rect | None
    column: Rect | None
    column_lane: Rect | None
    row_divider: Rect
    column_divider: Rect
    header: Rect
    corner: Rect
    sizes: SlicePixels  # after compression


def _compress(available: float, lane: int, size: int, *, lanes: bool) -> tuple[int, int]:
    """Shrink a lane, then its slice, to fit ``available`` pixels beyond the plot."""
    deficit = (lane if lanes and size else 0) + size - available
    if deficit > 0 and lanes and size:
        cut = max(0, min(int(deficit), lane - LANE_MIN))
        lane, deficit = lane - cut, deficit - cut
    if deficit > 0 and size:
        size = max(SLICE_MIN, size - int(deficit))
    return lane, size


def stored_share(value: object, default: float) -> float:
    """A share read from saved view state, within [0, SLICE_MAX]; anything else is the default."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
        return default
    return min(float(value), SLICE_MAX)


def _resolve(share: float, extent: float) -> int:
    """A share of ``extent`` in pixels, between the slice minimum and maximum; 0 stays 0."""
    if share <= 0:
        return 0
    return max(SLICE_MIN, round(min(share, SLICE_MAX) * extent))


def effective(width: float, height: float, sizes: SliceSizes, extents: Extents) -> SlicePixels:
    """The pixel sizes that fit: lanes compress first, then slices; collapsed stays so."""
    room_x = width - extents.left - MAIN_MIN[0] - GAP - PAD
    room_y = height - HEADER - MAIN_MIN[1] - GAP - extents.bottom
    column, row = _resolve(sizes.column, width), _resolve(sizes.row, height)
    lane_x, column = _compress(room_x, sizes.lane, column, lanes=sizes.lanes)
    lane_y, row = _compress(room_y, sizes.lane, row, lanes=sizes.lanes)
    return SlicePixels(row, column, sizes.lanes, min(lane_x, lane_y))


def layout(width: float, height: float, sizes: SliceSizes, extents: Extents) -> Layout:
    """Place the plot, slices, lanes and dividers; shared extents are equal by definition."""
    used = effective(width, height, sizes, extents)
    row_lane = used.lane if used.lanes and used.row else 0
    column_lane = used.lane if used.lanes and used.column else 0
    x0, y_main = extents.left, HEADER + used.row + row_lane + GAP
    main_w = max(1.0, width - x0 - GAP - column_lane - used.column - PAD)
    main_h = max(1.0, height - y_main - extents.bottom)
    x_right = x0 + main_w + GAP
    return Layout(
        main=(x0, y_main, main_w, main_h),
        row=(x0, HEADER, main_w, used.row) if used.row else None,
        row_lane=(x0, HEADER + used.row, main_w, row_lane) if row_lane else None,
        column=(x_right + column_lane, y_main, used.column, main_h) if used.column else None,
        column_lane=(x_right, y_main, column_lane, main_h) if column_lane else None,
        row_divider=(x0, y_main - GAP, main_w, GAP),
        column_divider=(x0 + main_w, y_main, GAP, main_h),
        header=(x0, 0, main_w, HEADER),
        corner=(x_right, 0, max(0.0, width - x_right), y_main - GAP),
        sizes=used,
    )


def minimum_size(sizes: SliceSizes, extents: Extents) -> tuple[int, int]:
    """The smallest view that keeps the 2D minimum with every visible part at its minimum."""
    lane = LANE_MIN if sizes.lanes else 0
    column = SLICE_MIN + lane if sizes.column else 0
    row = SLICE_MIN + lane if sizes.row else 0
    width = extents.left + MAIN_MIN[0] + GAP + column + PAD
    height = HEADER + row + GAP + MAIN_MIN[1] + extents.bottom
    return width, height


def drag(
    sizes: SliceSizes,
    which: str,
    value: float,
    width: float,
    height: float,
    extents: Extents,
) -> SliceSizes:
    """A divider dragged to ``value`` px, kept as a share of the view.

    Below the collapse threshold the slice collapses; the 2D minimum and the maximum
    share bound it.
    """
    if which == "row":
        extent = height
        room = height - HEADER - GAP - extents.bottom - MAIN_MIN[1]
    else:
        extent = width
        room = width - extents.left - GAP - PAD - MAIN_MIN[0]
    room -= sizes.lane if sizes.lanes else 0
    size = int(min(value, room, SLICE_MAX * extent))
    share = size / extent if size >= SLICE_COLLAPSE and extent > 0 else 0.0
    return replace(sizes, **{which: share})
