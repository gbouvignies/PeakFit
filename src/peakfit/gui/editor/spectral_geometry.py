"""Where the 2D spectrum, its slices and residual lanes go: one pure layout function.

The row slice sits above the 2D plot and the column slice to its right. Each shares
its spectral extent with the plot by construction: the row slice has the plot's x
and width, the column slice its y and height. Sizes are logical pixels the user
controls; when space is short, lanes compress first, then slices, so the 2D plot
keeps its minimum. Rectangles are (x, y, width, height) tuples; no Qt here.
"""

from dataclasses import dataclass, replace

MAIN_MIN = (360, 300)  # the 2D plot never shrinks below this when space allows
SLICE_MIN = 72  # a visible slice compresses no further
SLICE_COLLAPSE = 48  # a divider dragged below this collapses the slice
LANE_MIN = 24  # a visible residual lane compresses no further
SLICE_MAX = 0.45  # a slice takes at most this share of the view
GAP = 7  # divider thickness
HEADER = 22  # caption row above the row slice
PAD = 4  # right margin

Rect = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SliceSizes:
    """User-controlled sizes; 0 collapses a slice. Lanes follow their slice."""

    row: int = 150  # row-slice height
    column: int = 130  # column-slice width
    lanes: bool = False
    lane: int = 44  # residual lane thickness when shown


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
    sizes: SliceSizes  # after compression


def _compress(available: float, lane: int, size: int, *, lanes: bool) -> tuple[int, int]:
    """Shrink a lane, then its slice, to fit ``available`` pixels beyond the plot."""
    deficit = (lane if lanes and size else 0) + size - available
    if deficit > 0 and lanes and size:
        cut = max(0, min(int(deficit), lane - LANE_MIN))
        lane, deficit = lane - cut, deficit - cut
    if deficit > 0 and size:
        size = max(SLICE_MIN, size - int(deficit))
    return lane, size


def effective(width: float, height: float, sizes: SliceSizes, extents: Extents) -> SliceSizes:
    """The sizes that fit: lanes compress first, then slices; collapsed stays collapsed."""
    room_x = width - extents.left - MAIN_MIN[0] - GAP - PAD
    room_y = height - HEADER - MAIN_MIN[1] - GAP - extents.bottom
    lane_x, column = _compress(room_x, sizes.lane, sizes.column, lanes=sizes.lanes)
    lane_y, row = _compress(room_y, sizes.lane, sizes.row, lanes=sizes.lanes)
    return replace(sizes, row=row, column=column, lane=min(lane_x, lane_y))


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
    """A divider dragged to ``value`` px: collapse below the threshold, keep the 2D minimum."""
    if which == "row":
        room = height - HEADER - GAP - extents.bottom - MAIN_MIN[1]
        room -= sizes.lane if sizes.lanes else 0
        limit = min(room, SLICE_MAX * height)
    else:
        room = width - extents.left - GAP - PAD - MAIN_MIN[0]
        room -= sizes.lane if sizes.lanes else 0
        limit = min(room, SLICE_MAX * width)
    size = int(min(value, limit))
    return replace(sizes, **{which: size if size >= SLICE_COLLAPSE else 0})
