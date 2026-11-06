"""The spectral view's layout: aligned extents, compression order and limits."""

import pytest

from peakfit.gui.editor.spectral_geometry import (
    GAP,
    HEADER,
    LANE_MIN,
    MAIN_MIN,
    SLICE_COLLAPSE,
    SLICE_MIN,
    Extents,
    SliceSizes,
    drag,
    layout,
    minimum_size,
)

EXTENTS = Extents()
SIZES = [(1300, 900), (838, 787), (578, 615), (478, 615), (720, 560), (2000, 1200)]
SLICES = [
    SliceSizes(),
    SliceSizes(lanes=True),
    SliceSizes(row=0, lanes=True),
    SliceSizes(column=0),
    SliceSizes(row=220, column=84),
    SliceSizes(row=0, column=0, lanes=True),
]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("sizes", SLICES)
def test_slices_share_the_plot_extents_exactly(size, sizes):
    view = layout(size[0], size[1], sizes, EXTENTS)
    x, y, w, h = view.main
    for rect in (view.row, view.row_lane):
        if rect is not None:
            assert (rect[0], rect[2]) == (x, w)
    for rect in (view.column, view.column_lane):
        if rect is not None:
            assert (rect[1], rect[3]) == (y, h)
    # Stacked without overlap, inside the view.
    if view.row is not None:
        assert view.row[1] + view.row[3] <= y - GAP + (view.row_lane[3] if view.row_lane else 0)
    assert x + w <= size[0]
    assert y + h <= size[1]


def test_lanes_compress_before_slices_and_the_plot_keeps_its_minimum():
    sizes = SliceSizes(row=150, column=130, lanes=True, lane=44)
    width, height = minimum_size(sizes, EXTENTS)
    roomy = layout(width + 400, height + 400, sizes, EXTENTS)
    assert (roomy.sizes.lane, roomy.sizes.column, roomy.sizes.row) == (44, 130, 150)
    # A little short: only the lanes give way.
    tight = layout(EXTENTS.left + MAIN_MIN[0] + GAP + 44 + 130 - 10 + 4, 900, sizes, EXTENTS)
    assert tight.sizes.column == 130
    assert tight.sizes.lane == 44 - 10
    assert tight.main[2] >= MAIN_MIN[0]
    # At the minimum: lanes and slices at their floors, the plot at its minimum.
    floor = layout(width, height, sizes, EXTENTS)
    assert (floor.sizes.lane, floor.sizes.column, floor.sizes.row) == (
        LANE_MIN,
        SLICE_MIN,
        SLICE_MIN,
    )
    assert floor.main[2:] == (MAIN_MIN[0], MAIN_MIN[1])


def test_collapsed_slices_and_lanes_take_no_space():
    view = layout(900, 700, SliceSizes(row=0, column=0, lanes=True), EXTENTS)
    assert view.row is view.row_lane is view.column is view.column_lane is None
    assert view.main[1] == HEADER + GAP


@pytest.mark.parametrize("which", ["row", "column"])
def test_dividers_collapse_reopen_and_protect_the_plot(which):
    sizes = SliceSizes(lanes=True)
    assert getattr(drag(sizes, which, SLICE_COLLAPSE - 1, 900, 700, EXTENTS), which) == 0
    assert getattr(drag(sizes, which, 130, 900, 700, EXTENTS), which) == 130
    huge = drag(sizes, which, 10_000, 900, 700, EXTENTS)
    main = layout(900, 700, huge, EXTENTS).main
    assert main[2] >= MAIN_MIN[0]
    assert main[3] >= MAIN_MIN[1]
    # Dragging one divider never changes the other slice.
    other = "column" if which == "row" else "row"
    assert getattr(huge, other) == getattr(sizes, other)


def test_minimum_size_counts_only_visible_parts():
    full = minimum_size(SliceSizes(lanes=True), EXTENTS)
    bare = minimum_size(SliceSizes(row=0, column=0, lanes=True), EXTENTS)
    assert full[0] - bare[0] == SLICE_MIN + LANE_MIN
    assert full[1] - bare[1] == SLICE_MIN + LANE_MIN
