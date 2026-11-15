"""The spectral view's layout: aligned extents, compression order and limits."""

import pytest

from peakfit.gui.editor.spectral_geometry import (
    GAP,
    HEADER,
    LANE_MIN,
    MAIN_MIN,
    SLICE_COLLAPSE,
    SLICE_MAX,
    SLICE_MIN,
    Extents,
    SliceSizes,
    default_sizes,
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
    SliceSizes(row=0.40, column=0.08),
    default_sizes(focus=True),
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


def test_fractions_resolve_to_shares_of_the_view():
    # 22 % of 800 px and 18 % of 1000 px: the slices grow with the window.
    view = layout(1000, 800, SliceSizes(row=0.22, column=0.18), EXTENTS)
    assert (view.sizes.row, view.sizes.column) == (176, 180)
    large = layout(2000, 1600, SliceSizes(row=0.22, column=0.18), EXTENTS)
    assert (large.sizes.row, large.sizes.column) == (352, 360)


def test_overview_and_focus_have_separate_defaults_and_focus_gives_slices_more_room():
    overview, focus = default_sizes(focus=False), default_sizes(focus=True)
    assert SliceSizes() == overview
    assert focus.lanes
    assert not overview.lanes
    assert focus.row > overview.row
    assert focus.column > overview.column
    for sizes in (overview, focus):
        assert 0 < sizes.column <= sizes.row <= SLICE_MAX


def test_resolved_slices_respect_the_minimum_and_maximum_shares():
    # A small fraction never resolves below the slice minimum.
    small = layout(1300, 900, SliceSizes(row=0.01, column=0.01), EXTENTS)
    assert (small.sizes.row, small.sizes.column) == (SLICE_MIN, SLICE_MIN)
    # A fraction above the maximum share is held at it.
    big = layout(3000, 3000, SliceSizes(row=0.9, column=0.9), EXTENTS)
    assert (big.sizes.row, big.sizes.column) == (1350, 1350)


@pytest.mark.parametrize("focus", [False, True])
@pytest.mark.parametrize("size", [(1280, 800), (1512, 982), (1920, 1080), (2560, 1440)])
def test_defaults_keep_the_plot_above_its_minimum_at_target_sizes(focus, size):
    # The spectral area left by side panels at a target window size is narrower.
    width, height = size[0] - 560, size[1] - 120
    view = layout(width, height, default_sizes(focus=focus), EXTENTS)
    assert view.main[2] >= MAIN_MIN[0]
    assert view.main[3] >= MAIN_MIN[1]
    assert view.sizes.row >= SLICE_MIN
    assert view.sizes.column >= SLICE_MIN


def test_lanes_compress_before_slices_and_the_plot_keeps_its_minimum():
    sizes = SliceSizes(row=0.15, column=0.13, lanes=True, lane=44)
    # Roomy: each slice is its share; the lanes are whole.
    roomy = layout(1000, 1000, sizes, EXTENTS)
    assert (roomy.sizes.lane, roomy.sizes.column, roomy.sizes.row) == (44, 130, 150)
    # A little short: only the lanes give way.
    width = EXTENTS.left + MAIN_MIN[0] + GAP + 44 + 100 - 10 + 4
    tight = layout(width, 1000, SliceSizes(row=0.15, column=100 / width, lanes=True), EXTENTS)
    assert tight.sizes.column == 100
    assert tight.sizes.lane == 44 - 10
    assert tight.main[2] >= MAIN_MIN[0]
    # At the minimum: lanes and slices at their floors, the plot at its minimum.
    width, height = minimum_size(sizes, EXTENTS)
    floor = layout(width, height, sizes, EXTENTS)
    assert (floor.sizes.lane, floor.sizes.column, floor.sizes.row) == (
        LANE_MIN,
        SLICE_MIN,
        SLICE_MIN,
    )
    assert floor.main[2:] == (MAIN_MIN[0], MAIN_MIN[1])


def test_stored_fractions_re_resolve_after_a_size_change():
    sizes = drag(SliceSizes(), "row", 200, 1000, 800, EXTENTS)
    assert sizes.row == pytest.approx(0.25)
    assert layout(1000, 800, sizes, EXTENTS).sizes.row == 200
    assert layout(1000, 1200, sizes, EXTENTS).sizes.row == 300


def test_collapsed_slices_and_lanes_take_no_space():
    view = layout(900, 700, SliceSizes(row=0, column=0, lanes=True), EXTENTS)
    assert view.row is view.row_lane is view.column is view.column_lane is None
    assert view.main[1] == HEADER + GAP


@pytest.mark.parametrize("which", ["row", "column"])
def test_dividers_collapse_reopen_and_protect_the_plot(which):
    sizes = SliceSizes(lanes=True)
    assert getattr(drag(sizes, which, SLICE_COLLAPSE - 1, 900, 700, EXTENTS), which) == 0
    reopened = drag(sizes, which, 140, 900, 700, EXTENTS)
    assert getattr(layout(900, 700, reopened, EXTENTS).sizes, which) == 140
    huge = drag(sizes, which, 10_000, 900, 700, EXTENTS)
    main = layout(900, 700, huge, EXTENTS).main
    assert main[2] >= MAIN_MIN[0]
    assert main[3] >= MAIN_MIN[1]
    assert getattr(huge, which) <= SLICE_MAX
    # Dragging one divider never changes the other slice.
    other = "column" if which == "row" else "row"
    assert getattr(huge, other) == getattr(sizes, other)


def test_minimum_size_counts_only_visible_parts():
    full = minimum_size(SliceSizes(lanes=True), EXTENTS)
    bare = minimum_size(SliceSizes(row=0, column=0, lanes=True), EXTENTS)
    assert full[0] - bare[0] == SLICE_MIN + LANE_MIN
    assert full[1] - bare[1] == SLICE_MIN + LANE_MIN
