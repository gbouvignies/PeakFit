"""GUI acceptance captures and layout measurements on the bundled CEST dataset.

Run with ``uv run python tools/validate_gui_layout.py --output <directory>`` on a
desktop. It drives the real window through the agreed states (spec #178) and
writes WebP captures plus ``measurements.json``. Sizes are Qt logical pixels;
every record also states the device-pixel ratio and the Qt platform plugin.
"""

import argparse
import faulthandler
import json
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from PySide6 import QtCore, QtGui, QtTest, QtWidgets
from validate_gui_editor import settle, wait_until

from peakfit.gui.editor.window import EditorWindow
from peakfit.gui.failures import install_log
from peakfit.gui.spectrum import read_dataset
from peakfit.gui.theme import (
    DARK,
    LIGHT,
    Theme,
    TypeScale,
    contrast_ratio,
    type_scale,
    use_type_scale,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/02-advanced-fitting/data"
SIZES = ("1280x800", "1440x900", "1160x720")
APPEARANCES = ("Light", "Dark")
# An isolated peak, a two-member and a four-member cluster of the bundled dataset.
PEAKS = ("158EN-H", "53N-H", "22N-H")
CLUSTER = "22N-H"
REFERENCE = (0,)  # the off-resonance reference plane of the CEST series
PASSES = 2  # the default fit setup
FAILURE = "Validation tool: simulated worker failure on a first fit"
STALL = 300.0  # seconds one state may take before the watchdog ends the run
FIT_STALL = 900.0  # the real two-pass fit of the whole dataset
WALK = 80  # key presses each way; the fitted window has a few dozen focus stops
WALK_STOPS = 10  # fewer distinct stops means the keys never reached the window

# Text and control-border pairs of the theme, as (foreground, surface) token names;
# ``outline`` bounds editable controls, ``border`` buttons and tooltips.
_CONTRAST_PAIRS = (
    ("text", "panel"),
    ("text", "chrome"),
    ("text", "plot"),
    ("secondary", "panel"),
    ("secondary", "chrome"),
    ("faint", "panel"),
    ("faint", "chrome"),
    ("outline", "panel"),
    ("outline", "chrome"),
    ("outline", "plot"),
    ("border", "panel"),
    ("border", "chrome"),
)


def theme_contrast(theme: Theme) -> dict[str, float]:
    """Contrast of the theme's text/surface and control-border/surface pairs."""
    return {
        f"{fg}/{bg}": contrast_ratio(getattr(theme, fg), getattr(theme, bg))
        for fg, bg in _CONTRAST_PAIRS
    }


def dpr_text(dpr: float) -> str:
    """A device-pixel ratio with at least one decimal and no rounding of 1.25 or 1.75."""
    text = f"{dpr:.2f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def window_record(widget: QtWidgets.QWidget) -> dict[str, object]:
    """Platform, Qt plugin, logical size and device-pixel ratio of one capture."""
    width, height = widget.width(), widget.height()
    dpr = widget.devicePixelRatioF()
    return {
        "platform": platform.platform(),
        "qt_platform": QtGui.QGuiApplication.platformName(),
        "logical": [width, height],
        "dpr": dpr,
        "label": f"{width}×{height} logical @ DPR {dpr_text(dpr)}",
    }


def arm_watchdog(seconds: float) -> None:
    """End the process with every thread's traceback if it is still here in ``seconds``.

    The waits below time out on their own, but a modal dialog or a native deadlock
    blocks the event loop where no wait can see it; the watchdog thread still fires.
    Re-arming replaces the previous deadline.
    """
    faulthandler.dump_traceback_later(seconds, exit=True)


def steady(measure, interval=50, timeout=10.0):
    """A measurement once two readings ``interval`` ms apart agree.

    Inspector pages fill some rows after deferred deliveries, so a single reading
    right after a selection can describe a page that is still growing.
    """
    previous = measure()
    started = perf_counter()
    while perf_counter() - started < timeout:
        settle(interval)
        current = measure()
        if current == previous:
            return current
        previous = current
    raise TimeoutError("Layout measurement did not settle")


def spectral_measures(canvas) -> dict[str, list[int] | None]:
    """Logical sizes of the 2D plot, native slices and residual lanes; None when hidden."""

    def size(item) -> list[int] | None:
        if not item.isVisible():
            return None
        rect = item.sceneBoundingRect()
        return [round(rect.width()), round(rect.height())]

    return {
        "plot": size(canvas.box),
        "row_slice": size(canvas.row_box),
        "column_slice": size(canvas.column_box),
        "row_lane": size(canvas.row_lane),
        "column_lane": size(canvas.column_lane),
    }


def profiles_measures(profiles) -> dict[str, object]:
    """Columns of the Profiles grid at its current width, and the drawn tile size."""
    view = profiles.list
    width = view.viewport().width()
    tile = view.visualRect(view.model().index(0, 0)).size()
    return {
        "viewport": [width, view.viewport().height()],
        "columns": max(1, width // tile.width()) if tile.width() else 0,
        "tile": [tile.width(), tile.height()],
    }


def peak_page_measures(inspector) -> dict[str, int | None]:
    """Inspector viewport height and the Fitted table's top and bottom on the page.

    Positions are in the scrolled page's coordinates (logical pixels from the top
    of the inspector's content, comparable with the viewport height), so they do
    not depend on how far the page is scrolled. ``peak_page_top`` is the height of
    the identity header above the peak page; subtract it for positions relative
    to the peak page.

    The table is the grid's value rows (one cell per column: positions,
    linewidths, amplitude ± marginal σ); ``fitted_bottom`` is the bottom of its
    last row, as in the 14 November 2026 audit baseline. Captions spanning the
    grid below it end at ``fitted_notes_bottom``.
    """
    grid = inspector.fitted
    body = inspector.widget()

    def bottom(widget) -> int:
        return widget.mapTo(body, QtCore.QPoint(0, widget.height())).y()

    cells = []
    for index in range(grid.count()):
        widget = grid.itemAt(index).widget()
        if widget is not None and widget.isVisible():
            cells.append((grid.getItemPosition(index)[3] == 1, widget))
    rows = [w for single, w in cells if single]
    rect = grid.geometry()
    page = inspector.peak_page
    return {
        "viewport_height": inspector.viewport().height(),
        "peak_page_top": page.mapTo(body, QtCore.QPoint(0, 0)).y(),
        "fitted_top": page.mapTo(body, QtCore.QPoint(0, rect.top())).y(),
        "fitted_bottom": max((bottom(w) for w in rows), default=None),
        "fitted_notes_bottom": max((bottom(w) for _, w in cells), default=None),
    }


def clipped_labels(root: QtWidgets.QWidget) -> list[dict[str, object]]:
    """Visible labels whose text needs more room than they were given.

    A wrapping label is clipped when its text needs more height at its width; any
    other label when its size hint exceeds its size in either direction. Used with a
    simulated large system font to find fixed sizes that cut text.
    """
    found = []
    for label in root.findChildren(QtWidgets.QLabel):
        if not label.isVisible() or not label.text():
            continue
        size = label.size()
        if label.wordWrap():
            needed = QtCore.QSize(size.width(), label.heightForWidth(size.width()))
        else:
            needed = label.sizeHint()
        if needed.width() > size.width() or needed.height() > size.height():
            found.append(
                {
                    "text": label.text(),
                    "size": [size.width(), size.height()],
                    "needed": [needed.width(), needed.height()],
                }
            )
    return found


def noise_check_text(window: QtWidgets.QWidget) -> list[str]:
    """Visible texts that carry the noise-power check, in the run line or a row."""
    return [
        label.text()
        for label in window.findChildren(QtWidgets.QLabel)
        if "Noise check" in label.text() and label.isVisible()
    ]


class Session:
    """The real window on the bundled dataset, captured state by state."""

    def __init__(self, output: Path, stall: float = STALL) -> None:
        self.output = output
        self.stall = stall
        arm_watchdog(stall)
        self.window = EditorWindow()
        self.window.install_dataset(
            read_dataset(DATA / "pseudo3d.ft2", DATA / "pseudo3d.list", DATA / "b1_offsets.txt")
        )
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        workspace = self.window.workspace
        assert workspace is not None
        self.w = workspace
        wait_until(lambda: not workspace.canvas.contour_pending, timeout=60)
        self.records: list[dict[str, Any]] = []

    def peak_id(self, name: str) -> str:
        """The working ID of a peak list name."""
        return next(p.id for p in self.w.peaks if p.name == name)

    def capture(self, state: str, size: str, appearance: str, **measures) -> None:
        """Save the whole window once the spectral view has drawn and layout has settled."""
        wait_until(lambda: not self.w.canvas.contour_pending, timeout=60)
        settle(80)
        name = f"{state}-{size}-{appearance.lower()}.webp"
        self.window.grab().save(str(self.output / name), "WEBP", 95)
        self.records.append(
            {
                "state": state,
                "appearance": appearance,
                "file": name,
                "window": window_record(self.window),
                "clipped": clipped_labels(self.window),
                **measures,
            }
        )
        arm_watchdog(self.stall)

    def before_fit(self, size: str, appearance: str) -> None:
        """Loaded, unfitted: the inspector opens at the fit setup."""
        self.w.show_model_page()
        self.capture("fit-setup", size, appearance, spectral=self.spectral())

    def failure(self, size: str, appearance: str) -> None:
        """The Analysis page after a failed fit."""
        self.w.show_run_page()
        # Label and value texts of the failure rows, in reading order.
        rows = self.w.panel.run_page.failure_rows
        items = (rows.itemAt(i) for i in range(rows.count()))
        shown = (item.widget() for item in items if item is not None)
        texts = [w.text() for w in shown if isinstance(w, QtWidgets.QLabel)]
        self.capture("failure", size, appearance, failure=self.w.fitting.attempt, rows=texts)

    def spectral(self) -> dict[str, object]:
        """Settled spectral-view sizes."""
        return steady(lambda: spectral_measures(self.w.canvas))

    def fitted_states(self, size: str, appearance: str) -> None:
        """Overview, peak and cluster pages, cluster focus and Profiles after the fit."""
        w = self.w
        w.show_view(profiles=False)
        w.show_run_page()
        noise = noise_check_text(self.window)
        self.capture("overview", size, appearance, spectral=self.spectral(), noise_check=noise)
        if noise:
            self.capture("noise-check", size, appearance, noise_check=noise)
        for name in PEAKS:
            w.select_peak(self.peak_id(name), center=True)
            page = steady(lambda: peak_page_measures(w.panel))
            self.capture(f"peak-{name}", size, appearance, peak_page=page)
        w.select_peak(self.peak_id(CLUSTER), cluster=True, center=True)
        self.capture(f"cluster-{CLUSTER}", size, appearance)
        w.open_focus(self.peak_id(CLUSTER))
        for mode, label in ((1, "overlay"), (2, "residual")):
            w.set_mode(mode)
            self.capture(f"focus-{label}", size, appearance, spectral=self.spectral())
        w.close_focus()
        w.set_mode(1)
        w.show_view(profiles=True)
        wait_until(lambda: "Computing" not in w.profiles.caption.text(), timeout=60)
        for quantity, reference, normalized in (
            ("raw", (), False),
            ("normalized", REFERENCE, True),
        ):
            w.set_profile_quantity(reference, normalized)
            grid = steady(lambda: profiles_measures(w.profiles))
            self.capture(f"profiles-{quantity}", size, appearance, profiles=grid)
        w.set_profile_quantity((), False)
        w.show_view(profiles=False)

    def keyboard_walk(self) -> int:
        """Tab, then Backtab, through the fitted window from Fit; the focus stops reached.

        Native macOS crashed on Tab out of the inspector, a crash no offscreen test
        showed; on a desktop this walk is the smoke test for native focus handling.
        """
        stops: set[int] = set()
        for key in (QtCore.Qt.Key.Key_Tab, QtCore.Qt.Key.Key_Backtab):
            self.w.fit_button.setFocus(QtCore.Qt.FocusReason.TabFocusReason)
            for _ in range(WALK):
                QtTest.QTest.keyClick(QtWidgets.QApplication.focusWidget() or self.window, key)
                settle(15)
                stops.add(id(QtWidgets.QApplication.focusWidget()))
        if len(stops) < WALK_STOPS:
            raise RuntimeError(
                f"Keyboard walk reached {len(stops)} focus stops; is the window active?"
            )
        arm_watchdog(self.stall)
        return len(stops)

    def each(self, sizes, appearances, step) -> None:
        """Run ``step`` at every requested window size and appearance."""
        for size in sizes:
            width, height = (int(v) for v in size.split("x"))
            self.window.resize(width, height)
            for appearance in appearances:
                self.window.apply_appearance(appearance)
                step(size, appearance)

    def fit(self) -> None:
        """Run the default two-pass fit of the reviewed setup, as Fit does."""
        w = self.w
        arm_watchdog(max(self.stall, FIT_STALL))
        wait_until(lambda: w.fitting.prepared is not None, timeout=60)
        w.fit_button.click()
        wait_until(lambda: not w.fitting.busy, timeout=600)
        arm_watchdog(self.stall)
        assert w.fitting.current, w.fitting.failure
        assert w.fitting.result is not None
        assert w.fitting.result.outcome.n_optimizer_passes == PASSES


def run(output: Path, sizes, appearances, stall: float = STALL) -> list[dict[str, Any]]:
    """Drive every agreed state; the failure is delivered before the first fit."""
    output.mkdir(parents=True, exist_ok=True)
    # As launch does: the failure state then names the log file (#186).
    install_log(output / "peakfit.log")
    session = Session(output, stall)
    session.each(sizes, appearances, session.before_fit)
    # A first fit that fails: nothing is retained, which the failure state must say.
    session.w.fitting._finished(0, RuntimeError(FAILURE))
    session.each(sizes, appearances, session.failure)
    session.w.show_model_page()
    session.fit()
    session.each(sizes, appearances, session.fitted_states)
    walk = session.keyboard_walk()
    report = {
        "dataset": str(DATA.relative_to(ROOT)),
        "units": "Qt logical pixels; window.dpr converts to device pixels",
        "type_scale": asdict(type_scale()),
        "contrast": {"Light": theme_contrast(LIGHT), "Dark": theme_contrast(DARK)},
        "keyboard_walk": {"stops": walk},
        "captures": session.records,
    }
    (output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    faulthandler.cancel_dump_traceback_later()
    return session.records


def main() -> None:
    """Parse the matrix, refuse an output inside the repository, and run it."""
    parser = argparse.ArgumentParser(description="GUI acceptance captures and layout measurements.")
    parser.add_argument("--output", type=Path, required=True, help="outside the repository")
    parser.add_argument("--sizes", nargs="+", default=list(SIZES), metavar="WxH")
    parser.add_argument("--appearance", nargs="+", default=list(APPEARANCES), choices=APPEARANCES)
    parser.add_argument(
        "--body-px",
        type=int,
        metavar="PX",
        help="simulate a system font of this body size (logical px) to check clipping",
    )
    parser.add_argument(
        "--stall",
        type=float,
        default=STALL,
        metavar="S",
        help=f"end the run with tracebacks when one state takes longer (default {STALL:.0f} s)",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("captures stay outside the repository; choose another --output")
    _app = QtWidgets.QApplication([])  # alive for the run
    if args.body_px:
        use_type_scale(TypeScale.for_body(args.body_px))
    records = run(output, args.sizes, args.appearance, args.stall)
    for record in records:
        window = record["window"]
        extra = {k: v for k, v in record.items() if k not in ("state", "file", "window")}
        print(record["file"], window["label"], json.dumps(extra))  # noqa: T201
    print(f"Wrote {len(records)} captures to {output}")  # noqa: T201


if __name__ == "__main__":
    main()
