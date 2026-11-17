"""Native captures of the Peak page's Parameters: one peak's own constraints.

``docs/validation/gui-peak-parameters.md`` cites the figures and measurements this
tool writes. Run it on a desktop (natively, not offscreen)::

    uv run python tools/capture_peak_parameters.py --output <directory outside the repo>

It opens the real window on the bundled ``02-advanced-fitting`` dataset through
``validate_gui_layout.Session``, selects 2N-H and captures at 1280×800 (and 1160×720,
the narrowest supported window) in both appearances: Parameters as loaded; the
peak's own constraints set (a ¹⁵N window, a ¹H linewidth range, the ¹⁵N linewidth
held at a start); a typed range that cannot be used; a committed start outside the
settings' range; the Model page counting the peak; the GUI's own fit with those
constraints, its Run page; and an edit that differs from that run. Each state is
saved as the whole window (downscaled WebP) and the inspector alone (device pixels),
with ``measurements.json``: inspector and viewport sizes, sideways scrolling, the
section's minimum width, clipped labels and Fit's state.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtTest, QtWidgets
from validate_gui_editor import settle, wait_until
from validate_gui_layout import ROOT, Session, arm_watchdog, clipped_labels

from peakfit.gui import theme as themes
from peakfit.gui.failures import install_log

SIZES = ((1280, 800), (1160, 720))
APPEARANCES = ("Light", "Dark")
MAX_WIDTH = 1600  # px; wider window grabs are downscaled
STALL = 1200.0  # seconds one state, or the fit, may take
PEAK = "2N-H"


class Capture:
    """The window and inspector at each state, with the measurements of each."""

    def __init__(self, output: Path) -> None:
        self.output = output
        for folder in ("full", "inspector"):
            (output / folder).mkdir(parents=True, exist_ok=True)
        self.session = Session(output, stall=STALL)
        self.records: list[dict[str, Any]] = []

    def measures(self) -> dict[str, Any]:
        """What the inspector shows at this state, in logical px."""
        w = self.session.w
        panel = w.panel
        viewport, section = panel.viewport(), panel.parameters
        return {
            "inspector": [panel.width(), panel.height()],
            "viewport": [viewport.width(), viewport.height()],
            "sideways_scroll": panel.horizontalScrollBar().isVisible(),
            "section_min_width": section.minimumSizeHint().width(),
            "title": panel.parameters_toggle.text(),
            "fit_enabled": w.fit_button.isEnabled(),
            "fit_tooltip": w.fit_button.toolTip(),
            "invalid": section.invalid,
            "blocked": w.fitting.blocked,
            "undo": w.undo_action.text(),
            "run_line": w.summary.text(),
            "clipped": [label["text"] for label in clipped_labels(panel)],
        }

    def shot(self, state: str, appearance: str) -> None:
        """Save the window and the inspector in ``appearance``."""
        window, w = self.session.window, self.session.w
        window.apply_appearance(appearance)
        wait_until(lambda: not w.canvas.contour_pending, timeout=60)
        settle(250)
        name = f"{state}-{window.width()}x{window.height()}-{appearance.lower()}.webp"
        image = window.grab()
        if image.width() > MAX_WIDTH:  # device pixels
            image = image.scaledToWidth(
                MAX_WIDTH, QtCore.Qt.TransformationMode.SmoothTransformation
            )
        image.save(str(self.output / "full" / name), "WEBP", 82)
        w.panel.grab().save(str(self.output / "inspector" / name), "WEBP", 90)
        self.records.append({"file": name, **self.measures()})
        arm_watchdog(STALL)

    def matrix(self, state: str, scroll_to: QtWidgets.QWidget | None = None) -> None:
        """Save ``state`` at every size and appearance, then return to 1280×800."""
        for size in SIZES:
            for appearance in APPEARANCES:
                for _ in range(2):  # a first resize past the screen can be clamped
                    self.session.window.resize(*size)
                    settle(400)
                if scroll_to is not None:
                    self.session.w.panel.ensureWidgetVisible(scroll_to, 0, 24)
                    settle(150)
                self.shot(state, appearance)
        self.session.window.resize(1280, 800)
        settle(400)


def _type(edit: QtWidgets.QLineEdit, text: str) -> None:
    edit.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
    edit.selectAll()
    edit.insert(text)  # QTest.keyClicks refuses non-ASCII such as – and −
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key.Key_Return)
    settle(300)


def _row(c: Capture, key: str):
    return next(row for row in c.session.w.panel.parameters.rows if row.key == key)


def _open(c: Capture) -> None:
    """Select 2N-H with its Parameters open, the section's top in view."""
    w = c.session.w
    w.select_peak(c.session.peak_id(PEAK), center=True)
    w.panel.parameters_toggle.setChecked(True)
    wait_until(lambda: w.fitting.prepared is not None and w.panel.parameters.rows, timeout=120)
    w._update_inspector()
    settle(300)


def _prepared(c: Capture) -> None:
    w = c.session.w
    wait_until(lambda: w.fitting.prepared is not None or bool(w.fitting.blocked), timeout=120)
    w._update_inspector()
    settle(300)


def _set(c: Capture) -> None:
    """The peak's own constraints: a ¹⁵N window, a ¹H width range, the ¹⁵N width held."""
    _type(_row(c, "F2.cs").window_edit, "0.3")
    width = _row(c, "F3.lw")
    _type(width.low, "5")
    _type(width.high, "40")
    held = _row(c, "F2.lw")
    _type(held.start, "20")
    held.hold.setChecked(True)
    settle(300)
    _prepared(c)


def run(output: Path) -> list[dict[str, Any]]:
    """Walk the workflow once and write the figures and ``measurements.json``."""
    install_log(output / "peakfit.log")  # as launch does
    c = Capture(output)
    w = c.session.w
    wait_until(lambda: w.fitting.prepared is not None, timeout=120)
    _open(c)
    section = w.panel.parameters
    c.matrix("01-parameters-as-loaded", scroll_to=section)
    _set(c)
    c.matrix("02-own-constraints-set", scroll_to=section)
    # A typed range that cannot be used: a draft, explained, blocking Fit.
    width = _row(c, "F3.lw")
    _type(width.low, "60")
    c.matrix("03-invalid-draft", scroll_to=section)
    _type(width.low, "5")
    # A committed start the settings' range excludes: a preparation problem.
    _type(width.high, "")
    _type(width.start, "500")
    _prepared(c)
    c.matrix("04-start-outside-range", scroll_to=section)
    _type(width.start, "")
    _type(width.high, "40")
    _prepared(c)
    blocked = w.fitting.blocked or section.invalid
    if blocked or not w.fit_button.isEnabled():
        raise RuntimeError(f"Fit is blocked before the fit: {blocked or w.fit_button.toolTip()}")
    w.show_model_page()
    model = w.panel.model_page
    model.constraints_toggle.setChecked(True)
    settle(300)
    c.matrix("05-model-page-counts-own", scroll_to=model.constraints)
    model.constraints_toggle.setChecked(False)
    w.panel.verticalScrollBar().setValue(0)
    c.session.fit()
    w.select_peak(None)
    w.show_run_page()
    settle(1500)
    c.matrix("06-run-records-own")
    _open(c)
    c.matrix("07-after-fit-no-difference", scroll_to=section)
    _type(_row(c, "F3.lw").high, "30")
    _prepared(c)
    c.matrix("08-differs-from-run", scroll_to=section)
    (output / "measurements.json").write_text(
        json.dumps(
            {"body_px": themes.type_scale().body, "captures": c.records},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    c.session.window.close()
    return c.records


def main() -> None:
    """Refuse an output inside the repository, then capture."""
    parser = argparse.ArgumentParser(description="Native captures of the Peak page Parameters.")
    parser.add_argument("--output", type=Path, required=True, help="outside the repository")
    parser.add_argument("--body-px", type=int, metavar="PX", help="simulate a system font")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("captures stay outside the repository; choose another --output")
    _app = QtWidgets.QApplication([])  # alive for the run
    if args.body_px:
        themes.use_type_scale(themes.TypeScale.for_body(args.body_px))
    records = run(output)
    print(f"Wrote {len(records)} captures to {output}")  # noqa: T201


if __name__ == "__main__":
    main()
