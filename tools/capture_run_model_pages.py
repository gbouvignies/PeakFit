"""Native captures of the Run and Model pages through one real workflow (#215).

``docs/validation/gui-run-model-pages.md`` cites the figures and measurements this
tool writes. Run it on a desktop (natively, not offscreen)::

    uv run python tools/capture_run_model_pages.py --output <directory outside the repo>
    uv run python tools/capture_run_model_pages.py --output <dir> --body-px 17 --no-fit

It opens the real window on the bundled ``02-advanced-fitting`` dataset through
``validate_gui_layout.Session`` and walks the Model page before a fit, adding a ¹⁵N
doublet with **+ J**, Tab from J, an invalid J range, a phase on a family without
one and removing the doublet; then (unless ``--no-fit``) the GUI's own two-pass fit,
the Run page, and Model edits marked against the run, with Undo. Each state is
saved as the whole window (downscaled WebP) and the inspector alone (device
pixels), with ``measurements.json``: inspector and viewport sizes, sideways
scrolling, clipped labels, where the Planes row ends, and Fit's state.
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

SIZES = ((1160, 720), (1280, 800), (1920, 1080))
APPEARANCES = ("Light", "Dark")
MAX_WIDTH = 1600  # px; wider window grabs are downscaled
STALL = 1200.0  # seconds one state, or the fit, may take


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
        body, viewport, model = panel.widget(), panel.viewport(), panel.model_page

        def bottom(widget: QtWidgets.QWidget) -> int:
            return widget.mapTo(body, QtCore.QPoint(0, widget.height())).y()

        assert model.axes is not None
        return {
            "inspector": [panel.width(), panel.height()],
            "viewport": [viewport.width(), viewport.height()],
            "sideways_scroll": panel.horizontalScrollBar().isVisible(),
            "page": panel.page.value,
            "table_width": model.axes.minimumSizeHint().width(),
            "planes_bottom": bottom(model.planes) if model.isVisible() else None,
            "fit_enabled": w.fit_button.isEnabled(),
            "fit_tooltip": w.fit_button.toolTip(),
            "undo": w.undo_action.text(),
            "run_line": w.summary.text(),
            "clipped": [label["text"] for label in clipped_labels(panel)],
        }

    def shot(self, state: str, size=None, appearance: str = "Light") -> None:
        """Save the window and the inspector at ``size`` and ``appearance``."""
        window, w = self.session.window, self.session.w
        if size:
            window.resize(*size)
            settle(500)
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

    def matrix(self, state: str) -> None:
        """Save ``state`` at every size and appearance, then return to 1280×800."""
        for size in SIZES:
            for appearance in APPEARANCES:
                self.shot(state, size, appearance)
        self.session.window.resize(1280, 800)
        settle(400)


def _type(edit: QtWidgets.QLineEdit, text: str) -> None:
    edit.selectAll()
    QtTest.QTest.keyClicks(edit, text)
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key.Key_Return)
    settle(300)


def _after_fit(c: Capture, proton) -> None:
    """The GUI's own fit, the Run page, then Model edits marked against the run and Undo."""
    w = c.session.w
    c.session.fit()
    w.select_peak(None)
    settle(1500)
    c.matrix("08-run-after-fit")
    w.show_model_page()
    settle(300)
    w.panel.model_page.passes.setValue(3)
    proton.family.setCurrentIndex(proton.family.findData("sp1"))
    settle(600)
    for appearance in APPEARANCES:
        c.shot("09-model-differs-from-run", None, appearance)
    w.history(False)  # Undo the family change
    settle(600)
    c.shot("10-after-undo")


def run(output: Path, *, fit: bool) -> list[dict[str, Any]]:
    """Walk the workflow once and write the figures and ``measurements.json``."""
    install_log(output / "peakfit.log")  # as launch does
    c = Capture(output)
    w = c.session.w
    wait_until(lambda: w.fitting.prepared is not None, timeout=120)
    w.show_model_page()
    settle(300)
    table = w.panel.model_page.axes
    assert table is not None
    nitrogen, proton = table.rows
    c.matrix("01-model-before-fit")
    nitrogen.add.click()  # + J opens the J field, focused, with its hint
    settle(150)
    c.shot("02-adding-j")
    _type(nitrogen.j, "92")  # the doublet, with its default range
    c.matrix("03-doublet")
    nitrogen.j.setFocus(QtCore.Qt.FocusReason.TabFocusReason)
    QtTest.QTest.keyClick(nitrogen.j, QtCore.Qt.Key.Key_Tab)  # J → Fit J
    settle(200)
    c.shot("04-tab-from-j")
    _type(nitrogen.low, "100")  # J start outside its range: explained, Fit blocked
    for appearance in APPEARANCES:
        c.shot("05-invalid-range", None, appearance)
    c.shot("05-invalid-range", (1160, 720))
    c.session.window.resize(1280, 800)
    settle(400)
    _type(nitrogen.low, "46")
    proton.phase.setChecked(True)  # then a family without a phase
    proton.family.setCurrentIndex(proton.family.findData("gaussian"))
    settle(300)
    c.shot("06-phase-without-phase")
    proton.phase.setChecked(False)
    proton.family.setCurrentIndex(proton.family.findData("auto"))
    nitrogen.remove.click()  # the bundled spectrum has no doublet
    settle(300)
    c.shot("07-doublet-removed")
    if fit:
        _after_fit(c, proton)
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
    parser = argparse.ArgumentParser(description="Native captures of the Run and Model pages.")
    parser.add_argument("--output", type=Path, required=True, help="outside the repository")
    parser.add_argument("--body-px", type=int, metavar="PX", help="simulate a system font")
    parser.add_argument("--no-fit", action="store_true", help="stop before the fit")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("captures stay outside the repository; choose another --output")
    _app = QtWidgets.QApplication([])  # alive for the run
    if args.body_px:
        themes.use_type_scale(themes.TypeScale.for_body(args.body_px))
    records = run(output, fit=not args.no_fit)
    print(f"Wrote {len(records)} captures to {output}")  # noqa: T201


if __name__ == "__main__":
    main()
