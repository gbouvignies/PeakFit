"""Native captures of the Model page's W4 settings through one real workflow (#216).

``docs/validation/gui-model-settings.md`` cites the figures and measurements this
tool writes. Run it on a desktop (natively, not offscreen)::

    uv run python tools/capture_model_settings.py --output <directory outside the repo>

It opens the real window on the bundled ``02-advanced-fitting`` dataset through
``validate_gui_layout.Session`` and captures, at 1160×720, 1280×800 and 1920×1080 in
both appearances: the Model page as loaded; each new setting set (manual noise,
an absolute level, excluded planes, a position window and a rule); each invalid;
plane 41 excluded from the plane strip; then the GUI's own fit with that plane
excluded, its Run page, the excluded plane in Overlay with a peak's profile, and
Model edits that differ from the run, then Undo. ``--sizes`` limits the sizes, and
``--body-px 17`` simulates a large system font. Each state is saved as the whole window
(downscaled WebP) and the inspector alone (device pixels), with
``measurements.json``: inspector and viewport sizes, sideways scrolling, where the
first setting sits, clipped labels and Fit's state.
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
EXCLUDED = 41  # one-based plane number, excluded before the fit


class Capture:
    """The window and inspector at each state, with the measurements of each."""

    def __init__(self, output: Path) -> None:
        self.output = output
        for folder in ("full", "inspector"):
            (output / folder).mkdir(parents=True, exist_ok=True)
        self.session = Session(output, stall=STALL)
        self.records: list[dict[str, Any]] = []
        self.sizes = SIZES

    def measures(self) -> dict[str, Any]:
        """What the inspector shows at this state, in logical px."""
        w = self.session.w
        panel = w.panel
        body, viewport, model = panel.widget(), panel.viewport(), panel.model_page

        def top(widget: QtWidgets.QWidget) -> int:
            return widget.mapTo(body, QtCore.QPoint(0, 0)).y() - panel.verticalScrollBar().value()

        return {
            "inspector": [panel.width(), panel.height()],
            "viewport": [viewport.width(), viewport.height()],
            "sideways_scroll": panel.horizontalScrollBar().isVisible(),
            "page": panel.page.value,
            "page_min_width": model.minimumSizeHint().width(),
            # The first setting (the axis table) must start above the fold.
            "first_setting_top": top(model.axes) if model.isVisible() and model.axes else None,
            "noise_top": top(model.noise_mode) if model.isVisible() else None,
            "fit_enabled": w.fit_button.isEnabled(),
            "fit_tooltip": w.fit_button.toolTip(),
            "invalid": model.invalid,
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
        record = {"file": name, **self.measures()}
        self.records.append(record)
        # Acceptance, not just a record: nothing scrolls sideways or is cut.
        if record["sideways_scroll"] or record["clipped"]:
            raise RuntimeError(f"{name}: sideways scrolling or clipped text: {record}")
        arm_watchdog(STALL)

    def matrix(self, state: str, scroll_to: QtWidgets.QWidget | None = None) -> None:
        """Save ``state`` at every size and appearance, then return to 1280×800."""
        for size in self.sizes:
            for appearance in APPEARANCES:
                for _ in range(2):  # a first resize past the screen can be clamped
                    self.session.window.resize(*size)
                    settle(400)
                if scroll_to is not None:
                    self.session.w.panel.ensureWidgetVisible(scroll_to, 0, 24)
                    settle(150)
                self.shot(state, None, appearance)
        self.session.window.resize(1280, 800)
        settle(400)


def _type(edit: QtWidgets.QLineEdit, text: str) -> None:
    edit.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
    edit.selectAll()
    edit.insert(text)  # QTest.keyClicks refuses non-ASCII such as – and −
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key.Key_Return)
    settle(300)


def _choose(combo: QtWidgets.QComboBox | themes.MenuChoice, data: str) -> None:
    combo.setCurrentIndex(combo.findData(data))
    settle(300)


def _set(c: Capture) -> None:
    """Each new setting set to a valid value."""
    model = c.session.w.panel.model_page
    _choose(model.noise_mode, "manual")
    _choose(model.threshold_mode, "level")
    _type(model.excluded, "1, 41–42")
    model.constraints_toggle.setChecked(True)
    settle(200)
    editor = model.constraints
    assert editor is not None
    _type(editor.windows["F2"], "0.5")
    editor.add_rule.click()
    settle(200)
    row = editor.rules[-1]
    _choose(row.kind, "lw")
    _type(row.low, "5")
    _type(row.high, "80")


def _invalid(c: Capture) -> None:
    """Each new setting holding a value that cannot be fitted."""
    model = c.session.w.panel.model_page
    _type(model.noise_value, "0")
    _type(model.level_value, "−5")
    _type(model.excluded, "1–131")
    editor = model.constraints
    assert editor is not None
    _type(editor.rules[-1].start, "120")  # outside 5–80
    editor.add_rule.click()  # η: no parameter on these SP1/SP2 axes
    settle(200)
    row = editor.rules[-1]
    _choose(row.kind, "eta")
    row.hold.setChecked(True)
    settle(300)


def _reset(c: Capture) -> None:
    """Undo every settings edit; drafts follow the restored settings."""
    w = c.session.w
    while w.edits.can_undo:
        w.history(False)
    settle(600)
    model = w.panel.model_page
    model.constraints_toggle.setChecked(False)
    w.panel.verticalScrollBar().setValue(0)
    settle(300)


def _differs_and_undo(c: Capture) -> None:
    """Model edits that differ from the run, then Undo of the last one."""
    w = c.session.w
    model = w.panel.model_page
    w.set_mode(0)
    w.select_peak(None)
    w.show_model_page()
    settle(300)
    _choose(model.noise_mode, "manual")
    _choose(model.threshold_mode, "level")
    w.exclude_plane_action.trigger()  # include plane 41 again
    model.constraints_toggle.setChecked(True)
    settle(200)
    assert model.constraints is not None
    _type(model.constraints.windows["F3"], "0.05")
    wait_until(lambda: w.fitting.prepared is not None, timeout=120)
    c.matrix("09-model-differs-from-run")
    w.history(False)  # Undo the window: the run stays Outdated
    settle(600)
    c.matrix("10-after-undo")


def run(output: Path, sizes=SIZES) -> list[dict[str, Any]]:
    """Walk the workflow once and write the figures and ``measurements.json``."""
    install_log(output / "peakfit.log")  # as launch does
    c = Capture(output)
    c.sizes = sizes
    w = c.session.w
    wait_until(lambda: w.fitting.prepared is not None, timeout=120)
    w.show_model_page()
    settle(300)
    model = w.panel.model_page
    c.matrix("01-model-as-loaded")
    _set(c)
    wait_until(lambda: w.fitting.prepared is not None, timeout=120)
    c.matrix("02-settings-set")
    c.matrix("03-settings-set-constraints", scroll_to=model.constraints)
    _invalid(c)
    c.matrix("04-settings-invalid")
    c.matrix("05-settings-invalid-constraints", scroll_to=model.constraints)
    _reset(c)
    # Exclude plane 41 where it is seen: the plane strip, then Analysis ▸ Exclude.
    w.set_plane(EXCLUDED)
    w.exclude_plane_action.trigger()
    wait_until(lambda: w.fitting.prepared is not None, timeout=120)
    c.matrix("06-plane-excluded")
    blocked = model.invalid or w.fitting.blocked
    if blocked or not w.fit_button.isEnabled():
        raise RuntimeError(f"Fit is blocked before the fit: {blocked or w.fit_button.toolTip()}")
    c.session.fit()
    w.select_peak(None)
    w.show_run_page()
    settle(1500)
    c.matrix("07-run-after-fit")
    w.set_mode(1)
    w.select_peak(w.peaks[0].id, center=True)
    settle(1500)
    c.matrix("08-excluded-plane-overlay-profile")
    _differs_and_undo(c)
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
    parser = argparse.ArgumentParser(description="Native captures of the W4 Model settings.")
    parser.add_argument("--output", type=Path, required=True, help="outside the repository")
    parser.add_argument("--body-px", type=int, metavar="PX", help="simulate a system font")
    parser.add_argument(
        "--sizes", nargs="+", default=[f"{w}x{h}" for w, h in SIZES], help="e.g. 1160x720"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("captures stay outside the repository; choose another --output")
    _app = QtWidgets.QApplication([])  # alive for the run
    if args.body_px:
        themes.use_type_scale(themes.TypeScale.for_body(args.body_px))
    sizes = tuple(tuple(int(v) for v in size.split("x")) for size in args.sizes)
    records = run(output, sizes)
    print(f"Wrote {len(records)} captures to {output}")  # noqa: T201


if __name__ == "__main__":
    main()
