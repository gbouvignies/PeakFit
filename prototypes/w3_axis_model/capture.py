# ruff: noqa  -- throwaway prototype (W3 step 1, #215); delete with the folder
"""THROWAWAY PROTOTYPE (W3 step 1, issue #215): capture both per-axis layouts natively.

    uv run python prototypes/w3_axis_model/capture.py --output <directory outside the repo>

Opens the real window on the bundled ``02-advanced-fitting`` dataset (as
``tools/validate_gui_layout.py`` does), waits for the real preparation, then puts
the prototype Model page at the top of the inspector's analysis page and captures
every layout × scenario × window size × appearance: the whole window (downscaled
WebP) and the inspector alone (device pixels). ``measurements.json`` records the
fold, widths and keyboard order of each capture.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(Path(__file__).parent)]

from model_page import LAYOUTS, SCENARIOS, ModelPage  # noqa: E402
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402
from validate_gui_editor import settle, wait_until  # noqa: E402  # ty: ignore[unresolved-import]
from validate_gui_layout import Session, arm_watchdog, window_record  # noqa: E402  # ty: ignore[unresolved-import]

from peakfit.gui import theme as themes  # noqa: E402

SIZES = ("1160x720", "1280x800", "1920x1080")
APPEARANCES = ("Light", "Dark")
MAX_WIDTH = 1600


def focus_order(page: ModelPage) -> list[str]:
    """Accessible names of the per-axis block's Tab stops, in Tab order."""
    block = page.axes
    start = block.first
    assert start is not None
    names, widget = [], start
    for _ in range(2000):
        inside = block.isAncestorOf(widget)
        if (
            inside
            and widget.isVisible()
            and widget.isEnabled()
            and widget.focusPolicy() & QtCore.Qt.FocusPolicy.TabFocus
        ):
            names.append(widget.accessibleName() or type(widget).__name__)
        widget = widget.nextInFocusChain()
        if widget is start:
            break
    return names


def measure(s: Session, page: ModelPage) -> dict[str, object]:
    panel = s.w.panel
    body = panel.widget()
    viewport = panel.viewport()

    def top(widget: QtWidgets.QWidget) -> int:
        return widget.mapTo(body, QtCore.QPoint(0, 0)).y()

    def bottom(widget: QtWidgets.QWidget) -> int:
        return top(widget) + widget.height()

    axes = page.axes
    fold = viewport.height()
    return {
        "inspector": [panel.width(), panel.height()],
        "viewport": [viewport.width(), fold],
        "content_width": viewport.width() - 2 * themes.SPACE_L,
        "axes_block": {
            "width": axes.width(),
            "needed_width": axes.minimumSizeHint().width(),
            "top": top(axes),
            "bottom": bottom(axes),
            "height": axes.height(),
        },
        "first_setting_top": top(axes.first),  # ty: ignore[invalid-argument-type]
        "threshold_bottom": bottom(page.threshold),
        "planes_bottom": bottom(page.planes),
        "page_bottom": bottom(page),
        "horizontal_scroll": panel.horizontalScrollBar().isVisible(),
        "above_fold": {
            "axes_block": bottom(axes) <= fold,
            "threshold": bottom(page.threshold) <= fold,
            "planes": bottom(page.planes) <= fold,
        },
        "tab_order": focus_order(page),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--body-px", type=int, help="simulate a larger system font")
    parser.add_argument("--sizes", nargs="+", default=list(SIZES))
    parser.add_argument("--appearance", nargs="+", default=list(APPEARANCES))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("captures stay outside the repository")
    for sub in ("full", "inspector"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    _app = QtWidgets.QApplication([])
    if args.body_px:
        themes.use_type_scale(themes.TypeScale.for_body(args.body_px))
    s = Session(output)
    w = s.w
    w.show_fit_setup()
    wait_until(lambda: w.fitting.prepared is not None, timeout=120)
    prepared = w.fitting.prepared
    config = w.fitting.configuration
    values = {
        "noise": prepared.noise,
        "level": prepared.contour_level,
        "factor": config.clustering.contour_factor,
        "clusters": len(prepared.clusters),
        "samples": sum(len(c.grid_indices[0]) for c in prepared.clusters),
        "planes": int(w.dataset.spectra.data.shape[0]),
        "passes": config.fitting.refine_iterations,
    }
    panel = w.panel
    records = []
    current: list[ModelPage] = []

    def place(page: ModelPage) -> None:
        for old in current:
            old.hide()
            old.setParent(None)
            old.deleteLater()
        current[:] = [page]
        for block in (panel.failure_block, panel.setup_block, panel.review_block):
            block.hide()
        panel._analysis_layout.insertWidget(0, page)

    for size in args.sizes:
        width, height = (int(v) for v in size.split("x"))
        s.window.resize(width, height)
        settle(600)
        for appearance in args.appearance:
            s.window.apply_appearance(appearance)
            theme = s.window.theme
            for layout in LAYOUTS:
                for scenario in SCENARIOS:
                    page = ModelPage(layout, scenario, theme, **values)
                    place(page)
                    settle(250)
                    for block in (panel.failure_block, panel.setup_block, panel.review_block):
                        block.hide()
                    panel.verticalScrollBar().setValue(0)
                    settle(150)
                    name = f"{layout}-{scenario.key}-{size}-{appearance.lower()}.webp"
                    image = s.window.grab()
                    if image.width() > MAX_WIDTH:
                        image = image.scaledToWidth(
                            MAX_WIDTH, QtCore.Qt.TransformationMode.SmoothTransformation
                        )
                    image.save(str(output / "full" / name), "WEBP", 82)
                    panel.grab().save(str(output / "inspector" / name), "WEBP", 90)
                    records.append(
                        {
                            "file": name,
                            "layout": layout,
                            "scenario": scenario.key,
                            "size": size,
                            "appearance": appearance,
                            "window": window_record(s.window),
                            **measure(s, page),
                        }
                    )
                    arm_watchdog(s.stall)
    report = {
        "platform": platform.platform(),
        "qt_platform": QtGui.QGuiApplication.platformName(),
        "values": values,
        "scenarios": {sc.key: sc.title for sc in SCENARIOS},
        "captures": records,
    }
    (output / "measurements.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} captures to {output}")  # noqa: T201
    s.window.close()


if __name__ == "__main__":
    main()
