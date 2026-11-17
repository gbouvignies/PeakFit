"""Figures and measurements of the 2026-11 GUI product review.

``docs/validation/gui-product-review.md`` cites every figure this tool writes and
quotes its measurements through :func:`table`. Run it on a desktop (or headless
with ``QT_QPA_PLATFORM=offscreen``)::

    uv run python tools/capture_gui_review.py --output <directory outside the repo>

It opens the real window on the bundled ``02-advanced-fitting`` dataset through
``validate_gui_layout.Session`` and walks the review's states once, at 1280×800
light unless a state says otherwise: the dialogs, the Model page, a simulated
failed first fit, the GUI's two-pass fit, the overview, the unexplained-signal
list, the peak and cluster pages with every disclosure open, cluster focus in
Overlay and Residual, Profiles, window sizes from 1160×720 to 1920×1080, the
analysis history, the Outdated state after moving 22N-H by +0.02 ppm in ¹H, and
the refit. It writes the cited figures as WebP and ``measurements.json``, which
records the commit, the platform and every figure's window size.

The committed folder records the review commit. A run on a later commit describes
that commit; compare it with the folder, and replace the folder only together with
the note that quotes it.
"""

import argparse
import faulthandler
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets
from validate_gui_editor import settle, wait_until
from validate_gui_layout import (
    CLUSTER,
    DATA,
    FAILURE,
    ROOT,
    Session,
    arm_watchdog,
    clipped_labels,
    peak_page_measures,
    steady,
    window_record,
)

from peakfit.gui.editor.numerical import NumericalReview
from peakfit.gui.failures import install_log
from peakfit.gui.theme import type_scale

SIZE = "1280x800"
SIZES = ("1160x720", "1280x800", "1440x900", "1920x1080")
PEAK = "53N-H"  # a two-member cluster's peak: the reviewed peak page
MOVED = CLUSTER  # the four-member cluster's peak, moved to make the run Outdated
MOVE_PPM = 0.02  # in the direct (¹H) dimension
MAX_WIDTH = 1600  # px; wider grabs are downscaled, keeping the aspect ratio
QUALITY = 82  # WebP

# Every figure the note cites, in the order the tool writes them.
FIGURES = (
    "dialog-open-dataset.webp",
    "dialog-shortcuts.webp",
    "fit-setup.webp",
    "page-fit-setup.webp",
    "dialog-display.webp",
    "overview.webp",
    "page-analysis-unexplained.webp",
    f"peak-{PEAK}.webp",
    f"page-peak-{PEAK}.webp",
    f"page-cluster-{CLUSTER}.webp",
    "focus-overlay.webp",
    "focus-residual-dark.webp",
    "profiles.webp",
    "overview-1160x720.webp",
    "dialog-history.webp",
    "outdated-peak.webp",
    "page-analysis-refit.webp",
)


def window_size(size: str) -> tuple[int, int]:
    """Width and height of a ``WxH`` size."""
    width, height = (int(v) for v in size.split("x"))
    return width, height


def git_state() -> dict[str, object]:
    """The checkout's commit and whether tracked files differ from it."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "modified": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "modified": None}


def text_lines(label: QtWidgets.QLabel) -> int:
    """Rendered lines of a wrapped label."""
    return round(label.contentsRect().height() / label.fontMetrics().lineSpacing())


def lines_of(root: QtWidgets.QWidget, text: str) -> list[int]:
    """Rendered lines of every visible label under ``root`` that contains ``text``."""
    return [
        text_lines(label)
        for label in root.findChildren(QtWidgets.QLabel)
        if label.isVisible() and text in label.text()
    ]


def layout_texts(layout: QtWidgets.QLayout) -> list[str]:
    """Texts of the visible labels and buttons in a layout, in reading order."""
    texts = []
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item is None:
            continue
        if (nested := item.layout()) is not None:
            texts += layout_texts(nested)
        widget = item.widget()
        if isinstance(widget, QtWidgets.QLabel | QtWidgets.QAbstractButton) and widget.text():
            texts.append(widget.text())
    return texts


class ReviewSession(Session):
    """The layout tool's session, writing downscaled WebP figures and their records."""

    def __init__(self, output: Path) -> None:
        super().__init__(output)
        self.figures: dict[str, dict[str, Any]] = {}

    def figure(self, name: str, widget: QtWidgets.QWidget | None = None) -> None:
        """Save ``widget`` (the window by default) once the view has drawn."""
        wait_until(lambda: not self.w.canvas.contour_pending, timeout=60)
        settle(150)
        self._save(name, widget or self.window)

    def _save(self, name: str, widget: QtWidgets.QWidget) -> None:
        image = widget.grab()
        if image.width() > MAX_WIDTH:  # device pixels
            image = image.scaledToWidth(
                MAX_WIDTH, QtCore.Qt.TransformationMode.SmoothTransformation
            )
        file = f"{name}.webp"
        if not image.save(str(self.output / file), "WEBP", QUALITY):
            raise RuntimeError(f"Qt could not write {file}; is the WebP image plugin installed?")
        self.figures[file] = {
            "window": window_record(self.window),
            "widget": [widget.width(), widget.height()],
            "image": [image.width(), image.height()],
            "clipped": clipped_labels(self.window),
        }
        arm_watchdog(self.stall)

    def page(self, name: str) -> int:
        """Save the inspector's whole scrolling page; its height in logical px."""
        body = self.w.panel.widget()
        height = steady(body.height)  # deferred rows can still be growing the page
        self.figure(name, body)
        return height

    def dialog(self, name: str, show, ready=lambda _dialog: True) -> list[int]:
        """Open a modal dialog with ``show``, save it once ``ready``, close it; its size."""
        size: list[int] = []
        returned = False

        def poll() -> None:
            if returned:  # ``show`` came back without a modal dialog
                return
            dialog = QtWidgets.QApplication.activeModalWidget()
            if dialog is None or not ready(dialog):
                QtCore.QTimer.singleShot(50, poll)
                return
            QtCore.QTimer.singleShot(400, lambda: grab(dialog))

        def grab(dialog: QtWidgets.QWidget) -> None:
            self._save(name, dialog)
            size.extend((dialog.width(), dialog.height()))
            assert isinstance(dialog, QtWidgets.QDialog)
            dialog.reject()

        QtCore.QTimer.singleShot(0, poll)
        show()
        returned = True
        if not size:
            raise RuntimeError(f"No modal dialog was captured for {name}")
        return size

    def setup_top(self) -> int | None:
        """Where the Model page's settings start, in inspector content px; None off it."""
        block, body = self.w.panel.model_page, self.w.panel.widget()
        if not block.isVisible():
            return None
        return steady(lambda: block.mapTo(body, QtCore.QPoint(0, 0)).y())

    def noise_lines(self) -> list[int]:
        """Lines of each visible inspector label that contains the noise-check finding.

        The Run page's ⓘ holds the finding as written; reports and explanations
        quote it after a prefix, with its first letter lowercased, hence ``[1:]``.
        """
        result = self.w.fitting.result
        finding = result.outcome.noise_model.power_finding if result else None
        return lines_of(self.w.panel.widget(), finding[1:60]) if finding else []

    def run_line(self) -> str:
        """The toolbar's run summary (the separate "Modified" label is not part of it)."""
        return self.w.summary.text()


def history_ready(dialog: QtWidgets.QWidget) -> bool:
    """The history dialog has loaded the run it shows."""
    review = dialog.findChild(NumericalReview)
    return review is not None and not review.details.toPlainText().startswith(
        ("Loading", "Reconstructing")
    )


def before_fit(s: ReviewSession, record: dict[str, Any]) -> None:
    """Dialogs, the Model page and a failed first fit."""
    win, w = s.window, s.w
    s.dialog("dialog-open-dataset", win.open_dialog)
    s.dialog("dialog-shortcuts", win.shortcuts)
    w.show_model_page()
    wait_until(lambda: w.fitting.prepared is not None, timeout=120)
    s.figure("fit-setup")
    record["dataset_size"] = {
        "peaks": len(w.peaks),
        "planes": int(w.dataset.spectra.data.shape[0]),
    }
    record["preparation"] = w.panel.model_page.preparation.text()
    w.panel.model_page.limits_toggle.setChecked(True)
    s.page("page-fit-setup")
    w.panel.model_page.limits_toggle.setChecked(False)
    record["display_dialog"] = s.dialog("dialog-display", w.display_dialog)
    # A first fit that fails: nothing is retained, which the Run page must say.
    w.fitting._finished(0, RuntimeError(FAILURE))
    w.show_run_page()
    settle(300)
    record["failure"] = {"attempt": w.fitting.attempt, "run_line": s.run_line()}
    w.show_model_page()


def after_fit(s: ReviewSession, record: dict[str, Any]) -> None:
    """Overview, unexplained signal, peak and cluster pages, focus and Profiles."""
    w = s.w
    w.show_view(profiles=False)
    w.show_run_page()
    settle(300)
    s.figure("overview")
    record["overview"] = {
        "run_line": s.run_line(),
        "spectral": s.spectral(),
        "viewport_height": w.panel.viewport().height(),
        "setup_top": s.setup_top(),
        "noise_check_lines": s.noise_lines(),
    }
    w.find_unexplained()
    wait_until(lambda: isinstance(w._unexplained[1], list), timeout=120)
    regions = w._unexplained[1]
    assert isinstance(regions, list)
    strongest = max(regions, key=lambda region: abs(region.extreme))
    canvas = w.canvas
    record["unexplained"] = {
        "plane": canvas.plane + 1,
        "threshold": w.unexplained_threshold,
        "regions": len(regions),
        "listed": sum(1 for text in layout_texts(w.panel.run_page.unexplained) if "σ ·" in text),
        "title": w.panel.run_page.unexplained_title.text(),
        "strongest": {
            "extreme": round(strongest.extreme, 1),
            "ppm": {
                canvas.x_axis.label: round(float(canvas.ppm_x[strongest.column]), 3),
                canvas.y_axis.label: round(float(canvas.ppm_y[strongest.row]), 2),
            },
        },
        "page_height": s.page("page-analysis-unexplained"),
    }
    panel = w.panel
    w.select_peak(s.peak_id(PEAK), center=True)
    page = steady(lambda: peak_page_measures(panel))
    s.figure(f"peak-{PEAK}")
    amplitude = panel._amplitude
    assert amplitude is not None
    title, value, sigma = (cell.text() for cell in amplitude.cells)
    # Every disclosure open: the whole page.
    toggles = (
        panel.starts_toggle,
        panel.profile.about_toggle,
        panel.about_toggle,
        panel.provenance_toggle,
        panel.cluster_page.provenance_toggle,
    )
    for toggle in toggles:
        toggle.setChecked(True)
    height = s.page(f"page-peak-{PEAK}")
    record["peak"] = {
        "name": PEAK,
        **page,
        "amplitude": f"{title} {value} ± {sigma}".strip(),
        "page_height": height,
        "error_bars_lines": text_lines(panel.profile.about),
        "noise_check_lines": s.noise_lines(),
    }
    w.select_peak(s.peak_id(CLUSTER), cluster=True, center=True)
    record["cluster"] = {
        "name": CLUSTER,
        "page_height": s.page(f"page-cluster-{CLUSTER}"),
        "noise_check_lines": s.noise_lines(),
    }
    for toggle in toggles:
        toggle.setChecked(False)
    w.open_focus(s.peak_id(CLUSTER))
    w.set_mode(1)
    record["focus_overlay"] = s.spectral()
    s.figure("focus-overlay")
    w.set_mode(2)
    s.window.apply_appearance("Dark")
    record["focus_residual"] = s.spectral()
    s.figure("focus-residual-dark")
    s.window.apply_appearance("Light")
    w.close_focus()
    w.set_mode(1)
    w.select_peak(s.peak_id(CLUSTER), cluster=True, center=True)
    w.show_view(profiles=True)
    wait_until(lambda: "Computing" not in w.profiles.caption.text(), timeout=60)
    s.figure("profiles")
    w.show_view(profiles=False)


def window_sizes(s: ReviewSession, record: dict[str, Any]) -> None:
    """The overview at each window size; the 1160×720 one is a figure."""
    w, sizes = s.w, []
    for size in SIZES:
        s.window.resize(*window_size(size))
        settle(600)
        w.show_run_page()
        if size == "1160x720":
            s.figure(f"overview-{size}")
        sizes.append(
            {
                "requested": size,
                "window": window_record(s.window),
                "inspector_width": w.panel.width(),
                "plot": s.spectral()["plot"],
            }
        )
    record["sizes"] = sizes
    screen = s.window.screen().size()
    record["screen"] = [screen.width(), screen.height()]
    s.window.resize(*window_size(SIZE))
    settle(600)


def outdated_and_refit(s: ReviewSession, record: dict[str, Any]) -> None:
    """History, then an edit that makes the run Outdated, then a refit."""
    w = s.w
    record["history_dialog"] = s.dialog("dialog-history", s.window.show_history, history_ready)
    w.navigator.tab_buttons[0].click()  # the Peaks tab
    identity = s.peak_id(MOVED)
    peak = next(p for p in w.peaks if p.id == identity)
    w.move_peak(identity, (peak.positions[0], peak.positions[1] + MOVE_PPM))
    w.select_peak(identity, center=True)
    settle(500)
    s.figure("outdated-peak")
    record["outdated"] = {"peak": MOVED, "moved_ppm": MOVE_PPM, "run_line": s.run_line()}
    s.fit()
    w.show_run_page()
    settle(1500)
    record["refit"] = {
        "run_line": s.run_line(),
        "changes": layout_texts(w.panel.run_page.changes),
        "viewport_height": w.panel.viewport().height(),
        "setup_top": s.setup_top(),
        "page_height": s.page("page-analysis-refit"),
    }


def run(output: Path) -> dict[str, Any]:
    """Walk every state once and write the figures and ``measurements.json``."""
    output.mkdir(parents=True, exist_ok=True)
    install_log(output / "peakfit.log")  # as launch does
    s = ReviewSession(output)
    s.window.resize(*window_size(SIZE))
    s.window.apply_appearance("Light")
    settle(300)
    states: dict[str, Any] = {}
    before_fit(s, states)
    s.w.show_model_page()
    s.fit()
    after_fit(s, states)
    window_sizes(s, states)
    outdated_and_refit(s, states)
    faulthandler.cancel_dump_traceback_later()
    assert set(s.figures) == set(FIGURES), sorted(set(s.figures) ^ set(FIGURES))
    record = {
        "tool": "tools/capture_gui_review.py",
        **git_state(),
        "dataset": str(DATA.relative_to(ROOT)),
        "platform": platform.platform(),
        "qt_platform": QtGui.QGuiApplication.platformName(),
        "units": "Qt logical pixels unless named; image sizes are file pixels",
        "type_scale": asdict(type_scale()),
        **states,
        "figures": {name: s.figures[name] for name in FIGURES},
    }
    (output / "measurements.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return record


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _size(size: list[int]) -> str:
    return f"{size[0]}×{size[1]}"


def table(record: dict[str, Any]) -> str:
    """The measurements the note quotes, as the note's Markdown table."""
    overview, peak, cluster = record["overview"], record["peak"], record["cluster"]
    unexplained, refit = record["unexplained"], record["refit"]
    strongest = unexplained["strongest"]
    where = ", ".join(f"{axis} {ppm}" for axis, ppm in strongest["ppm"].items())
    # What changed between the runs; the full list is in the record. A one-line
    # value follows its label, which is the text before it.
    texts = refit["changes"]
    changes = " · ".join(
        f"{texts[index - 1]} {line}" if index and "\n" not in text else line
        for index, text in enumerate(texts)
        for line in text.split("\n")
        if "now reported" in line or "→" in line
    )
    sizes = "; ".join(
        f"{s['window']['logical'][0]}×{s['window']['logical'][1]}: inspector "
        f"{s['inspector_width']}, plot {_size(s['plot'])}"
        for s in record["sizes"]
    )
    rows = [
        (
            "M1",
            "Dataset; preparation before the first fit",
            f"{record['dataset_size']['peaks']} peaks · {record['dataset_size']['planes']} "
            f"planes; {record['preparation']}",
        ),
        ("M2", "Failed first fit: run line", record["failure"]["run_line"]),
        ("M3", "Run line after the fit", overview["run_line"]),
        (
            "M4",
            "Fit setup top / inspector viewport, after the fit",
            f"{overview['setup_top']} / {overview['viewport_height']} px",
        ),
        (
            "M5",
            "Lines of the labels with the noise-check finding: Analysis, peak, cluster page",
            ", ".join(
                " + ".join(map(str, page["noise_check_lines"])) or "none"
                for page in (overview, peak, cluster)
            ),
        ),
        (
            "M6",
            f"{peak['name']} Fitted table top, bottom / viewport",
            f"{peak['fitted_top']}, {peak['fitted_bottom']} / {peak['viewport_height']} px",
        ),
        ("M7", f"{peak['name']} amplitude row", peak["amplitude"]),
        ("M8", "About these error bars, lines", peak["error_bars_lines"]),
        (
            "M9",
            f"Whole pages, disclosures open: peak {peak['name']}, cluster {cluster['name']}",
            f"{peak['page_height']}, {cluster['page_height']} px",
        ),
        (
            "M10",
            f"Focus {cluster['name']}: 2D plot, Overlay → Residual",
            f"{_size(record['focus_overlay']['plot'])} → {_size(record['focus_residual']['plot'])}",
        ),
        (
            "M11",
            f"Unexplained signal on plane {unexplained['plane']}: regions, rows listed, "
            "strongest (σ_stat, ppm)",
            f"{unexplained['regions']}, {unexplained['listed']}, "
            f"{strongest['extreme']:+.1f} at {where}",
        ),
        (
            "M12",
            f"Run line after moving {record['outdated']['peak']}",
            record["outdated"]["run_line"],
        ),
        ("M13", "Changes from parent run after the refit", changes),
        (
            "M14",
            "After the refit: Analysis page height, fit setup top / viewport",
            f"{refit['page_height']}, {refit['setup_top']} / {refit['viewport_height']} px",
        ),
        ("M15", "Window sizes (logical): inspector width, overview plot", sizes),
        ("M16", "Screen (logical)", _size(record["screen"])),
        (
            "M17",
            "Type scale: body, small, caption",
            "{body}, {small}, {caption} px".format(**record["type_scale"]),
        ),
    ]
    lines = ["| | Measurement | Value |", "| --- | --- | --- |"]
    lines += [f"| {key} | {_cell(name)} | {_cell(value)} |" for key, name, value in rows]
    return "\n".join(lines)


def main() -> None:
    """Refuse an output inside the repository, then capture."""
    parser = argparse.ArgumentParser(description="Figures and measurements of the GUI review.")
    parser.add_argument("--output", type=Path, required=True, help="outside the repository")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("captures stay outside the repository; choose another --output")
    _app = QtWidgets.QApplication([])  # alive for the run
    record = run(output)
    print(table(record))  # noqa: T201
    print(f"Wrote {len(FIGURES)} figures and measurements.json to {output}")  # noqa: T201


if __name__ == "__main__":
    main()
