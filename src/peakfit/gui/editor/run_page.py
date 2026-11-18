"""The inspector's Run page: the active run as recorded, and what to do about it.

A failed attempt leads, then the run's identity and outcome counts with the warning
action right under them, the changes from a baseline run (its parent unless another
retained run is chosen in the section's heading) and, on request, the signal not
explained on the current plane. Every value comes from the recorded outcome; the
page never refits or recomputes statistics, and choosing a baseline changes nothing
but what the changes section compares.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.constraints import ParameterConfig
from peakfit.gui import formatting
from peakfit.gui import theme as themes
from peakfit.gui.editor.constraints import rule_text
from peakfit.gui.editor.labels import InfoButton, Row, clear_layout, label_rows, section
from peakfit.gui.editor.labels import text_label as _label
from peakfit.gui.editor.model_page import constraints_text, planes_fitted
from peakfit.gui.editor.peak_parameters import limit_text
from peakfit.gui.failures import log_path
from peakfit.gui.fitting import parameter_title

NBSP = "\u00a0"

if TYPE_CHECKING:
    from peakfit.gui.dataset_fit import DatasetFitResult


def submitted_rows(result: DatasetFitResult, names: dict[str, str]) -> list[Row]:
    """The settings a run was submitted with, exactly as recorded, as Run page rows.

    Noise and threshold always; the planes and constraints when the run had any.
    A manual noise or level shows the value entered; an estimate, as computed.
    """
    settings = PeakFitConfig.model_validate_json(result.inputs.snapshot.configuration)
    clustering = settings.clustering
    level = result.inputs.contour_level
    rows = [
        Row(
            "Noise",
            f"{formatting.setting(settings.noise_level)} · manual"
            if settings.noise_level is not None
            else f"{formatting.scaled(result.inputs.noise)} · estimated",
            tip="The statistical noise this run used",
        ),
        Row(
            "Threshold",
            f"level {formatting.setting(level)}"
            if clustering.contour_level is not None
            else f"{formatting.setting(clustering.contour_factor)} × noise = "
            f"{formatting.scaled(level)}",
            tip="The clustering threshold this run used",
        ),
    ]
    if excluded := settings.exclude_planes:
        total = result.inputs.snapshot.data_shape[0]
        rows.append(
            Row(
                "Planes",
                # Each phrase stays whole: the line breaks only at the dot.
                f"{planes_fitted(total, excluded)} · "
                f"{formatting.plane_numbers(excluded)}{NBSP}excluded",
                tip="Excluded planes are measured, with no fitted amplitude",
            )
        )
    parameters = settings.parameters
    own = [p for p in result.inputs.snapshot.peaks if p.limits]
    if parameters != ParameterConfig() or own:
        rules = [rule_text(p, c) for p, c in parameters.defaults.patterns.items()]
        engine = {i.working_id: i.engine_name for i in result.inputs.identities}
        submitted = result.inputs.parameters
        for peak in own:
            for key, limit in peak.limits:
                axis, _, kind = key.partition(".")
                title = parameter_title(kind, names.get(axis) or axis)
                name = f"{engine[peak.id]}.{key}"
                unit = submitted[name].unit if name in submitted else ""
                rules.append(f"{peak.name} {title}: {limit_text(limit, unit)}")
        text = constraints_text(parameters, names, len(own))
        rows.append(Row("Constraints", text, detail="\n".join(rules)))
    return rows


@dataclass(frozen=True, slots=True)
class BaselineChoice:
    """A run the active run can be compared with, as the heading's menu lists it."""

    label: str  # "Run 3", shown on the menu button
    text: str  # "Run 3 · <time> · parent", the menu item
    run_id: str


def _rows() -> QtWidgets.QGridLayout:
    grid = QtWidgets.QGridLayout()
    grid.setHorizontalSpacing(themes.SPACE_S)
    grid.setVerticalSpacing(3)
    return grid


class RunPage(QtWidgets.QWidget):
    """Failure, active run, changes from a baseline run and unexplained signal."""

    next_warning = QtCore.Signal()
    baseline_chosen = QtCore.Signal(str)  # the run ID to compare the active run with
    change_chosen = QtCore.Signal(str, bool)  # working ID, open its cluster page
    unexplained_requested = QtCore.Signal()
    region_chosen = QtCore.Signal(int)  # index in the unexplained-signal list

    def __init__(self) -> None:  # noqa: PLR0915
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(themes.SPACE_S)
        # A failed attempt leads the page.
        self.failure_block = QtWidgets.QWidget()
        failure = QtWidgets.QVBoxLayout(self.failure_block)
        failure.setContentsMargins(0, 0, 0, 0)
        failure.setSpacing(themes.SPACE_S)
        section(failure, "Last fit attempt")
        self.failure_rows = _rows()
        failure.addLayout(self.failure_rows)
        self.copy_details = themes.command("Copy details")
        self.copy_details.setToolTip("Copy the stage, message and diagnostic context")
        self.copy_details.clicked.connect(self._copy_failure_details)
        failure.addWidget(self.copy_details, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self._failure_details = ""
        self.failure_block.hide()
        layout.addWidget(self.failure_block)
        self.run_block = QtWidgets.QWidget()
        run = QtWidgets.QVBoxLayout(self.run_block)
        run.setContentsMargins(0, 0, 0, 0)
        run.setSpacing(themes.SPACE_S)
        section(run, "Active run")
        self.run_rows = _rows()
        run.addLayout(self.run_rows)
        # The warnings' action sits under the run rows that count them.
        self.next_button = themes.command("Select next cluster with a warning")
        self.next_button.clicked.connect(self.next_warning)
        run.addWidget(self.next_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        # The heading names the baseline and chooses it: "CHANGES FROM  Run 3 ▾".
        run.addSpacing(themes.SPACE_M)
        self.changes_heading = QtWidgets.QWidget()
        heading = QtWidgets.QHBoxLayout(self.changes_heading)
        heading.setContentsMargins(0, 0, 0, 0)
        heading.setSpacing(themes.SPACE_S)
        self.changes_title = _label("CHANGES FROM", "section")
        heading.addWidget(self.changes_title)
        self.baseline = themes.MenuChoice(
            "Compare with",
            "The run the active run is compared with; nothing else changes",
            [("", "", "")],  # until a run has earlier runs; the heading is hidden
        )
        self.baseline.currentIndexChanged.connect(
            lambda _index: self.baseline_chosen.emit(self.baseline.currentData())
        )
        heading.addWidget(self.baseline, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        heading.addStretch(1)
        self.changes_heading.hide()
        run.addWidget(self.changes_heading)
        self.changes = QtWidgets.QVBoxLayout()
        self.changes.setSpacing(2)
        run.addLayout(self.changes)
        self._changes_key = None
        self.unexplained_section = section(run, "Signal not explained on this plane")
        # What the list is sits beside the button that computes it, not on the heading,
        # which would then need more than the inspector's width.
        request = QtWidgets.QHBoxLayout()
        request.setSpacing(4)
        self.unexplained_button = themes.command("")
        self.unexplained_button.clicked.connect(self.unexplained_requested)
        request.addWidget(self.unexplained_button)
        self.unexplained_info = InfoButton("Signal not explained")
        request.addWidget(self.unexplained_info)
        request.addStretch(1)
        run.addLayout(request)
        self.unexplained_title = _label("", "muted", wrap=True)
        run.addWidget(self.unexplained_title)
        self.unexplained = QtWidgets.QVBoxLayout()
        self.unexplained.setSpacing(2)
        run.addLayout(self.unexplained)
        self.unexplained_note = _label("", "caption", wrap=True)
        run.addWidget(self.unexplained_note)
        self._unexplained_key = None
        layout.addWidget(self.run_block)
        self._key = None
        self._fitted = False
        self._about = ""

    def show_run(self, rows: list[Row], *, fitted: bool, warned: bool) -> None:
        """The active run's rows; without a run and rows, only a failure can show."""
        self.run_block.setVisible(fitted or bool(rows))
        self._fitted = fitted
        for widget in (self.unexplained_section, self.unexplained_button, self.unexplained_note):
            widget.setVisible(fitted)
        self._show_info()
        key = (tuple(rows), warned)
        if key == self._key:
            return
        self._key = key
        clear_layout(self.run_rows)
        label_rows(self.run_rows, rows)
        self.next_button.setVisible(warned)

    def show_failure(self, rows: list[Row], details: str) -> None:
        """Lead the page with a failed attempt, or hide it with no rows.

        ``details`` is the plain text **Copy details** puts on the clipboard.
        """
        self._failure_details = details
        self.failure_block.setVisible(bool(rows))
        path = log_path()
        if rows and path is not None:
            rows = [
                *rows,
                Row("Log", str(path), style="caption", tip="The traceback is in this file"),
            ]
        clear_layout(self.failure_rows)
        label_rows(self.failure_rows, rows)

    def _copy_failure_details(self) -> None:
        QtGui.QGuiApplication.clipboard().setText(self._failure_details)

    def show_changes(
        self, choices: list[BaselineChoice], chosen: str, entries: list[tuple]
    ) -> None:
        """Show run-to-run changes against the ``chosen`` baseline run (its run ID).

        ``choices`` are the runs to compare with; with no baseline chosen the
        heading hides. Entries are ("row", label, value,
        tooltip), ("heading", text[, tooltip]), ("note", text) or ("link", text,
        detail, working ID, opens a cluster page[, warning]). Consecutive rows share
        one label grid.
        """
        key = (tuple(choices), chosen, tuple(entries))
        if key == self._changes_key:
            return
        self._changes_key = key
        ids = [c.run_id for c in choices]
        shown = chosen in ids
        self.changes_heading.setVisible(shown)
        if shown:
            self.baseline.set_choices(
                [(c.label, c.text, c.run_id) for c in choices], ids.index(chosen)
            )
        clear_layout(self.changes)
        rows: list[Row] = []

        def flush() -> None:
            if rows:
                grid = _rows()
                label_rows(grid, rows)
                self.changes.addLayout(grid)
                rows.clear()

        for entry in entries:
            kind = entry[0]
            if kind == "row":
                _kind, label, value, tip = entry
                rows.append(Row(label, value, tip=tip))
                continue
            flush()
            if kind == "link":
                _kind, text, detail, identity, cluster, *flag = entry
                row = QtWidgets.QHBoxLayout()
                row.setSpacing(themes.SPACE_S)
                link = themes.link(text)
                link.clicked.connect(
                    lambda _checked=False, i=identity, c=cluster: self.change_chosen.emit(i, c)
                )
                row.addWidget(link, 0, QtCore.Qt.AlignmentFlag.AlignTop)
                style = "warning" if flag and flag[0] else "caption"
                row.addWidget(_label(detail, style, wrap=True), 1)
                self.changes.addLayout(row)
            elif kind == "heading":
                self.changes.addSpacing(themes.SPACE_XS)
                heading = _label(entry[1], "muted")
                heading.setToolTip(entry[2] if len(entry) > 2 else "")  # noqa: PLR2004
                self.changes.addWidget(heading)
            else:
                self.changes.addWidget(
                    _label(entry[1], "" if kind == "line" else "caption", wrap=True)
                )
        flush()

    def show_unexplained(
        self,
        button: str,
        enabled: bool,
        title: str,
        rows: list[tuple[str, str]],
        note: str,
        about: str = "",
    ) -> None:
        """The on-request list of residual regions; each row's link frames its region."""
        key = (button, enabled, title, tuple(rows), note, about)
        if key == self._unexplained_key:
            return
        self._unexplained_key = key
        self._about = about
        self.unexplained_info.set_text(about)
        self._show_info()
        self.unexplained_button.setText(button)
        self.unexplained_button.setEnabled(enabled)
        self.unexplained_title.setText(title)
        self.unexplained_title.setVisible(bool(title))
        self.unexplained_note.setText(note)
        clear_layout(self.unexplained)
        for index, (text, detail) in enumerate(rows):
            if not text:
                self.unexplained.addWidget(_label(detail, "caption"))
                continue
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(themes.SPACE_S)
            link = themes.link(text)
            link.setToolTip("Frame this region in the spectrum")
            link.clicked.connect(lambda _checked=False, i=index: self.region_chosen.emit(i))
            row.addWidget(link, 0, QtCore.Qt.AlignmentFlag.AlignTop)
            row.addWidget(_label(detail, "caption", wrap=True), 1)
            self.unexplained.addLayout(row)

    def _show_info(self) -> None:
        """The unexplained-signal ⓘ shows with its section, when it has something to say."""
        self.unexplained_info.setVisible(self._fitted and bool(self._about))
