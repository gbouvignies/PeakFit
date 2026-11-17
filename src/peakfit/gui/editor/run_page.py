"""The inspector's Run page: the active run as recorded, and what to do about it.

A failed attempt leads, then the run's identity and outcome counts with the warning
action right under them, the changes from its parent run and, on request, the
signal not explained on the current plane. Every value comes from the recorded
outcome; the page never refits or recomputes statistics.
"""

from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui import theme as themes
from peakfit.gui.editor.labels import InfoButton, Row, clear_layout, label_rows, section
from peakfit.gui.editor.labels import text_label as _label
from peakfit.gui.failures import log_path


def _rows() -> QtWidgets.QGridLayout:
    grid = QtWidgets.QGridLayout()
    grid.setHorizontalSpacing(themes.SPACE_S)
    grid.setVerticalSpacing(3)
    return grid


class RunPage(QtWidgets.QWidget):
    """Failure, active run, changes from the parent run and unexplained signal."""

    next_warning = QtCore.Signal()
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
        self.changes_title = section(run, "Changes from parent run")
        self.changes_title.hide()
        self.changes = QtWidgets.QVBoxLayout()
        self.changes.setSpacing(2)
        run.addLayout(self.changes)
        self._changes_key = None
        heading = QtWidgets.QHBoxLayout()
        heading.setSpacing(4)
        run.addSpacing(themes.SPACE_M)
        self.unexplained_section = _label("SIGNAL NOT EXPLAINED ON THIS PLANE", "section")
        heading.addWidget(self.unexplained_section)
        self.unexplained_info = InfoButton("Signal not explained")
        heading.addWidget(self.unexplained_info)
        heading.addStretch(1)
        run.addLayout(heading)
        self.unexplained_button = themes.command("")
        self.unexplained_button.clicked.connect(self.unexplained_requested)
        run.addWidget(self.unexplained_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
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

    def show_run(self, rows: list[Row], *, fitted: bool, warned: bool) -> None:
        """The active run's rows; without a run and rows, only a failure can show."""
        self.run_block.setVisible(fitted or bool(rows))
        for widget in (self.unexplained_section, self.unexplained_button, self.unexplained_note):
            widget.setVisible(fitted)
        self.unexplained_info.setVisible(fitted and bool(self.unexplained_info.body.text()))
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

    def show_changes(self, parent: str, entries: list[tuple]) -> None:
        """Show run-to-run changes; ``parent`` is the parent run's short ID, or "".

        Entries are ("row", label, value, tooltip), ("heading", text[, tooltip]),
        ("note", text) or ("link", text, detail, working ID, opens a cluster
        page[, warning]). Consecutive rows share one label grid.
        """
        key = (parent, tuple(entries))
        if key == self._changes_key:
            return
        self._changes_key = key
        self.changes_title.setVisible(bool(parent))
        clear_layout(self.changes)
        rows: list[Row] = [Row("Parent run", parent)] if parent else []

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
        self.unexplained_info.set_text(about)
        self.unexplained_info.setVisible(bool(about) and self.unexplained_section.isVisibleTo(self))
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
