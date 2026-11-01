"""Right inspector: the selected peak's results and review, or the fit setup.

With nothing selected the inspector shows the fit setup and the active run.

Every value shown comes from the completed outcome as recorded; the inspector
never refits, re-solves amplitudes or recomputes statistics.
"""

import math
from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.engine.domain.spectrum import get_dimension_label
from peakfit.gui import formatting
from peakfit.gui import theme as themes
from peakfit.gui.editing import PeakInspector
from peakfit.gui.editor.cluster_page import ClusterPage
from peakfit.gui.editor.labels import clear_layout as _clear
from peakfit.gui.editor.labels import review_item
from peakfit.gui.editor.labels import text_label as _label
from peakfit.gui.editor.profile import ProfilePlot, peak_profile
from peakfit.gui.editor.review import BOUND_TOLERANCE, correlation_matrix
from peakfit.gui.editor.setup import FitSetup

if TYPE_CHECKING:
    from peakfit.engine.scientific.final_outcome import FinalClusterOutcome
    from peakfit.gui.dataset_fit import DatasetFitResult
    from peakfit.gui.editor.review import ClusterReview
    from peakfit.gui.peaks import WorkingPeak

_KINDS = {"cs": "δ", "lw": "LW"}


class PreciseInspector(PeakInspector):
    """Display-precision labels with exact unchanged values and round-trip focus/copy."""

    decimals: tuple[int, int] = (4, 3)  # (horizontal, vertical) display decimals

    def show_peak(self, peak, displayed=None) -> None:
        """Populate a new committed group without feeding formatted values into science."""
        self._peak = peak
        self._baseline = {}
        self.setTitle("")
        for field in (self.name, self.x_field, self.y_field, self.delete):
            field.setEnabled(peak is not None)
        dx, dy = self.decimals
        values = (
            (peak.name, f"{peak.positions[1]:.{dx}f}", f"{peak.positions[0]:.{dy}f}")
            if peak
            else ("", "", "")
        )
        for field, initial_text in zip(
            (self.name, self.x_field, self.y_field), values, strict=True
        ):
            text = initial_text
            if peak and field.hasFocus() and field in (self.x_field, self.y_field):
                text = repr(peak.positions[1 if field is self.x_field else 0])
            with QtCore.QSignalBlocker(field):
                field.setText(text)
            self._baseline[field] = text
        alias = (
            f"Displayed alias: {displayed[0]:.{dx}f}, {displayed[1]:.{dy}f} ppm\n"
            if peak
            and displayed
            and any(
                not math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
                for a, b in zip(displayed, reversed(peak.positions), strict=True)
            )
            else ""
        )
        self.details.setText(alias + "Unfolded ppm · Enter commits · Escape discards")
        self.fields_changed.emit()

    @property
    def pending(self) -> bool:
        """Formatting/focus alone is never a scientific draft."""
        return self._peak is not None and any(
            field.text() != value for field, value in self._baseline.items()
        )

    def values(self):
        """Keep untouched float64 values exact; accept the local decimal separator."""
        if not self.name.text().strip():
            self.name.setFocus()
            raise ValueError("Give the peak a name. Correct it or press Escape.")
        assert self._peak is not None
        positions = []
        for field, original in (
            (self.y_field, self._peak.positions[0]),
            (self.x_field, self._peak.positions[1]),
        ):
            if field.text() == self._baseline[field]:
                positions.append(original)
                continue
            text = field.text().strip()
            if QtCore.QLocale().decimalPoint() == "," and "." not in text:
                text = text.replace(",", ".")
            try:
                value = float(text)

                if not math.isfinite(value) or "_" in text:
                    raise ValueError
            except ValueError as exc:
                field.setFocus()
                raise ValueError(
                    f"{field.accessibleName()}: enter finite ppm; correct it or press Escape."
                ) from exc
            positions.append(value)
        return self.name.text(), (positions[0], positions[1])

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        """Reveal round-trip coordinates on focus without creating a draft."""
        if (
            event.type() == QtCore.QEvent.Type.ShortcutOverride
            and isinstance(event, QtGui.QKeyEvent)
            and not self.pending
            and (
                event.matches(QtGui.QKeySequence.StandardKey.Undo)
                or event.matches(QtGui.QKeySequence.StandardKey.Redo)
            )
        ):
            event.ignore()
            return True  # Let the workspace's named peak action handle this key.
        peak = getattr(self, "_peak", None)
        if (
            peak is not None
            and watched in (self.x_field, self.y_field)
            and event.type() in (QtCore.QEvent.Type.FocusIn, QtCore.QEvent.Type.FocusOut)
            and watched.text() == self._baseline.get(watched)
        ):
            index = 1 if watched is self.x_field else 0
            value = peak.positions[index]
            decimals = self.decimals[0 if index == 1 else 1]
            text = (
                repr(value)
                if event.type() == QtCore.QEvent.Type.FocusIn
                else f"{value:.{decimals}f}"
            )
            with QtCore.QSignalBlocker(watched):
                watched.setText(text)
            self._baseline[watched] = text
        return super().eventFilter(watched, event)


def _run_text(result: DatasetFitResult | None, *, current: bool, busy: bool) -> str:
    if result is None:
        return ""
    return f"run {result.run_id[:8]} · {'Current' if current else 'Outdated'}" + (
        " · new fit running" if busy else ""
    )


def _state(cluster: FinalClusterOutcome) -> str:
    if not cluster.usable:
        return "Unusable"
    return (
        "Converged · usable" if cluster.optimizer_provenance.converged else "Not converged · usable"
    )


def _provenance(cluster: FinalClusterOutcome, result: DatasetFitResult) -> str:
    provenance = cluster.optimizer_provenance
    return "\n".join(
        [
            f"Optimizer: {provenance.optimizer_kind or 'not recorded'}",
            f"Termination: {provenance.termination_message or 'not recorded'}",
            f"Function evaluations: {provenance.function_evaluations}",
            f"Correction revision: {cluster.correction_revision}",
            f"Statistical noise: {formatting.intensity(result.outcome.noise)} a.u.",
            f"Run {result.run_id}",
            f"Completed {result.completed_at} · PeakFit {result.software_version}",
        ]
    )


def _dimension_and_kind(name: str) -> tuple[str, str]:
    """`p_20.F2.cs` → (`F2`, `cs`); names without both parts keep what they have."""
    parts = name.split(".")
    return (parts[1] if len(parts) > 1 else "", parts[-1] if len(parts) > 2 else "")  # noqa: PLR2004


def find_cluster(
    result: DatasetFitResult, working_id: str
) -> tuple[str, FinalClusterOutcome] | None:
    """The engine name of a working peak and the recorded cluster that fitted it."""
    identity = next((i for i in result.inputs.identities if i.working_id == working_id), None)
    if identity is None:
        return None
    cluster = next(
        (c for c in result.outcome.clusters if identity.engine_name in c.peak_names), None
    )
    return (identity.engine_name, cluster) if cluster is not None else None


class Inspector(QtWidgets.QScrollArea):
    """One scrolling page chosen by selection: the selected peak, or the analysis."""

    plane_requested = QtCore.Signal(int)
    next_warning = QtCore.Signal()
    cluster_requested = QtCore.Signal()
    change_chosen = QtCore.Signal(str, bool)  # working ID, open its cluster page
    member_chosen = QtCore.Signal(str)
    lens_requested = QtCore.Signal()

    def __init__(self, x_axis, y_axis, decimals: tuple[int, int]) -> None:  # noqa: PLR0915
        super().__init__()
        self.setObjectName("inspector")
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(300)
        self._axes = {
            axis.label: formatting.nucleus(axis.nucleus, axis.label) for axis in (x_axis, y_axis)
        }
        self._axis_order = [x_axis.label, y_axis.label]
        body = QtWidgets.QWidget()
        body.setObjectName("inspectorBody")
        self.setWidget(body)
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(themes.SPACE_L, themes.SPACE_M, themes.SPACE_L, themes.SPACE_L)
        layout.setSpacing(themes.SPACE_S)
        self.kind = _label("", "section")
        self.title = _label("", "title")
        self.status = _label("", "muted", wrap=True)
        for widget in (self.kind, self.title, self.status):
            layout.addWidget(widget)
        # Peak page
        self.peak_page = QtWidgets.QWidget()
        peak = QtWidgets.QVBoxLayout(self.peak_page)
        peak.setContentsMargins(0, themes.SPACE_XS, 0, 0)
        peak.setSpacing(themes.SPACE_S)
        self.cluster_link = QtWidgets.QPushButton()
        self.cluster_link.setObjectName("link")
        self.cluster_link.setFlat(True)
        self.cluster_link.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.cluster_link.clicked.connect(self.cluster_requested)
        self.cluster_link.hide()
        peak.addWidget(self.cluster_link, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        x_title = f"{self._axes[x_axis.label]} · {x_axis.label} ppm"
        y_title = f"{self._axes[y_axis.label]} · {y_axis.label} ppm"
        self.editor = PreciseInspector(x_title, y_title)
        self.editor.decimals = decimals
        self.editor.details.setObjectName("caption")
        self.editor.setFlat(True)
        grid = self.editor.layout()
        assert isinstance(grid, QtWidgets.QGridLayout)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(themes.SPACE_S)
        grid.setVerticalSpacing(6)
        # Same validated group-commit component, composed vertically in the inspector.
        grid.removeWidget(self.editor.delete)
        grid.addWidget(self.editor.delete, 4, 0, 1, 2, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.editor.delete.setObjectName("chip")
        self.editor.details.setMinimumWidth(0)
        self.draft = _label("", "warning")
        grid.addWidget(self.draft, 5, 0, 1, 2)
        self.editor.fields_changed.connect(self._draft_changed)
        peak.addWidget(self.editor)
        self.starts_toggle = self._disclosure("Starting values and bounds")
        self.starts_toggle.toggled.connect(lambda shown: self.starts.setVisible(shown))
        peak.addWidget(self.starts_toggle, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.starts = QtWidgets.QWidget()
        starts = QtWidgets.QVBoxLayout(self.starts)
        starts.setContentsMargins(0, 0, 0, 0)
        starts.setSpacing(4)
        self.starts_note = _label("", "caption", wrap=True)
        starts.addWidget(self.starts_note)
        self.start_grid = QtWidgets.QGridLayout()
        self.start_grid.setHorizontalSpacing(themes.SPACE_S)
        self.start_grid.setVerticalSpacing(3)
        starts.addLayout(self.start_grid)
        self.starts.hide()
        peak.addWidget(self.starts)
        self._starts_key = None
        # One amplitude profile, moved between the peak and cluster pages, so its
        # normalization reference is shared by both.
        self.profile_block = QtWidgets.QWidget()
        block = QtWidgets.QVBoxLayout(self.profile_block)
        block.setContentsMargins(0, 0, 0, 0)
        block.setSpacing(themes.SPACE_S)
        self.profile_title = self._section(block, "Amplitude profile")
        self.profile = ProfilePlot()
        self.profile.plane_requested.connect(self.plane_requested)
        block.addWidget(self.profile)
        self.profile_reason = _label("", "muted", wrap=True)
        block.addWidget(self.profile_reason)
        self.peak_profile_slot = QtWidgets.QVBoxLayout()
        self.peak_profile_slot.setContentsMargins(0, 0, 0, 0)
        self.peak_profile_slot.addWidget(self.profile_block)
        peak.addLayout(self.peak_profile_slot)
        self.fitted_title = self._section(peak, "Fitted")
        self.fitted = QtWidgets.QGridLayout()
        self.fitted.setHorizontalSpacing(themes.SPACE_S)
        self.fitted.setVerticalSpacing(3)
        peak.addLayout(self.fitted)
        self.review_title = self._section(peak, "Cluster review")
        self.review = QtWidgets.QVBoxLayout()
        self.review.setSpacing(2)
        peak.addLayout(self.review)
        self.provenance_toggle = self._disclosure("Provenance")
        self.provenance_toggle.toggled.connect(lambda shown: self.provenance.setVisible(shown))
        self.provenance = _label("", "caption", wrap=True)
        self.provenance.hide()
        peak.addSpacing(themes.SPACE_XS)
        peak.addWidget(self.provenance_toggle, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        peak.addWidget(self.provenance)
        layout.addWidget(self.peak_page)
        # Cluster page: parameters keyed by the engine's axis names, horizontal first.
        decimals_x, decimals_y = decimals
        self.cluster_page = ClusterPage(
            [
                (get_dimension_label(1), self._axes[x_axis.label], decimals_x),
                (get_dimension_label(0), self._axes[y_axis.label], decimals_y),
            ]
        )
        self.cluster_page.member_chosen.connect(self.member_chosen)
        self.cluster_page.lens_requested.connect(self.lens_requested)
        layout.addWidget(self.cluster_page)
        # Analysis page
        self.analysis_page = QtWidgets.QWidget()
        analysis = QtWidgets.QVBoxLayout(self.analysis_page)
        analysis.setContentsMargins(0, themes.SPACE_XS, 0, 0)
        analysis.setSpacing(themes.SPACE_S)
        self._section(analysis, "Fit setup")
        self.setup = FitSetup()
        analysis.addWidget(self.setup)
        self._section(analysis, "Active run")
        self.run_text = _label("", "", wrap=True)
        analysis.addWidget(self.run_text)
        self.changes_title = _label("", "caption")
        analysis.addWidget(self.changes_title)
        self.changes = QtWidgets.QVBoxLayout()
        self.changes.setSpacing(2)
        analysis.addLayout(self.changes)
        self._changes_key = None
        self.next_button = QtWidgets.QPushButton("Select next cluster with a warning")
        self.next_button.clicked.connect(self.next_warning)
        analysis.addWidget(self.next_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.analysis_note = _label("", "caption", wrap=True)
        analysis.addWidget(self.analysis_note)
        layout.addWidget(self.analysis_page)
        layout.addStretch(1)
        self._peak: WorkingPeak | None = None
        self.cluster_view = False
        self._key = None
        self.theme = themes.LIGHT
        self.set_peak(None, None)

    @staticmethod
    def _section(layout: QtWidgets.QVBoxLayout, text: str) -> QtWidgets.QLabel:
        label = _label(text.upper(), "section")
        layout.addSpacing(themes.SPACE_S)
        layout.addWidget(label)
        return label

    def apply_theme(self, theme: themes.Theme) -> None:
        """Retint the profile; labels follow the stylesheet."""
        self.theme = theme
        self.cluster_page.theme = theme
        self.profile.apply_theme(theme)
        self._key = None

    @staticmethod
    def _disclosure(text: str) -> QtWidgets.QToolButton:
        """A collapsed-by-default section toggle with a turning arrow."""
        button = QtWidgets.QToolButton()
        button.setObjectName("text")
        button.setText(text)
        button.setCheckable(True)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        button.toggled.connect(
            lambda shown: button.setArrowType(
                QtCore.Qt.ArrowType.DownArrow if shown else QtCore.Qt.ArrowType.RightArrow
            )
        )
        return button

    def show_starts(self, rows: list[tuple[str, str, str, str]] | None, note: str) -> None:
        """The next fit's starting value, range and freedom per parameter (read-only)."""
        key = (tuple(rows) if rows is not None else None, note)
        if key == self._starts_key:
            return
        self._starts_key = key
        self.starts_note.setText(note)
        _clear(self.start_grid)
        for row, (name, start, detail, unit) in enumerate(rows or ()):
            value = _label(start)
            value.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            self.start_grid.addWidget(_label(name, "muted"), 2 * row, 0)
            self.start_grid.addWidget(value, 2 * row, 1)
            self.start_grid.addWidget(_label(unit, "caption"), 2 * row, 2)
            self.start_grid.addWidget(_label(detail, "caption"), 2 * row + 1, 1, 1, 2)
        self.start_grid.setColumnStretch(1, 1)

    def _draft_changed(self) -> None:
        self.draft.setText("Draft · Enter commits · Escape discards" if self.editor.pending else "")

    def set_peak(
        self,
        peak: WorkingPeak | None,
        displayed: tuple[float, float] | None,
        *,
        cluster: bool = False,
    ) -> None:
        """Show the committed selection as its peak page, or as its cluster's page.

        The editor resets only on selection or commit; the cluster page keeps the
        selected peak, so the spectrum and the editor stay on it.
        """
        cluster = cluster and peak is not None
        if (
            peak is None
            or self._peak is None
            or peak.id != self._peak.id
            or cluster != self.cluster_view
        ):
            self.verticalScrollBar().setValue(0)
        self._peak = peak
        self.cluster_view = cluster
        self.editor.show_peak(peak, displayed)
        self.peak_page.setVisible(peak is not None and not cluster)
        self.cluster_page.setVisible(cluster)
        self.analysis_page.setVisible(peak is None)
        source, target = (
            (self.peak_profile_slot, self.cluster_page.profile_slot)
            if cluster
            else (self.cluster_page.profile_slot, self.peak_profile_slot)
        )
        if source.indexOf(self.profile_block) >= 0:
            source.removeWidget(self.profile_block)
            target.addWidget(self.profile_block)
        if peak is not None and not cluster:
            self.title.setText(peak.name)
        self._key = None

    def set_plane(self, plane: int) -> None:
        """Move the profile's plane marker only."""
        self.profile.set_plane(plane)

    def show_analysis(self, title: str, summary: str, run: str, note: str) -> None:
        """No selection: the fit setup and what the active run recorded."""
        if self._peak is not None:
            return
        self.kind.setText("ANALYSIS")
        self.title.setText(title)
        self.status.setText(summary)
        key = ("analysis", run, note)
        if key == self._key:
            return
        self._key = key
        self.run_text.setText(run)
        self.next_button.setVisible("warning" in run)
        self.analysis_note.setText(note)

    def show_result(
        self,
        result: DatasetFitResult | None,
        *,
        current: bool,
        busy: bool,
        plane_values: tuple[float, ...] | None,
        plane: int,
        review: ClusterReview | None = None,
    ) -> None:
        """Recorded profile, parameters and review of the selected peak's cluster."""
        peak = self._peak
        if peak is None or self.cluster_view:
            return
        key = (
            peak.id,
            peak.name,
            result.run_id if result else None,
            current,
            busy,
            plane_values,
            review,
        )
        if key == self._key:
            self.profile.set_plane(plane)
            return
        self._key = key
        found = find_cluster(result, peak.id) if result is not None else None
        run = _run_text(result, current=current, busy=busy)
        self.cluster_link.setVisible(found is not None and len(found[1].peak_names) > 1)
        if found is not None:
            self.cluster_link.setText(f"Review cluster of {len(found[1].peak_names)} ›")
        for widget in (self.profile_title, self.fitted_title, self.review_title):
            widget.setVisible(found is not None)
        self.provenance_toggle.setVisible(found is not None)
        self.provenance.setVisible(found is not None and self.provenance_toggle.isChecked())
        _clear(self.fitted)
        _clear(self.review)
        if result is None or found is None:
            self.kind.setText("PEAK")
            self.status.setText(
                "Not fitted yet"
                if result is None
                else f"Not part of {run}: added or renamed after that fit"
            )
            self.profile.show_profile(None, plane)
            self.profile.hide()
            self.profile_reason.setText("")
            return
        engine, cluster = found
        members = len(cluster.peak_names)
        self.kind.setText(f"PEAK · CLUSTER OF {members}" if members > 1 else "PEAK")
        self.status.setText(f"{_state(cluster)} · {run}")
        self._show_profile(result, peak.id, plane_values, plane, current=current)
        self.fitted_title.setText(f"FITTED · RUN {result.run_id[:8].upper()}")
        self._fill_parameters(engine, cluster)
        start = next((p for p in result.inputs.snapshot.peaks if p.id == peak.id), None)
        if start is not None:
            # The run's submitted position, for inspection; movement is not a diagnostic.
            dx, dy = self.editor.decimals
            x_title, y_title = (self._axes[label] for label in self._axis_order)
            started = _label(
                f"Started from {x_title} {start.positions[1]:.{dx}f}, "
                f"{y_title} {start.positions[0]:.{dy}f} ppm",
                "caption",
                wrap=True,
            )
            self.fitted.addWidget(started, self.fitted.rowCount(), 0, 1, 3)
        self._fill_review(engine, cluster, result, review)
        self.provenance.setText(_provenance(cluster, result))

    def show_changes(self, title: str, entries: list[tuple]) -> None:
        """Show run-to-run changes as a list of entries.

        Entries are ("line", text), ("heading", text), ("note", text) or
        ("link", text, detail, working ID, opens a cluster page).
        """
        key = (title, tuple(entries))
        if key == self._changes_key:
            return
        self._changes_key = key
        self.changes_title.setText(title)
        self.changes_title.setVisible(bool(title))
        _clear(self.changes)
        for entry in entries:
            kind = entry[0]
            if kind == "link":
                _kind, text, detail, identity, cluster = entry
                row = QtWidgets.QHBoxLayout()
                row.setSpacing(themes.SPACE_S)
                link = QtWidgets.QPushButton(text)
                link.setObjectName("link")
                link.setFlat(True)
                link.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
                link.clicked.connect(
                    lambda _checked=False, i=identity, c=cluster: self.change_chosen.emit(i, c)
                )
                row.addWidget(link, 0, QtCore.Qt.AlignmentFlag.AlignTop)
                row.addWidget(_label(detail, "caption", wrap=True), 1)
                self.changes.addLayout(row)
            elif kind == "heading":
                self.changes.addSpacing(4)
                self.changes.addWidget(_label(entry[1], "muted"))
            else:
                self.changes.addWidget(
                    _label(entry[1], "" if kind == "line" else "caption", wrap=True)
                )

    def show_cluster_result(
        self,
        result: DatasetFitResult | None,
        review: ClusterReview | None,
        *,
        current: bool,
        busy: bool,
        plane_values: tuple[float, ...] | None,
        plane: int,
    ) -> None:
        """The selected peak's recorded cluster as one unit of review."""
        peak = self._peak
        if peak is None or not self.cluster_view:
            return
        key = (
            "cluster",
            peak.id,
            result.run_id if result else None,
            current,
            busy,
            plane_values,
            review,
        )
        if key == self._key:
            self.profile.set_plane(plane)
            return
        self._key = key
        found = find_cluster(result, peak.id) if result is not None else None
        if result is None or found is None or review is None:
            self.kind.setText("CLUSTER")
            self.title.setText(peak.name)
            self.status.setText("Not part of the active run")
            return
        _engine, cluster = found
        names = review.names
        count = len(names)
        self.kind.setText(f"CLUSTER · {count} PEAK{'S' if count != 1 else ''}")
        self.title.setText(names[0] + (f"  +{count - 1}" if count > 1 else ""))
        self.status.setText(f"{_state(cluster)} · {_run_text(result, current=current, busy=busy)}")
        self._show_profile(result, peak.id, plane_values, plane, current=current)
        engines = {i.working_id: i.engine_name for i in result.inputs.identities}
        self.cluster_page.show_cluster(
            review, cluster, tuple(engines[i] for i in review.ids), peak.id
        )
        self.cluster_page.provenance.setText(_provenance(cluster, result))

    def _show_profile(
        self,
        result: DatasetFitResult,
        peak_id: str,
        plane_values: tuple[float, ...] | None,
        plane: int,
        *,
        current: bool,
    ) -> None:
        profile = peak_profile(result, peak_id, plane_values)
        self.profile_title.show()
        if isinstance(profile, str):
            self.profile.show_profile(None, plane)
            self.profile.hide()
            self.profile_reason.setText(profile)
        else:
            self.profile.show()
            self.profile.show_profile(profile, plane)
            self.profile_reason.setText("" if current else "Recorded by an outdated run.")

    def _fill_parameters(self, engine: str, cluster: FinalClusterOutcome) -> None:
        if cluster.analytical_evaluation is None:
            self.fitted.addWidget(
                _label(cluster.unusable_reason or "No fitted values recorded", "muted", wrap=True),
                0,
                0,
                1,
                3,
            )
            return
        parameters = [p for p in cluster.final_nonlinear_parameters if p.name.startswith(engine)]

        def order(parameter) -> tuple[int, str]:
            dim, kind = _dimension_and_kind(parameter.name)
            axis = self._axis_order.index(dim) if dim in self._axis_order else 9
            return axis, kind

        for row, parameter in enumerate(sorted(parameters, key=order)):
            dim, kind = _dimension_and_kind(parameter.name)
            name = f"{_KINDS.get(kind, kind)} {self._axes.get(dim, dim)}"
            value, sigma = formatting.with_sigma(parameter.value, parameter.standard_error)
            span = parameter.max - parameter.min
            at_bound = (
                parameter.vary
                and math.isfinite(span)
                and min(parameter.value - parameter.min, parameter.max - parameter.value)
                <= BOUND_TOLERANCE * span
            )
            text = value if not parameter.vary else f"{value} ± {sigma}"
            if sigma == "σ unavailable" and parameter.vary:
                text = f"{value} · σ unavailable"
            cell = _label(text)
            cell.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            unit = _label(parameter.unit + (" · fixed" if not parameter.vary else ""), "caption")
            title = _label(name, "warning" if at_bound else "muted")
            if at_bound:
                title.setToolTip("At a recorded bound")
            self.fitted.addWidget(title, row, 0)
            self.fitted.addWidget(cell, row, 1)
            self.fitted.addWidget(unit, row, 2)
        self.fitted.setColumnStretch(1, 1)

    def _item(self, family: str, glyph: str, text: str, detail: str = "", *, warn=False) -> None:
        review_item(self.review, family, glyph, text, detail, warn=warn)

    def _fill_review(
        self,
        engine: str,
        cluster: FinalClusterOutcome,
        result: DatasetFitResult,
        review: ClusterReview | None,
    ) -> None:
        """Three separate families: numerical warnings, measures and highlights."""
        warnings = review.warnings if review is not None else ()
        for index, warning in enumerate(warnings):
            self._item(
                "Numerical warnings" if index == 0 else "",
                "▲",
                warning.kind,
                warning.detail,
                warn=True,
            )
        if not warnings:
            self._item("Numerical warnings", "–", "None recorded")
        evaluation = cluster.analytical_evaluation
        if evaluation is None or review is None:
            return
        samples = review.samples
        self._item(
            "Measures",
            "·",
            f"χ²ᵣ {review.chi2:.3g}",
            f"{samples:,} fitting samples · relative to the noise estimate".replace(",", " "),
        )
        if math.isfinite(review.largest_residual) and samples > 1:
            self._item(
                "",
                "·",
                f"Largest recorded residual {review.largest_residual:.1f} σ",
                f"typical maximum for {samples:,} Gaussian samples ≈ "
                f"{math.sqrt(2 * math.log(samples)):.1f} σ".replace(",", " "),
            )
        names = {i.engine_name: i.assignment for i in result.inputs.identities}
        matrix = correlation_matrix(cluster)
        if len(cluster.peak_names) == 1:
            self._item("", "·", "Amplitude correlation not applicable", "single-peak cluster")
        elif matrix is None:
            self._item("", "·", "Amplitude correlation unavailable", "covariance not recorded")
        else:
            index = cluster.peak_names.index(engine)
            row = matrix[index].copy()
            row[index] = 0.0
            partner = int(np.nanargmax(np.abs(row)))
            self._item(
                "",
                "·",
                f"Amplitude correlation ρ {formatting.number(float(row[partner]), 2)}"
                f" with {names.get(cluster.peak_names[partner], cluster.peak_names[partner])}",
                "conditional on the fitted lineshapes",
            )
        assignments = dict(zip(review.ids, review.names, strict=True))
        for index, highlight in enumerate(review.highlights):
            prefix = f"{assignments[highlight.peak]}: " if highlight.peak in assignments else ""
            self._item("Highlights" if index == 0 else "", "◇", prefix + highlight.rule)
        if not review.highlights:
            self._item("Highlights", "–", "None at the current thresholds")
