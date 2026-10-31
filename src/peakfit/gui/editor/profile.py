"""Selection-linked amplitude profiles of the active run, shown without recomputation."""

from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui.editor.labels import ElidedLabel

if TYPE_CHECKING:
    from peakfit.gui.dataset_fit import DatasetFitResult

_MAX_ISOLATED = 2
_SIDE_WIDTH = 300
_MEMBER_COLORS = ("#7f77dd", "#1d9e75", "#d4537e", "#888780")


@dataclass(frozen=True, slots=True)
class ProfileSeries:
    """One peak's recorded amplitudes and scaled analytical standard error."""

    assignment: str
    amplitudes: np.ndarray
    error: float


@dataclass(frozen=True, slots=True)
class PeakProfile:
    """The selected peak and its cluster members, exactly as one run recorded them."""

    run_id: str
    series: tuple[ProfileSeries, ...]  # selected peak first
    source_planes: np.ndarray  # zero-based source plane of each amplitude column
    x: np.ndarray  # recorded plane values, or one-based plane numbers
    plane_values: bool
    reduced_chi_squared: float
    uncertainty_scale: float
    warning: str | None


def peak_profile(
    result: DatasetFitResult, working_id: str, plane_values: tuple[float, ...] | None
) -> PeakProfile | str:
    """Return the selected peak's cluster profiles, or why the run has none.

    Plane values are used only when supplied and identical to those the run
    recorded; otherwise columns are labeled by one-based source plane number.
    """
    identity = next((i for i in result.inputs.identities if i.working_id == working_id), None)
    cluster = identity and next(
        (c for c in result.outcome.clusters if identity.engine_name in c.peak_names), None
    )
    if identity is None or cluster is None:
        return f"Not part of run {result.run_id[:8]}"
    evaluation = cluster.analytical_evaluation
    if not cluster.usable or evaluation is None:
        return cluster.unusable_reason or "No analytical evaluation for this cluster"
    names = {i.engine_name: i.assignment for i in result.inputs.identities}
    order = sorted(
        range(len(cluster.peak_names)),
        key=lambda k: cluster.peak_names[k] != identity.engine_name,
    )
    series = tuple(
        ProfileSeries(
            names[cluster.peak_names[k]],
            evaluation.amplitudes[k],
            float(evaluation.scaled_amplitude_standard_errors[k]),
        )
        for k in order
    )
    snapshot = result.inputs.snapshot
    planes = np.asarray(
        snapshot.source_plane_indices or range(evaluation.amplitudes.shape[1]), dtype=int
    )
    recorded = plane_values is not None and tuple(snapshot.plane_values) == tuple(plane_values)
    x = np.asarray(snapshot.plane_values, dtype=float)[planes] if recorded else planes + 1.0
    return PeakProfile(
        result.run_id,
        series,
        planes,
        x,
        recorded,
        evaluation.statistics.reduced_chi_squared,
        evaluation.statistics.amplitude_uncertainty_scale,
        evaluation.uncertainty_warning,
    )


def _sigma(error: float) -> str:
    """Name an unavailable standard error instead of formatting NaN as a value."""
    return f"σ = {error:.3g}" if np.isfinite(error) else "σ unavailable"


def framed_range(x: np.ndarray) -> tuple[float, float] | None:
    """Return the main x-range when at most two observations lie beyond a dominant gap.

    A gap dominates when it exceeds half of the total span, as for a distant
    reference plane. Isolation on both sides, or none, leaves the full range.
    """
    values = np.unique(x)
    if values.size < 3:  # noqa: PLR2004
        return None
    gaps = np.diff(values)
    index = int(np.argmax(gaps))
    if gaps[index] <= (values[-1] - values[0]) / 2:
        return None
    below = int(np.sum(x <= values[index]))
    above = x.size - below
    if below <= _MAX_ISOLATED < above:
        return float(values[index + 1]), float(values[-1])
    if above <= _MAX_ISOLATED < below:
        return float(values[0]), float(values[index])
    return None


class ProfilePanel(QtWidgets.QWidget):
    """Collapsible profile of the selected peak's cluster; header always visible."""

    plane_requested = QtCore.Signal(int)
    expanded_changed = QtCore.Signal(bool)

    def __init__(self) -> None:  # noqa: PLR0915
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)
        header = QtWidgets.QHBoxLayout()
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setText("Profile")
        self.toggle.setCheckable(True)
        self.toggle.setAutoRaise(True)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setAccessibleName("Show amplitude profile")
        self.toggle.toggled.connect(self._toggled)
        self.title = ElidedLabel("No fit results yet")
        self.badge = QtWidgets.QLabel("▲ Uncertainty unreliable")
        self.badge.setObjectName("warning")
        self.show_all = QtWidgets.QPushButton("Show all points")
        self.show_all.setCheckable(True)
        self.show_all.toggled.connect(self._draw)
        header.addWidget(self.toggle)
        header.addWidget(self.title, 1)
        header.addWidget(self.badge)
        header.addWidget(self.show_all)
        layout.addLayout(header)
        # Key, off-scale points, error basis and warning sit beside the plot, so
        # the plot keeps the panel's height and no text covers observations.
        body = QtWidgets.QHBoxLayout()
        self.basis = QtWidgets.QLabel()
        self.basis.setObjectName("muted")
        self.basis.setWordWrap(True)
        self.basis.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.basis.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.basis.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.side = QtWidgets.QScrollArea()
        self.side.setWidget(self.basis)
        self.side.setWidgetResizable(True)
        self.side.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.side.setFixedWidth(_SIDE_WIDTH)
        self.side.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.plot = pg.PlotWidget()
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMinimumHeight(60)
        self.plot.setAccessibleName("Signed fitted amplitudes of the selected peak's cluster")
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.plot.getAxis("left").setWidth(64)
        body.addWidget(self.plot, 1)
        body.addWidget(self.side)
        layout.addLayout(body, 1)
        self.zero = pg.InfiniteLine(angle=0, pos=0)
        self.plane_line = pg.InfiniteLine(angle=90)
        self._profile: PeakProfile | None = None
        self._key = None
        self._plane = 0
        self._visible = np.array([], dtype=bool)
        self._note_text = ""
        self._basis_html = ""
        self._key_html = ""
        self._framed = False
        self.apply_colors(dark=False)
        self.set_expanded(False)
        self.badge.hide()
        self.show_all.hide()

    def set_expanded(self, expanded: bool) -> None:
        """Apply a restored or toggled state without reporting a user choice."""
        with QtCore.QSignalBlocker(self.toggle):
            self.toggle.setChecked(expanded)
        self._apply_expanded(expanded)

    def _toggled(self, expanded: bool) -> None:
        self._apply_expanded(expanded)
        self.expanded_changed.emit(expanded)

    def _apply_expanded(self, expanded: bool) -> None:
        self.toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow
        )
        self.side.setVisible(expanded and self._profile is not None)
        self.plot.setVisible(expanded)
        self.show_all.setVisible(expanded and self._framed)
        height = self.toggle.sizeHint().height() + 8
        self.setMaximumHeight(16777215 if expanded else height)

    def apply_colors(self, *, dark: bool) -> None:
        """Follow the application appearance; no scientific values change."""
        self._colors = {
            "bg": "#171d24" if dark else "#fbfcfd",
            "muted": "#aab7c4" if dark else "#586875",
            "selected": "#73b5d5" if dark else "#23688a",
            "highlight": "#e2ad65" if dark else "#875619",
        }
        self.plot.setBackground(self._colors["bg"])
        for name in ("left", "bottom"):
            axis = self.plot.getAxis(name)
            axis.setPen(self._colors["muted"])
            axis.setTextPen(self._colors["muted"])
        self.zero.setPen(pg.mkPen(self._colors["muted"], width=1))
        self.plane_line.setPen(
            pg.mkPen(self._colors["selected"], width=1, style=QtCore.Qt.PenStyle.DashLine)
        )
        if self._profile is not None:
            self._draw()

    def show_selection(
        self,
        result: DatasetFitResult | None,
        working_id: str | None,
        *,
        current: bool,
        busy: bool,
        plane_values: tuple[float, ...] | None,
        plane: int,
    ) -> None:
        """Follow the accepted selection and active run; redraw only on change."""
        key = (result.run_id if result else None, working_id, current, busy, plane_values)
        if key != self._key:
            self._key = key
            self._plane = plane
            self._rebuild(result, working_id, current=current, busy=busy, values=plane_values)
        elif plane != self._plane:
            self._plane = plane
            self._place_plane_marker()

    def _rebuild(self, result, working_id, *, current, busy, values) -> None:
        self._profile = None
        self.plot.clear()
        self.badge.hide()
        self.show_all.hide()
        self._framed = False
        self.side.hide()
        if result is None:
            self.title.setText("No fit results yet")
            return
        run = f"Run {result.run_id[:8]} · {'Current' if current else 'Outdated'}" + (
            " · new fit running" if busy else ""
        )
        if working_id is None:
            self.title.setText(f"Select a peak to see its amplitude profile · {run}")
            return
        profile = peak_profile(result, working_id, values)
        if isinstance(profile, str):
            self.title.setText(f"{profile} · {run}")
            return
        self._profile = profile
        selected, *members = profile.series
        unavailable = [s.assignment for s in profile.series if not np.isfinite(s.error)]
        self.title.setText(
            selected.assignment
            + (f" with {', '.join(s.assignment for s in members)}" if members else "")
            + f" · {run} · ±1σ scaled analytical"
            + (f" · σ unavailable: {', '.join(unavailable)}" if unavailable else "")
        )
        if profile.warning:
            self.badge.setToolTip(profile.warning)
            self.badge.show()
        chi2 = profile.reduced_chi_squared
        scaling = (
            f"multiplied by √χ²ᵣ = {profile.uncertainty_scale:.3g} (cluster χ²ᵣ = {chi2:.3g})"
            if profile.uncertainty_scale > 1
            else f"not scaled (cluster χ²ᵣ = {chi2:.3g} ≤ 1)"
        )
        self._basis_html = escape(
            "Errors are analytical amplitude standard errors computed with the fitted "
            "lineshape parameters held fixed, so their uncertainty and correlations are "
            f"not propagated; {scaling}. One σ per peak applies to every plane "
            f"({selected.assignment}: {_sigma(selected.error)}). They can underestimate "
            "the uncertainty of overlapping peaks."
            + (" No error bar is drawn where σ is unavailable." if unavailable else "")
        ) + (f"<br><b>▲ {escape(profile.warning)}</b>" if profile.warning else "")
        self.side.setVisible(self.toggle.isChecked())
        self.plot.setLabel(
            "bottom", "Plane value (file units)" if profile.plane_values else "Plane number"
        )
        self.plot.setLabel("left", "Signed amplitude")
        self._draw()

    def _draw(self) -> None:
        profile = self._profile
        if profile is None:
            return
        self.plot.clear()
        x = profile.x
        frame = framed_range(x)
        self._framed = frame is not None
        self.show_all.setVisible(self._framed and self.toggle.isChecked())
        if self.show_all.isChecked():
            frame = None
        visible = (
            np.ones(x.size, dtype=bool) if frame is None else (x >= frame[0]) & (x <= frame[1])
        )
        self._visible = visible
        style = QtCore.Qt.PenStyle.DashLine if profile.warning else QtCore.Qt.PenStyle.SolidLine
        marker = "○" if profile.warning else "●"
        self._key_html = " ".join(
            f'<span style="color:{self._color(index)}">{marker}</span> {escape(s.assignment)}'
            + (" (selected)" if index == 0 else "")
            + ("" if np.isfinite(s.error) else " · σ unavailable")
            for index, s in enumerate(profile.series)
        )
        for index, series in reversed(list(enumerate(profile.series))):
            color = self._color(index)
            if np.isfinite(series.error):
                self.plot.addItem(
                    pg.ErrorBarItem(
                        x=x,
                        y=series.amplitudes,
                        height=np.full(x.size, 2 * series.error),
                        beam=0,
                        pen=pg.mkPen(color, width=1, style=style),
                    )
                )
            points = pg.ScatterPlotItem(
                x=x,
                y=series.amplitudes,
                data=np.arange(x.size),
                symbol="o",
                size=7 if index == 0 else 5,
                pen=pg.mkPen(color, width=1.2),
                brush=None if profile.warning else pg.mkBrush(color),
            )
            points.sigClicked.connect(self._clicked)
            self.plot.addItem(points)
        self.plot.getPlotItem().addItem(self.zero, ignoreBounds=True)
        self.plot.getPlotItem().addItem(self.plane_line, ignoreBounds=True)
        low, high = float(x[visible].min()), float(x[visible].max())
        pad = 0.03 * (high - low) or 0.5
        # Bounds cover amplitudes, finite error bars and zero; unavailable σ adds none.
        ys = np.concatenate(
            [s.amplitudes[visible] for s in profile.series]
            + [
                np.concatenate((s.amplitudes[visible] - s.error, s.amplitudes[visible] + s.error))
                for s in profile.series
                if np.isfinite(s.error)
            ]
            + [np.zeros(1)]
        )
        ys = ys[np.isfinite(ys)]
        bottom, top = float(ys.min()), float(ys.max())
        margin = 0.08 * (top - bottom) or 1.0
        self.plot.getViewBox().setRange(
            xRange=(low - pad, high + pad), yRange=(bottom - margin, top + margin), padding=0
        )
        hidden = np.flatnonzero(~visible)
        self._note_text = ""
        if hidden.size:
            left = x[hidden[0]] < low
            self._note_text = ("◂ " if left else "") + (
                f"{hidden.size} {'point' if hidden.size == 1 else 'points'} off scale: "
                + ", ".join(f"{x[j]:g} (plane {profile.source_planes[j] + 1})" for j in hidden)
                + ("" if left else " ▸")
            )
            self.show_all.setToolTip(
                "Off-scale observations: "
                + "; ".join(
                    f"{s.assignment} {s.amplitudes[j]:.4g} ({_sigma(s.error)}) at {x[j]:g} "
                    f"(plane {profile.source_planes[j] + 1})"
                    for j in hidden
                    for s in profile.series
                )
            )
        self._place_plane_marker()

    def _place_plane_marker(self) -> None:
        profile = self._profile
        if profile is None:
            return
        columns = np.flatnonzero(profile.source_planes == self._plane)
        shown = columns.size > 0 and bool(self._visible[columns[0]])
        self.plane_line.setVisible(shown)
        if shown:
            self.plane_line.setValue(float(profile.x[columns[0]]))
        off = bool(self._note_text) and columns.size > 0 and not shown
        note = (
            f'<br><span style="color:{self._colors["highlight" if off else "muted"]}">'
            f"{escape(self._note_text)}{' · current plane' if off else ''}</span>"
            if self._note_text
            else ""
        )
        self.basis.setText(f"{self._key_html}{note}<br>{self._basis_html}")

    def _color(self, index: int) -> str:
        return (
            self._colors["selected"]
            if index == 0
            else _MEMBER_COLORS[(index - 1) % len(_MEMBER_COLORS)]
        )

    def _clicked(self, _item, points, _event) -> None:
        profile = self._profile
        if profile is None:
            return
        columns = sorted({int(point.data()) for point in points})
        planes = sorted({int(profile.source_planes[j]) for j in columns})
        if len(planes) == 1:
            self.plane_requested.emit(planes[0] + 1)
            return
        menu = QtWidgets.QMenu(self)
        for plane in planes:
            j = int(np.flatnonzero(profile.source_planes == plane)[0])
            label = f"Plane {plane + 1}" + (
                f" · value {profile.x[j]:g}" if profile.plane_values else ""
            )
            action = menu.addAction(label)
            action.triggered.connect(
                lambda _checked=False, p=plane: self.plane_requested.emit(p + 1)
            )
        menu.popup(QtGui.QCursor.pos())
        self._chooser = menu
