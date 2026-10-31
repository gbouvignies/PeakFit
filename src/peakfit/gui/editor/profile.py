"""Selection-linked amplitude profiles of the active run, shown without recomputation."""

from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui import formatting
from peakfit.gui.theme import FONT_CAPTION, LIGHT, Theme, color

if TYPE_CHECKING:
    from peakfit.gui.dataset_fit import DatasetFitResult

_MAX_ISOLATED = 2


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


class ProfilePlot(QtWidgets.QWidget):
    """The selected peak's recorded amplitudes with its cluster members, in the inspector.

    Points are recorded amplitudes, unconnected, with the scaled analytical σ that
    exports write. The selected peak is drawn first-class; members are lighter on the
    same scale. Clicking a point opens its plane.
    """

    plane_requested = QtCore.Signal(int)

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.plot = pg.PlotWidget()
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setFixedHeight(176)
        self.plot.setAccessibleName("Signed fitted amplitudes of the selected peak's cluster")
        font = QtGui.QFont(QtWidgets.QApplication.font())
        font.setPixelSize(FONT_CAPTION)
        for name in ("left", "bottom"):
            axis = self.plot.getAxis(name)
            axis.setStyle(tickFont=font, tickLength=-4, maxTickLevel=0)
            axis.enableAutoSIPrefix(False)
        self.plot.getAxis("left").setWidth(40)
        layout.addWidget(self.plot)
        self.legend = QtWidgets.QLabel()
        self.legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.legend.setWordWrap(True)
        layout.addWidget(self.legend)
        notes = QtWidgets.QVBoxLayout()
        notes.setSpacing(4)
        self.note = QtWidgets.QLabel()
        self.note.setObjectName("caption")
        self.note.setWordWrap(True)
        self.show_all = QtWidgets.QPushButton("Show all points")
        self.show_all.setObjectName("chip")
        self.show_all.setCheckable(True)
        self.show_all.toggled.connect(self._draw)
        self.show_all.hide()
        notes.addWidget(self.note)
        notes.addWidget(self.show_all, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(notes)
        self.zero = pg.InfiniteLine(angle=0, pos=0)
        self.plane_line = pg.InfiniteLine(angle=90)
        self._profile: PeakProfile | None = None
        self._plane = 0
        self._visible = np.array([], dtype=bool)
        self._hidden_text = ""
        self.theme: Theme = LIGHT

    def apply_theme(self, theme: Theme) -> None:
        """Follow the application appearance; no scientific values change."""
        self.theme = theme
        self.plot.setBackground(theme.plot)
        for name in ("left", "bottom"):
            axis = self.plot.getAxis(name)
            axis.setPen(pg.mkPen(theme.divider))
            axis.setTextPen(pg.mkPen(theme.secondary))
        self.zero.setPen(pg.mkPen(theme.divider, width=1))
        self.plane_line.setPen(
            pg.mkPen(color(theme.faint), width=1, style=QtCore.Qt.PenStyle.DashLine)
        )
        self._draw()

    def show_profile(self, profile: PeakProfile | None, plane: int) -> None:
        """Draw a new selection or run; plane changes alone only move the marker."""
        self._profile = profile
        self._plane = plane
        self.show_all.setChecked(False)
        self._draw()

    def set_plane(self, plane: int) -> None:
        """Move the current-plane marker without redrawing observations."""
        self._plane = plane
        self._place_plane_marker()

    def _colors(self) -> list[str]:
        return list(self.theme.members)

    def _draw(self) -> None:
        profile = self._profile
        self.plot.clear()
        if profile is None:
            return
        x = profile.x
        frame = framed_range(x)
        self.show_all.setVisible(frame is not None)
        if self.show_all.isChecked():
            frame = None
        visible = (
            np.ones(x.size, dtype=bool) if frame is None else (x >= frame[0]) & (x <= frame[1])
        )
        self._visible = visible
        warned = profile.warning is not None
        colors = self._colors()
        style = QtCore.Qt.PenStyle.DashLine if warned else QtCore.Qt.PenStyle.SolidLine
        for index, series in reversed(list(enumerate(profile.series))):
            value = colors[index % len(colors)]
            selected = index == 0
            alpha = 1.0 if selected else 0.5
            if np.isfinite(series.error):
                self.plot.addItem(
                    pg.ErrorBarItem(
                        x=x,
                        y=series.amplitudes,
                        height=np.full(x.size, 2 * series.error),
                        beam=0,
                        pen=pg.mkPen(color(value, alpha * 0.7), width=1, style=style),
                    )
                )
            points = pg.ScatterPlotItem(
                x=x,
                y=series.amplitudes,
                data=np.arange(x.size),
                symbol="o",
                size=6 if selected else 4,
                pen=pg.mkPen(color(value, alpha), width=1.2),
                brush=None if warned else pg.mkBrush(color(value, alpha)),
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
        power = formatting.exponent(max(abs(bottom), abs(top)))
        style_text = {"color": self.theme.faint, "font-size": f"{FONT_CAPTION}px"}
        left = self.plot.getAxis("left")
        left.setScale(10.0**-power)
        left.setLabel(f"×10{formatting.superscript(str(power))}" if power else "", **style_text)
        self.plot.setLabel(
            "bottom", "Plane value" if profile.plane_values else "Plane number", **style_text
        )
        marker = "○" if warned else "●"
        self.legend.setText(
            "&nbsp;&nbsp;".join(
                f'<span style="color:{colors[i % len(colors)]}">{marker}</span>&nbsp;'
                + (f"<b>{escape(s.assignment)}</b>" if i == 0 else escape(s.assignment))
                + ("" if np.isfinite(s.error) else " · σ unavailable")
                for i, s in enumerate(profile.series)
            )
        )
        hidden = np.flatnonzero(~visible)
        self._hidden_text = ""
        if hidden.size:
            self._hidden_text = (
                f"{hidden.size} {'point' if hidden.size == 1 else 'points'} off scale: "
                + ", ".join(
                    f"{formatting.intensity(float(x[j]))} (plane {profile.source_planes[j] + 1})"
                    for j in hidden
                )
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
            self.note.setText("")
            return
        columns = np.flatnonzero(profile.source_planes == self._plane)
        shown = columns.size > 0 and bool(self._visible[columns[0]])
        self.plane_line.setVisible(shown)
        if shown:
            self.plane_line.setValue(float(profile.x[columns[0]]))
        off = bool(self._hidden_text) and columns.size > 0 and not shown
        scale = (
            f"scaled by √χ²ᵣ = {profile.uncertainty_scale:.3g}"
            if profile.uncertainty_scale > 1
            else "not scaled (χ²ᵣ ≤ 1)"
        )
        self.note.setText(
            "±1σ analytical, lineshapes held fixed, "
            + scale
            + (f" · {self._hidden_text}" if self._hidden_text else "")
            + (" · current plane is off scale" if off else "")
        )
        self.note.setToolTip(
            "Analytical amplitude standard errors computed with the fitted lineshape "
            "parameters held fixed, so their uncertainty and correlations are not "
            "propagated. One σ per peak applies to every plane. They can underestimate "
            "the uncertainty of overlapping peaks."
        )

    def _clicked(self, _item, points, event) -> None:
        profile = self._profile
        if profile is None:
            return
        columns = sorted({int(point.data()) for point in points})
        if event is not None and len(columns) > 1:
            # Dense series put several points under the pointer: take the nearest one,
            # then offer a choice only among planes that share its plane value.
            width, height = self.plot.getViewBox().viewPixelSize()
            click = event.pos()
            nearest = min(
                points,
                key=lambda point: ((point.pos().x() - click.x()) / width) ** 2
                + ((point.pos().y() - click.y()) / height) ** 2,
            )
            value = profile.x[int(nearest.data())]
            columns = [j for j in columns if profile.x[j] == value]
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
