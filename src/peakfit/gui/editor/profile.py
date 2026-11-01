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


RELIABILITY_LIMIT = 0.1  # largest σ_R/|R| for which first-order propagation is used


@dataclass(frozen=True, slots=True)
class Normalized:
    """Display-only ratios r_k = A_k / R to a reference mean, with propagated σ."""

    values: np.ndarray
    errors: np.ndarray  # first-order σ(r_k); zero for an exact reference point
    reference: float  # R
    reference_error: float  # σ_R
    in_reference: np.ndarray  # columns belonging to S
    exact: bool  # one reference plane: its ratio is exactly 1, with no error bar


def normalize(
    amplitudes: np.ndarray,
    errors: np.ndarray,
    reference: np.ndarray,
    limit: float = RELIABILITY_LIMIT,
) -> Normalized | str:
    """Normalize one peak's amplitudes to the mean of its reference columns S.

    Uses first-order propagation with the full gradient (specification §9):
    ∂r_k/∂A_k = 1/R − [k∈S]·A_k/(nR²) and ∂r_k/∂A_j = −A_k/(nR²) for j∈S, j≠k.
    Amplitudes of different planes are treated as uncorrelated, as the recorded
    analytical uncertainties (lineshapes held fixed) imply. Returns the reason
    instead when the ratio is not reliable enough to show.
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    errors = np.asarray(errors, dtype=float)
    columns = np.asarray(reference, dtype=int)
    n = columns.size
    if n == 0:
        return "no reference plane chosen"
    if not np.all(np.isfinite(errors)):
        return "σ unavailable for this peak"
    mean = float(np.mean(amplitudes[columns]))
    if not np.isfinite(mean) or mean == 0:
        return "reference amplitude is zero or not finite"
    sigma_r = float(np.sqrt(np.sum(errors[columns] ** 2)) / n)
    ratio = sigma_r / abs(mean)
    if ratio > limit:
        return f"reference not distinguishable from its uncertainty: σ_R/|R| = {ratio:.2f}"
    values = amplitudes / mean
    in_reference = np.zeros(amplitudes.size, dtype=bool)
    in_reference[columns] = True
    coupling = amplitudes / (n * mean**2)  # |∂r_k/∂A_j| for j in S, j ≠ k
    reference_variance = float(np.sum(errors[columns] ** 2))
    own = 1 / mean - in_reference * coupling  # ∂r_k/∂A_k
    others = reference_variance - in_reference * errors**2  # Σ σ_j² over S without k
    variance = own**2 * errors**2 + coupling**2 * others
    exact = n == 1
    if exact:
        values[columns] = 1.0
        variance[columns] = 0.0
    return Normalized(values, np.sqrt(variance), mean, sigma_r, in_reference, exact)


def parse_planes(text: str, count: int) -> tuple[int, ...]:
    """One-based plane numbers from "1, 3-5"; raise ValueError naming the problem."""
    planes: set[int] = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        low, _dash, high = part.partition("-")
        try:
            first, last = int(low), int(high or low)
        except ValueError:
            raise ValueError(f"“{part}” is not a plane number or range") from None
        if not 1 <= first <= last <= count:
            raise ValueError(f"Planes must lie between 1 and {count}: “{part}”")
        planes.update(range(first, last + 1))
    if not planes:
        raise ValueError("Enter at least one reference plane")
    return tuple(sorted(planes))


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
        notes.addWidget(self.note)
        notes.addLayout(self._chips())
        layout.addLayout(notes)
        self.reference: tuple[int, ...] = ()  # zero-based source planes; session only
        self._normalized: Normalized | None = None
        self.normalized = False
        self.zero = pg.InfiniteLine(angle=0, pos=0)
        self.plane_line = pg.InfiniteLine(angle=90)
        self._profile: PeakProfile | None = None
        self._plane = 0
        self._visible = np.array([], dtype=bool)
        self._hidden_text = ""
        self.theme: Theme = LIGHT

    def _chips(self) -> QtWidgets.QHBoxLayout:
        """Quantity choice and the off-scale toggle, side by side under the note."""
        self.show_all = QtWidgets.QPushButton("Show all points")
        self.show_all.setObjectName("chip")
        self.show_all.setCheckable(True)
        self.show_all.toggled.connect(self._draw)
        self.show_all.hide()
        # Raw amplitudes by default; a reference is only ever chosen explicitly.
        self.mode = QtWidgets.QPushButton("Raw amplitudes")
        self.mode.setObjectName("chip")
        self.mode.setToolTip("Profile quantity: raw amplitudes or normalized to reference planes")
        self.mode.clicked.connect(self._mode_menu)
        chips = QtWidgets.QHBoxLayout()
        chips.setSpacing(6)
        chips.addWidget(self.mode)
        chips.addWidget(self.show_all)
        chips.addStretch(1)
        return chips

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

    def _rows(self, profile: PeakProfile) -> tuple[list, list[str], str]:
        """Series as drawn: raw amplitudes, or ratios to the reference when available.

        Returns rows of (series index, values, errors or None, reference mask), the
        members whose normalization is unavailable, and the selected peak's reason
        when it is unavailable (then every row is raw).
        """
        raw = [
            (
                index,
                s.amplitudes,
                np.full(s.amplitudes.size, s.error) if np.isfinite(s.error) else None,
                None,
            )
            for index, s in enumerate(profile.series)
        ]
        if not (self.normalized and self.reference):
            return raw, [], ""
        if profile.warning is not None:
            # A degenerate decomposition can have amplitudes and σ inflated by the
            # same arbitrary scale; the ratio would cancel it and look well determined.
            return raw, [], "the engine marks this cluster's uncertainties unreliable"
        columns = np.flatnonzero(np.isin(profile.source_planes, self.reference))
        if columns.size != len(self.reference):
            return raw, [], "a reference plane is not in this run"
        rows, unavailable = [], []
        for index, series in enumerate(profile.series):
            shown = normalize(
                series.amplitudes, np.full(series.amplitudes.size, series.error), columns
            )
            if isinstance(shown, str):
                if index == 0:
                    return raw, [], shown
                unavailable.append(f"{series.assignment}: {shown}")
                continue
            if index == 0:
                self._normalized = shown
            rows.append((index, shown.values, shown.errors, shown.in_reference))
        return rows, unavailable, ""

    def _draw(self) -> None:  # noqa: PLR0915
        profile = self._profile
        self.plot.clear()
        self._normalized = None
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
        rows, unavailable, reason = self._rows(profile)
        normalized = self._normalized is not None
        warned = profile.warning is not None
        colors = self._colors()
        style = QtCore.Qt.PenStyle.DashLine if warned else QtCore.Qt.PenStyle.SolidLine
        for index, values, errors, in_reference in reversed(rows):
            value = colors[index % len(colors)]
            selected = index == 0
            alpha = 1.0 if selected else 0.5
            if errors is not None:
                self.plot.addItem(
                    pg.ErrorBarItem(
                        x=x,
                        y=values,
                        height=2 * errors,
                        beam=0,
                        pen=pg.mkPen(color(value, alpha * 0.7), width=1, style=style),
                    )
                )
            symbols = (
                ["d" if member else "o" for member in in_reference]
                if in_reference is not None
                else "o"
            )
            points = pg.ScatterPlotItem(
                x=x,
                y=values,
                data=np.arange(x.size),
                symbol=symbols,
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
        # Bounds cover values, finite error bars and zero; unavailable σ adds none.
        ys = np.concatenate(
            [values[visible] for _i, values, _e, _r in rows]
            + [
                np.concatenate(
                    (values[visible] - errors[visible], values[visible] + errors[visible])
                )
                for _i, values, errors, _r in rows
                if errors is not None
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
        scale = f"×10{formatting.superscript(str(power))}" if power else ""
        quantity = f"A / A_ref ({self.reference_text()})" if normalized else ""
        left.setLabel(" ".join(t for t in (quantity, scale) if t), **style_text)
        left.setWidth(52 if normalized else 40)
        self.plot.setLabel(
            "bottom", "Plane value" if profile.plane_values else "Plane number", **style_text
        )
        marker = "○" if warned else "●"
        shown = {index for index, *_ in rows}
        entries = [
            f'<span style="color:{colors[i % len(colors)]}">{marker}</span>&nbsp;'
            + (f"<b>{escape(s.assignment)}</b>" if i == 0 else escape(s.assignment))
            + ("" if np.isfinite(s.error) else " · σ unavailable")
            for i, s in enumerate(profile.series)
            if i in shown
        ] + (["◆&nbsp;reference"] if normalized else [])
        # Wrap between entries only, never inside an assignment such as 159N-H.
        self.legend.setText(
            " &nbsp;".join(f'<span style="white-space:nowrap">{e}</span>' for e in entries)
        )
        self.mode.setText(f"÷ {self.reference_text()}" if self.normalized else "Raw amplitudes")
        self._normalization_note = (
            f"Normalization unavailable: {reason}; showing raw amplitudes"
            if reason
            else "; ".join(f"{text}, not shown" for text in unavailable)
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

    def reference_text(self) -> str:
        """Name the reference set by one-based source plane number."""
        planes = [p + 1 for p in self.reference]
        if len(planes) == 1:
            return f"plane {planes[0]}"
        return "planes " + ", ".join(str(p) for p in planes) + ", mean"

    def _mode_menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        raw = menu.addAction("Raw amplitudes")
        raw.setCheckable(True)
        raw.setChecked(not self.normalized)
        raw.triggered.connect(lambda: self.set_normalized(False))
        if self.reference:
            ratio = menu.addAction(f"Normalized to {self.reference_text()}")
            ratio.setCheckable(True)
            ratio.setChecked(self.normalized)
            ratio.triggered.connect(lambda: self.set_normalized(True))
        menu.addSeparator()
        menu.addAction("Choose reference planes…").triggered.connect(self.choose_reference)
        menu.popup(self.mode.mapToGlobal(QtCore.QPoint(0, self.mode.height())))
        self._menu = menu

    def set_normalized(self, normalized: bool) -> None:
        """Switch the display quantity; recorded amplitudes never change."""
        self.normalized = normalized and bool(self.reference)
        self._draw()

    def set_reference(self, planes: tuple[int, ...]) -> None:
        """Adopt an explicit reference set (zero-based source planes) and show ratios."""
        self.reference = tuple(sorted(set(planes)))
        self.set_normalized(bool(self.reference))

    def choose_reference(self) -> None:
        """Ask for reference planes by one-based source plane number; never by value."""
        profile = self._profile
        count = int(profile.source_planes.max()) + 1 if profile is not None else 0
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Normalize to reference")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(
            QtWidgets.QLabel("Reference planes by plane number, for example 1 or 1-3, 131:")
        )
        field = QtWidgets.QLineEdit(", ".join(str(p + 1) for p in self.reference))
        field.setAccessibleName("Reference plane numbers")
        layout.addWidget(field)
        hint = QtWidgets.QLabel(self._isolated_hint())
        hint.setObjectName("caption")
        hint.setWordWrap(True)
        hint.setVisible(bool(hint.text()))
        layout.addWidget(hint)
        note = QtWidgets.QLabel(
            "A / A_ref with R the mean reference amplitude of each peak. Display only: "
            "recorded amplitudes, uncertainties and exports are unchanged. Unavailable "
            f"when σ_R/|R| exceeds {RELIABILITY_LIMIT:g}."
        )
        note.setObjectName("caption")
        note.setWordWrap(True)
        layout.addWidget(note)
        error = QtWidgets.QLabel()
        error.setObjectName("error")
        error.hide()
        layout.addWidget(error)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(buttons)

        def accept() -> None:
            try:
                planes = parse_planes(field.text(), count)
            except ValueError as exc:
                error.setText(str(exc))
                error.show()
                return
            dialog.accept()
            self.set_reference(tuple(p - 1 for p in planes))

        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        dialog.open()
        self._dialog = dialog

    def _isolated_hint(self) -> str:
        """Point out planes isolated from the rest; never choose them."""
        profile = self._profile
        if profile is None:
            return ""
        frame = framed_range(profile.x)
        if frame is None:
            return ""
        isolated = [j for j in range(profile.x.size) if not frame[0] <= profile.x[j] <= frame[1]]
        planes = ", ".join(
            f"plane {profile.source_planes[j] + 1}"
            + (
                f" (value {formatting.intensity(float(profile.x[j]))})"
                if profile.plane_values
                else ""
            )
            for j in isolated
        )
        return f"Isolated from the other planes: {planes}."

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
        shown = self._normalized
        quantity = (
            f"±1σ propagated to the ratio (first order, full gradient) from analytical σ, "
            f"lineshapes held fixed; σ_R/|R| = {shown.reference_error / abs(shown.reference):.3f}"
            f" (limit {RELIABILITY_LIMIT:g})"
            + (" · reference amplitude negative" if shown.reference < 0 else "")
            if shown is not None
            else "±1σ analytical, lineshapes held fixed, " + scale
        )
        extra = getattr(self, "_normalization_note", "")
        self.note.setText(
            quantity
            + (f" · {extra}" if extra else "")
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
