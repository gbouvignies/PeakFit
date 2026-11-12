"""Selection-linked amplitude profiles of the active run, shown without recomputation."""

from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.engine.diagnostics.identifiability import (
    AmplitudeStatus,
    Cause,
    MemberAmplitude,
    amplitude_identifiability,
)
from peakfit.engine.scientific.profiles import (
    RELIABILITY_LIMIT,
    Normalized,
    Unavailable,
    normalize,
)
from peakfit.gui import formatting
from peakfit.gui.editor.labels import about, disclosure, phrases
from peakfit.gui.theme import FONT_CAPTION, LIGHT, Theme, color

if TYPE_CHECKING:
    from peakfit.engine.algorithms.noise_correlation import NoiseModel
    from peakfit.gui.dataset_fit import DatasetFitResult

_MAX_ISOLATED = 2


@dataclass(frozen=True, slots=True)
class ProfileSeries:
    """One peak's recorded amplitudes, analytical standard error and identifiability."""

    assignment: str
    amplitudes: np.ndarray
    error: float
    identifiability: MemberAmplitude
    partners: tuple[str, ...] = ()  # assignments of near-collinear members


@dataclass(frozen=True, slots=True)
class PeakProfile:
    """The selected peak and its cluster members, exactly as one run recorded them."""

    run_id: str
    series: tuple[ProfileSeries, ...]  # selected peak first
    source_planes: np.ndarray  # zero-based source plane of each amplitude column
    x: np.ndarray  # recorded plane values, or one-based plane numbers
    plane_values: bool
    reduced_chi_squared: float
    noise_model: NoiseModel  # the noise correlation the error bars assume


def peak_profile(
    result: DatasetFitResult, working_id: str, plane_values: tuple[float, ...] | None
) -> PeakProfile | str:
    """Return the selected peak's cluster profiles, or why the run has none.

    Plane values are used only when supplied and identical to those the run
    recorded; otherwise columns are labeled by one-based source plane number.
    """
    found = result.index.find(working_id)
    if found is None:
        return f"Not part of run {result.run_id[:8]}"
    engine_name, cluster = found
    evaluation = cluster.analytical_evaluation
    if not cluster.usable or evaluation is None:
        return cluster.unusable_reason or "No analytical evaluation for this cluster"
    by_engine = result.index.by_engine
    members = amplitude_identifiability(
        evaluation.shapes, evaluation.amplitudes, evaluation.amplitude_standard_errors
    )
    order = sorted(
        range(len(cluster.peak_names)),
        key=lambda k: cluster.peak_names[k] != engine_name,
    )
    series = tuple(
        ProfileSeries(
            by_engine[cluster.peak_names[k]].assignment,
            evaluation.amplitudes[k],
            float(evaluation.amplitude_standard_errors[k]),
            members[k],
            tuple(by_engine[cluster.peak_names[j]].assignment for j in members[k].partners),
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
        result.outcome.noise_model,
    )


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


def _scientific(value: float, digits: int = 2) -> str:
    """A small or large diagnostic number as 3.3×10⁻⁸; zero stays 0."""
    if not np.isfinite(value):
        return "–"
    if value == 0:
        return "0"
    mantissa, power = f"{value:.{digits - 1}e}".split("e")
    exponent = int(power)
    sign = mantissa.replace("-", "−")
    return sign if exponent == 0 else f"{sign}×10{formatting.superscript(str(exponent))}"


def withheld_reason(series: ProfileSeries) -> str:
    """Why one member's amplitudes are not shown, naming which class applies."""
    m = series.identifiability
    if m.cause is Cause.NULL_SPACE:
        return (
            "not identifiable: the fitted shapes are linearly dependent (numerical rank "
            f"{m.rank} of {m.peaks}) and this amplitude lies partly in their null space"
        )
    if m.cause is Cause.NEAR_COLLINEAR:
        return (
            "reliability not established: near-collinear fitted shapes with "
            f"{', '.join(series.partners)} (1 − |cos| = {_scientific(m.dissimilarity)})"
        )
    if m.cause is Cause.ILL_CONDITIONED:
        return (
            "reliability not established: ill-conditioned fitted shapes (smallest to "
            f"largest singular value {_scientific(m.condition)})"
        )
    if m.cause is Cause.INVALID_SHAPE:
        return "reliability not established: the fitted shape cannot be assessed"
    return ""


def withheld_text(profile: PeakProfile) -> str:
    """The statement that replaces withheld individual profiles; empty when none."""
    hidden = [s for s in profile.series if not s.identifiability.determined]
    if not hidden:
        return ""
    parts, stated = [], set()
    for s in hidden:
        if id(s) in stated:
            continue
        group = [
            t
            for t in hidden
            if t.identifiability.cause is Cause.NEAR_COLLINEAR
            and (t is s or t.assignment in s.partners)
        ]
        names = {t.assignment for t in group}
        if (
            s.identifiability.cause is Cause.NEAR_COLLINEAR
            and len(group) > 1
            and all(set(t.partners) <= names for t in group)
        ):
            # Mutual partners share one reason: state it once for the pair.
            closest = min(t.identifiability.dissimilarity for t in group)
            parts.append(
                " and ".join(t.assignment for t in group)
                + ": reliability not established: near-collinear fitted shapes "
                + f"(1 − |cos| = {_scientific(closest)})"
            )
            stated.update(id(t) for t in group)
        else:
            parts.append(f"{s.assignment}: {withheld_reason(s)}")
            stated.add(id(s))
    parts = "; ".join(parts)
    # The total model stays evaluable; that says nothing about its decomposition.
    decomposition = (
        "the data do not determine the individual amplitude decomposition"
        if any(s.identifiability.cause is Cause.NULL_SPACE for s in hidden)
        else "the individual amplitude decomposition is unreliable"
    )
    return (
        f"Individual amplitude profile not shown. {parts}. The total fitted model remains "
        f"available for comparison, but {decomposition}. Recorded values are under "
        "Numerical diagnostics."
    )


def diagnostics_text(profile: PeakProfile) -> str:
    """Recorded values and the numbers behind every member that is not identifiable."""
    lines = []
    for s in profile.series:
        m = s.identifiability
        if m.status is AmplitudeStatus.IDENTIFIABLE:
            continue
        low, high = float(np.min(s.amplitudes)), float(np.max(s.amplitudes))
        sigma = f"σ {_scientific(s.error, 3)}" if np.isfinite(s.error) else "σ not available"
        lines += [
            f"{s.assignment} · {m.status.value} ({m.cause.value})",
            f"  recorded amplitudes {_scientific(low, 3)} to {_scientific(high, 3)} · {sigma}",
            f"  rank {m.rank} of {m.peaks} · smallest/largest singular value "
            f"{_scientific(m.condition)} · null-space projection "
            f"{_scientific(m.null_projection)} · ill-conditioned projection "
            f"{_scientific(m.weak_projection)}"
            + (
                f" · 1 − |cos| to {', '.join(s.partners)} {_scientific(m.dissimilarity)}"
                if s.partners
                else ""
            ),
        ]
    if lines:
        lines.append("Recorded values are unchanged; Export results writes them.")
    return "\n".join(lines)


def _sigma(error: float) -> str:
    """Name an unavailable standard error instead of formatting NaN as a value."""
    return f"σ = {formatting.scaled(error, 3)}" if np.isfinite(error) else "σ unavailable"


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

    Points are recorded amplitudes, unconnected, with the analytical σ that exports
    write, from the noise level alone; the cluster's χ²ᵣ is shown beside it. The
    selected peak is drawn first-class; members are lighter on the same scale.
    Clicking a point opens its plane.
    """

    plane_requested = QtCore.Signal(int)

    def __init__(self) -> None:  # noqa: PLR0915
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
        self.fallback = QtWidgets.QLabel()
        self.fallback.setObjectName("warning")
        self.fallback.setWordWrap(True)
        self.fallback.hide()
        layout.addWidget(self.fallback)
        self.legend = QtWidgets.QLabel()
        self.legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.legend.setWordWrap(True)
        layout.addWidget(self.legend)
        notes = QtWidgets.QVBoxLayout()
        notes.setSpacing(4)
        self.note = QtWidgets.QLabel()
        self.note.setObjectName("caption")
        self.note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.note.setWordWrap(True)
        notes.addWidget(self.note)
        notes.addLayout(self._chips())
        layout.addLayout(notes)
        self.about_toggle, self.about = about(layout, "About these error bars")
        # Members whose individual amplitudes are withheld, and why; then the numbers.
        self.withheld = QtWidgets.QLabel()
        self.withheld.setObjectName("warning")
        self.withheld.setWordWrap(True)
        self.withheld.hide()
        layout.addWidget(self.withheld)
        self.diagnostics_toggle = disclosure("Numerical diagnostics")
        self.diagnostics_toggle.hide()
        layout.addWidget(self.diagnostics_toggle, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.diagnostics = QtWidgets.QLabel()
        self.diagnostics.setObjectName("caption")
        self.diagnostics.setWordWrap(True)
        self.diagnostics.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.diagnostics.hide()
        self.diagnostics_toggle.toggled.connect(self.diagnostics.setVisible)
        layout.addWidget(self.diagnostics)
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

    def _rows(self, profile: PeakProfile) -> tuple[list, list[str], Unavailable | None]:
        """Series as drawn: raw amplitudes, or ratios to the reference when available.

        Returns rows of (series index, values, errors or None, reference mask), the
        members whose normalization is unavailable, and why the selected peak's is
        unavailable (then every row is raw).
        """
        # Only amplitudes the data determine are drawn; withheld members are explained.
        raw = [
            (
                index,
                s.amplitudes,
                np.full(s.amplitudes.size, s.error) if np.isfinite(s.error) else None,
                None,
            )
            for index, s in enumerate(profile.series)
            if s.identifiability.determined
        ]
        if not (self.normalized and self.reference):
            return raw, [], None
        selected = profile.series[0]
        if not selected.identifiability.determined:
            # Nothing is normalized; with members still drawn, say so (raw values shown).
            if not raw:
                return raw, [], None
            null = selected.identifiability.cause is Cause.NULL_SPACE
            reason = f"the individual amplitudes of {selected.assignment} are " + (
                "not identifiable" if null else "not established"
            )
            return raw, [], Unavailable(reason, undefined=null)
        columns = np.flatnonzero(np.isin(profile.source_planes, self.reference))
        if columns.size != len(self.reference):
            return raw, [], Unavailable("a reference plane is not in this run", undefined=True)
        rows, unavailable = [], []
        for index, series in enumerate(profile.series):
            if not series.identifiability.determined:
                continue
            shown = normalize(
                series.amplitudes, np.full(series.amplitudes.size, series.error), columns
            )
            if isinstance(shown, Unavailable):
                if index == 0:
                    return raw, [], shown
                unavailable.append(f"{series.assignment}: {shown.reason}")
                continue
            if index == 0:
                self._normalized = shown
            rows.append((index, shown.values, shown.errors, shown.in_reference))
        return rows, unavailable, None

    def _draw(self) -> None:  # noqa: PLR0915
        profile = self._profile
        self.plot.clear()
        self._normalized = None
        withheld = withheld_text(profile) if profile is not None else ""
        self.withheld.setText(withheld)
        self.withheld.setVisible(bool(withheld))
        diagnostics = diagnostics_text(profile) if profile is not None else ""
        self.diagnostics.setText(diagnostics)
        self.diagnostics_toggle.setVisible(bool(diagnostics))
        self.diagnostics.setVisible(bool(diagnostics) and self.diagnostics_toggle.isChecked())
        if profile is None:
            return
        x = profile.x
        frame = framed_range(x)
        self.show_all.setVisible(
            frame is not None and any(s.identifiability.determined for s in profile.series)
        )
        if self.show_all.isChecked():
            frame = None
        visible = (
            np.ones(x.size, dtype=bool) if frame is None else (x >= frame[0]) & (x <= frame[1])
        )
        self._visible = visible
        rows, unavailable, reason = self._rows(profile)
        normalized = self._normalized is not None
        colors = self._colors()
        drawn = bool(rows)
        # With nothing drawn, the error-model note, legend and chips would describe no plot.
        for widget in (self.plot, self.legend, self.note, self.mode, self.about_toggle):
            widget.setVisible(drawn)
        self.about.setVisible(drawn and self.about_toggle.isChecked())
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
                        pen=pg.mkPen(color(value, alpha * 0.7), width=1),
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
                brush=pg.mkBrush(color(value, alpha)),
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
        marker = "●"
        shown = {index for index, *_ in rows}
        entries = [
            f'<span style="color:{colors[i % len(colors)]}">{marker}</span>&nbsp;'
            + (f"<b>{escape(s.assignment)}</b>" if i == 0 else escape(s.assignment))
            + ("" if np.isfinite(s.error) else " · σ not available")
            for i, s in enumerate(profile.series)
            if i in shown
        ] + (["◆&nbsp;reference"] if normalized else [])
        # Wrap between entries only, never inside an assignment such as 159N-H.
        self.legend.setText(
            " &nbsp;".join(f'<span style="white-space:nowrap">{e}</span>' for e in entries)
        )
        # The chip names the requested quantity; the axis and the fallback line the shown one.
        requested = f"÷ {self.reference_text()}"
        self.mode.setText(
            "Raw amplitudes"
            if not self.normalized
            else f"{requested} · not applied"
            if reason
            else requested
        )
        self.fallback.setText(
            f"▲ Showing raw amplitudes. Normalization to {self.reference_text()} is "
            + ("undefined" if reason.undefined else "withheld as unreliable")
            + f": {reason.reason}."
            if reason
            else ""
        )
        self.fallback.setVisible(reason is not None)
        self._normalization_note = "; ".join(f"{text}, not shown" for text in unavailable)
        hidden = np.flatnonzero(~visible)
        self._hidden_text = ""
        if hidden.size:
            planes = ", ".join(str(profile.source_planes[j] + 1) for j in hidden)
            self._hidden_text = (
                f"{hidden.size} {'point' if hidden.size == 1 else 'points'} off scale "
                f"(plane{'s' if hidden.size > 1 else ''} {planes})"
            )
            self.show_all.setToolTip(
                "Off-scale observations: "
                + "; ".join(
                    f"{s.assignment} {formatting.scaled(s.amplitudes[j])} ({_sigma(s.error)}) "
                    f"at {formatting.intensity(float(x[j]))} "
                    f"(plane {profile.source_planes[j] + 1})"
                    for j in hidden
                    for s in profile.series
                    if s.identifiability.determined
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
        # The qualifiers that change what a bar means, in a few words (About has the
        # full explanation); what is unusual about the data on the next line. χ²ᵣ is
        # with the cluster's measures.
        noise = "modelled" if profile.noise_model.modelled else "not modelled"
        shown = self._normalized
        quantity = (
            f"±1σ of the ratio, conditional on lineshapes · noise correlation {noise} · "
            f"σ_R/|R| = {shown.reference_error / abs(shown.reference):.3f} "
            f"(limit {RELIABILITY_LIMIT:g})"
            + (" · reference amplitude negative" if shown.reference < 0 else "")
            if shown is not None
            else f"±1σ conditional on lineshapes · noise correlation {noise}"
        )
        extra = getattr(self, "_normalization_note", "")
        states = [
            part
            for part in (self._hidden_text, "current plane is off scale" if off else "")
            if part
        ]
        # Wraps only between phrases, so no qualifier is split across lines.
        self.note.setText(phrases((*quantity.split(" · "), extra), tuple(states)))
        # Say what the bars on show are: raw amplitude σ, or the ratio's propagated σ.
        if shown is None:
            quantity_text = (
                "Error bars are the recorded analytical standard errors of the amplitudes: "
                "one σ per peak, the same on every plane."
            )
        else:
            quantity_text = (
                "Error bars are the first-order (full-gradient) standard errors of the "
                "ratio A_k / R, where R is the mean amplitude of the reference planes, "
                "propagated from the analytical amplitude standard errors with amplitudes "
                "of different planes treated as uncorrelated. They differ from plane to "
                "plane. "
                + (
                    "With a single reference plane its ratio is exactly 1 and has no error bar."
                    if shown.exact
                    else "Each reference plane enters R, so its own ratio has a propagated "
                    "error too."
                )
            )
        explanation = (
            quantity_text + " The amplitude standard errors are computed with the fitted lineshape "
            "parameters held fixed, so their uncertainty and correlations are not "
            "propagated; they can underestimate the uncertainty of overlapping peaks. "
            "They are never rescaled by χ²ᵣ. "
            + profile.noise_model.description
            + " χ²ᵣ is a goodness-of-fit diagnostic: well above 1, the noise level, the "
            "lineshape model or the noise assumptions do not describe the data, so these "
            "error bars need not match the observed scatter."
        )
        self.note.setToolTip(explanation)
        self.about.setText(explanation)

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
