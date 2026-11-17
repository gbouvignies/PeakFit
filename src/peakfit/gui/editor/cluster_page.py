"""Inspector cluster page: one recorded cluster as the unit of review.

Members with their recorded parameters, one line each; the recorded amplitude
correlation; and the cluster's review, all read from the completed outcome. The
amplitude profile is the inspector's shared plot, placed here.
"""

import math
from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.engine.algorithms.covariance_factors import (
    PeakProfileCorrelation,
    ProfileCorrelation,
    profile_correlations,
)
from peakfit.gui import formatting
from peakfit.gui import theme as themes
from peakfit.gui.dataset_fit import parameter_name
from peakfit.gui.editor.labels import (
    Row,
    about,
    clear_layout,
    disclosure,
    label_rows,
    section,
    show_text,
    text_label,
)
from peakfit.gui.editor.profile import (
    CORRELATED_LIMIT,
    CORRELATED_MARK,
    NOT_ASSESSED_REASON,
    WITHIN_TOLERANCE_TEXT,
    correlation_findings,
    flagged,
    partner_name,
)
from peakfit.gui.editor.review import correlation_matrix

if TYPE_CHECKING:
    from peakfit.engine.scientific.final_outcome import FinalClusterOutcome
    from peakfit.gui.editor.review import ClusterReview

_KINDS = {"cs": "δ", "lw": "LW"}
_KIND_ORDER = ("cs", "lw")
_NUMBER_WIDTH = 16


ABOUT_MEASURES = (
    "χ²ᵣ and the recorded residuals are relative to the statistical noise estimate "
    "σ_stat, over the cluster's fitting samples. χ²ᵣ well above 1 means the noise "
    "level, the lineshape model or the noise assumptions do not describe the data. "
    "The largest recorded residual is set beside √(2 ln N), the typical largest of N "
    "independent standard Gaussian values, which grows slowly with N. When the noise "
    "is correlated, as after apodization, neighbouring samples are not independent: "
    "the comparison is a rough guide, not a significance threshold. Amplitude "
    "correlations ρ are recorded once for all planes and are conditional on the fitted "
    "lineshape parameters; |ρ| near 1 means the data do not determine how the signal "
    "splits between those members."
)


CORRELATION_QUALIFIER = "conditional on lineshapes"
CORRELATION_TIP = (
    "Recorded once for all planes, conditional on the fitted lineshape parameters; "
    "see About these measures"
)


def samples_caption(samples: int) -> str:
    """What χ²ᵣ is computed over."""
    return f"Over {formatting.count(samples, 'fitting sample')}"


def maximum_guide(samples: int) -> float:
    """The largest |residual| expected from that many independent Gaussian samples."""
    return math.sqrt(2 * math.log(samples))


def maximum_caption(samples: int) -> str:
    """The reference for the largest residual, with what it assumes."""
    return (
        f"Largest recorded residual on the fitting samples. Rough guide ≈ "
        f"{formatting.sigma_multiple(maximum_guide(samples))} σ for "
        f"{formatting.count(samples, 'independent Gaussian sample')}; "
        "not a significance threshold"
    )


def review_rows(
    review: ClusterReview,
    names: dict[str, str],
    *,
    collinear=None,
    correlation: Row | None = None,
) -> list[Row]:
    """The review as label rows: flags first (warnings, highlights), then measures.

    Peak and cluster pages share it; the peak page adds its strongest correlation
    and keeps only the near-collinear pairs it belongs to.
    """
    rows: list[Row] = []
    for index, warning in enumerate(review.warnings):
        label = "Warnings" if index == 0 else ""
        rows.append(Row(label, f"▲ {warning.kind}", style="warning", detail=warning.detail))
    if not review.warnings:
        rows.append(Row("Warnings", "none", tip="No numerical warnings recorded"))
    for index, highlight in enumerate(review.highlights):
        prefix = f"{names[highlight.peak]}: " if highlight.peak in names else ""
        rows.append(Row("Highlights" if index == 0 else "", f"◇ {prefix}{highlight.rule}"))
    if not review.highlights:
        rows.append(Row("Highlights", "none", tip="None at the current highlight thresholds"))
    if math.isfinite(review.chi2):
        samples = review.samples
        rows.append(Row("χ²ᵣ", formatting.chi2(review.chi2), tip=samples_caption(samples)))
        if math.isfinite(review.largest_residual) and samples > 1:
            largest = formatting.sigma_multiple(review.largest_residual)
            text = f"{largest} σ (guide {formatting.sigma_multiple(maximum_guide(samples))} σ)"
            rows.append(Row("Max residual", text, tip=maximum_caption(samples)))
    if correlation is not None:
        rows.append(correlation)
    for pair in review.collinear if collinear is None else collinear:
        text, detail = pair.describe(names)
        shapes = text.removeprefix("Near-collinear fitted shapes: ")
        rows.append(Row("Collinear", shapes, detail=detail, tip="Near-collinear fitted shapes"))
    return rows


def correlated_profiles(
    cluster: FinalClusterOutcome, engine_names: tuple[str, ...], names: tuple[str, ...]
) -> dict[str, tuple[PeakProfileCorrelation, str]]:
    """Each member's correlated-profile flag and its most correlated partner, by name.

    ``engine_names`` and ``names`` list the members in the review's order. Empty
    without an analytical evaluation or with one plane: there is no profile.
    """
    evaluation = cluster.analytical_evaluation
    if evaluation is None or evaluation.amplitudes.shape[1] < 2:  # noqa: PLR2004
        return {}
    flags = profile_correlations(evaluation.covariance_factors, evaluation.amplitudes)
    shown = dict(zip(engine_names, names, strict=True))
    by_index = [shown.get(name, name) for name in cluster.peak_names]
    return {
        engine: (flags[k], partner_name(flags[k], by_index))
        for k, engine in enumerate(cluster.peak_names)
    }


def correlation_row(
    correlations: dict[str, tuple[PeakProfileCorrelation, str]],
    engine_names: tuple[str, ...],
    names: tuple[str, ...],
) -> Row | None:
    """The cluster's correlated-profile flag as a review row; None without a profile."""
    if not correlations:
        return None
    members = [
        (name, correlations[e][0], correlations[e][1])
        for e, name in zip(engine_names, names, strict=True)
    ]
    flagged = [(n, c, p) for n, c, p in members if c.flagged]
    if flagged:
        findings = "; ".join(f"{n}: {correlation_findings(c, p)}" for n, c, p in flagged)
        detail = ", ".join(n for n, _c, _p in flagged) if len(members) > 1 else ""
        tip = f"Errors correlated across planes. {findings}. {CORRELATED_LIMIT}"
        return Row(
            "Profile errors", f"{CORRELATED_MARK} correlated across planes", detail=detail, tip=tip
        )
    if any(c.status is ProfileCorrelation.NOT_ASSESSED for _n, c, _p in members):
        return Row("Profile errors", "not assessed", tip="Not assessed: " + NOT_ASSESSED_REASON)
    return Row("Profile errors", "within tolerance", tip=WITHIN_TOLERANCE_TEXT)


class ClusterPage(QtWidgets.QWidget):
    """Review of one cluster; choosing a member opens that peak's page."""

    member_chosen = QtCore.Signal(str)
    focus_requested = QtCore.Signal()

    def __init__(self, axes: list[tuple[str, str]]) -> None:
        """``axes`` lists (engine axis name, display name), horizontal first."""
        super().__init__()
        self._axes = axes
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, themes.SPACE_XS, 0, 0)
        layout.setSpacing(themes.SPACE_S)
        self.focus_button = themes.command(
            "Focus cluster",
            "Frame this cluster in the spectrum; Escape returns to the view as it was "
            "(Enter in the cluster list)",
        )
        self.focus_button.clicked.connect(self.focus_requested)
        layout.addWidget(self.focus_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        # Numerical warnings named at the top; their explanations stay under Review.
        self.warnings = text_label("", "warning", wrap=True)
        self.warnings.hide()
        layout.addWidget(self.warnings)
        # One line per member: its number (the key to the matrix), a link to its
        # page, then its recorded parameters.
        section(layout, "Members")
        self.members = QtWidgets.QGridLayout()
        self.members.setHorizontalSpacing(themes.SPACE_S)
        self.members.setVerticalSpacing(0)
        layout.addLayout(self.members)
        self.members_note = text_label("", "caption", wrap=True)
        self.members_note.hide()
        layout.addWidget(self.members_note)
        # The inspector moves its amplitude profile here while this page is shown.
        self.profile_slot = QtWidgets.QVBoxLayout()
        self.profile_slot.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.profile_slot)
        self.correlation_title = section(layout, "Amplitude correlation")
        self.matrix = QtWidgets.QGridLayout()
        self.matrix.setHorizontalSpacing(2)
        self.matrix.setVerticalSpacing(2)
        layout.addLayout(self.matrix)
        self.matrix_note = text_label("", "caption", wrap=True)
        layout.addWidget(self.matrix_note)
        section(layout, "Review")
        self.review = QtWidgets.QGridLayout()
        self.review.setHorizontalSpacing(themes.SPACE_S)
        self.review.setVerticalSpacing(3)
        layout.addLayout(self.review)
        self.about_toggle, self.about = about(layout, "About these measures")
        self.about.setText(ABOUT_MEASURES)
        self.provenance_toggle = disclosure("Provenance")
        self.provenance = text_label("", "caption", wrap=True)
        self.provenance.hide()
        self.provenance_toggle.toggled.connect(self.provenance.setVisible)
        layout.addWidget(self.provenance_toggle, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.provenance)
        self.theme = themes.LIGHT

    def show_cluster(
        self,
        review: ClusterReview,
        cluster: FinalClusterOutcome,
        engine_names: tuple[str, ...],
        selected: str | None,
    ) -> None:
        """Fill every section from the recorded outcome; ``engine_names`` match ``review.ids``."""
        for grid in (self.members, self.matrix, self.review):
            clear_layout(grid)
            # Column sizes from the previous cluster would otherwise stay.
            for column in range(grid.columnCount()):
                grid.setColumnMinimumWidth(column, 0)
                grid.setColumnStretch(column, 0)
        self.members_note.hide()
        kinds = list(dict.fromkeys(w.kind for w in review.warnings))
        self.warnings.setText(f"▲ {' · '.join(kinds)} · see Review" if kinds else "")
        self.warnings.setVisible(bool(kinds))
        correlations = correlated_profiles(cluster, engine_names, review.names)
        self._fill_members(review, cluster, engine_names, selected, correlations)
        self._fill_matrix(review, cluster, engine_names)
        self._fill_review(review, correlation_row(correlations, engine_names, review.names))

    def _fill_members(
        self,
        review: ClusterReview,
        cluster: FinalClusterOutcome,
        engine_names: tuple[str, ...],
        selected: str | None,
        correlations: dict[str, tuple[PeakProfileCorrelation, str]],
    ) -> None:
        grid = self.members
        # Names keep their natural width: the parameter columns need the room.
        grid.setColumnMinimumWidth(0, _NUMBER_WIDTH)
        marked = {h.peak for h in review.highlights if h.peak}
        engines = dict(zip(review.ids, engine_names, strict=True))
        # Two header rows (axis name, then quantity and unit) above the members.
        for number, (identity, name) in enumerate(zip(review.ids, review.names, strict=True), 1):
            row = number + 1
            grid.addWidget(text_label(str(number), "caption"), row, 0)
            link = themes.link(name)
            link.setToolTip(f"Open {name}: position, profile and fitted values")
            link.setAccessibleName(f"Open peak {name}")
            if identity == selected:
                font = link.font()
                font.setWeight(QtGui.QFont.Weight.DemiBold)
                link.setFont(font)
            link.clicked.connect(lambda _checked=False, i=identity: self.member_chosen.emit(i))
            cell = QtWidgets.QHBoxLayout()
            cell.setSpacing(themes.SPACE_XS)
            cell.setContentsMargins(0, 0, themes.SPACE_S, 0)  # names apart from values
            cell.addWidget(link)
            if identity in marked:
                glyph = text_label("◇", "caption")
                glyph.setToolTip("; ".join(h.rule for h in review.highlights if h.peak == identity))
                cell.addWidget(glyph)
            correlation, partner = correlations.get(engines[identity], (None, ""))
            if flagged(correlation):
                mark = text_label(CORRELATED_MARK, "caption")
                mark.setToolTip(
                    f"Errors correlated across planes: {correlation_findings(correlation, partner)}"
                )
                cell.addWidget(mark)
            cell.addStretch(1)
            grid.addLayout(cell, row, 1)
        if cluster.analytical_evaluation is None:
            grid.setColumnStretch(2, 1)
            show_text(self.members_note, cluster.unusable_reason or "No fitted values recorded")
            return
        columns = self._fill_parameters(review, cluster, engine_names)
        # Columns keep their natural width; the spare room stays to the right.
        grid.setColumnStretch(len(columns) + 2, 1)

    def _fill_parameters(
        self, review: ClusterReview, cluster: FinalClusterOutcome, engine_names: tuple[str, ...]
    ) -> list[tuple[str, str, str]]:
        """The recorded parameters, one column each, from grid column 2."""
        grid = self.members
        values = {p.name: p for p in cluster.final_nonlinear_parameters}
        bounded = {
            name.strip()
            for w in review.warnings
            if w.kind == "At bound"
            for name in w.detail.split(",")
        }
        found = {tuple(name.split(".")[1:]) for name in values if name.count(".") == 2}  # noqa: PLR2004
        columns = [
            (axis, kind, title)
            for axis, title in self._axes
            for kind in (
                *_KIND_ORDER,
                *sorted({k for a, k in found if a == axis} - set(_KIND_ORDER)),
            )
            if (axis, kind) in found
        ]
        units = {
            (p.name.split(".")[1], p.name.split(".")[2]): p.unit
            for p in values.values()
            if p.name.count(".") == 2  # noqa: PLR2004
        }
        right = QtCore.Qt.AlignmentFlag.AlignRight
        for column, (axis, kind, title) in enumerate(columns, 2):
            # The axis name once over its columns, then each quantity with its unit.
            if column == 2 or columns[column - 3][0] != axis:  # noqa: PLR2004
                span = sum(1 for a, _k, _t in columns if a == axis)
                name = text_label(title, "caption")
                name.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(name, 0, column, 1, span)
            heading = text_label(_KINDS.get(kind, kind), "caption")
            heading.setAlignment(right)
            grid.addWidget(heading, 1, column)
        for number, engine in enumerate(engine_names, 1):
            for column, (axis, kind, _title) in enumerate(columns, 2):
                parameter = values.get(parameter_name(engine, axis, kind))
                if parameter is None:
                    continue
                shown = formatting.fitted(
                    parameter.value, parameter.standard_error, parameter.unit, varied=parameter.vary
                )
                at_bound = parameter.name in bounded
                cell = text_label(
                    formatting.quantity(parameter.value, parameter.unit),
                    "warning" if at_bound else "",
                )
                cell.setAlignment(right)
                cell.setToolTip(
                    f"{parameter.name}: {shown.text} {parameter.unit}"
                    if parameter.vary
                    else f"{parameter.name}: {shown.value_text} {parameter.unit} · fixed"
                )
                if at_bound:
                    cell.setToolTip(cell.toolTip() + " · at a recorded bound")
                grid.addWidget(cell, number + 1, column)
        # Units once under the table: per-column units would widen it past the inspector.
        quantities = dict.fromkeys(
            f"{_KINDS.get(kind, kind)} ({units[axis, kind]})"
            for axis, kind, _title in columns
            if units.get((axis, kind))
        )
        show_text(
            self.members_note,
            " · ".join([*quantities, *(["amber: at a recorded bound"] if bounded else [])]),
        )
        self.members_note.setToolTip("Recorded values; hover one for its ± σ")
        return columns

    def _fill_matrix(
        self, review: ClusterReview, cluster: FinalClusterOutcome, engine_names: tuple[str, ...]
    ) -> None:
        """Two members: one ρ row; more: the matrix. A single peak has no section."""
        for widget in (self.correlation_title, self.matrix_note):
            widget.setVisible(not review.single)
        self.matrix_note.setToolTip(CORRELATION_TIP)
        if review.single:
            return
        matrix = correlation_matrix(cluster)
        if matrix is None:
            self.matrix_note.setText("Unavailable: the amplitude covariance was not recorded.")
            return
        order = [cluster.peak_names.index(name) for name in engine_names]
        matrix = matrix[np.ix_(order, order)]
        size = len(order)
        if size == 2:  # noqa: PLR2004
            # One value: a row like the review's, its qualifier underneath.
            rho = float(matrix[0, 1])
            text = formatting.correlation(rho)
            tip = (
                f"ρ({review.names[0]}, {review.names[1]}) = {formatting.number(rho, 4)} · "
                f"{CORRELATION_TIP}"
            )
            self.matrix.setHorizontalSpacing(themes.SPACE_S)
            label_rows(self.matrix, [Row("ρ", text, detail=CORRELATION_QUALIFIER, tip=tip)])
            self.matrix_note.hide()
            return
        self.matrix_note.setText(CORRELATION_QUALIFIER)
        self.matrix.setHorizontalSpacing(2)
        for index in range(size):
            for row, column in ((0, index + 1), (index + 1, 0)):
                heading = text_label(str(index + 1), "caption")
                heading.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                heading.setToolTip(review.names[index])
                self.matrix.addWidget(heading, row, column)
        shade = QtGui.QColor(self.theme.secondary)
        for i in range(size):
            for j in range(size):
                rho = float(matrix[i, j])
                # The formatter never shows a tiny negative as "−0.00".
                text = "1" if i == j else formatting.correlation(rho)
                cell = text_label(text)
                cell.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                cell.setMinimumWidth(40)
                if i != j and math.isfinite(rho):
                    shade.setAlphaF(0.08 + 0.32 * min(abs(rho), 1.0))
                    cell.setStyleSheet(
                        f"background:{shade.name(QtGui.QColor.NameFormat.HexArgb)};"
                        "border-radius:3px;padding:2px 0;"
                    )
                    cell.setToolTip(
                        f"ρ({review.names[i]}, {review.names[j]}) = {formatting.number(rho, 4)}"
                    )
                self.matrix.addWidget(cell, i + 1, j + 1)
        self.matrix.setColumnStretch(size + 1, 1)

    def _fill_review(self, review: ClusterReview, correlation: Row | None) -> None:
        names = dict(zip(review.ids, review.names, strict=True))
        rows = review_rows(review, names)
        label_rows(self.review, rows if correlation is None else [*rows, correlation])
