"""Inspector cluster page: one recorded cluster as the unit of review.

Members with their recorded parameters, one line each; the recorded amplitude
correlation; and the cluster's review, all read from the completed outcome. The
amplitude profile is the inspector's shared plot, placed here.
"""

import math
from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui import formatting
from peakfit.gui import theme as themes
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
    return f"Over {samples:,} fitting samples".replace(",", " ")


def maximum_guide(samples: int) -> float:
    """The largest |residual| expected from that many independent Gaussian samples."""
    return math.sqrt(2 * math.log(samples))


def maximum_caption(samples: int) -> str:
    """The reference for the largest residual, with what it assumes."""
    return (
        f"Largest recorded residual on the fitting samples. Rough guide ≈ "
        f"{maximum_guide(samples):.1f} σ for {samples:,} independent Gaussian samples; "
        "not a significance threshold"
    ).replace(",", " ")


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
        rows.append((label, f"▲ {warning.kind}", "warning", warning.detail, ""))
    if not review.warnings:
        rows.append(("Warnings", "none", "", "", "No numerical warnings recorded"))
    for index, highlight in enumerate(review.highlights):
        prefix = f"{names[highlight.peak]}: " if highlight.peak in names else ""
        rows.append(("Highlights" if index == 0 else "", f"◇ {prefix}{highlight.rule}", "", "", ""))
    if not review.highlights:
        rows.append(("Highlights", "none", "", "", "None at the current highlight thresholds"))
    if math.isfinite(review.chi2):
        samples = review.samples
        rows.append(("χ²ᵣ", f"{review.chi2:.3g}", "", "", samples_caption(samples)))
        if math.isfinite(review.largest_residual) and samples > 1:
            text = f"{review.largest_residual:.1f} σ (guide {maximum_guide(samples):.1f} σ)"
            rows.append(("Max residual", text, "", "", maximum_caption(samples)))
    if correlation is not None:
        rows.append(correlation)
    for pair in review.collinear if collinear is None else collinear:
        text, detail = pair.describe(names)
        shapes = text.removeprefix("Near-collinear fitted shapes: ")
        rows.append(("Collinear", shapes, "", detail, "Near-collinear fitted shapes"))
    return rows


class ClusterPage(QtWidgets.QWidget):
    """Review of one cluster; choosing a member opens that peak's page."""

    member_chosen = QtCore.Signal(str)
    focus_requested = QtCore.Signal()

    def __init__(self, axes: list[tuple[str, str, int]]) -> None:
        """``axes`` lists (engine axis name, display title, ppm decimals), horizontal first."""
        super().__init__()
        self._axes = axes
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, themes.SPACE_XS, 0, 0)
        layout.setSpacing(themes.SPACE_S)
        self.focus_button = QtWidgets.QPushButton("Focus cluster")
        self.focus_button.setObjectName("chip")
        self.focus_button.setToolTip(
            "Frame this cluster in the spectrum; Escape returns to the view as it was "
            "(Enter in the cluster list)"
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
        self._fill_members(review, cluster, engine_names, selected)
        self._fill_matrix(review, cluster, engine_names)
        self._fill_review(review)

    def _fill_members(
        self,
        review: ClusterReview,
        cluster: FinalClusterOutcome,
        engine_names: tuple[str, ...],
        selected: str | None,
    ) -> None:
        grid = self.members
        # Names keep their natural width: the parameter columns need the room.
        grid.setColumnMinimumWidth(0, _NUMBER_WIDTH)
        marked = {h.peak for h in review.highlights if h.peak}
        for row, (identity, name) in enumerate(zip(review.ids, review.names, strict=True), 1):
            grid.addWidget(text_label(str(row), "caption"), row, 0)
            link = QtWidgets.QPushButton(name)
            link.setObjectName("link")
            link.setFlat(True)
            link.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
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
    ) -> list[tuple[str, str, str, int]]:
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
            (axis, kind, title, decimals)
            for axis, title, decimals in self._axes
            for kind in (
                *_KIND_ORDER,
                *sorted({k for a, k in found if a == axis} - set(_KIND_ORDER)),
            )
            if (axis, kind) in found
        ]
        for column, (_axis, kind, title, _decimals) in enumerate(columns, 2):
            heading = text_label(f"{_KINDS.get(kind, kind)} {title}", "caption")
            heading.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            grid.addWidget(heading, 0, column)
        for row, engine in enumerate(engine_names, 1):
            for column, (axis, kind, _title, decimals) in enumerate(columns, 2):
                parameter = values.get(f"{engine}.{axis}.{kind}")
                if parameter is None:
                    continue
                places = decimals if parameter.unit == "ppm" else 1 if parameter.unit == "Hz" else 3
                value, sigma = formatting.with_sigma(parameter.value, parameter.standard_error)
                at_bound = parameter.name in bounded
                cell = text_label(
                    formatting.number(parameter.value, places), "warning" if at_bound else ""
                )
                cell.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
                cell.setToolTip(
                    f"{parameter.name}: {value} ± {sigma} {parameter.unit}"
                    if parameter.vary
                    else f"{parameter.name}: {value} {parameter.unit} · fixed"
                )
                if at_bound:
                    cell.setToolTip(cell.toolTip() + " · at a recorded bound")
                grid.addWidget(cell, row, column)
        units = sorted({p.unit for p in values.values()})
        show_text(
            self.members_note,
            "δ in ppm · LW in Hz"
            + (f" ({', '.join(units)})" if set(units) - {"ppm", "Hz"} else "")
            + (" · amber: at a recorded bound" if bounded else ""),
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
            text = formatting.number(round(rho, 2) + 0.0, 2) if math.isfinite(rho) else "–"
            tip = f"ρ({review.names[0]}, {review.names[1]}) = {rho:.4f} · {CORRELATION_TIP}"
            self.matrix.setHorizontalSpacing(themes.SPACE_S)
            label_rows(self.matrix, [("ρ", text, "", CORRELATION_QUALIFIER, tip)])
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
                # Round before formatting so a tiny negative never reads "−0.00".
                text = (
                    "1"
                    if i == j
                    else formatting.number(round(rho, 2) + 0.0, 2)
                    if math.isfinite(rho)
                    else "–"
                )
                cell = text_label(text)
                cell.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                cell.setMinimumWidth(40)
                if i != j and math.isfinite(rho):
                    shade.setAlphaF(0.08 + 0.32 * min(abs(rho), 1.0))
                    cell.setStyleSheet(
                        f"background:{shade.name(QtGui.QColor.NameFormat.HexArgb)};"
                        "border-radius:3px;padding:2px 0;"
                    )
                    cell.setToolTip(f"ρ({review.names[i]}, {review.names[j]}) = {rho:.4f}")
                self.matrix.addWidget(cell, i + 1, j + 1)
        self.matrix.setColumnStretch(size + 1, 1)

    def _fill_review(self, review: ClusterReview) -> None:
        names = dict(zip(review.ids, review.names, strict=True))
        label_rows(self.review, review_rows(review, names))
