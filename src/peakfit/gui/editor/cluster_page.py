"""Inspector cluster page: one recorded cluster as the unit of review.

Members, their recorded parameters side by side, the recorded amplitude
correlation matrix and the cluster's review, all read from the completed
outcome. The amplitude profile is the inspector's shared plot, placed here.
"""

import math
from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui import formatting
from peakfit.gui import theme as themes
from peakfit.gui.editor.labels import clear_layout, review_item, text_label
from peakfit.gui.editor.review import correlation_matrix

if TYPE_CHECKING:
    from peakfit.engine.scientific.final_outcome import FinalClusterOutcome
    from peakfit.gui.editor.review import ClusterReview

_KINDS = {"cs": "δ", "lw": "LW"}
_KIND_ORDER = ("cs", "lw")


def _section(layout: QtWidgets.QVBoxLayout, text: str) -> QtWidgets.QLabel:
    label = text_label(text.upper(), "section")
    layout.addSpacing(themes.SPACE_S)
    layout.addWidget(label)
    return label


class ClusterPage(QtWidgets.QWidget):
    """Review of one cluster; choosing a member opens that peak's page."""

    member_chosen = QtCore.Signal(str)
    lens_requested = QtCore.Signal()

    def __init__(self, axes: list[tuple[str, str, int]]) -> None:  # noqa: PLR0915
        """``axes`` lists (engine axis name, display title, ppm decimals), horizontal first."""
        super().__init__()
        self._axes = axes
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, themes.SPACE_XS, 0, 0)
        layout.setSpacing(themes.SPACE_S)
        self.lens_button = QtWidgets.QPushButton("Open lens")
        self.lens_button.setObjectName("chip")
        self.lens_button.setToolTip(
            "Judge this cluster as Experimental, Overlay and whole-spectrum residual (Enter in the "
            "cluster list)"
        )
        self.lens_button.clicked.connect(self.lens_requested)
        layout.addWidget(self.lens_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        # Numerical warnings named at the top; their explanations stay under Review.
        self.warnings = text_label("", "warning", wrap=True)
        self.warnings.hide()
        layout.addWidget(self.warnings)
        _section(layout, "Members")
        self.members = QtWidgets.QGridLayout()
        self.members.setHorizontalSpacing(themes.SPACE_S)
        self.members.setVerticalSpacing(0)
        layout.addLayout(self.members)
        # The inspector moves its amplitude profile here while this page is shown.
        self.profile_slot = QtWidgets.QVBoxLayout()
        self.profile_slot.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.profile_slot)
        _section(layout, "Parameters")
        self.parameters = QtWidgets.QGridLayout()
        self.parameters.setHorizontalSpacing(themes.SPACE_S)
        self.parameters.setVerticalSpacing(3)
        layout.addLayout(self.parameters)
        self.parameters_note = text_label("", "caption", wrap=True)
        layout.addWidget(self.parameters_note)
        _section(layout, "Amplitude correlation")
        self.matrix = QtWidgets.QGridLayout()
        self.matrix.setHorizontalSpacing(2)
        self.matrix.setVerticalSpacing(2)
        layout.addLayout(self.matrix)
        self.matrix_note = text_label("", "caption", wrap=True)
        layout.addWidget(self.matrix_note)
        _section(layout, "Review")
        self.review = QtWidgets.QVBoxLayout()
        self.review.setSpacing(2)
        layout.addLayout(self.review)
        self.provenance_toggle = QtWidgets.QToolButton()
        self.provenance_toggle.setObjectName("text")
        self.provenance_toggle.setText("Provenance")
        self.provenance_toggle.setCheckable(True)
        self.provenance_toggle.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.provenance_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.provenance = text_label("", "caption", wrap=True)
        self.provenance.hide()
        self.provenance_toggle.toggled.connect(self._toggle_provenance)
        layout.addSpacing(themes.SPACE_XS)
        layout.addWidget(self.provenance_toggle, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.provenance)
        self.theme = themes.LIGHT

    def _toggle_provenance(self, shown: bool) -> None:
        self.provenance.setVisible(shown)
        self.provenance_toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if shown else QtCore.Qt.ArrowType.RightArrow
        )

    def show_cluster(
        self,
        review: ClusterReview,
        cluster: FinalClusterOutcome,
        engine_names: tuple[str, ...],
        selected: str | None,
    ) -> None:
        """Fill every section from the recorded outcome; ``engine_names`` match ``review.ids``."""
        for layout in (self.members, self.parameters, self.matrix, self.review):
            clear_layout(layout)
        kinds = list(dict.fromkeys(w.kind for w in review.warnings))
        self.warnings.setText(f"▲ {' · '.join(kinds)} · see Review" if kinds else "")
        self.warnings.setVisible(bool(kinds))
        self._fill_members(review, selected)
        self._fill_parameters(review, cluster, engine_names)
        self._fill_matrix(review, cluster, engine_names)
        self._fill_review(review)

    def _fill_members(self, review: ClusterReview, selected: str | None) -> None:
        marked = {h.peak for h in review.highlights if h.peak}
        for row, (identity, name) in enumerate(zip(review.ids, review.names, strict=True)):
            number = text_label(str(row + 1), "caption")
            number.setFixedWidth(16)
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
            self.members.addWidget(number, row, 0)
            self.members.addWidget(link, row, 1, QtCore.Qt.AlignmentFlag.AlignLeft)
            if identity in marked:
                glyph = text_label("◇", "caption")
                glyph.setToolTip("; ".join(h.rule for h in review.highlights if h.peak == identity))
                self.members.addWidget(glyph, row, 2)
        self.members.setColumnStretch(1, 1)

    def _fill_parameters(
        self, review: ClusterReview, cluster: FinalClusterOutcome, engine_names: tuple[str, ...]
    ) -> None:
        if cluster.analytical_evaluation is None:
            self.parameters.addWidget(
                text_label(
                    cluster.unusable_reason or "No fitted values recorded", "muted", wrap=True
                ),
                0,
                0,
            )
            self.parameters_note.setText("")
            return
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
        self.parameters.addWidget(text_label("#", "caption"), 0, 0)
        for column, (_axis, kind, title, _decimals) in enumerate(columns, 1):
            heading = text_label(f"{_KINDS.get(kind, kind)} {title}", "caption")
            heading.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            self.parameters.addWidget(heading, 0, column)
        for row, engine in enumerate(engine_names, 1):
            self.parameters.addWidget(text_label(str(row), "caption"), row, 0)
            for column, (axis, kind, _title, decimals) in enumerate(columns, 1):
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
                self.parameters.addWidget(cell, row, column)
        # Columns keep their natural width; the spare room stays to the right.
        self.parameters.setColumnStretch(len(columns) + 1, 1)
        units = sorted({p.unit for p in values.values()})
        self.parameters_note.setText(
            "Recorded values; δ in ppm, LW in Hz"
            + (f" ({', '.join(units)})" if set(units) - {"ppm", "Hz"} else "")
            + ". Hover a value for ± σ; amber marks a recorded bound."
        )

    def _fill_matrix(
        self, review: ClusterReview, cluster: FinalClusterOutcome, engine_names: tuple[str, ...]
    ) -> None:
        if review.single:
            self.matrix_note.setText("Not applicable: a single-peak cluster.")
            return
        matrix = correlation_matrix(cluster)
        if matrix is None:
            self.matrix_note.setText("Unavailable: the amplitude covariance was not recorded.")
            return
        order = [cluster.peak_names.index(name) for name in engine_names]
        matrix = matrix[np.ix_(order, order)]
        size = len(order)
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
        self.matrix_note.setText(
            "Recorded amplitude correlations, one matrix for all planes, conditional on the "
            "fitted lineshape parameters. |ρ| near 1 means the decomposition between those "
            "members is not determined by the data."
        )

    def _fill_review(self, review: ClusterReview) -> None:
        for index, warning in enumerate(review.warnings):
            review_item(
                self.review,
                "Numerical warnings" if index == 0 else "",
                "▲",
                warning.kind,
                warning.detail,
                warn=True,
            )
        if not review.warnings:
            review_item(self.review, "Numerical warnings", "–", "None recorded")
        if math.isfinite(review.chi2):
            samples = review.samples
            review_item(
                self.review,
                "Measures",
                "·",
                f"χ²ᵣ {review.chi2:.3g}",
                f"{samples:,} fitting samples · relative to the noise estimate".replace(",", " "),
            )
            if math.isfinite(review.largest_residual) and samples > 1:
                review_item(
                    self.review,
                    "",
                    "·",
                    f"Largest recorded residual {review.largest_residual:.1f} σ",
                    f"typical maximum for {samples:,} Gaussian samples ≈ "
                    f"{math.sqrt(2 * math.log(samples)):.1f} σ".replace(",", " "),
                )
        names = dict(zip(review.ids, review.names, strict=True))
        for index, highlight in enumerate(review.highlights):
            prefix = f"{names[highlight.peak]}: " if highlight.peak in names else ""
            review_item(
                self.review, "Highlights" if index == 0 else "", "◇", prefix + highlight.rule
            )
        if not review.highlights:
            review_item(self.review, "Highlights", "–", "None at the current thresholds")
