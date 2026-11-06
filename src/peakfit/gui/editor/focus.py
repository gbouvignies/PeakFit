"""The bar above the spectral view while a cluster is in focus.

Cluster focus reframes the one spectral view in place; it has no plots of its own.
The bar names the cluster and the run it comes from, offers the way back and the
fitting-region toggle, and, after a refit that regrouped the cluster, says what
changed with links to wherever previous members are now fitted.
"""

import html

from PySide6 import QtCore, QtWidgets

from peakfit.gui import theme as themes
from peakfit.gui.editor.labels import ElidedLabel


class FocusBar(QtWidgets.QWidget):
    """Title, provenance, return and fitting-region toggle of a focused cluster."""

    close_requested = QtCore.Signal()
    selected = QtCore.Signal(str)  # a membership link: a peak to follow

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("focusBar")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(themes.SPACE_M, themes.SPACE_XS, themes.SPACE_M, 0)
        layout.setSpacing(themes.SPACE_XS)
        header = QtWidgets.QHBoxLayout()
        self.back = QtWidgets.QPushButton("‹ Spectrum")
        self.back.setObjectName("chip")
        self.back.setToolTip("Return to the spectrum as it was (Escape)")
        self.back.clicked.connect(self.close_requested)
        self.title = QtWidgets.QLabel()
        self.title.setObjectName("value")
        self.note = ElidedLabel()
        self.note.setObjectName("muted")
        self.regions = QtWidgets.QPushButton("Fitting region")
        self.regions.setObjectName("chip")
        self.regions.setCheckable(True)
        self.regions.setToolTip("Tint the samples each cluster was fitted on")
        header.addWidget(self.back)
        header.addSpacing(themes.SPACE_S)
        header.addWidget(self.title)
        header.addSpacing(themes.SPACE_S)
        header.addWidget(self.note, 1)
        header.addWidget(self.regions)
        layout.addLayout(header)
        self.membership = QtWidgets.QLabel()
        self.membership.setObjectName("caption")
        self.membership.setWordWrap(True)
        self.membership.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.membership.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse
            | QtCore.Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        )
        self.membership.setToolTip("Links show the cluster each peak is now fitted in")
        self.membership.linkActivated.connect(self.selected)
        self.membership_plain = ""
        self.membership.hide()
        layout.addWidget(self.membership)

    def show_cluster(
        self,
        names: tuple[str, ...],
        run_id: str,
        *,
        current: bool,
        historical: bool,
        outside: str | None,
    ) -> None:
        """Name the cluster, its run and whether the selection is outside it."""
        cluster = names[0] + (f" +{len(names) - 1}" if len(names) > 1 else "") if names else ""
        self.title.setText(f"Focus · {cluster}")
        # The run's identity is in the toolbar; the bar adds only what differs.
        parts = (
            "" if current else "Outdated",
            "model and residual are historical" if historical and not current else "",
            f"{outside} is not in this cluster of the run" if outside else "",
        )
        self.note.setText(" · ".join(part for part in parts if part))
        self.note.setToolTip(f"Run {run_id[:8]}")

    def set_membership(self, text: str, links: list[tuple[str, str]]) -> None:
        """Say how a refit regrouped the cluster; links open where members went."""
        self.membership_plain = " ".join([text, *(f"{name} ›" for name, _i in links)])
        anchors = [
            f'<a href="{html.escape(identity)}">{html.escape(name)} ›</a>'
            for name, identity in links
        ]
        self.membership.setText(" ".join([html.escape(text), *anchors]))
        self.membership.setVisible(bool(text))
