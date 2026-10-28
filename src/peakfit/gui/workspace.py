"""Resizable presentation of existing scientific plots and contextual details."""

from PySide6 import QtCore, QtGui, QtWidgets


class SpectrumWorkspace(QtWidgets.QWidget):
    """Keep viewing widgets alive while details and slices are hidden or resized.

    This module owns geometry only. Callers decide whether scientific state permits
    changing details; background deliveries must never call ``show_details``.
    """

    close_requested = QtCore.Signal()

    def __init__(  # noqa: PLR0915
        self,
        spectrum: QtWidgets.QWidget,
        slices: QtWidgets.QWidget,
        editor: QtWidgets.QWidget,
        fit: QtWidgets.QWidget,
        profile: QtWidgets.QWidget,
    ) -> None:
        super().__init__()
        self._details_height = 240
        self._slices_width = 340
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.commands = QtWidgets.QHBoxLayout()
        self.commands.setSpacing(8)
        self.edit = QtWidgets.QPushButton("Edit peak")
        self.fit = QtWidgets.QPushButton("Local fit")
        self.profile = QtWidgets.QPushButton("Across planes")
        self.edit.setToolTip("Edit the selected peak in the lower analysis panel")
        self.fit.setToolTip("Inspect or prepare a local fit; Run local fit starts optimization")
        self.profile.setToolTip(
            "Open crosshair intensity and available fitted amplitudes across planes"
        )
        for button in (self.edit, self.fit, self.profile):
            self.commands.addWidget(button)
        self.commands.addStretch()
        layout.addLayout(self.commands)

        self.vertical = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.vertical.setChildrenCollapsible(False)
        self.horizontal = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.horizontal.setChildrenCollapsible(False)
        spectrum.setMinimumSize(560, 360)
        slices.setMinimumSize(280, 360)
        self.horizontal.addWidget(spectrum)
        self.horizontal.addWidget(slices)
        self.horizontal.setStretchFactor(0, 1)
        self.horizontal.setStretchFactor(1, 0)
        self.horizontal.setSizes([900, self._slices_width])
        self.slices = slices
        self.vertical.addWidget(self.horizontal)

        self.details = QtWidgets.QWidget()
        details_layout = QtWidgets.QVBoxLayout(self.details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(8)
        header = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel()
        title_font = self.title.font()
        title_font.setBold(True)
        self.title.setFont(title_font)
        header.addWidget(self.title)
        header.addStretch()
        self.close_button = QtWidgets.QPushButton("Close details")
        self.close_button.clicked.connect(self.close_requested)
        header.addWidget(self.close_button)
        details_layout.addLayout(header)
        self.pages = QtWidgets.QStackedWidget()
        for widget in (editor, fit):
            scroll = QtWidgets.QScrollArea()
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            self.pages.addWidget(scroll)
        self.pages.addWidget(profile)
        details_layout.addWidget(self.pages, 1)
        self.vertical.addWidget(self.details)
        self.vertical.setStretchFactor(0, 1)
        self.vertical.setStretchFactor(1, 0)
        self.details.hide()
        self.vertical.splitterMoved.connect(self._remember_height)
        layout.addWidget(self.vertical, 1)

        self.slices_action = QtGui.QAction("1D slices", self)
        self.slices_action.setCheckable(True)
        self.slices_action.setChecked(True)
        self.slices_action.toggled.connect(self._set_slices_visible)

    def show_details(self, index: int, title: str) -> None:
        """Open one existing section, restoring the user's expanded height."""
        self.pages.setCurrentIndex(index)
        self.title.setText(title)
        if self.details.isHidden():
            height = min(self._details_height, max(180, self.vertical.height() - 400))
            self.details.show()
            self.vertical.setSizes([self.vertical.height() - height, height])

    def hide_details(self) -> None:
        """Collapse without destroying fields, plots, selection or computations."""
        self._remember_height()
        self.details.hide()
        self.close_button.clearFocus()

    def _remember_height(self, *_args) -> None:
        if not self.details.isHidden():
            self._details_height = self.vertical.sizes()[1]

    def _set_slices_visible(self, visible: bool) -> None:
        if visible:
            self.slices.show()
            self.horizontal.setSizes(
                [self.horizontal.width() - self._slices_width, self._slices_width]
            )
        else:
            self._slices_width = self.horizontal.sizes()[1]
            self.slices.hide()
