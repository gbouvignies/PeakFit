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
        self._slices_width = 260
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.commands = QtWidgets.QToolBar()
        self.commands.setMovable(False)
        self.commands.setFloatable(False)
        self.commands.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.dataset_fit = QtWidgets.QPushButton("Fit dataset…")
        self.dataset_fit.setDefault(True)
        self.dataset_fit.setAutoDefault(False)
        primary_font = self.dataset_fit.font()
        primary_font.setBold(True)
        self.dataset_fit.setFont(primary_font)
        self.dataset_fit.setToolTip(
            "Review settings and fit every cluster across all experimental planes"
        )
        self.commands.addWidget(self.dataset_fit)
        self.commands.addSeparator()
        self.edit = QtWidgets.QToolButton()
        self.edit.setText("Edit peak")
        self.fit = QtWidgets.QToolButton()
        self.fit.setText("Local fit")
        self.profile = QtWidgets.QToolButton()
        self.profile.setText("Across planes")
        self.edit.setToolTip("Edit the selected peak in the lower analysis panel")
        self.fit.setToolTip("Inspect or prepare a local fit; Run local fit starts optimization")
        self.profile.setToolTip(
            "Open crosshair intensity and available fitted amplitudes across planes"
        )
        for button in (self.edit, self.fit, self.profile):
            button.setAutoRaise(True)
            self.commands.addWidget(button)
        space = QtWidgets.QWidget()
        space.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred
        )
        self.commands.addWidget(space)
        comparison = QtWidgets.QWidget()
        self.comparison_controls = QtWidgets.QHBoxLayout()
        self.comparison_controls.setContentsMargins(0, 0, 8, 0)
        self.comparison_controls.setSpacing(8)
        comparison.setLayout(self.comparison_controls)
        self.commands.addWidget(comparison)
        layout.addWidget(self.commands)
        summary_row = QtWidgets.QHBoxLayout()
        self.summary = QtWidgets.QLabel()
        self.summary.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.summary.setMinimumHeight(self.summary.fontMetrics().lineSpacing() + 4)
        self.summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
        )
        summary_row.addWidget(self.summary, 1)
        self.result_details = QtWidgets.QToolButton()
        self.result_details.setText("Result details")
        self.result_details.setAutoRaise(True)
        policy = self.result_details.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.result_details.setSizePolicy(policy)
        self.result_details.hide()
        summary_row.addWidget(self.result_details)
        layout.addLayout(summary_row)

        self.vertical = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.vertical.setChildrenCollapsible(False)
        self.horizontal = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.horizontal.setChildrenCollapsible(False)
        spectrum.setMinimumSize(560, 360)
        slices.setMinimumSize(220, 360)
        self.horizontal.setHandleWidth(7)
        self.horizontal.setToolTip("Drag the divider to resize the spectrum and slices")
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
        self.slices_button = QtWidgets.QToolButton()
        self.slices_button.setAutoRaise(True)
        self.slices_button.setDefaultAction(self.slices_action)
        self.slices_widget_action = self.commands.addWidget(self.slices_button)

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
