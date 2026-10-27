"""Retained spectrum contours and synchronized read-only inspection tools."""

from typing import TYPE_CHECKING, cast

import numpy as np
from PySide6 import QtCore, QtWidgets

# isort: split
# PySide6 must be imported before PyQtGraph selects a binding.
import pyqtgraph as pg

from peakfit.engine.algorithms.noise import estimate_noise
from peakfit.gui.spectrum import contour_paths, display_position, validate_plane_values

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.engine.domain.peaks import Peak
    from peakfit.engine.domain.spectrum import Spectra, SpectralParameters


class SpectrumViewer(QtWidgets.QWidget):
    """Inspect one Spectra object; visualization never changes scientific data."""

    def __init__(self, spectra: Spectra, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.spectra = spectra
        self.plane = 0
        self.iy, self.ix = (size // 2 for size in spectra.data.shape[1:])
        self.y_axis, self.x_axis = spectra.spectral_params
        self.ppm_x = self.x_axis.pts2ppm(np.arange(self.x_axis.size, dtype=float))
        self.ppm_y = self.y_axis.pts2ppm(np.arange(self.y_axis.size, dtype=float))
        self.noise = estimate_noise(spectra.data[0])
        self.peaks: Sequence[Peak] = ()
        self.contours: list[QtWidgets.QGraphicsPathItem] = []
        self.peak_labels: list[pg.TextItem] = []
        self.labels_visible = False
        self._build_ui()
        self._contour_timer = QtCore.QTimer(self)
        self._contour_timer.setSingleShot(True)
        self._contour_timer.setInterval(30)
        self._contour_timer.timeout.connect(self._update_contours)
        for control in (self.threshold, self.factor, self.levels):
            control.valueChanged.connect(self._schedule_contours)
        self._update_contours()
        self.reset_view()
        self._update_slices()

    def _plot(self, title: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(title=title, background="w")
        for name in ("left", "bottom"):
            plot.getAxis(name).setPen("#52606d")
            plot.getAxis(name).setTextPen("#243746")
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setMenuEnabled(False)
        return plot

    @staticmethod
    def _axis_title(axis: SpectralParameters) -> str:
        return f"{axis.nucleus or axis.label} ({axis.label})"

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        splitter = QtWidgets.QSplitter()
        layout.addWidget(splitter, 1)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.spectrum_plot = self._plot("Spectrum")
        self.spectrum_plot.setLabel("bottom", self._axis_title(self.x_axis), units="ppm")
        self.spectrum_plot.setLabel("left", self._axis_title(self.y_axis), units="ppm")
        view = self.spectrum_plot.getViewBox()
        view.invertX(True)
        view.invertY(True)
        view.setDefaultPadding(0.02)
        self.spectrum_plot.setToolTip(
            "Drag to pan; scroll to zoom. Click to position the crosshair; "
            "drag either crosshair line."
        )
        left_layout.addWidget(self.spectrum_plot, 1)
        self.cross_x = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("#a65e00"))
        self.cross_y = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen("#a65e00"))
        for line in (self.cross_x, self.cross_y):
            self.spectrum_plot.getPlotItem().addItem(line, ignoreBounds=True)
            line.sigPositionChanged.connect(self._crosshair_dragged)
        cast("pg.GraphicsScene", self.spectrum_plot.scene()).sigMouseClicked.connect(self._clicked)
        self.peak_scatter = pg.ScatterPlotItem(
            symbol="+",
            size=10,
            pen=pg.mkPen("#202020", width=1.5),
            hoverable=True,
            tip=lambda x, y, data: f"{data}\n{x:.4f}, {y:.4f} ppm",
        )
        self.spectrum_plot.addItem(self.peak_scatter)
        self._build_plane_controls(left_layout)
        self._build_contour_controls(left_layout)
        splitter.addWidget(left)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.trace_x = self._plot(self._axis_title(self.x_axis))
        self.trace_y = self._plot(self._axis_title(self.y_axis))
        self.profile = self._plot("Raw intensity across planes")
        for plot, axis in ((self.trace_x, self.x_axis), (self.trace_y, self.y_axis)):
            plot.setLabel("bottom", self._axis_title(axis), units="ppm")
            plot.setLabel("left", "Intensity")
            plot.getViewBox().invertX(True)
            plot.addLine(y=0, pen=pg.mkPen("#999999", style=QtCore.Qt.PenStyle.DotLine))
        self.trace_x.setXLink(self.spectrum_plot)
        self.curve_x = self.trace_x.plot(pen=pg.mkPen("#1265a8", width=1.5))
        self.curve_y = self.trace_y.plot(pen=pg.mkPen("#1265a8", width=1.5))
        self.profile.setLabel("bottom", "Plane value")
        self.profile.setLabel("left", "Raw intensity")
        # Points do not imply interpolation across unordered/repeated experimental values.
        self.curve_profile = self.profile.plot(
            pen=None, symbol="o", symbolSize=5, symbolBrush="#a65e00", symbolPen=None
        )
        self.profile_marker = pg.ScatterPlotItem(size=11, brush="#a65e00", pen="#202020")
        self.profile.addItem(self.profile_marker)
        for plot in (self.trace_x, self.trace_y, self.profile):
            right_layout.addWidget(plot, 1)
        splitter.addWidget(right)
        splitter.setSizes([850, 450])
        self.position_label = QtWidgets.QLabel()
        layout.addWidget(self.position_label)

    def _build_plane_controls(self, layout: QtWidgets.QVBoxLayout) -> None:
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Plane"))
        self.plane_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.plane_spin = QtWidgets.QSpinBox()
        n_planes = self.spectra.data.shape[0]
        self.plane_slider.setRange(1, n_planes)
        self.plane_spin.setRange(1, n_planes)
        for control in (self.plane_slider, self.plane_spin):
            control.setValue(1)
            control.setEnabled(n_planes > 1)
            control.valueChanged.connect(self.set_plane)
        row.addWidget(self.plane_slider, 1)
        row.addWidget(self.plane_spin)
        self.plane_label = QtWidgets.QLabel()
        row.addWidget(self.plane_label)
        layout.addLayout(row)

    def _build_contour_controls(self, layout: QtWidgets.QVBoxLayout) -> None:
        row = QtWidgets.QHBoxLayout()
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setRange(0.1, 1000)
        self.threshold.setValue(5)
        self.threshold.setSingleStep(0.5)
        self.threshold.setSuffix(" × noise")
        self.factor = QtWidgets.QDoubleSpinBox()
        self.factor.setRange(1.05, 3)
        self.factor.setValue(1.3)
        self.factor.setSingleStep(0.05)
        self.levels = QtWidgets.QSpinBox()
        self.levels.setRange(1, 30)
        self.levels.setValue(10)
        for title, control in (
            ("Threshold", self.threshold),
            ("Spacing", self.factor),
            ("Levels", self.levels),
        ):
            row.addWidget(QtWidgets.QLabel(title))
            control.setKeyboardTracking(False)
            row.addWidget(control)
        reset = QtWidgets.QPushButton("Reset view")
        reset.clicked.connect(self.reset_view)
        row.addWidget(reset)
        layout.addLayout(row)

    def reset_view(self) -> None:
        """Show the entire sampled spectrum without recalculating contours."""
        self.spectrum_plot.getViewBox().setRange(
            xRange=(float(self.ppm_x[-1]), float(self.ppm_x[0])),
            yRange=(float(self.ppm_y[-1]), float(self.ppm_y[0])),
        )
        self.trace_y.getViewBox().setXRange(
            float(self.ppm_y[-1]), float(self.ppm_y[0]), padding=0.02
        )

    def set_plane(self, number: int) -> None:
        """Select a one-based plane while retaining the current viewing range."""
        plane = int(np.clip(number - 1, 0, self.spectra.data.shape[0] - 1))
        if plane == self.plane:
            return
        self.plane = plane
        for control in (self.plane_slider, self.plane_spin):
            with QtCore.QSignalBlocker(control):
                control.setValue(plane + 1)
        self._schedule_contours()
        self._update_slices()

    def _schedule_contours(self) -> None:
        # Throttle rather than debounce: scrubbing still displays intermediate planes.
        if not self._contour_timer.isActive():
            self._contour_timer.start()

    def _update_contours(self) -> None:
        n_levels = self.levels.value()
        while len(self.contours) < 2 * n_levels:
            item = QtWidgets.QGraphicsPathItem()
            item.setZValue(-10)
            self.spectrum_plot.addItem(item)
            self.contours.append(item)
        base = max(self.noise * self.threshold.value(), np.finfo(float).tiny)
        positive = base * self.factor.value() ** np.arange(n_levels)
        levels = np.concatenate((positive, -positive))
        paths = contour_paths(self.spectra.data[self.plane], self.spectra.spectral_params, levels)
        for index, item in enumerate(self.contours):
            item.setVisible(index < len(paths))
            if index < len(paths):
                negative = index >= n_levels
                item.setPen(
                    pg.mkPen(
                        "#b34244" if negative else "#1265a8",
                        width=1,
                        style=QtCore.Qt.PenStyle.DashLine
                        if negative
                        else QtCore.Qt.PenStyle.SolidLine,
                    )
                )
                item.setPath(paths[index])

    def _clicked(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        view = self.spectrum_plot.getViewBox()
        if view.sceneBoundingRect().contains(event.scenePos()):
            point = view.mapSceneToView(event.scenePos())
            self._select_position(point.x(), point.y())

    def _crosshair_dragged(self) -> None:
        self._select_position(self.cross_x.value(), self.cross_y.value())

    def _select_position(self, x: float, y: float) -> None:
        self.ix = int(np.clip(round(self.x_axis.ppm2pts(x)), 0, self.x_axis.size - 1))
        self.iy = int(np.clip(round(self.y_axis.ppm2pts(y)), 0, self.y_axis.size - 1))
        self._update_slices()

    def _update_slices(self) -> None:
        plane = self.spectra.data[self.plane]
        for line, value in (
            (self.cross_x, self.ppm_x[self.ix]),
            (self.cross_y, self.ppm_y[self.iy]),
        ):
            with QtCore.QSignalBlocker(line):
                line.setValue(float(value))
        self.curve_x.setData(self.ppm_x, plane[self.iy, :])
        self.curve_y.setData(self.ppm_y, plane[:, self.ix])
        profile = self.spectra.data[:, self.iy, self.ix]
        values = self.spectra.z_values
        self.curve_profile.setData(values, profile)
        self.profile_marker.setData([values[self.plane]], [profile[self.plane]])
        self.plane_label.setText(f"/ {len(values)}   value: {values[self.plane]:g}")
        self.position_label.setText(
            f"{self._axis_title(self.x_axis)}: {self.ppm_x[self.ix]:.4f} ppm    "
            f"{self._axis_title(self.y_axis)}: {self.ppm_y[self.iy]:.4f} ppm    "
            f"Raw intensity: {plane[self.iy, self.ix]:.5g}"
        )

    def set_plane_values(self, values: np.ndarray) -> None:
        """Replace experimental plane coordinates without changing spectral samples."""
        validate_plane_values(values, self.spectra.data.shape[0])
        self.spectra.z_values = np.array(values, dtype=float, copy=True)
        self.profile.enableAutoRange()
        self._update_slices()

    def set_peaks(self, peaks: Sequence[Peak]) -> None:
        """Display imported peaks, folding only indirect coordinates."""
        self.peaks = peaks
        for label in self.peak_labels:
            self.spectrum_plot.removeItem(label)
        self.peak_labels.clear()
        positions = [
            (
                display_position(self.x_axis, peak.positions[1]),
                display_position(self.y_axis, peak.positions[0]),
            )
            for peak in peaks
        ]
        self.peak_scatter.setData(
            x=[position[0] for position in positions],
            y=[position[1] for position in positions],
            data=[peak.name for peak in peaks],
        )
        for peak, (x, y) in zip(peaks, positions, strict=True):
            label = pg.TextItem(peak.name, color="#303030", anchor=(0, 1))
            label.setPos(x, y)
            label.setVisible(self.labels_visible)
            self.spectrum_plot.getPlotItem().addItem(label, ignoreBounds=True)
            self.peak_labels.append(label)

    def show_peak_labels(self, visible: bool) -> None:
        """Keep crowded survey views readable while allowing explicit assignments."""
        self.labels_visible = visible
        for label in self.peak_labels:
            label.setVisible(visible)
