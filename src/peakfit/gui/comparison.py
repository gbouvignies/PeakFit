"""Completed local-fit comparison in the viewer's existing plots and title bands."""

from collections import OrderedDict
from threading import Event
from typing import TYPE_CHECKING, cast

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# isort: split
import pyqtgraph as pg

from peakfit.gui.fit_display import (
    ModelContours,
    ModelSlices,
    local_model_plane,
    model_contours,
    model_slices,
)
from peakfit.gui.spectrum import paths_from_geometry

if TYPE_CHECKING:
    from peakfit.gui.fit_process import FitProcessResult
    from peakfit.gui.viewer import SpectrumViewer


_EXPERIMENT, _OVERLAY, _RESIDUAL = range(3)
_CACHE_ENTRIES = 8
_CACHE_BYTES = 16 * 1024**2
_DARK_LIGHTNESS = 128


class FitComparison(QtCore.QObject):
    """Own graphical comparison and bounded caches, never editable fitting state."""

    def __init__(self, viewer: SpectrumViewer) -> None:
        super().__init__(viewer)
        self.viewer = viewer
        self.result: FitProcessResult | None = None
        self.active = False
        self._revision = 0
        self._slice_revision = 0
        self._cancel = Event()
        self._slice_cancel = Event()
        self._planes: OrderedDict[tuple, ModelContours] = OrderedDict()
        self._slices: OrderedDict[tuple, ModelSlices] = OrderedDict()
        self._current: ModelContours | None = None
        self._slice: ModelSlices | None = None
        self.paths: list[QtWidgets.QGraphicsPathItem] = []
        self.residual = pg.ImageItem(axisOrder="row-major")
        self.residual.setZValue(-5)
        self.residual.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        viewer.spectrum_plot.addItem(self.residual)
        self.residual.hide()
        self._build_controls(viewer)

    def _build_controls(self, viewer: SpectrumViewer) -> None:
        self.mode = QtWidgets.QComboBox(viewer.spectrum_plot.viewport())
        self.mode.addItems(["Experiment", "Overlay", "Residual"])
        self.mode.setFixedWidth(self.mode.fontMetrics().horizontalAdvance("Experiment") + 30)
        self.mode.setAccessibleName("Spectrum comparison")
        self.mode.setToolTip(
            "Local group only: other signals are not subtracted. Model contours/lines are "
            "solid in sampled cells, dotted outside them. Residuals use exact fitted samples "
            "only, scaled by retained statistical noise. Unmodeled neighboring signals may remain."
        )
        self.mode.hide()
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.samples = QtWidgets.QCheckBox("Show exact fitting samples")
        self.samples.setToolTip("Amber cells are the samples that constrained this completed fit.")
        self.samples.hide()
        popup_layout = viewer.contour_popup.layout()
        assert isinstance(popup_layout, QtWidgets.QGridLayout)
        popup_layout.addWidget(self.samples, 3, 0, 1, 3)
        self.samples.toggled.connect(self._support_visibility)
        self.profile_mode = QtWidgets.QComboBox(viewer.profile.viewport())
        self.profile_mode.addItem("Crosshair intensity · ● exp / ▼ group")
        self.profile_mode.setAccessibleName("Plane profile quantity")
        self.profile_mode.setToolTip(
            "Intensity compares experiment and summed local model at the exact crosshair. "
            "Peak amplitudes are separate recovered quantities, never raw-coordinate intensities."
        )
        self.profile_mode.currentIndexChanged.connect(self._profile_changed)
        viewer.profile.setTitle("Across planes")
        self.profile_mode.hide()
        self.model_x = viewer.trace_x.plot()
        self.predicted_x = viewer.trace_x.plot()
        self.model_y = viewer.trace_y.plot()
        self.predicted_y = viewer.trace_y.plot()
        self.profile_model = viewer.profile.plot(pen=None, symbol="t", symbolSize=6)
        self.profile_model.sigPointsClicked.connect(self._profile_clicked)
        viewer.curve_profile.sigPointsClicked.connect(self._profile_clicked)
        viewer.spectrum_plot.getViewBox().sigResized.connect(self.position_controls)
        viewer.profile.getViewBox().sigResized.connect(self.position_controls)
        viewer._computation.completed.connect(self._completed)
        self.position_controls()
        self.palette_changed()

    def position_controls(self) -> None:
        """Keep controls in existing title cells, without extra rows or data occlusion."""
        for plot, control in (
            (self.viewer.spectrum_plot, self.mode),
            (self.viewer.profile, self.profile_mode),
        ):
            title = plot.getPlotItem().titleLabel
            rect = plot.mapFromScene(title.sceneBoundingRect()).boundingRect()
            control.setFixedHeight(min(26, max(20, rect.height() - 2)))
            if control is self.profile_mode:
                control.setFixedWidth(max(100, rect.width() - 8))
            control.move(rect.left() + 4, rect.top() + (rect.height() - control.height()) // 2)

    def palette_changed(self) -> None:
        """Restyle retained graphics without scientific evaluation or contour extraction."""
        background = self.viewer._colors["background"]
        dark = QtGui.QColor(background).lightness() < _DARK_LIGHTNESS
        self.color = "#d8a6ff" if dark else "#7435a0"
        for curve in (self.model_x, self.model_y):
            curve.setPen(pg.mkPen(self.color, width=2))
            curve.setZValue(2)
        for curve in (self.predicted_x, self.predicted_y):
            curve.setPen(pg.mkPen(self.color, width=2, style=QtCore.Qt.PenStyle.DotLine))
            curve.setZValue(1)
        self.profile_model.setSymbolBrush(self.color)
        self.profile_model.setSymbolPen(self.color)
        if self.active:
            self._render_plane()
            self._render_slices()
        self.position_controls()

    def show_result(self, result: FitProcessResult, *, outdated: bool) -> None:
        """Only current usable outcomes may be compared to the working spectrum."""
        terminal = result.scientific.outcome.cluster(result.scientific.inputs.scope.cluster_id)
        if outdated or not terminal.usable:
            self.clear()
            return
        changed = self.result is not result
        if changed:
            self.clear()
            self.result = result
            self._planes.clear()
            self._slices.clear()
            with QtCore.QSignalBlocker(self.profile_mode):
                self.profile_mode.clear()
                self.profile_mode.addItem("Crosshair intensity · ● exp / ▼ group")
                labels = {i.engine_name: i.assignment for i in result.scientific.inputs.identities}
                for number, name in enumerate(terminal.peak_names, 1):
                    self.profile_mode.addItem(f"Amplitude · {number}. {labels[name]}")
            with QtCore.QSignalBlocker(self.mode):
                self.mode.setCurrentIndex(1)
        self.active = True
        self.viewer.profile.setTitle(" ")
        self.profile_mode.show()
        self.mode.show()
        self.samples.show()
        self._support_visibility()
        self._mode_changed()

    def clear(self) -> None:
        """Invalidate pending graphics immediately on preparation or scientific edits."""
        self.active = False
        self.result = None
        self._current = None
        self._slice = None
        self._planes.clear()
        self._slices.clear()
        with QtCore.QSignalBlocker(self.profile_mode):
            self.profile_mode.clear()
            self.profile_mode.addItem("Crosshair intensity · ● exp / ▼ group")
        self._revision += 1
        self._slice_revision += 1
        self._cancel.set()
        self._slice_cancel.set()
        self.viewer._computation.cancel("model")
        self.viewer._computation.cancel("slices")
        self._hide_paths()
        self.residual.hide()
        self.mode.hide()
        self.samples.hide()
        self.samples.setChecked(False)
        self.profile_mode.hide()
        self.viewer.profile.setTitle("Across planes")
        self.viewer.fit_mask.hide()
        for curve in (
            self.model_x,
            self.model_y,
            self.predicted_x,
            self.predicted_y,
            self.profile_model,
        ):
            curve.clear()
        self.viewer.spectrum_plot.setTitle("Spectrum")
        self.viewer.trace_x.setLabel("left", "Intensity")
        self.viewer.trace_y.setLabel("left", "Intensity")
        self.viewer.profile.setLabel("left", "Raw intensity")
        self.viewer.curve_x.show()
        self.viewer.curve_y.show()
        self.viewer.curve_profile.show()
        self.viewer._update_slices()
        for item in self.viewer.contours:
            item.setOpacity(1)

    def _support_visibility(self) -> None:
        if self.active:
            self.viewer.fit_mask.setVisible(
                self.samples.isChecked() and self.mode.currentIndex() != _RESIDUAL
            )

    def _mode_changed(self) -> None:
        if not self.active:
            return
        self._support_visibility()
        self.contours_changed()
        self.viewer.trace_x.enableAutoRange(axis="y")
        self.viewer.trace_y.enableAutoRange(axis="y")
        self.viewer.profile.enableAutoRange()
        self.viewer._update_slices()

    def contours_changed(self) -> None:
        """Drop mismatched graphics before any plane/level change, including cached delivery."""
        if not self.active or self.result is None:
            return
        v = self.viewer
        v._computation.cancel("model")
        self._revision += 1
        self._cancel.set()
        self._cancel = Event()
        self._hide_paths()
        self.residual.hide()
        if getattr(v, "_contour_plane", -1) != v.plane:
            for item in v.contours:
                item.hide()
        if getattr(v, "_contour_applied_revision", -1) != v._contour_revision:
            # The experimental debounce must enqueue first. Starting an expensive
            # current-plane projection here would bypass worker queue priority.
            self._slice_cancel.set()
            v._computation.cancel("slices")
            self._current = None
            self._render_plane()
            return
        if self.mode.currentIndex() != _OVERLAY:
            self._current = None
            self._render_plane()
            return
        key = (v.plane, v.lowest_contour_level, v.factor.value(), v.levels.value())
        if key in self._planes:
            self._planes.move_to_end(key)
            self._current = self._planes[key]
            self._render_plane()
            return
        self._current = None
        axes = [axis.model_copy(deep=True) for axis in v.spectra.spectral_params]
        result = self.result.scientific
        plane, level, factor, count = key
        cancel = self._cancel
        reusable = next(
            (
                cached.grid
                for cached_key, cached in reversed(self._planes.items())
                if cached_key[0] == plane and (cached_key[1] <= level or cached.grid.edge_visible)
            ),
            None,
        )

        def calculate():
            grid = reusable or local_model_plane(result, axes, plane, level, cancel=cancel)
            levels = level * factor ** np.arange(count)
            return key, model_contours(
                grid, axes, result.inputs.scope.grid_indices, np.r_[levels, -levels], cancel=cancel
            )

        v._computation.submit("model", self._revision, calculate)
        self._title("Exp ± · group prediction…")

    def update_slices(self) -> None:
        """Raw crosshairs update synchronously; coalesce only the calculated additions."""
        if not self.active or self.result is None:
            return
        self._slice_revision += 1
        self._slice_cancel.set()
        self._slice_cancel = Event()
        cancel = self._slice_cancel
        v = self.viewer
        v._computation.cancel("slices")
        for curve in (
            self.model_x,
            self.model_y,
            self.predicted_x,
            self.predicted_y,
            self.profile_model,
        ):
            curve.clear()
        self._slice = None
        self._render_slices()
        if self.mode.currentIndex() == _EXPERIMENT:
            return
        key = (v.iy, v.ix)
        if key in self._slices:
            self._slices.move_to_end(key)
            self._slice = self._slices[key]
            self._render_slices()
            return
        if getattr(v, "_contour_applied_revision", -1) != v._contour_revision:
            return  # Experimental samples update immediately; calculations follow.
        result = self.result.scientific
        axes = [axis.model_copy(deep=True) for axis in v.spectra.spectral_params]
        v._computation.submit(
            "slices",
            self._slice_revision,
            lambda: (key, model_slices(result, axes, *key, cancel=cancel)),
        )

    def _completed(self, kind: str, revision: int, value: object) -> None:
        if not self.active or self.viewer._closing:
            return
        if kind == "contours":
            if revision == self.viewer._contour_revision:
                self.update_slices()
                self.contours_changed()
        elif kind in ("model", "slices") and revision == (
            self._revision if kind == "model" else self._slice_revision
        ):
            if isinstance(value, Exception):
                from concurrent.futures import CancelledError  # noqa: PLC0415

                if not isinstance(value, CancelledError):
                    self._title("Comparison unavailable · see tooltip")
                    self.mode.setToolTip(str(value))
                return
            assert isinstance(value, tuple)
            key, projected = value
            assert isinstance(key, tuple)
            if kind == "model":
                projected = cast("ModelContours", projected)
                self._planes[key] = projected
                self._current = projected
                self._trim(self._planes)
                self._render_plane()
            else:
                projected = cast("ModelSlices", projected)
                self._slices[key] = projected
                self._slice = projected
                self._trim(self._slices)
                self._render_slices()

    @staticmethod
    def _trim[T: ModelContours | ModelSlices](cache: OrderedDict[tuple, T]) -> None:
        while cache and (
            len(cache) > _CACHE_ENTRIES or sum(v.nbytes for v in cache.values()) > _CACHE_BYTES
        ):
            cache.popitem(last=False)

    def _hide_paths(self) -> None:
        for item in self.paths:
            item.hide()

    def _title(self, text: str) -> None:
        self.viewer.spectrum_plot.setTitle(text)
        self.position_controls()
        self.viewer._position_contour_control()

    def _render_plane(self) -> None:
        if not self.active or self.result is None:
            return
        v = self.viewer
        mode = self.mode.currentIndex()
        self._hide_paths()
        self.residual.hide()
        for item in v.contours:
            item.setOpacity(0.2 if mode == _RESIDUAL else 1)
        if mode == _EXPERIMENT:
            self._title("Spectrum")
            return
        if mode == _RESIDUAL:
            self._render_residual()
            return
        if (
            self._current is None
            or getattr(v, "_contour_applied_revision", -1) != v._contour_revision
        ):
            self._title("Exp ± · group prediction…")
            return
        geometry = self._current.sampled + self._current.predicted
        paths = paths_from_geometry(geometry)
        while len(self.paths) < len(paths):
            item = QtWidgets.QGraphicsPathItem()
            item.setZValue(-6)
            v.spectrum_plot.addItem(item)
            self.paths.append(item)
        count = len(self._current.sampled) // 2
        for index, path in enumerate(paths):
            predicted = index >= 2 * count
            negative = (index % (2 * count)) >= count
            style = (
                (QtCore.Qt.PenStyle.DashDotLine if negative else QtCore.Qt.PenStyle.DotLine)
                if predicted
                else (QtCore.Qt.PenStyle.DashLine if negative else QtCore.Qt.PenStyle.SolidLine)
            )
            self.paths[index].setPen(pg.mkPen(self.color, width=1.5, style=style))
            self.paths[index].setPath(path)
            self.paths[index].show()
        suffix = " · edge clipped" if self._current.grid.edge_visible else ""
        self._title(
            f'<span style="color:{self.color}">Group: sampled ━ · predicted ··</span>{suffix}'
        )

    def _render_residual(self) -> None:
        v = self.viewer
        assert self.result is not None
        scientific = self.result.scientific
        evaluation = scientific.outcome.cluster(
            scientific.inputs.scope.cluster_id
        ).analytical_evaluation
        assert evaluation is not None
        sy, sx = scientific.inputs.scope.grid_indices
        y0, x0 = int(sy.min()), int(sx.min())
        rgba = np.zeros((int(sy.max()) - y0 + 1, int(sx.max()) - x0 + 1, 4), dtype=np.uint8)
        neutral = (
            "#30343a"
            if QtGui.QColor(v._colors["background"]).lightness() < _DARK_LIGHTNESS
            else "#eeeeee"
        )
        colors = pg.ColorMap([0, 0.5, 1], ["#2474c6", neutral, "#d74b24"])
        rgba[sy - y0, sx - x0] = colors.map(
            np.clip(
                evaluation.raw_residuals[:, v.plane] / scientific.outcome.noise / 6 + 0.5, 0, 1
            ),
            mode="byte",
        )
        self.residual.setImage(rgba, autoLevels=False, autoDownsample=False)
        self.residual.setTransform(
            QtGui.QTransform(
                v.x_axis.delta,
                0,
                0,
                v.y_axis.delta,
                float(v.x_axis.pts2ppm(x0)) - v.x_axis.delta / 2,
                float(v.y_axis.pts2ppm(y0)) - v.y_axis.delta / 2,
            )
        )
        self.residual.show()
        self._title(
            'Local residual /σ: <span style="color:#2474c6">−3</span> · 0 · '
            '<span style="color:#d74b24">+3</span>'
        )

    def _render_slices(self) -> None:
        if not self.active or self.result is None:
            return
        v = self.viewer
        residual = self.mode.currentIndex() == _RESIDUAL
        for raw in (v.curve_x, v.curve_y):
            raw.setVisible(not residual)
        for plot in (v.trace_x, v.trace_y):
            plot.setLabel("left", "Residual /σ" if residual else "Intensity")
        if self._slice is not None and self.mode.currentIndex() != _EXPERIMENT:
            for points, values, support, error, curve, prediction in (
                (
                    v.ppm_x,
                    self._slice.x,
                    self._slice.support_x,
                    self._slice.residual_x,
                    self.model_x,
                    self.predicted_x,
                ),
                (
                    v.ppm_y,
                    self._slice.y,
                    self._slice.support_y,
                    self._slice.residual_y,
                    self.model_y,
                    self.predicted_y,
                ),
            ):
                if residual:
                    curve.setData(
                        points,
                        error[:, v.plane] / self.result.scientific.outcome.noise,
                        pen=None,
                        symbol="o",
                        symbolSize=4,
                        symbolBrush=self.color,
                    )
                else:
                    curve.setData(
                        points,
                        np.where(support, values[:, v.plane], np.nan),
                        pen=pg.mkPen(self.color, width=2),
                        symbol="o",
                        symbolSize=3,
                        symbolBrush=self.color,
                        symbolPen=self.color,
                        connect="finite",
                    )
                    # The model is evaluated at every native coordinate. Draw it
                    # continuously underneath the sampled portion so adjacent
                    # styles share their endpoints. Actual NaNs still break paths.
                    prediction.setData(points, values[:, v.plane], connect="finite")
        self._render_profile()

    def _render_profile(self) -> None:
        v = self.viewer
        amplitude = self.profile_mode.currentIndex() - 1
        assert self.result is not None
        scientific = self.result.scientific
        evaluation = scientific.outcome.cluster(
            scientific.inputs.scope.cluster_id
        ).analytical_evaluation
        assert evaluation is not None
        values = self.result.scientific.inputs.snapshot.plane_values
        if amplitude >= 0:
            data = evaluation.amplitudes[amplitude]
            v.curve_profile.hide()
            v.profile.setLabel("left", "Signed peak amplitude")
            self.profile_model.setData(values, data, data=np.arange(len(values)))
            v.profile_marker.setData([values[v.plane]], [data[v.plane]])
        elif self.mode.currentIndex() == _RESIDUAL:
            self.profile_mode.setItemText(0, "Crosshair residual · fitted samples only")
            v.curve_profile.hide()
            v.profile.setLabel("left", "Residual /σ")
            data = None if self._slice is None else self._slice.residual_profile
            if data is None:
                self.profile_mode.setItemText(0, "No fitted sample at crosshair")
                v.profile_marker.clear()
            else:
                data = data / self.result.scientific.outcome.noise
                self.profile_model.setData(values, data, data=np.arange(len(values)))
                v.profile_marker.setData([values[v.plane]], [data[v.plane]])
        else:
            label = (
                "Crosshair intensity · ● exp / ▼ group"
                if self.mode.currentIndex() == _OVERLAY
                else "Crosshair intensity · experiment"
            )
            self.profile_mode.setItemText(0, label)
            v.curve_profile.show()
            v.profile.setLabel("left", "Coordinate intensity")
            if self._slice is not None and self.mode.currentIndex() == _OVERLAY:
                self.profile_model.setData(values, self._slice.profile, data=np.arange(len(values)))

    def _profile_changed(self) -> None:
        if self.active:
            self.viewer.profile.enableAutoRange()
            self.viewer._update_slices()

    def _profile_clicked(self, _item, points, _event) -> None:
        if points:
            self.viewer.set_plane(int(points[0].index()) + 1)
