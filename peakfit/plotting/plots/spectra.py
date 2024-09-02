import argparse
import sys
from dataclasses import dataclass
from typing import Self

import nmrglue as ng
import numpy as np
import pandas as pd
from matplotlib.backend_bases import Event
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from peakfit.noise import estimate_noise
from peakfit.typing import FloatArray

# Configuration
CONTOUR_NUM = 25
CONTOUR_FACTOR = 1.40
CONTOUR_COLORS = {
    "spectrum_exp": "C0",
    "spectrum_sim": "C1",
    "difference": "C2",
}


@dataclass
class NMRData:
    filename: str
    dic: dict
    data: FloatArray
    xlim: tuple[float, float]
    ylim: tuple[float, float]

    @classmethod
    def from_file(cls, filename: str) -> Self:
        dic, data = ng.pipe.read(filename)
        data = data.astype(np.float32)
        data, xlim, ylim = cls._process_data(dic, data)
        return cls(filename, dic, data, xlim, ylim)

    @staticmethod
    def _process_data(
        dic: dict, data: FloatArray
    ) -> tuple[FloatArray, tuple[float, float], tuple[float, float]]:
        if data.ndim == 3:
            uc_y, uc_x = (
                ng.pipe.make_uc(dic, data, dim=1),
                ng.pipe.make_uc(dic, data, dim=2),
            )
        elif data.ndim == 2:
            uc_y, uc_x = (
                ng.pipe.make_uc(dic, data, dim=0),
                ng.pipe.make_uc(dic, data, dim=1),
            )
            data = data.reshape(1, *data.shape)
        else:
            raise ValueError(f"Unsupported data dimensionality: {data.ndim}")

        return data, uc_x.ppm_limits(), uc_y.ppm_limits()

    def unalias_y(self, y0: FloatArray) -> FloatArray:
        y_scale = (
            (self.ylim[1] - self.ylim[0])
            * (self.data.shape[1] + 1)
            / self.data.shape[1]
        )
        return (y0 - self.ylim[0]) % y_scale + self.ylim[0]


class PlotWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.current_xlim = self.current_ylim = None

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

        self.canvas.mpl_connect("draw_event", self._update_limits)

    def _update_limits(self, _event: Event) -> None:
        self.current_xlim = self.ax.get_xlim()
        self.current_ylim = self.ax.get_ylim()

    def plot(
        self,
        data1: FloatArray,
        data2: FloatArray,
        data_diff: FloatArray,
        plist: pd.DataFrame | None,
        show_spectra: dict[str, bool],
        contour_level: float,
        noise_level: float,
        current_plane: int,
        xlim: list[float],
        ylim: list[float],
        *,
        reset_view: bool = False,
    ) -> None:
        self.ax.clear()
        levels = contour_level * noise_level * CONTOUR_FACTOR ** np.arange(CONTOUR_NUM)
        levels = np.concatenate((-levels[::-1], levels))

        for key, data in [
            ("spectrum_exp", data1),
            ("spectrum_sim", data2),
            ("difference", data_diff),
        ]:
            if show_spectra[key]:
                self.ax.contour(
                    data[current_plane],
                    levels=levels,
                    colors=CONTOUR_COLORS[key],
                    alpha=0.7,
                    extent=[*xlim, *ylim],
                )

        if plist is not None:
            self.ax.scatter(plist["x0_ppm"], plist["y0_ppm"], color="black", s=10)
            for label, y, x in plist.itertuples(index=False):
                self.ax.annotate(
                    label, (x, y), textcoords="offset points", xytext=(5, 5)
                )

        self.ax.set_title(f"NMR Spectrum - Plane {current_plane + 1}")
        self.ax.set_xlabel("Dimension 1 [ppm]")
        self.ax.set_ylabel("Dimension 2 [ppm]")

        if reset_view or self.current_xlim is None:
            self.ax.set_xlim(*sorted(xlim, reverse=True))
            self.ax.set_ylim(*sorted(ylim, reverse=True))
        else:
            self.ax.set_xlim(self.current_xlim)
            self.ax.set_ylim(self.current_ylim)

        self.figure.tight_layout()
        self.canvas.draw_idle()


class ControlWidget(QWidget):
    plane_changed = pyqtSignal(int)
    contour_level_changed = pyqtSignal(int)
    spectrum_toggled = pyqtSignal(str, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout()
        layout.addLayout(self._create_navigation_layout())
        layout.addLayout(self._create_slider_layout())
        layout.addLayout(self._create_checkbox_layout())
        self.setLayout(layout)

    def _create_navigation_layout(self) -> QHBoxLayout:
        nav_layout = QHBoxLayout()
        self.plane_slider = QSlider(Qt.Orientation.Horizontal)
        self.plane_spinbox = QSpinBox()

        nav_layout.addWidget(QLabel("Plane:"))
        nav_layout.addWidget(self.plane_slider)
        nav_layout.addWidget(self.plane_spinbox)
        return nav_layout

    def _create_slider_layout(self) -> QHBoxLayout:
        slider_layout = QHBoxLayout()
        self.contour_slider = QSlider(Qt.Orientation.Horizontal)
        self.contour_spinbox = QSpinBox()

        slider_layout.addWidget(QLabel("Contour:"))
        slider_layout.addWidget(self.contour_slider)
        slider_layout.addWidget(self.contour_spinbox)
        return slider_layout

    def _create_checkbox_layout(self) -> QHBoxLayout:
        checkbox_layout = QHBoxLayout()
        self.checkboxes = {}
        for key, label in [
            ("spectrum_exp", "Exp"),
            ("spectrum_sim", "Sim"),
            ("difference", "Diff"),
        ]:
            checkbox = QCheckBox(label)
            checkbox.setChecked(key != "difference")
            self.checkboxes[key] = checkbox
            checkbox_layout.addWidget(checkbox)
        return checkbox_layout

    def update_plane_label(self, current_plane: int, total_planes: int) -> None:
        self.plane_spinbox.setRange(1, total_planes)
        self.plane_slider.setRange(1, total_planes)
        self.plane_spinbox.setValue(current_plane + 1)
        self.plane_slider.setValue(current_plane + 1)


class SpectraViewer(QMainWindow):
    def __init__(
        self, data1: NMRData, data2: NMRData, plist: pd.DataFrame | None
    ) -> None:
        super().__init__()
        self.data1, self.data2 = data1, data2
        self.data_diff = self.data1.data - self.data2.data
        self.plist = plist
        self.current_plane = 0
        self.show_spectra = {
            "spectrum_exp": True,
            "spectrum_sim": True,
            "difference": False,
        }
        self.noise_level = float(estimate_noise(self.data1.data))
        self.contour_level = 5

        self._init_ui()

    def _init_ui(self) -> None:
        self.setWindowTitle("NMR Pseudo-3D Spectra Viewer")
        self.setGeometry(100, 100, 1000, 800)

        self._create_menu_bar()
        self._create_central_widget()
        self._create_status_bar()

        self.control_widget.contour_slider.setValue(self.contour_level)
        self.control_widget.contour_spinbox.setValue(self.contour_level)

        self.update_view(reset_view=True)

    def _create_menu_bar(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        view_menu = menubar.addMenu("View")

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        reset_view_action = QAction("Reset View", self)
        reset_view_action.setShortcut("Ctrl+R")
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)

    def _create_central_widget(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.plot_widget = PlotWidget()
        self.control_widget = ControlWidget()

        splitter.addWidget(self.plot_widget)
        splitter.addWidget(self.control_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        self._connect_signals()

    def _connect_signals(self) -> None:
        self.control_widget.plane_slider.valueChanged.connect(self._change_plane)
        self.control_widget.plane_spinbox.valueChanged.connect(self._change_plane)
        self.control_widget.contour_slider.valueChanged.connect(
            self._update_contour_level
        )
        self.control_widget.contour_spinbox.valueChanged.connect(
            self._update_contour_level
        )
        for key, checkbox in self.control_widget.checkboxes.items():
            checkbox.stateChanged.connect(lambda state, k=key: self._toggle_spectrum(k))

    def _create_status_bar(self) -> None:
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")

    def update_view(self, *, reset_view: bool = False) -> None:
        xlim = sorted(self.data1.xlim, reverse=True)
        ylim = sorted(self.data1.ylim, reverse=True)

        self.plot_widget.plot(
            self.data1.data,
            self.data2.data,
            self.data_diff,
            self.plist,
            self.show_spectra,
            self.contour_level,
            self.noise_level,
            self.current_plane,
            xlim,
            ylim,
            reset_view=reset_view,
        )

    def reset_view(self) -> None:
        self.update_view(reset_view=True)

    def _change_plane(self, value: int) -> None:
        self.current_plane = value - 1  # Adjust for 0-based indexing
        self.control_widget.update_plane_label(
            self.current_plane, self.data1.data.shape[0]
        )
        self.update_view()

    def _update_contour_level(self, value: int) -> None:
        self.contour_level = value
        self.control_widget.contour_slider.setValue(value)
        self.control_widget.contour_spinbox.setValue(value)
        self.update_view()

    def _toggle_spectrum(self, spectrum: str) -> None:
        self.show_spectra[spectrum] = self.control_widget.checkboxes[
            spectrum
        ].isChecked()
        self.update_view()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.plot_widget.figure.tight_layout()
        self.plot_widget.canvas.draw_idle()


def plot_spectra(args: argparse.Namespace) -> None:
    try:
        data1 = NMRData.from_file(args.data_exp)
        data2 = NMRData.from_file(args.data_sim)
        plist = None
        if args.peak_list:
            plist = pd.read_table(
                args.peak_list,
                sep=r"\s+",
                comment="#",
                header=None,
                names=("name", "y0_ppm", "x0_ppm"),
            )
            plist["y0_ppm"] = data1.unalias_y(
                plist["y0_ppm"].to_numpy().astype(np.float32)
            )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if data1.data.shape != data2.data.shape:
        print("Error: Data shapes do not match")
        sys.exit(1)

    app = QApplication(sys.argv)
    viewer = SpectraViewer(data1, data2, plist)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NMR Spectra Viewer")
    parser.add_argument("data_exp", help="Experimental data file")
    parser.add_argument("data_sim", help="Simulated data file")
    parser.add_argument("--peak-list", help="Peak list file")
    args = parser.parse_args()
    plot_spectra(args)
