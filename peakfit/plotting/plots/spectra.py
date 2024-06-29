import argparse
import sys

import nmrglue as ng
import numpy as np
import numpy.typing as npt
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from peakfit.noise import estimate_noise

CONTOUR_NUM = 25
CONTOUR_FACTOR = 1.30
CONTOUR_COLORS = {
    "spectrum_exp": "C0",
    "spectrum_sim": "C1",
    "difference": "C2",
}

FloatArray = npt.NDArray[np.float64]


class NMRData:
    def __init__(self, filename: str) -> None:
        self.dic, self.data = ng.pipe.read(filename)
        self.data, self.xlim, self.ylim = self._process_data()

    def _process_data(
        self,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        if self.data.ndim == 3:
            uc_y = ng.pipe.make_uc(self.dic, self.data, dim=1)
            uc_x = ng.pipe.make_uc(self.dic, self.data, dim=2)
        elif self.data.ndim == 2:
            uc_y = ng.pipe.make_uc(self.dic, self.data, dim=0)
            uc_x = ng.pipe.make_uc(self.dic, self.data, dim=1)
            self.data = self.data.reshape(1, *self.data.shape)
        else:
            msg = f"Unsupported data dimensionality: {self.data.ndim}"
            raise ValueError(msg)

        return self.data, uc_x.ppm_limits(), uc_y.ppm_limits()


class PlotWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

    def plot(
        self,
        data1: FloatArray,
        data2: FloatArray,
        data_diff: FloatArray,
        show_spectra: dict[str, bool],
        contour_level: int,
        current_plane: int,
        xlim: list[float],
        ylim: list[float],
    ) -> None:
        self.ax.clear()
        levels = contour_level * CONTOUR_FACTOR ** np.arange(CONTOUR_NUM)
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

        self.ax.set_title(f"NMR Spectrum - Plane {current_plane + 1}")
        self.ax.set_xlabel("Dimension 1 [ppm]")
        self.ax.set_ylabel("Dimension 2 [ppm]")
        self.ax.set_xlim(*sorted(xlim, reverse=True))
        self.ax.set_ylim(*sorted(ylim, reverse=True))

        self.figure.tight_layout()
        self.canvas.draw_idle()


class ControlWidget(QWidget):
    plane_changed = Signal(int)
    contour_level_changed = Signal(int)
    spectrum_toggled = Signal(str, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout()
        layout.addLayout(self._create_navigation_layout())
        layout.addLayout(self._create_slider_layout())
        layout.addLayout(self._create_checkbox_layout())
        self.setLayout(layout)

    def _create_navigation_layout(self) -> QHBoxLayout:
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous Plane")
        self.next_button = QPushButton("Next Plane")
        self.plane_label = QLabel()

        self.prev_button.clicked.connect(lambda: self.plane_changed.emit(-1))
        self.next_button.clicked.connect(lambda: self.plane_changed.emit(1))

        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.plane_label)
        return nav_layout

    def _create_slider_layout(self) -> QHBoxLayout:
        slider_layout = QHBoxLayout()
        self.contour_slider = QSlider(Qt.Horizontal)
        self.contour_slider.setRange(1, 100)
        self.contour_slider.valueChanged.connect(self.contour_level_changed.emit)

        slider_layout.addWidget(QLabel("Contour Level:"))
        slider_layout.addWidget(self.contour_slider)
        return slider_layout

    def _create_checkbox_layout(self) -> QHBoxLayout:
        checkbox_layout = QHBoxLayout()
        for key, label in [
            ("spectrum_exp", "Show Spectrum Exp"),
            ("spectrum_sim", "Show Spectrum Sim"),
            ("difference", "Show Difference Spectrum"),
        ]:
            checkbox = QCheckBox(label)
            checkbox.setChecked(key != "difference")
            checkbox.stateChanged.connect(
                lambda state, k=key: self.spectrum_toggled.emit(k, state == Qt.Checked)
            )
            checkbox_layout.addWidget(checkbox)
        return checkbox_layout

    def update_plane_label(self, current_plane: int, total_planes: int) -> None:
        self.plane_label.setText(f"Plane: {current_plane + 1}/{total_planes}")


class SpectraViewer(QMainWindow):
    def __init__(self, data1: NMRData, data2: NMRData) -> None:
        super().__init__()
        self.data1, self.data2 = data1, data2
        self.data_diff = self.data1.data - self.data2.data
        self.current_plane = 0
        self.show_spectra = {
            "spectrum_exp": True,
            "spectrum_sim": True,
            "difference": False,
        }
        self.noise_level = estimate_noise(self.data1.data)
        self.contour_level = 5
        self.current_xlim = None
        self.current_ylim = None

        self.setWindowTitle("NMR Pseudo-3D Spectra Viewer")
        self.setGeometry(100, 100, 1000, 800)

        # Create menu bar
        self.create_menu_bar()

        # Create central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create and add other widgets
        splitter = QSplitter(Qt.Vertical)
        self.plot_widget = self.create_plot_widget()
        self.control_widget = self.create_control_widget()

        splitter.addWidget(self.plot_widget)
        splitter.addWidget(self.control_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        # Set initial values for sliders and spinboxes
        self.plane_slider.setValue(1)
        self.plane_spinbox.setValue(1)
        self.contour_slider.setValue(self.contour_level)
        self.contour_spinbox.setValue(self.contour_level)

        self.update_view(True)  # True flag for initial view

    def create_menu_bar(self) -> None:
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

    def init_ui(self) -> None:
        self.setWindowTitle("NMR Pseudo-3D Spectra Viewer")
        self.setGeometry(100, 100, 1000, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        splitter = QSplitter(Qt.Vertical)
        self.plot_widget = self.create_plot_widget()
        self.control_widget = self.create_control_widget()

        splitter.addWidget(self.plot_widget)
        splitter.addWidget(self.control_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def create_plot_widget(self) -> QWidget:
        plot_widget = QWidget()
        layout = QVBoxLayout(plot_widget)

        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, plot_widget)

        self.canvas.mpl_connect("motion_notify_event", self.on_navigation)

        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        return plot_widget

    def create_control_widget(self) -> QWidget:
        control_widget = QWidget()
        layout = QHBoxLayout(control_widget)

        # Navigation controls
        nav_group = QGroupBox("Plane")
        nav_layout = QHBoxLayout(nav_group)
        self.plane_slider = QSlider(Qt.Horizontal)
        self.plane_slider.setRange(1, self.data1.data.shape[0])
        self.plane_spinbox = QSpinBox()
        self.plane_spinbox.setRange(1, self.data1.data.shape[0])
        nav_layout.addWidget(self.plane_slider)
        nav_layout.addWidget(self.plane_spinbox)

        # Contour level controls
        contour_group = QGroupBox("Contour Level")
        contour_layout = QHBoxLayout(contour_group)
        self.contour_slider = QSlider(Qt.Horizontal)
        self.contour_slider.setRange(1, 100)
        self.contour_spinbox = QSpinBox()
        self.contour_spinbox.setRange(1, 100)
        contour_layout.addWidget(self.contour_slider)
        contour_layout.addWidget(self.contour_spinbox)

        # Spectrum visibility controls
        visibility_group = QGroupBox("Visibility")
        visibility_layout = QHBoxLayout(visibility_group)
        self.checkboxes = {}
        for key, label in [
            ("spectrum_exp", "Exp"),
            ("spectrum_sim", "Sim"),
            ("difference", "Diff"),
        ]:
            checkbox = QCheckBox(label)
            checkbox.setChecked(key != "difference")
            self.checkboxes[key] = checkbox
            visibility_layout.addWidget(checkbox)

        # Add the group boxes to the main horizontal layout
        layout.addWidget(nav_group)
        layout.addWidget(contour_group)
        layout.addWidget(visibility_group)

        # Connect signals
        self.plane_slider.valueChanged.connect(self.change_plane)
        self.plane_spinbox.valueChanged.connect(self.change_plane)
        self.contour_slider.valueChanged.connect(self.update_contour_level)
        self.contour_spinbox.valueChanged.connect(self.update_contour_level)
        for key, checkbox in self.checkboxes.items():
            checkbox.stateChanged.connect(
                lambda state, k=key: self.toggle_spectrum(k, state == Qt.Checked)
            )

        self.plane_slider.setFixedWidth(150)
        self.plane_spinbox.setFixedWidth(50)
        self.contour_slider.setFixedWidth(150)
        self.contour_spinbox.setFixedWidth(50)

        return control_widget

    def create_status_bar(self) -> None:
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")

    def update_view(self, reset_view=False) -> None:
        # Store current limits if they exist and we're not resetting the view
        if not reset_view and self.current_xlim and self.current_ylim:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
        else:
            xlim = sorted(self.data1.xlim, reverse=True)
            ylim = sorted(self.data1.ylim, reverse=True)

        self.ax.clear()
        levels = (
            self.contour_level
            * self.noise_level
            * CONTOUR_FACTOR ** np.arange(CONTOUR_NUM)
        )
        levels = np.concatenate((-levels[::-1], levels))

        for key, data in [
            ("spectrum_exp", self.data1.data),
            ("spectrum_sim", self.data2.data),
            ("difference", self.data_diff),
        ]:
            if self.show_spectra[key]:
                self.ax.contour(
                    data[self.current_plane],
                    levels=levels,
                    colors=CONTOUR_COLORS[key],
                    alpha=0.7,
                    extent=[*self.data1.xlim, *self.data1.ylim],
                )

        self.ax.set_title(f"NMR Spectrum - Plane {self.current_plane + 1}")
        self.ax.set_xlabel("Dimension 1 [ppm]")
        self.ax.set_ylabel("Dimension 2 [ppm]")

        # Restore the previous view limits
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        self.figure.tight_layout()
        self.canvas.draw_idle()

        # self.plane_label.setText(
        #     f"Plane: {self.current_plane + 1}/{self.data1.data.shape[0]}"
        # )

        # Store the current limits for next update
        self.current_xlim = self.ax.get_xlim()
        self.current_ylim = self.ax.get_ylim()

    def change_plane(self, value: int) -> None:
        self.current_plane = value - 1  # Adjust for 0-based indexing
        self.plane_slider.setValue(value)
        self.plane_spinbox.setValue(value)
        self.update_view()

    def update_contour_level(self, value: int) -> None:
        self.contour_level = value
        self.contour_slider.setValue(value)
        self.contour_spinbox.setValue(value)
        self.update_view()

    def toggle_spectrum(self, spectrum: str, show: bool) -> None:
        self.show_spectra[spectrum] = show
        self.update_view()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    # Add a method to reset the view
    def reset_view(self) -> None:
        self.update_view(True)

    def update_limits(self) -> None:
        self.current_xlim = self.ax.get_xlim()
        self.current_ylim = self.ax.get_ylim()

    def resize_event(self, event) -> None:
        super().resizeEvent(event)
        self.update_limits()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def on_navigation(self, event) -> None:
        if (
            event.name == "motion_notify"
        ):  # This event is triggered when panning or zooming
            self.update_limits()

    def key_press_event(self, event) -> None:
        if event.key() == Qt.Key_Left:
            self.change_plane(-1)
        elif event.key() == Qt.Key_Right:
            self.change_plane(1)
        else:
            super().keyPressEvent(event)


def plot_spectra(args: argparse.Namespace) -> None:
    try:
        data1 = NMRData(args.data_exp)
        data2 = NMRData(args.data_sim)
    except FileNotFoundError:
        sys.exit(1)
    except ValueError:
        sys.exit(1)

    if data1.data.shape != data2.data.shape:
        sys.exit(1)

    app = QApplication(sys.argv)
    viewer = SpectraViewer(data1, data2)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NMR Pseudo-3D Spectra Viewer")
    parser.add_argument("data_exp", help="Path to experimental data file")
    parser.add_argument("data_sim", help="Path to simulated data file")
    args = parser.parse_args()
    plot_spectra(args)
