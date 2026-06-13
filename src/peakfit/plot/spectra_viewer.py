import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import nmrglue as ng
import numpy as np
import pandas as pd

from peakfit.plot.widgets.matplotlib_backend import MatplotlibBackend

if TYPE_CHECKING:
    from peakfit.plot.reconstruction import SpectraReconstructor
    from peakfit.shared.typing import FloatArray

from peakfit.engine.algorithms.noise import estimate_noise
from peakfit.plot.qt_core import (
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
    Qt,
    QVBoxLayout,
    QWidget,
    Signal,
)
from peakfit.shared.paths import format_path

# Configuration
CONTOUR_NUM = 25
CONTOUR_FACTOR = 1.40
_MIN_VIEWER_NDIM = 3
CONTOUR_COLORS = {
    "spectrum_exp": "C0",
    "spectrum_sim": "C1",
    "difference": "C2",
}


def _report_error(message: str) -> None:
    """Report viewer errors without relying on the CLI UI layer."""
    print(message, file=sys.stderr)


@dataclass
class NMRData:
    """Container for loaded NMR data and related metadata."""

    filename: str
    dic: dict[str, Any]
    data: FloatArray
    xlim: tuple[float, float]
    ylim: tuple[float, float]

    @classmethod
    def from_file(cls, filename: str) -> Self:
        """Load an NMR file and return a populated NMRData instance."""
        dic, data = ng.pipe.read(filename)
        data = data.astype(np.float32)
        data, xlim, ylim = cls._process_data(dic, data)
        return cls(filename, dic, data, xlim, ylim)

    @staticmethod
    def _process_data(
        dic: dict[str, Any], data: FloatArray
    ) -> tuple[FloatArray, tuple[float, float], tuple[float, float]]:
        udic = ng.pipe.guess_udic(dic, data)
        first_is_freq = udic[0]["freq"] if udic else True
        if first_is_freq:
            data = data.reshape(1, *data.shape)

        if data.ndim < _MIN_VIEWER_NDIM:
            msg = f"Unsupported data dimensionality after normalization: {data.ndim}"
            raise ValueError(msg)

        dim_y = data.ndim - 2
        dim_x = data.ndim - 1
        uc_y, uc_x = (
            ng.pipe.make_uc(dic, data, dim=dim_y),
            ng.pipe.make_uc(dic, data, dim=dim_x),
        )
        return data, uc_x.ppm_limits(), uc_y.ppm_limits()

    def unalias_y(self, y0: FloatArray) -> FloatArray:
        """Unwrap Y-axis positions if needed, keeping values within y-limits."""
        y_scale = (self.ylim[1] - self.ylim[0]) * (self.data.shape[1] + 1) / self.data.shape[1]
        return cast(
            "FloatArray", np.asarray((y0 - self.ylim[0]) % y_scale + self.ylim[0], dtype=np.float64)
        )


class ControlWidget(QWidget):
    """Control panel for adjusting view and contour levels in the spectra viewer."""

    plane_changed = Signal(int)
    contour_level_changed = Signal(int)
    spectrum_toggled = Signal(str, bool)

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
        """Update plane label and slider/spinbox ranges for the UI."""
        self.plane_spinbox.setRange(1, total_planes)
        self.plane_slider.setRange(1, total_planes)
        self.plane_spinbox.setValue(current_plane + 1)
        self.plane_slider.setValue(current_plane + 1)


class SpectraViewer(QMainWindow):
    """Top-level Qt application window containing spectra viewer and controls."""

    def __init__(
        self,
        data1: NMRData,
        data2: NMRData | None,
        plist: pd.DataFrame | None,
        reconstructor: SpectraReconstructor | None = None,
    ) -> None:
        super().__init__()
        self.data1 = data1
        self.data2 = data2
        self.reconstructor = reconstructor

        # If a simulated spectrum is supplied, calculate the difference once.
        if self.data2 is not None:
            self.data_diff: FloatArray | None = self.data1.data - self.data2.data
        else:
            self.data_diff = None  # Calculated on fly

        self.plist = plist
        self.current_plane = 0
        self.show_spectra = {
            "spectrum_exp": True,
            "spectrum_sim": True,
            "difference": False,
        }
        self.noise_level = float(estimate_noise(self.data1.data))
        self.contour_level = 5

        self.backend = MatplotlibBackend()

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
        file_menu = menubar.addMenu("File") if menubar is not None else None
        view_menu = menubar.addMenu("View") if menubar is not None else None

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")

        def _exit_slot(*_args: Any) -> None:
            self.close()

        exit_action.triggered.connect(_exit_slot)
        if file_menu is not None:
            file_menu.addAction(exit_action)

        reset_view_action = QAction("Reset View", self)
        reset_view_action.setShortcut("Ctrl+R")

        def _reset_slot(*_args: Any) -> None:
            self.reset_view()

        reset_view_action.triggered.connect(_reset_slot)
        if view_menu is not None:
            view_menu.addAction(reset_view_action)

    def _create_central_widget(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Backend provides the widget
        self.plot_widget = self.backend.get_widget()
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
        self.control_widget.contour_slider.valueChanged.connect(self._update_contour_level)
        self.control_widget.contour_spinbox.valueChanged.connect(self._update_contour_level)
        for key, checkbox in self.control_widget.checkboxes.items():
            checkbox.stateChanged.connect(lambda _state, k=key: self._toggle_spectrum(k))

    def _create_status_bar(self) -> None:
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")

    def update_view(self, *, reset_view: bool = False) -> None:
        """Refresh the plot view, optionally resetting zoom to defaults."""
        xlim = sorted(self.data1.xlim, reverse=True)
        ylim = sorted(self.data1.ylim, reverse=True)

        # Get simulated data for current plane
        sim_plane = None
        diff_plane = None
        exp_plane = self.data1.data[self.current_plane]

        if self.data2 is not None:
            sim_plane = self.data2.data[self.current_plane]
            assert self.data_diff is not None
            diff_plane = self.data_diff[self.current_plane]
        elif self.reconstructor is not None:
            # On-the-fly reconstruction
            data_shape = self.data1.data.shape
            grid_shape = data_shape[1:]
            sim_plane = self.reconstructor.reconstruct_plane(self.current_plane, grid_shape)
            diff_plane = exp_plane - sim_plane

        # Delegate drawing to the plotting backend.
        self.backend.plot(
            exp_plane,  # Slice experimental data here to pass only 2D array
            sim_plane,
            diff_plane,
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
        """Reset the plot view to default limits and refresh view."""
        self.update_view(reset_view=True)

    def _change_plane(self, value: int) -> None:
        self.current_plane = value - 1  # Adjust for 0-based indexing
        self.control_widget.update_plane_label(self.current_plane, self.data1.data.shape[0])
        self.update_view()

    def _update_contour_level(self, value: int) -> None:
        self.contour_level = value
        self.control_widget.contour_slider.setValue(value)
        self.control_widget.contour_spinbox.setValue(value)
        self.update_view()

    def _toggle_spectrum(self, spectrum: str) -> None:
        self.show_spectra[spectrum] = self.control_widget.checkboxes[spectrum].isChecked()
        self.update_view()

    def resizeEvent(self, a0: Any) -> None:  # noqa: N802
        """Handle window resize events and adjust layouts accordingly."""
        super().resizeEvent(a0)
        # Assuming backend widget handles resize or we might need a notify method in backend
        # MatplotlibBackend (QWidget) handles its own layout, but we can call a refresh if needed
        pass


def _validate_files(args: argparse.Namespace) -> None:
    """Validate existence of input files."""
    for path, desc in [
        (args.data_exp, "Experimental data"),
        (args.data_sim, "Simulated data"),
    ]:
        if not Path(path).exists():
            _report_error(f"{desc} file not found: {format_path(path)}")
            sys.exit(1)

    if args.peak_list and not Path(args.peak_list).exists():
        _report_error(f"Peak list file not found: {format_path(args.peak_list)}")
        sys.exit(1)


def _load_datasets(args: argparse.Namespace) -> tuple[NMRData, NMRData, pd.DataFrame | None]:
    """Load experimental, simulated data and peak list."""
    try:
        data1 = NMRData.from_file(args.data_exp)
        data2 = NMRData.from_file(args.data_sim)

        if data1.data.shape != data2.data.shape:
            # We catch this here or let caller handle? Caller handles exit logic usually.
            # But simpler to raise ValueError
            msg = "Data shapes do not match between experimental and simulated data"
            raise ValueError(msg)

        plist = None
        if args.peak_list:
            plist = pd.read_table(
                args.peak_list,
                sep=r"\s+",
                comment="#",
                header=None,
                names=("name", "y0_ppm", "x0_ppm"),
            )
            plist["y0_ppm"] = data1.unalias_y(plist["y0_ppm"].to_numpy().astype(np.float32))

        return data1, data2, plist

    except (FileNotFoundError, ValueError, OSError) as e:
        _report_error(f"Error loading data files: {e}")
        sys.exit(1)


def plot_spectra(args: argparse.Namespace) -> None:
    """Plot spectra with improved error handling for missing files."""
    _validate_files(args)
    data1, data2, plist = _load_datasets(args)

    _app = QApplication(sys.argv)
    viewer = SpectraViewer(data1, data2, plist)
    viewer.show()


def main() -> None:
    """Run the spectra viewer command-line interface and display plots."""
    parser = argparse.ArgumentParser(description="NMR Spectra Viewer")
    parser.add_argument("data_exp", help="Experimental data file")
    parser.add_argument("--sim", dest="data_sim", required=True, help="Simulated data file")
    parser.add_argument("--peak-list", help="Peak list file")
    args = parser.parse_args()
    plot_spectra(args)


if __name__ == "__main__":
    main()
