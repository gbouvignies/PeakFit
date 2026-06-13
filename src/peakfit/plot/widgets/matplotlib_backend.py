import os
import sys
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from matplotlib.figure import Figure

from peakfit.plot.qt_core import QVBoxLayout, QWidget

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from peakfit.shared.typing import FloatArray

# Configure Backend
if (
    sys.platform.startswith("linux")
    and not os.environ.get("DISPLAY")
    and not os.environ.get("QT_QPA_PLATFORM")
):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Matplotlib Qt Agg backend - works with PySide6 if installed


class _CanvasProtocol(Protocol):
    def __init__(self, figure: Figure) -> None: ...

    def mpl_connect(self, event: str, callback: Callable[..., object]) -> object: ...

    def draw_idle(self) -> None: ...


class _ToolbarProtocol(Protocol):
    def __init__(self, canvas: QWidget, parent: QWidget) -> None: ...


FigureCanvas: type[_CanvasProtocol]
NavigationToolbar: type[_ToolbarProtocol] | None

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as _FigureCanvasQTAgg

    _navigation_toolbar: type[_ToolbarProtocol] | None
    try:
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as _NavigationToolbar2QT
    except (ModuleNotFoundError, ImportError):
        _navigation_toolbar = None
    else:
        _navigation_toolbar = _NavigationToolbar2QT

    FigureCanvas = _FigureCanvasQTAgg
    NavigationToolbar = _navigation_toolbar
except (ModuleNotFoundError, ImportError):
    # Fallback or headless
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        class _AggCanvasWidget(QWidget):
            def __init__(self, figure: Figure) -> None:
                super().__init__()
                self._canvas = FigureCanvasAgg(figure)

            def mpl_connect(self, event: str, callback: Callable[..., object]) -> object:
                return self._canvas.mpl_connect(event, callback)

            def draw_idle(self) -> None:
                self._canvas.draw_idle()

        FigureCanvas = _AggCanvasWidget
        NavigationToolbar = None
    except (ModuleNotFoundError, ImportError):
        # If even backend_agg is not available, define a dummy FigureCanvas
        class _DummyFigureCanvas(QWidget):
            """Dummy FigureCanvas for headless environments."""

            def __init__(self, figure: object) -> None:
                super().__init__()

            def mpl_connect(self, event: str, callback: Callable[..., object]) -> object:
                """Register a matplotlib event callback (no-op fallback)."""
                return object()

            def draw_idle(self) -> None:
                """Request a redraw (no-op fallback)."""
                return

        FigureCanvas = _DummyFigureCanvas
        NavigationToolbar = None


CONTOUR_NUM = 25
CONTOUR_FACTOR = 1.40
CONTOUR_COLORS = {
    "spectrum_exp": "C0",
    "spectrum_sim": "C1",
    "difference": "C2",
}


def _calculate_1d_slice(
    view_min: float,
    view_max: float,
    full_min: float,
    full_max: float,
    n_points: int,
    data_min_to_max: bool = False,
) -> tuple[slice, float, float]:
    """Return slice object and new extent for the slice based on view limits.

    Parameters
    ----------
    view_min : float
        Minimum value of current view (e.g. min ppm).
    view_max : float
        Maximum value of current view (e.g. max ppm).
    full_min : float
        Minimum value of full dataset.
    full_max : float
        Maximum value of full dataset.
    n_points : int
        Number of points in this dimension.
    data_min_to_max : bool, optional
        If True, data index 0 corresponds to full_min.
        If False, data index 0 corresponds to full_max. Default is False.

    Returns:
    -------
    tuple[slice, float, float]
        A tuple containing (slice_obj, new_start, new_end).
    """
    # Add 10% margin to view to avoid edge artifacts during small pans
    margin = (view_max - view_min) * 0.1
    v_min = max(view_min - margin, full_min)
    v_max = min(view_max + margin, full_max)

    full_range = full_max - full_min
    if full_range == 0:
        # Fallback
        return (
            slice(0, n_points),
            (full_min if data_min_to_max else full_max),
            (full_max if data_min_to_max else full_min),
        )

    if data_min_to_max:
        # Index 0 = Min PPM (full_min)
        frac_start = (v_min - full_min) / full_range
        frac_end = (v_max - full_min) / full_range
    else:
        # Index 0 = Max PPM (full_max)
        frac_start = (full_max - v_max) / full_range
        frac_end = (full_max - v_min) / full_range

    i_start = int(np.floor(frac_start * n_points))
    i_end = int(np.ceil(frac_end * n_points))

    i_start = max(0, i_start)
    i_end = min(n_points, i_end)

    if i_start >= i_end:
        # Fallback
        s_start = full_min if data_min_to_max else full_max
        s_end = full_max if data_min_to_max else full_min
        return slice(0, n_points), s_start, s_end

    # Calculate new extent for this slice
    actual_frac_start = i_start / n_points
    actual_frac_end = i_end / n_points

    if data_min_to_max:
        new_start = full_min + (actual_frac_start * full_range)
        new_end = full_min + (actual_frac_end * full_range)
    else:
        new_start = full_max - (actual_frac_start * full_range)
        new_end = full_max - (actual_frac_end * full_range)

    return slice(i_start, i_end), new_start, new_end


class MatplotlibBackend(QWidget):
    """Matplotlib widget for the spectrum viewer with view-slicing optimization."""

    def __init__(self, parent: QWidget | None = None) -> None:
        QWidget.__init__(self, parent)
        self.figure = Figure(figsize=(5, 5), dpi=100)

        canvas = FigureCanvas(self.figure)
        # `FigureCanvas` is a Qt widget for the Qt backend, and a QWidget wrapper
        # for the Agg fallback; validate this assumption for type safety.
        if not isinstance(canvas, QWidget):
            raise TypeError("Matplotlib FigureCanvas must be a QWidget")

        self.canvas = canvas
        self.ax = self.figure.add_subplot(111)

        self.toolbar: Any = None
        if NavigationToolbar is not None:
            self.toolbar = NavigationToolbar(self.canvas, self)

        self.current_xlim: tuple[float, float] | None = None
        self.current_ylim: tuple[float, float] | None = None

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        if self.toolbar is not None:
            layout.addWidget(self.toolbar)
        self.setLayout(layout)

        self.canvas.mpl_connect("draw_event", self._update_limits)

    def get_widget(self) -> QWidget:
        """Return the Matplotlib canvas widget."""
        return self

    def _update_limits(self, _event: Any) -> None:
        self.current_xlim = self.ax.get_xlim()
        self.current_ylim = self.ax.get_ylim()

    def clear(self) -> None:
        """Clear the current plot axis."""
        self.ax.clear()

    def _plot_contours_layer(
        self,
        data_map: list[tuple[str, FloatArray | None]],
        show_spectra: dict[str, bool],
        levels: FloatArray,
        slices: tuple[slice, slice],
        extent: list[float],
    ) -> None:
        """Plot contour layers for experimental, simulated, and difference spectra."""
        for key, data in data_map:
            if data is None:
                continue

            if show_spectra.get(key, False):
                # Slice data
                data_view = data[slices[0], slices[1]]

                self.ax.contour(
                    data_view,
                    levels=levels,
                    colors=CONTOUR_COLORS[key],
                    alpha=0.7,
                    extent=extent,
                )

    def _plot_peaks(
        self,
        plist: pd.DataFrame,
        view_xlim: tuple[float, ...] | list[float],
        view_ylim: tuple[float, ...] | list[float],
    ) -> None:
        """Filter and plot peaks on the spectrum."""
        # Filter plist to only show peaks within the current view for performance
        current_x_min, current_x_max = sorted(view_xlim)
        current_y_min, current_y_max = sorted(view_ylim)

        # Filter peaks that are within the current view (including margin)
        filtered_plist = plist[
            (plist["x0_ppm"] >= current_x_min)
            & (plist["x0_ppm"] <= current_x_max)
            & (plist["y0_ppm"] >= current_y_min)
            & (plist["y0_ppm"] <= current_y_max)
        ]

        self.ax.scatter(
            filtered_plist["x0_ppm"], filtered_plist["y0_ppm"], color="black", s=10, zorder=100
        )
        for label, y, x in filtered_plist.itertuples(index=False):
            self.ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), zorder=100)

    def plot(
        self,
        data1: FloatArray,
        data2: FloatArray | None,
        data_diff: FloatArray | None,
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
        """Update the plot with new data or settings using view-slicing optimization."""
        self.clear()

        levels = contour_level * noise_level * CONTOUR_FACTOR ** np.arange(CONTOUR_NUM)
        levels = np.concatenate((-levels[::-1], levels))

        # View-Slicing Optimization
        view_xlim = self.current_xlim if (not reset_view and self.current_xlim) else tuple(xlim)
        view_ylim = self.current_ylim if (not reset_view and self.current_ylim) else tuple(ylim)

        # Sort limits to be safe (ppm can be descending)
        full_x_min, full_x_max = min(xlim), max(xlim)
        full_y_min, full_y_max = min(ylim), max(ylim)

        # Data shape is (Y, X)
        n_y, n_x = data1.shape

        # X Slice (corresponds to columns, second dimension of data array)
        x_slice, ex_left, ex_right = _calculate_1d_slice(
            min(view_xlim),
            max(view_xlim),
            full_x_min,
            full_x_max,
            n_x,
            data_min_to_max=False,
        )

        # Y Slice (corresponds to rows, first dimension of data array)
        y_slice, ex_bottom, ex_top = _calculate_1d_slice(
            min(view_ylim),
            max(view_ylim),
            full_y_min,
            full_y_max,
            n_y,
            data_min_to_max=False,
        )

        # Apply slicing
        # extent format for contour: [left, right, bottom, top]
        s_extent = [ex_left, ex_right, ex_bottom, ex_top]

        data_map = [
            ("spectrum_exp", data1),
            ("spectrum_sim", data2),
            ("difference", data_diff),
        ]

        self._plot_contours_layer(data_map, show_spectra, levels, (y_slice, x_slice), s_extent)

        if plist is not None:
            self._plot_peaks(plist, view_xlim, view_ylim)

        self.ax.set_title(f"NMR Spectrum - Plane {current_plane + 1}")
        self.ax.set_xlabel("Dimension 1 [ppm]")
        self.ax.set_ylabel("Dimension 2 [ppm]")

        # Restore request view
        if reset_view or self.current_xlim is None:
            # Set to Full Extent
            self.ax.set_xlim(*sorted(xlim, reverse=True))
            self.ax.set_ylim(*sorted(ylim, reverse=True))
        else:
            # Restore previous user zoom
            self.ax.set_xlim(self.current_xlim)
            self.ax.set_ylim(self.current_ylim)

        self.figure.tight_layout()
        self.canvas.draw_idle()
