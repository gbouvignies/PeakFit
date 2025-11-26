import importlib
import sys
from types import ModuleType
from typing import ClassVar

import matplotlib
import numpy as np


def import_spectra(monkeypatch) -> ModuleType:
    """Import the spectra module, useful for reloading in tests."""
    if "peakfit.plotting.spectra" in sys.modules:
        del sys.modules["peakfit.plotting.spectra"]
    return importlib.import_module("peakfit.plotting.spectra")


def test_plotting_fallback_to_agg(monkeypatch, tmp_path):
    # Use Agg backend for headless testing
    matplotlib.use("Agg")

    mod = import_spectra(monkeypatch)
    plot_widget_cls = mod.PlotWidget
    # Build small synthetic data and call plot
    data1 = np.zeros((1, 10, 10), dtype=float)
    data2 = np.zeros_like(data1)
    data_diff = np.zeros_like(data1)

    # Create a dummy NMRData-like object with required attributes
    class Dummy:
        filename = "test"
        dic: ClassVar[dict] = {}
        data = data1
        xlim = (0.0, 1.0)
        ylim = (0.0, 1.0)

    # Create PlotWidget and call plot without Qt available
    # Ensure QApplication exists if PyQt5 is available
    try:
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
    except RuntimeError:
        app = None
    try:
        pw = plot_widget_cls()
        # Call plot with minimal inputs — this should not raise
        pw.plot(
            data1,
            data2,
            data_diff,
            None,
            {"spectrum_exp": True, "spectrum_sim": True, "difference": False},
            1.0,
            1.0,
            0,
            [0.0, 1.0],
            [0.0, 1.0],
        )
    finally:
        if app is not None:
            app.quit()
    # Call plot with minimal inputs — this should not raise
    pw.plot(
        data1,
        data2,
        data_diff,
        None,
        {"spectrum_exp": True, "spectrum_sim": True, "difference": False},
        1.0,
        1.0,
        0,
        [0.0, 1.0],
        [0.0, 1.0],
    )


def test_plot_widget_with_pyqt(monkeypatch):
    # If PyQt5 is present, test that PlotWidget exists and plotting works.
    import matplotlib

    matplotlib.use("Agg")
    import importlib

    mod = import_spectra(monkeypatch)
    importlib.reload(mod)
    plot_widget_cls = mod.PlotWidget

    # Ensure a QApplication exists to safely construct widgets
    try:
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
    except RuntimeError:
        app = None
    try:
        pw = plot_widget_cls()
        data = np.zeros((1, 10, 10), dtype=float)
        pw.plot(
            data,
            data,
            data,
            None,
            {"spectrum_exp": True, "spectrum_sim": True, "difference": False},
            1.0,
            1.0,
            0,
            [0.0, 1.0],
            [0.0, 1.0],
        )
    finally:
        if app is not None:
            app.quit()
    # Also ensure 2D arrays are rejected (must be 3D), but not required
