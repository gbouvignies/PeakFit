"""Small scientific handoff regressions for the read-only spectrum viewer."""

from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from peakfit.gui.spectrum import contour_paths, display_position, load_peaks, load_spectrum
from peakfit.gui.viewer import SpectrumViewer

DATA = Path(__file__).resolve().parents[1] / "examples" / "02-advanced-fitting" / "data"
pytestmark = pytest.mark.gui


def test_experimental_spectrum_inspection_preserves_coordinates_and_raw_data() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    spectra = load_spectrum(DATA / "pseudo3d.ft2", DATA / "b1_offsets.txt")
    spectra.data.flags.writeable = False
    viewer = SpectrumViewer(spectra)
    viewer.resize(1200, 800)
    viewer.show()
    app.processEvents()
    try:
        # Click in the real plotted coordinate system, then inspect a different plane.
        point = viewer.spectrum_plot.getViewBox().mapViewToScene(
            QtCore.QPointF(float(viewer.ppm_x[250]), float(viewer.ppm_y[100]))
        )
        QtTest.QTest.mouseClick(
            viewer.spectrum_plot.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=viewer.spectrum_plot.mapFromScene(point),
        )
        viewer.set_plane(8)
        iy, ix = viewer.iy, viewer.ix
        assert abs(ix - 250) <= 1
        assert abs(iy - 100) <= 1
        assert viewer.plane == 7
        np.testing.assert_array_equal(viewer.curve_x.getData()[1], spectra.data[7, iy, :])
        np.testing.assert_array_equal(viewer.curve_y.getData()[1], spectra.data[7, :, ix])
        np.testing.assert_array_equal(viewer.curve_profile.getData()[0], spectra.z_values)
        np.testing.assert_array_equal(viewer.curve_profile.getData()[1], spectra.data[:, iy, ix])
        assert viewer.cross_x.value() == viewer.ppm_x[ix]
        assert viewer.cross_y.value() == viewer.ppm_y[iy]
        viewer.set_peaks(load_peaks(DATA / "pseudo3d.list", spectra))
        assert len(viewer.peaks) == 166
        peak_x, peak_y = viewer.peak_scatter.getData()
        first = viewer.peaks[0]
        assert peak_x[0] == pytest.approx(first.positions[1])
        assert peak_y[0] == pytest.approx(display_position(viewer.y_axis, first.positions[0]))
    finally:
        viewer.close()
        viewer.deleteLater()
        app.processEvents()


def test_2d_loading_signed_contours_and_indirect_peak_aliasing(tmp_path: Path) -> None:
    # Write a genuine 2D NMRPipe file from one bundled experimental plane.
    spectra = load_spectrum(DATA / "pseudo3d.ft2")
    udic = ng.pipe.guess_udic(spectra.dic, spectra.data)
    plane_udic = {"ndim": 2, 0: udic[1], 1: udic[2]}
    path = tmp_path / "plane.ft2"
    ng.pipe.write(str(path), ng.pipe.create_dic(plane_udic), spectra.data[0])
    single = load_spectrum(path)
    assert single.data.shape == (1, *spectra.data.shape[1:])
    np.testing.assert_array_equal(single.data[0], spectra.data[0])
    y_axis, x_axis = single.spectral_params
    plane = np.zeros(single.data.shape[1:])
    plane[80, 120] = 10
    plane[100, 200] = -10
    paths = contour_paths(plane, single.spectral_params, np.array([5, -5]))
    for contour, (iy, ix) in zip(paths, [(80, 120), (100, 200)], strict=True):
        bounds = contour.boundingRect()
        assert bounds.center().x() == pytest.approx(x_axis.pts2ppm(ix))
        assert bounds.center().y() == pytest.approx(y_axis.pts2ppm(iy))
        assert bounds.width() == pytest.approx(abs(x_axis.delta))
        assert bounds.height() == pytest.approx(abs(y_axis.delta))
    unfolded = y_axis.pts2ppm(y_axis.size + 3.25)
    assert display_position(y_axis, unfolded) == pytest.approx(y_axis.pts2ppm(3.25))
    direct = x_axis.pts2ppm(x_axis.size + 3.25)
    assert display_position(x_axis, direct) == direct
    bad_values = tmp_path / "values.txt"
    bad_values.write_text("42\n")
    np.testing.assert_array_equal(load_spectrum(path, bad_values).z_values, [42])
    bad_values.write_text("1\n2\n")
    with pytest.raises(ValueError, match="exactly 1"):
        load_spectrum(path, bad_values)
