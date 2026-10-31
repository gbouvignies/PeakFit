"""Working-coordinate invariants without a Qt application."""

from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest

from peakfit.gui.peaks import PeakEdits, WorkingPeak, export_peaks
from peakfit.gui.spectrum import (
    contour_geometry,
    display_position,
    load_peaks,
    load_spectrum,
    paths_from_geometry,
)

DATA = Path(__file__).resolve().parents[1] / "examples" / "02-advanced-fitting" / "data"


def test_edit_history_keeps_identity_validates_atomically_and_tracks_saved_state() -> None:
    peak = WorkingPeak.new("duplicate", (120.1234567890123, 8.23456789012345))
    other = WorkingPeak.new(peak.name, peak.positions)
    edits = PeakEdits((peak, other))
    with pytest.raises(ValueError, match="finite"):
        edits.update(peak.id, "renamed", (float("nan"), 8.0))
    assert edits.peaks == (peak, other)
    assert not edits.dirty
    edits.update(peak.id, "renamed", (121.2345678901234, 8.12345678901234))
    assert edits.peaks[0].id == peak.id
    assert edits.peaks[1] == other
    edits.mark_saved()
    edits.delete(peak.id)
    edits.undo()
    assert not edits.dirty
    edits.undo()
    assert edits.peaks == (peak, other)
    assert edits.dirty
    edits.redo()
    assert not edits.dirty
    edits.add((122.0, 8.0))
    assert not edits.can_redo
    edits.undo()
    assert not edits.dirty


def test_export_round_trips_unfolded_coordinates_assignments_and_empty_list(tmp_path: Path) -> None:
    spectra = load_spectrum(DATA / "pseudo3d.ft2")
    peaks = load_peaks(DATA / "pseudo3d.list", spectra)
    assert len(peaks) == 166
    special = WorkingPeak.new('A, "quoted" assignment', (350.1234567890123, 8.123456789012345))
    special_names = tuple(WorkingPeak.new(name, special.positions) for name in ("NA", "001"))
    exported = (*peaks, special, *special_names)
    path = tmp_path / "edited.csv"
    export_peaks(path, exported)
    imported = load_peaks(path, spectra)
    for original, restored in zip(exported, imported, strict=True):
        assert restored.name == original.name
        assert restored.positions == original.positions
    export_peaks(path, ())
    assert load_peaks(path, spectra) == []


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
    paths = paths_from_geometry(contour_geometry(plane, single.spectral_params, np.array([5, -5])))
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
