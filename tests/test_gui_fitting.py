"""Scientific preparation and one end-to-end scope preview, without optimization."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from peakfit.engine.algorithms.clustering import create_clusters
from peakfit.engine.algorithms.noise import prepare_noise_level
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.spectrum import get_shape_names
from peakfit.gui.fitting import prepare_local_fit
from peakfit.gui.spectrum import load_peaks, load_spectrum
from peakfit.gui.viewer import SpectrumViewer
from peakfit.io.readers.peaks import read_list

DATA = Path(__file__).resolve().parents[1] / "examples" / "02-advanced-fitting" / "data"
pytestmark = pytest.mark.gui


def test_all_plane_region_identity_aliasing_and_read_only_preview(monkeypatch) -> None:  # noqa: PLR0915
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    spectra = load_spectrum(DATA / "pseudo3d.ft2", DATA / "b1_offsets.txt")
    original_data = spectra.data.copy()
    peaks = tuple(load_peaks(DATA / "pseudo3d.list", spectra))
    config = PeakFitConfig()
    # Use the existing reader + engine clustering as the reference, independently
    # of the GUI's identity conversion. Select a real overlapping group.

    reference_peaks = read_list(
        DATA / "pseudo3d.list", spectra, get_shape_names(config.fitting, spectra), config.fitting
    )

    noise = prepare_noise_level(None, spectra)
    groups = create_clusters(spectra, reference_peaks, 5 * noise)
    reference = next(group for group in groups if len(group.peaks) > 1)
    selected = next(peak for peak in peaks if peak.name == reference.peaks[0].name)
    expected_ids = {peak.id for peak in peaks if peak.name in {p.name for p in reference.peaks}}
    # Rename two members to the same assignment, preserving separate identities;
    # unfold one indirect position by exactly a spectral width.
    y_axis, x_axis = spectra.spectral_params
    duplicate_id = next(identity for identity in expected_ids if identity != selected.id)
    peaks = tuple(
        replace(
            peak,
            name="Duplicate",
            positions=(peak.positions[0] + y_axis.delta * y_axis.size, peak.positions[1]),
        )
        if peak.id == selected.id
        else replace(peak, name="Duplicate")
        if peak.id == duplicate_id
        else peak
        for peak in peaks
    )
    prepared = prepare_local_fit(spectra, peaks, selected.id)
    assert {item.working_id for item in prepared.identities} == expected_ids
    assert len({item.engine_name for item in prepared.identities}) == len(expected_ids)
    assert sum(item.assignment == "Duplicate" for item in prepared.identities) == 2
    for actual, expected in zip(prepared.cluster.grid_indices, reference.grid_indices, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(prepared.cluster.data, reference.data)
    assert prepared.cluster.n_series == spectra.data.shape[0]
    assert prepared.noise == noise
    assert prepared.contour_level == 5 * noise
    identity = next(item for item in prepared.identities if item.working_id == selected.id)
    scientific = next(peak for peak in prepared.cluster.peaks if peak.name == identity.engine_name)
    working = next(peak for peak in peaks if peak.id == selected.id)
    np.testing.assert_array_equal(scientific.positions, working.positions)
    # The engine's indirect alias sign must be retained as well as folded indices.
    original = reference.peaks[0].shapes[0]
    offsets, signs = scientific.shapes[0].dim_ctx.compute_offsets(
        np.array([0, y_axis.size - 1]), np.array([working.positions[0]])
    )
    old_offsets, old_signs = original.dim_ctx.compute_offsets(
        np.array([0, y_axis.size - 1]), np.array([selected.positions[0]])
    )
    np.testing.assert_allclose(offsets, old_offsets)
    np.testing.assert_array_equal(signs, -old_signs if y_axis.p180 else old_signs)
    outside = replace(peaks[0], positions=(peaks[0].positions[0], x_axis.pts2ppm(x_axis.size)))
    with pytest.raises(ValueError, match="direct-axis position"):
        prepare_local_fit(spectra, (outside, *peaks[1:]), outside.id)

    viewer = SpectrumViewer(spectra)
    viewer.set_peaks(peaks)
    viewer.select_peak(selected.id)
    viewer.resize(1400, 900)
    viewer.show()
    app.processEvents()

    # A preview must not invoke contour extraction or optimization.
    def unexpected(*_args, **_kwargs):
        pytest.fail("Scope preview must not rebuild contours or execute fitting")

    monkeypatch.setattr("peakfit.gui.viewer.contour_paths", unexpected)
    monkeypatch.setattr("peakfit.fit.pipeline.fit_with_optimizer", unexpected)
    paths = [item.path() for item in viewer.contours]
    try:
        viewer.workspace_tabs.setCurrentIndex(1)
        app.processEvents()
        assert viewer.fit_preparation is not None
        assert viewer.workspaces.currentIndex() == viewer.workspace_tabs.currentIndex() == 1
        assert viewer.fit_preview.parameters.rowCount() == len(viewer.fit_preparation.parameters)
        assert len(viewer.fit_members.points()) == len(expected_ids)
        assert viewer.fit_mask.image is not None
        mask = viewer.fit_mask.image[..., 3] > 0
        expected_mask = np.zeros(spectra.data.shape[1:], dtype=bool)
        expected_mask[tuple(reference.grid_indices)] = True
        np.testing.assert_array_equal(mask, expected_mask)
        iy, ix = (indices[0] for indices in reference.grid_indices)
        point = viewer.fit_mask.mapToParent(QtCore.QPointF(ix + 0.5, iy + 0.5))
        assert point.x() == pytest.approx(x_axis.pts2ppm(ix))
        assert point.y() == pytest.approx(y_axis.pts2ppm(iy))
        viewer.spectrum_plot.getViewBox().setRange(xRange=(7.5, 9), yRange=(115, 120))
        assert viewer.fit_mask.isVisible()
        assert paths == [item.path() for item in viewer.contours]
        assert viewer.peaks == peaks
        assert not viewer.edits.dirty
        assert not viewer.edits.can_undo
        # In-group selection preserves the exact preparation without segmentation;
        # outside selection and deselection return to ordinary inspection.
        group_preparation = viewer.fit_preparation
        with monkeypatch.context() as selection_patch:
            selection_patch.setattr("peakfit.gui.viewer.prepare_local_fit", unexpected)
            viewer.select_peak(duplicate_id, inspect=True)
            assert viewer.selected_id == duplicate_id
            assert viewer.fit_preparation is group_preparation
            assert viewer.workspaces.currentIndex() == 1
            outsider = next(peak for peak in peaks if peak.id not in expected_ids)
            viewer.select_peak(outsider.id, inspect=True)
            assert viewer.selected_id == outsider.id
            assert viewer.fit_preparation is None
            assert viewer.workspaces.currentIndex() == viewer.workspace_tabs.currentIndex() == 0
            assert not viewer.fit_mask.isVisible()
            assert len(viewer.fit_members.points()) == 0
        viewer.preview_fit_group()
        viewer.select_peak(None)
        assert viewer.selected_id is None
        assert viewer.fit_preparation is None
        assert viewer.workspaces.currentIndex() == 0
        assert not viewer.fit_mask.isVisible()
        viewer.select_peak(selected.id)
        viewer.preview_fit_group()
        assert viewer.fit_preparation is not None
        # Changing plane labels invalidates the view, but cannot mutate its snapshot.
        snapshot = viewer.fit_preparation.snapshot
        viewer.set_plane_values(spectra.z_values + 1)
        assert viewer.fit_preparation is None
        assert snapshot.plane_values != tuple(spectra.z_values)
        assert not viewer.fit_mask.isVisible()
        viewer.preview_fit_group()
        viewer.dismiss_fit_preview()
        assert viewer.peaks == peaks
        np.testing.assert_array_equal(spectra.data, original_data)
    finally:
        viewer.close()
        viewer.deleteLater()
        app.processEvents()
