"""Scientific preparation and one end-to-end scope preview, without optimization."""

from dataclasses import replace
from pathlib import Path
from threading import Event, get_ident

import numpy as np
import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from peakfit.engine.algorithms.clustering import create_clusters
from peakfit.engine.algorithms.noise import estimate_noise, prepare_noise_level
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.spectrum import get_shape_names
from peakfit.gui.fitting import prepare_local_fit
from peakfit.gui.spectrum import contour_geometry, load_peaks, load_spectrum
from peakfit.gui.viewer import SpectrumViewer
from peakfit.io.readers.peaks import read_list

DATA = Path(__file__).resolve().parents[1] / "examples" / "02-advanced-fitting" / "data"
pytestmark = pytest.mark.gui


def _process_events(milliseconds: int = 10) -> None:
    # Unlike QtTest.qWait, exec releases the GIL so Python workers can progress.
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


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
    contour_level = 5 * estimate_noise(spectra.data[0])
    groups = create_clusters(spectra, reference_peaks, contour_level)
    high_groups = create_clusters(spectra, reference_peaks, 1000 * estimate_noise(spectra.data[0]))
    high_sizes = {peak.name: len(group.peaks) for group in high_groups for peak in group.peaks}
    reference = next(
        group for group in groups if high_sizes[group.peaks[0].name] < len(group.peaks)
    )
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
    prepared = prepare_local_fit(spectra, peaks, selected.id, contour_level=contour_level)
    assert prepared.cluster is not None
    assert {item.working_id for item in prepared.identities} == expected_ids
    assert len({item.engine_name for item in prepared.identities}) == len(expected_ids)
    assert sum(item.assignment == "Duplicate" for item in prepared.identities) == 2
    for actual, expected in zip(prepared.cluster.grid_indices, reference.grid_indices, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(prepared.cluster.data, reference.data)
    assert prepared.cluster.n_series == spectra.data.shape[0]
    assert prepared.noise == noise
    assert prepared.contour_level == contour_level
    assert prepared.snapshot.contour_level == contour_level
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
        prepare_local_fit(spectra, (outside, *peaks[1:]), outside.id, contour_level=contour_level)

    viewer = SpectrumViewer(spectra)
    viewer.set_peaks(peaks)
    viewer.select_peak(selected.id)
    viewer.resize(1400, 900)
    viewer.show()
    app.processEvents()
    elapsed = QtCore.QElapsedTimer()
    elapsed.start()
    while not viewer.contours and elapsed.elapsed() < 4000:
        _process_events(10)

    def wait_for_preview():
        elapsed.restart()
        while viewer.fit_preparation is None and elapsed.elapsed() < 4000:
            _process_events(10)
        assert viewer.fit_preparation is not None, viewer.fit_preview.summary.text()
        assert viewer.fit_preparation.parameters is not None
        return viewer.fit_preparation

    # A preview must not invoke contour extraction or optimization.
    def unexpected(*_args, **_kwargs):
        pytest.fail("Scope preview must not rebuild contours or execute fitting")

    monkeypatch.setattr("peakfit.gui.viewer.contour_geometry", unexpected)
    monkeypatch.setattr("peakfit.fit.pipeline.fit_with_optimizer", unexpected)
    paths = [item.path() for item in viewer.contours]
    plots = (viewer.trace_x, viewer.trace_y)

    def plot_positions():
        return tuple((plot.mapTo(viewer, QtCore.QPoint()), plot.size()) for plot in plots)

    positions = plot_positions()
    spectrum_geometry = viewer.spectrum_plot.geometry()
    assert not viewer.threshold.isVisible()
    assert "5.0×" in viewer.contour_button.text()
    title_button = viewer.contour_button.geometry()
    data_top = viewer.spectrum_plot.mapFromScene(
        viewer.spectrum_plot.getViewBox().sceneBoundingRect().topLeft()
    ).y()
    assert title_button.bottom() <= data_top
    # Contour controls are transient, precise and keyboard accessible; opening them
    # never reflows either the spectrum or inspection plots.
    QtTest.QTest.mouseClick(viewer.contour_button, QtCore.Qt.MouseButton.LeftButton)
    app.processEvents()
    assert viewer.contour_popup.isVisible()
    assert viewer.threshold.hasFocus()
    assert viewer.threshold.isVisible()
    assert viewer.spectrum_plot.geometry() == spectrum_geometry
    assert plot_positions() == positions
    QtTest.QTest.keyClick(viewer.threshold, QtCore.Qt.Key.Key_Escape)
    assert not viewer.contour_popup.isVisible()
    try:
        viewer.workspace.fit.click()
        wait_for_preview()
        app.processEvents()
        assert viewer.fit_preparation is not None
        assert viewer.fit_preparation.parameters is not None
        assert all(plot.isVisible() for plot in plots)
        assert viewer.workspace.details.isVisible()
        assert viewer.fit_preview.detail_content.isVisible()
        viewer.close_details()
        app.processEvents()
        assert all(plot.isVisible() for plot in plots)
        assert plot_positions() == positions
        viewer.open_details(1)
        assert viewer._fit_context
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
            assert viewer._fit_context
            outsider = next(peak for peak in peaks if peak.id not in expected_ids)
            viewer.select_peak(outsider.id, inspect=True)
            assert viewer.selected_id == outsider.id
            assert viewer.fit_preparation is None
            assert not viewer._fit_context
            assert not viewer.fit_mask.isVisible()
            assert len(viewer.fit_members.points()) == 0
        viewer.preview_fit_group()
        viewer.select_peak(None)
        assert viewer.selected_id is None
        assert viewer.fit_preparation is None
        assert not viewer._fit_context
        assert not viewer.fit_mask.isVisible()
        viewer.select_peak(selected.id)
        viewer.preview_fit_group()
        wait_for_preview()
        assert viewer.fit_preparation is not None
        # Changing plane labels invalidates the view, but cannot mutate its snapshot.
        snapshot = viewer.fit_preparation.snapshot
        viewer.set_plane_values(spectra.z_values + 1)
        assert viewer.fit_preparation is None
        assert snapshot.plane_values != tuple(spectra.z_values)
        assert not viewer.fit_mask.isVisible()
        viewer.preview_fit_group()
        baseline = wait_for_preview()
        assert baseline is not None
        calls = []

        def record_preparation(*args, **kwargs):
            calls.append(kwargs["contour_level"])
            return prepare_local_fit(*args, **kwargs)

        monkeypatch.setattr("peakfit.gui.viewer.prepare_local_fit", record_preparation)
        monkeypatch.setattr("peakfit.gui.viewer.contour_geometry", contour_geometry)
        monkeypatch.setattr("peakfit.engine.algorithms.noise.estimate_noise", unexpected)
        # Display-only controls keep the same scientific preparation.
        viewer.factor.setValue(1.7)
        viewer.levels.setValue(11)
        viewer.set_plane(2)
        _process_events(200)
        assert viewer.fit_preparation is baseline
        assert not calls
        # A burst of slider changes produces one preparation at the final absolute
        # base intensity. Statistical noise is reused rather than re-estimated.
        for value in (390, 395, 400):
            viewer.threshold_slider.setValue(value)
        assert viewer.fit_preparation is None
        assert viewer.fit_mask.isVisible()
        assert not viewer.fit_preview.parameters.isEnabled()
        _process_events(300)
        updated = wait_for_preview()
        assert updated is not None
        assert calls == [viewer.lowest_contour_level]
        assert (
            updated.contour_level == updated.snapshot.contour_level == viewer.lowest_contour_level
        )
        effective = PeakFitConfig.model_validate_json(updated.snapshot.configuration)
        assert effective.clustering.contour_level == viewer.lowest_contour_level
        assert updated.noise == baseline.noise == noise
        reference_groups = create_clusters(spectra, reference_peaks, viewer.lowest_contour_level)
        reference_group = next(
            g for g in reference_groups if selected.name in {p.name for p in g.peaks}
        )
        by_name = {p.name: w.id for p, w in zip(reference_peaks, peaks, strict=True)}
        updated_ids = {by_name[p.name] for p in reference_group.peaks}
        assert {member.working_id for member in updated.identities} == updated_ids
        assert updated_ids != expected_ids
        for actual, expected in zip(
            updated.cluster.grid_indices, reference_group.grid_indices, strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
        expected_mask[:] = False
        expected_mask[tuple(reference_group.grid_indices)] = True
        np.testing.assert_array_equal(viewer.fit_mask.image[..., 3] > 0, expected_mask)
        assert len(viewer.fit_members.points()) == len(updated_ids)
        assert viewer.fit_preview.parameters.rowCount() == len(updated.parameters)
        assert viewer.fit_preview.parameters.isEnabled()
        viewer.dismiss_fit_preview()
        assert viewer.peaks == peaks
        np.testing.assert_array_equal(spectra.data, original_data)
    finally:
        viewer.close()
        viewer.deleteLater()
        app.processEvents()


def test_low_threshold_preparation_keeps_qt_responsive_and_rejects_obsolete_work(  # noqa: PLR0915
    monkeypatch,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    spectra = load_spectrum(DATA / "pseudo3d.ft2")
    viewer = SpectrumViewer(spectra)
    viewer.set_peaks(load_peaks(DATA / "pseudo3d.list", spectra))
    viewer.select_peak(viewer.peaks[0].id)
    viewer.show()
    entered, release = Event(), Event()
    blocked = [True]
    threads = []
    heartbeats = []
    heartbeat = QtCore.QTimer(viewer)
    heartbeat.setInterval(10)
    heartbeat.timeout.connect(lambda: heartbeats.append(1))
    heartbeat.start()

    def delayed_preparation(*args, **kwargs):
        threads.append(get_ident())
        if blocked[0]:
            entered.set()
            release.wait(2)
        return prepare_local_fit(*args, **kwargs)

    def wait_until(condition):
        elapsed = QtCore.QElapsedTimer()
        elapsed.start()
        while not condition() and elapsed.elapsed() < 4000:
            _process_events()
        assert condition()

    monkeypatch.setattr("peakfit.gui.viewer.prepare_local_fit", delayed_preparation)
    try:
        viewer.preview_fit_group()
        wait_until(entered.is_set)
        _process_events(60)
        assert len(heartbeats) >= 3
        assert len(threads) == 1
        assert threads[0] != get_ident()
        # One running computation; selection/slider changes replace pending work.
        newer = viewer.peaks[-1]
        viewer.select_peak(newer.id)
        assert not viewer._fit_context
        viewer.preview_fit_group()
        viewer.threshold.setValue(1000)
        _process_events(200)
        assert len(threads) == 1
        blocked[0] = False
        release.set()
        wait_until(lambda: viewer.fit_preparation is not None)
        current = viewer.fit_preparation
        assert current is not None
        assert {member.working_id for member in current.identities} == {newer.id}
        assert current.contour_level == viewer.lowest_contour_level
        assert len(set(threads)) == 1
        # Large numerical inputs do not prevent viewing exact lightweight scope
        # geometry or the complete low-threshold contours. Status never reflows UI.
        geometry = (
            viewer.spectrum_plot.geometry(),
            viewer.threshold_slider.geometry(),
            viewer.fit_preview.parameters.geometry(),
        )
        button_geometry = viewer.contour_button.geometry()
        viewer.threshold.setValue(0.1)
        assert "0.1×" in viewer.contour_button.text()
        assert viewer.contour_button.geometry() == button_geometry
        assert any(item.isVisible() for item in viewer.contours)
        _process_events()
        assert (
            viewer.spectrum_plot.geometry(),
            viewer.threshold_slider.geometry(),
            viewer.fit_preview.parameters.geometry(),
        ) == geometry
        wait_until(lambda: viewer.fit_preparation is not None)
        wait_until(lambda: viewer.contour_status.text() == "")
        large = viewer.fit_preparation
        assert large is not None
        assert large.cluster is None
        assert large.parameters is None
        assert large.limitation is not None
        assert "279 MiB" in large.limitation
        assert large.scope.n_points == np.prod(spectra.data.shape[1:])
        assert len(large.identities) == len(viewer.peaks)
        assert viewer.fit_mask.isVisible()
        assert len(viewer.fit_members.points()) == len(viewer.peaks)
        assert (
            sum(item.path().elementCount() for item in viewer.contours if item.isVisible())
            > 100_000
        )
        assert (
            viewer.spectrum_plot.geometry(),
            viewer.threshold_slider.geometry(),
            viewer.fit_preview.parameters.geometry(),
        ) == geometry
        # Closing never waits for a running operation or installs its late result.
        blocked[0] = True
        entered.clear()
        release.clear()
        viewer.threshold.setValue(1000)
        viewer.preview_fit_group()
        wait_until(entered.is_set)
        viewer.close()
        release.set()
        _process_events(100)
    finally:
        release.set()
        viewer.close()
        viewer.deleteLater()
        app.processEvents()
