"""Position projection, atomic peak edits and retained-result GUI semantics."""

from dataclasses import replace
from zipfile import ZipFile

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from test_gui_fit_execution import _example, _wait

from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.gui.fit_positions import fitted_position_updates
from peakfit.gui.fit_process import FitProcessResult
from peakfit.gui.local_fit import execute_local_fit
from peakfit.gui.peaks import PeakEdits, WorkingPeak
from peakfit.gui.viewer import SpectrumViewer

pytestmark = pytest.mark.gui


@pytest.fixture
def fitted():
    spectra, working, prepared, config = _example()
    return spectra, working, execute_local_fit(prepared), config


def _terminal(result, **changes):
    cluster = replace(result.outcome.clusters[0], **changes)
    return replace(
        result,
        outcome=replace(
            result.outcome, clusters=(cluster,), by_cluster_id={cluster.cluster_id: cluster}
        ),
    )


def test_mapping_unfolded_axes_duplicates_and_atomic_history(fitted):
    _, working, result, _ = fitted
    updates = fitted_position_updates(result, working)
    assert set(updates) == {p.id for p in working}
    # Deliberately reverse identity and parameter iteration: neither is coordinate order.
    reversed_result = replace(
        result, inputs=replace(result.inputs, identities=result.inputs.identities[::-1])
    )
    reversed_result = _terminal(
        reversed_result,
        final_nonlinear_parameters=result.outcome.clusters[0].final_nonlinear_parameters[::-1],
    )
    assert fitted_position_updates(reversed_result, working) == updates
    for identity in result.inputs.identities:
        for dim, shape in enumerate(
            next(p for p in result.inputs.scope.peaks if p.name == identity.engine_name).shapes
        ):
            parameter = next(
                p
                for p in result.outcome.final_nonlinear_parameters
                if p.name == f"{identity.engine_name}.{shape.axis}.cs"
            )
            assert updates[identity.working_id][dim] == parameter.value
    assert updates[working[0].id][0] > 1  # Unfolded indirect coordinate, never wrapped.
    outside = WorkingPeak.new("Unaffected", (0.1, 0.1))
    original = (*working, outside)
    edits = PeakEdits(original)
    edits.update_positions(updates)
    applied = edits.peaks
    assert edits.dirty
    assert applied[-1] is outside
    assert [(p.id, p.name) for p in applied] == [(p.id, p.name) for p in original]
    edits.undo()
    assert edits.peaks == original
    assert not edits.can_undo
    edits.redo()
    assert edits.peaks == applied
    assert not edits.can_redo
    edits.update_positions(updates)
    edits.undo()
    assert edits.peaks == original  # A no-op did not add a history entry.


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -100.0])
def test_invalid_bulk_edit_is_atomic(bad):
    _, working, _, _ = _example()
    edits = PeakEdits(working)
    with pytest.raises(ValueError, match="finite"):
        edits.update_positions({working[0].id: (0.2, 0.3), working[1].id: (bad, float("nan"))})
    assert edits.peaks == working
    assert not edits.can_undo
    with pytest.raises(ValueError, match="IDs"):
        edits.update_positions({working[0].id: (0.2, 0.3), "missing": (0.2, 0.3)})
    assert edits.peaks == working
    assert not edits.can_undo


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "duplicate",
        "nonfinite",
        "bounds",
        "direct",
        "axis",
        "identity",
        "unusable",
        "outdated",
    ],
)
def test_projection_rejects_invalid_results(fitted, invalid):
    _, working, result, _ = fitted
    terminal = result.outcome.clusters[0]
    parameters = list(terminal.final_nonlinear_parameters)
    index = next(i for i, p in enumerate(parameters) if p.name.endswith(".cs"))
    if invalid == "missing":
        parameters.pop(index)
    elif invalid == "duplicate":
        parameters.append(parameters[index])
    elif invalid == "nonfinite":
        parameters[index] = replace(parameters[index], value=np.nan)
    elif invalid == "bounds":
        parameters[index] = replace(parameters[index], value=parameters[index].max + 1)
    elif invalid == "direct":
        index = next(i for i, p in enumerate(parameters) if p.name == "p_0.F3.cs")
        parameters[index] = replace(parameters[index], value=100, min=-np.inf, max=np.inf)
    elif invalid == "axis":
        parameters[index] = replace(parameters[index], name="p_0.F9.cs")
    elif invalid == "identity":
        result = replace(
            result, inputs=replace(result.inputs, identities=(result.inputs.identities[0],) * 2)
        )
    elif invalid == "unusable":
        result = _terminal(result, classification=FitOutcomeClassification.UNUSABLE)
    elif invalid == "outdated":
        working = (replace(working[0], name="changed"), *working[1:])
    result = _terminal(result, final_nonlinear_parameters=tuple(parameters))
    with pytest.raises(ValueError, match=r"changed|identities|usable|shift|parameters|axes"):
        fitted_position_updates(result, working)


def test_fixed_positions_use_final_values_and_noop():
    _, working, prepared, config = _example()
    config.fitting.steps[0].fix = ["*.*.cs"]
    from peakfit.gui.fitting import prepare_local_fit  # noqa: PLC0415

    spectra, _, _, _ = _example()
    prepared = prepare_local_fit(spectra, working, working[0].id, config, contour_level=0.01)
    result = execute_local_fit(prepared)
    assert result.outcome.clusters[0].usable
    assert not fitted_position_updates(result, working)


@pytest.mark.parametrize("exported", [False, True])
def test_apply_confirmation_retains_result_and_export_state(  # noqa: PLR0915
    fitted, monkeypatch, exported, tmp_path
):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    spectra, working, scientific, _ = fitted
    viewer = SpectrumViewer(spectra)
    viewer.set_peaks(working)
    viewer.select_peak(working[0].id)
    viewer.fit_result = retained = FitProcessResult(scientific, 0, 0, 0, 0)
    viewer._result_revision = viewer._scientific_revision
    if exported:
        viewer.exported_result_id = scientific.run_id
    from peakfit.gui.fit_export import export_fit_result  # noqa: PLC0415

    original_export = export_fit_result(scientific, tmp_path / "before")
    viewer.show_fit_result()
    viewer.show()
    app.processEvents()
    selection = viewer.selected_id
    viewport = viewer.spectrum_plot.viewRange()
    amplitudes = scientific.outcome.clusters[0].analytical_evaluation.amplitudes.copy()
    dialogs = []

    def confirm(dialog):
        dialogs.append(dialog)
        assert "2 peaks" in dialog.text()
        assert "Undo" in dialog.informativeText()
        assert "Indirect:" in dialog.informativeText()
        assert "Direct:" in dialog.informativeText()
        return QtWidgets.QMessageBox.StandardButton.Apply

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", confirm)
    try:
        assert viewer.fit_preview.apply_positions.isEnabled()
        viewer.apply_fitted_positions()
        assert dialogs
        applied = viewer.peaks
        assert applied != working
        assert viewer.edits.dirty
        assert viewer.fit_result is retained
        assert viewer.result_exported == exported
        assert viewer.selected_id == selection
        assert viewer.spectrum_plot.viewRange() == viewport
        assert not viewer.comparison.active
        assert "graphics are hidden" in viewer.fit_preview.note.text()
        assert not viewer.fit_preview.apply_positions.isEnabled()
        np.testing.assert_array_equal(
            amplitudes, scientific.outcome.clusters[0].analytical_evaluation.amplitudes
        )
        after_export = export_fit_result(scientific, tmp_path / "after")
        with (
            ZipFile(after_export / "scientific.npz") as after,
            ZipFile(original_export / "scientific.npz") as before,
        ):
            assert {n: after.read(n) for n in after.namelist()} == {
                n: before.read(n) for n in before.namelist()
            }
        assert (after_export / "tables/intensities.csv").read_bytes() == (
            original_export / "tables/intensities.csv"
        ).read_bytes()
        viewer.undo()
        assert viewer.peaks == working
        assert not viewer.edits.can_undo
        viewer.redo()
        assert viewer.peaks == applied
        assert viewer.fit_result is retained
        assert viewer.result_exported == exported
        with pytest.raises(ValueError, match="outdated"):
            viewer._fitted_position_updates()
    finally:
        _wait(lambda: not viewer._computation._busy)
        viewer.stop_preparation()
        viewer.close()
        viewer.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()


def test_live_guards_and_modal_revalidation(fitted, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    spectra, working, scientific, _ = fitted
    viewer = SpectrumViewer(spectra)
    viewer.set_peaks(working)
    viewer.fit_result = FitProcessResult(scientific, 0, 0, 0, 0)
    viewer._result_revision = viewer._scientific_revision
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: errors.append(args[-1]))
    try:
        monkeypatch.setattr(SpectrumViewer, "fit_busy", property(lambda _: True))
        with pytest.raises(ValueError, match="current fit"):
            viewer._fitted_position_updates()
        monkeypatch.setattr(SpectrumViewer, "fit_busy", property(lambda _: False))
        spectra.z_values[0] += 1
        with pytest.raises(ValueError, match="Scientific inputs changed"):
            viewer._fitted_position_updates()
        spectra.z_values[0] -= 1
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "exec", lambda _: QtWidgets.QMessageBox.StandardButton.Cancel
        )
        viewer.apply_fitted_positions()
        assert viewer.peaks == working
        assert not viewer.edits.can_undo

        def change_during_confirmation(_):
            viewer._scientific_revision += 1
            return QtWidgets.QMessageBox.StandardButton.Apply

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", change_during_confirmation)
        viewer.apply_fitted_positions()
        assert errors
        assert "outdated" in errors[-1]
        assert viewer.peaks == working
        assert not viewer.edits.can_undo
    finally:
        _wait(lambda: not viewer._computation._busy)
        viewer.stop_preparation()
        viewer.close()
        viewer.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()


def test_custom_axis_labels_follow_coordinate_order():
    spectra, working, _, config = _example()
    # Labels deliberately sort in the opposite order to working coordinates.
    spectra.params[1].label = "F3"
    spectra.params[2].label = "F2"
    from peakfit.gui.fitting import prepare_local_fit  # noqa: PLC0415

    prepared = prepare_local_fit(spectra, working, working[0].id, config, contour_level=0.01)
    result = execute_local_fit(prepared)
    updates = fitted_position_updates(result, working)
    final = {p.name: p.value for p in result.outcome.final_nonlinear_parameters}
    for identity in prepared.identities:
        assert updates[identity.working_id] == (
            final[f"{identity.engine_name}.F3.cs"],
            final[f"{identity.engine_name}.F2.cs"],
        )


def test_unreliable_uncertainties_block_application_but_keep_result(fitted):
    _, working, result, _ = fitted
    evaluation = result.outcome.clusters[0].analytical_evaluation
    assert evaluation is not None
    uncertain = _terminal(
        result,
        analytical_evaluation=replace(evaluation, uncertainty_warning="Unidentifiable peak group"),
    )
    with pytest.raises(ValueError, match="Unidentifiable"):
        fitted_position_updates(uncertain, working)
    assert uncertain.outcome.clusters[0].usable
    np.testing.assert_array_equal(
        uncertain.outcome.clusters[0].analytical_evaluation.model_values, evaluation.model_values
    )
