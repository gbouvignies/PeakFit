"""Coordinate provenance and scientific adoption remain explicit in shared exports."""

import csv
import json
from dataclasses import replace
from zipfile import ZipFile

import numpy as np
import pytest
from test_gui_dataset import _dataset
from test_scientific_storage import completed_result

from peakfit.fit import fitting, pipeline
from peakfit.fit.output_metadata import RunMetadata
from peakfit.fit.run_models import FitRun
from peakfit.fit.storage import read_scientific
from peakfit.gui.dataset_fit import DatasetFitAttempt, finish_dataset_fit
from peakfit.gui.fit_export import export_fit_result
from peakfit.gui.fit_positions import dataset_completion


def test_export_retains_initial_fitted_and_adopted_coordinates(tmp_path):
    _spectra, _config, _pipeline, result = completed_result(["csv"], False)
    completion = dataset_completion(
        result, result.inputs.snapshot, submitted_revision=0, manual_revision=0
    )
    path = export_fit_result(result, tmp_path / "result", completion=completion)
    _, _, context = read_scientific(path / "scientific.npz")
    table = context["coordinate_table"]
    assert table["run_id"] == result.run_id
    assert table["schema_version"] == "1.0.0"
    for peak in result.inputs.snapshot.peaks:
        row = table["peaks"][peak.id]
        np.testing.assert_array_equal(row["initial_unfolded_ppm"], peak.positions)
        assert row["adoption_status"] == "adopted_fitted"
        adopted = next(p for p in completion.working_peaks if p.id == peak.id)
        np.testing.assert_array_equal(row["adopted_unfolded_ppm"], adopted.positions)
        np.testing.assert_array_equal(row["fitted_unfolded_ppm"], adopted.positions)


def test_intensity_csv_has_both_plane_identities(tmp_path):
    _spectra, _config, _pipeline, result = completed_result(["csv"], False)
    path = export_fit_result(result, tmp_path / "result")
    with (path / "tables/intensities.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["source_plane_index"]) for r in rows[:3]] == [0, 1, 2]
    assert [int(r["plane_index"]) for r in rows[:3]] == [0, 1, 2]
    assert [float(r["z_value"]) for r in rows[:3]] == [2.0, 0.0, 2.0]


def _completed_case(mode, monkeypatch):
    spectra, _peaks, config, prepared = _dataset()
    config.output.formats = ["csv"]
    if mode == "non_converged":
        config.fitting.max_iterations = 1
    prepared = replace(
        prepared, snapshot=replace(prepared.snapshot, configuration=config.model_dump_json())
    )
    if mode == "partially_unusable":
        actual_fit = pipeline.fit_single_cluster_task
        unusable_id = prepared.clusters[0].cluster_id

        def fit(task):
            result = actual_fit(task)
            if task.cluster_id == unusable_id:
                result.cost = np.nan
            return result

        monkeypatch.setattr(pipeline, "fit_single_cluster_task", fit)
    completed = pipeline.run_pipeline(
        config,
        prepared.clusters,
        prepared.noise,
        prepared.parameters,
        [p for c in prepared.clusters for p in c.peaks],
    )
    result = finish_dataset_fit(prepared, DatasetFitAttempt(completed.completed_fit, (), 1.0))
    completion = dataset_completion(
        result, result.inputs.snapshot, submitted_revision=0, manual_revision=0
    )
    return spectra, config, prepared, completed, result, completion


@pytest.mark.parametrize("mode", ["usable", "partially_unusable", "non_converged", "outdated"])
def test_shared_coordinate_provenance_and_actual_adoption(tmp_path, monkeypatch, mode):
    spectra, config, prepared, completed, result, completion = _completed_case(mode, monkeypatch)
    if mode == "non_converged":
        assert any(
            c.classification.value == "usable_non_converged" for c in result.outcome.clusters
        )
    if mode == "partially_unusable":
        assert any(c.usable for c in result.outcome.clusters)
        assert any(not c.usable for c in result.outcome.clusters)
        assert completion.warnings
    path = export_fit_result(
        result, tmp_path / "gui", completion=completion, current=mode != "outdated"
    )
    restored, _samples, context = read_scientific(path / "scientific.npz")
    assert context["export_state"]["currentness"] == (
        "outdated" if mode == "outdated" else "current"
    )
    assert context["export_state"]["adoption_warnings"] == list(completion.warnings)
    rows = context["coordinate_table"]["peaks"]
    _assert_coordinate_rows(rows, prepared, completion, restored)
    metadata = RunMetadata(
        run_id=result.run_id,
        timestamp=result.completed_at,
        configuration=json.loads(result.inputs.snapshot.configuration),
        run_duration_seconds=result.execution_seconds,
    )
    monkeypatch.setattr(fitting, "capture_output_metadata", lambda *_args: metadata)
    run = FitRun(
        result.outcome,
        completed.continuation_state,
        tmp_path / "cli",
        spectra,
        result.simulation_snapshot,
        result.completed_fit,
        tuple(
            {"engine_name": i.engine_name, "peak_id": i.working_id, "assignment": i.assignment}
            for i in result.inputs.identities
        ),
    )
    fitting.write_fit_run_outputs(run, spectra, config, {})
    cli_outcome, _, cli_context = read_scientific(tmp_path / "cli/scientific.npz")
    assert cli_context["export_state"] == {
        "scope": "dataset",
        "currentness": "not_applicable",
        "adoption_warnings": [],
    }
    for peak_id, row in cli_context["coordinate_table"]["peaks"].items():
        gui_row = rows[peak_id]
        for name in ("initial_unfolded_ppm", "fitted_unfolded_ppm"):
            np.testing.assert_array_equal(row[name], gui_row[name])
        assert row["parameter_names"] == gui_row["parameter_names"]
        assert row["adoption_status"] == "not_recorded"
        assert row["adopted_unfolded_ppm"] is None
    assert cli_outcome.final_nonlinear_parameters == result.outcome.final_nonlinear_parameters
    if mode == "outdated":
        # Only export-time state changes; immutable evidence and completion are untouched.
        current_path = export_fit_result(
            result, tmp_path / "current", completion=completion, current=True
        )

        with (
            ZipFile(path / "scientific.npz") as old,
            ZipFile(current_path / "scientific.npz") as current,
        ):
            for name in old.namelist():
                if name != "metadata.json":
                    assert old.read(name) == current.read(name)
                else:
                    a, b = json.loads(old.read(name)), json.loads(current.read(name))
                    a["context"]["mapping"].pop("export_state")
                    b["context"]["mapping"].pop("export_state")
                    assert a == b


def _assert_coordinate_rows(rows, prepared, completion, restored):
    for identity in completion.result.inputs.identities:
        row = rows[identity.working_id]
        seed = next(p for p in prepared.snapshot.peaks if p.id == identity.working_id)
        accepted = next(p for p in completion.working_peaks if p.id == identity.working_id)
        assert row["engine_name"] == identity.engine_name
        assert row["assignment"] == identity.assignment
        assert row["parameter_names"] == [
            f"{identity.engine_name}.{axis}.cs" for axis in row["axes"]
        ]
        np.testing.assert_array_equal(row["initial_unfolded_ppm"], seed.positions)
        np.testing.assert_array_equal(row["adopted_unfolded_ppm"], accepted.positions)
        terminal = restored.cluster(row["cluster_id"])
        if terminal.usable:
            assert row["adoption_status"] == "adopted_fitted"
            final = {p.name: p.value for p in terminal.final_nonlinear_parameters}
            np.testing.assert_array_equal(
                row["fitted_unfolded_ppm"], [final[name] for name in row["parameter_names"]]
            )
        else:
            assert row["fitted_unfolded_ppm"] is None
            assert row["adoption_status"] == "retained_initial"
            assert row["adoption_reason"] == "unusable cluster"
            np.testing.assert_array_equal(row["adopted_unfolded_ppm"], seed.positions)


def test_coordinate_table_rejects_false_adoption(tmp_path):
    _spectra, _config, _pipeline, result = completed_result([], False)
    completion = dataset_completion(
        result, result.inputs.snapshot, submitted_revision=0, manual_revision=0
    )
    wrong = (
        replace(completion.working_peaks[0], positions=(0.0, 0.0)),
        *completion.working_peaks[1:],
    )
    with pytest.raises(ValueError, match="adoption disagrees"):
        export_fit_result(
            result, tmp_path / "wrong", completion=replace(completion, working_peaks=wrong)
        )
    assert not (tmp_path / "wrong").exists()


def test_coordinate_table_rejects_falsified_parameter_mapping(tmp_path):
    from peakfit.fit.storage import write_scientific  # noqa: PLC0415

    _spectra, _config, _pipeline, result = completed_result([], False)
    path = export_fit_result(result, tmp_path / "original")
    outcome, samples, context = read_scientific(path / "scientific.npz")
    row = next(iter(context["coordinate_table"]["peaks"].values()))
    row["parameter_names"][0] = "unrelated.F9.cs"
    with pytest.raises(ValueError, match="Coordinate table resolution"):
        write_scientific(tmp_path / "broken.npz", outcome, samples, context=context)
