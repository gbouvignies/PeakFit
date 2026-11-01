"""Measure the real retention policy, source-free workflows and interface parity.

The same driver can run against 1051ceb via PYTHONPATH for a measured baseline.
Temporary input copies are deleted, never the bundled source data.
"""

import argparse
import hashlib
import json
import resource
import shutil
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any
from zipfile import ZipFile

import numpy as np
from threadpoolctl import threadpool_limits

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.fit.fitting import write_fit_run_outputs
from peakfit.fit.pipeline import run_pipeline
from peakfit.fit.run_models import FitRun
from peakfit.gui import project
from peakfit.gui.dataset_fit import DatasetFitAttempt, finish_dataset_fit
from peakfit.gui.fit_export import export_fit_result
from peakfit.gui.fit_positions import dataset_completion
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.project_results import verify_model
from peakfit.gui.session_evidence import SessionEvidence
from peakfit.gui.spectrum import read_dataset

ROOT = Path(__file__).resolve().parents[1]


def sizes(path):
    """Logical bytes for every file, independent of filesystem allocation blocks."""
    files = {str(p.relative_to(path)): p.stat().st_size for p in path.rglob("*") if p.is_file()}
    return {"bytes": sum(files.values()), "files": files}


def main():  # noqa: PLR0915
    """Run the real writers and, optionally, a native source-free adjust/refit workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-file", type=Path, required=True)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "project_version": project.VERSION,
        "timings_seconds": {},
        "rss_high_water_bytes": {},
    }

    def timed(name, action):
        args.phase_file.write_text(json.dumps({"phase": name, "stage": "running"}))
        start = perf_counter()
        value = action()
        report["timings_seconds"][name] = perf_counter() - start
        report["rss_high_water_bytes"][name] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value

    with (
        TemporaryDirectory(prefix="peakfit-policy-validation-") as temporary,
        threadpool_limits(limits=1),
    ):
        root = Path(temporary)
        data = ROOT / "examples/02-advanced-fitting/data"
        paths = {}
        for key, filename in (
            ("spectrum", "pseudo3d.ft2"),
            ("peaklist", "pseudo3d.list"),
            ("z_values", "b1_offsets.txt"),
        ):
            paths[key] = root / filename
            shutil.copyfile(data / filename, paths[key])
        dataset = timed("load", lambda: read_dataset(*paths.values()))
        config = PeakFitConfig()
        config.fitting.refine_iterations = 2
        prepared = prepare_dataset_fit(dataset.spectra, dataset.peaks, config)
        prepared = replace(
            prepared, snapshot=replace(prepared.snapshot, source_files=dataset.sources)
        )
        pipeline = timed(
            "fit",
            lambda: run_pipeline(
                config,
                prepared.clusters,
                prepared.noise,
                prepared.parameters,
                [p for c in prepared.clusters for p in c.peaks],
            ),
        )
        result = finish_dataset_fit(
            prepared,
            DatasetFitAttempt(pipeline.completed_fit, (), report["timings_seconds"]["fit"]),
        )
        completion = dataset_completion(
            result, prepared.snapshot, submitted_revision=0, manual_revision=0
        )
        analysis = project.Analysis(
            dataset,
            project.dataset_header(dataset),
            completion.working_peaks,
            config.model_dump_json(),
            0,
            (completion,),
        )
        owner = SessionEvidence()
        try:
            accepted = timed("durable_acceptance", lambda: owner.preserve(analysis))
            report["fit_to_durable_acceptance_seconds"] = (
                report["timings_seconds"]["fit"] + report["timings_seconds"]["durable_acceptance"]
            )
            permanent = timed(
                "first_save", lambda: project.save_analysis(accepted, root / "Analysis.peakfit")
            )
            saved = project.stored_analysis(accepted, permanent)
            timed("routine_save", lambda: project.save_analysis(saved, permanent))
            loaded = timed("verified_reopen", lambda: project.open_analysis(permanent))
            assert loaded.dataset is not None, loaded.reason
            historical = timed("historical_result", lambda: loaded.runs[0].load())
            verify_model(historical.result)
            report["project"] = sizes(permanent)
            identities = tuple(
                {"engine_name": i.engine_name, "peak_id": i.working_id, "assignment": i.assignment}
                for i in prepared.identities
            )
            run = FitRun(
                result.outcome,
                pipeline.continuation_state,
                root / "cli",
                dataset.spectra,
                result.simulation_snapshot,
                result.completed_fit,
                identities,
            )
            timed(
                "cli_default_export",
                lambda: write_fit_run_outputs(run, dataset.spectra, config, paths),
            )
            gui = timed(
                "gui_default_export",
                lambda: export_fit_result(
                    result, root / "gui", completion=completion, current=True
                ),
            )
            report["cli_default"] = sizes(run.output_dir)
            report["gui_default"] = sizes(gui)
            with (
                ZipFile(gui / "scientific.npz") as g,
                ZipFile(run.output_dir / "scientific.npz") as c,
            ):
                scientific = [n for n in g.namelist() if n.startswith(("outcome/", "samples/"))]
                assert all(g.read(n) == c.read(n) for n in scientific)
                assert (
                    json.loads(g.read("metadata.json"))["outcome"]
                    == json.loads(c.read("metadata.json"))["outcome"]
                )
                report["equal_scientific_members"] = len(scientific)
            for name in ("parameters", "intensities", "clusters", "shifts"):
                assert (gui / f"tables/{name}.csv").read_bytes() == (
                    run.output_dir / f"tables/{name}.csv"
                ).read_bytes()
            report["csv_parity"] = True
            if project.VERSION == "3.0.0":
                from peakfit.io.readers.continuation import (  # noqa: PLC0415
                    continuation_state,
                    load_completed,
                )

                native = timed(
                    "compact_continuation", lambda: load_completed(gui / "scientific.npz")
                )
                state = continuation_state(native)
                assert all(
                    a.corrections.tobytes() == b.corrections.tobytes()
                    and a.corrected_data.tobytes() == b.corrected_data.tobytes()
                    for a, b in zip(
                        state.clusters, pipeline.continuation_state.clusters, strict=True
                    )
                )
                report["exact_compact_parent_inputs"] = True
                detailed = timed(
                    "detailed_export",
                    lambda: export_fit_result(
                        result,
                        root / "detailed",
                        completion=completion,
                        current=True,
                        detailed_evidence=True,
                    ),
                )
                report["detailed"] = sizes(detailed)
            # Remove only benchmark-owned input copies. The bundled originals are untouched.
            for path in paths.values():
                path.unlink()
            unlinked = timed("source_free_reopen", lambda: project.open_analysis(permanent))
            report["source_free_spectra"] = unlinked.dataset is not None
            if project.VERSION == "3.0.0":
                assert unlinked.dataset is not None
                assert unlinked.dataset.spectra.data.tobytes() == dataset.spectra.data.tobytes()
                portable = timed(
                    "detailed_continuation_without_sources",
                    lambda: load_completed(detailed / "scientific.npz"),
                )
                report["exact_detailed_parent_inputs"] = all(
                    a.neighbor_corrections.tobytes() == b.neighbor_corrections.tobytes()
                    and a.data.tobytes() == b.data.tobytes()
                    for a, b in zip(portable.samples, result.completed_fit.samples, strict=True)
                )
                if args.native:
                    native_refit(unlinked, args, report, timed)
                else:
                    peaks = list(unlinked.peaks)
                    peaks[0] = replace(
                        peaks[0], positions=(peaks[0].positions[0], peaks[0].positions[1] + 0.00001)
                    )
                    again = prepare_dataset_fit(unlinked.dataset.spectra, tuple(peaks), config)
                    timed(
                        "adjusted_source_free_refit",
                        lambda: run_pipeline(
                            config,
                            again.clusters,
                            again.noise,
                            again.parameters,
                            [p for c in again.clusters for p in c.peaks],
                        ),
                    )
                report["source_free_adjust_refit"] = True
            for a, b in zip(
                result.outcome.clusters, historical.result.outcome.clusters, strict=True
            ):
                if a.analytical_evaluation:
                    for field in (
                        "amplitudes",
                        "shapes",
                        "model_values",
                        "raw_residuals",
                        "normalized_residuals",
                    ):
                        assert (
                            getattr(a.analytical_evaluation, field).tobytes()
                            == getattr(b.analytical_evaluation, field).tobytes()
                        )
            report["exact_scientific_round_trip"] = True
            report["nonzero_corrections"] = sum(
                int(np.count_nonzero(s.neighbor_corrections)) for s in result.completed_fit.samples
            )
            report["outcome_parameter_digest"] = hashlib.sha256(
                repr(
                    tuple(
                        (p.name, p.value, p.standard_error)
                        for p in result.outcome.final_nonlinear_parameters
                    )
                ).encode()
            ).hexdigest()
        finally:
            owner.close()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def native_refit(analysis, args, report, timed):
    """Exercise actual Qt actions with a missing source, including fit publication."""
    from PySide6 import QtCore, QtTest, QtWidgets  # noqa: PLC0415
    from validate_gui_editor import settle, wait_until  # noqa: PLC0415

    from peakfit.gui.editor.window import EditorWindow  # noqa: PLC0415

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert isinstance(app, QtWidgets.QApplication)
    window = EditorWindow()
    window.install_analysis(analysis)
    window.show()
    workspace = window.workspace
    assert workspace is not None
    assert workspace.fitting.current
    wait_until(lambda: not workspace.canvas.contour_pending)
    workspace.set_mode(1)
    wait_until(lambda: workspace.canvas._model is not None and not workspace.canvas.contour_pending)
    settle()
    report["native_model_visible_without_sources"] = True
    workspace.set_mode(2)
    wait_until(lambda: workspace.canvas._model is not None and not workspace.canvas.contour_pending)
    settle()
    report["native_residual_visible_without_sources"] = True
    workspace.set_mode(1)
    wait_until(lambda: workspace.canvas._model is not None and not workspace.canvas.contour_pending)
    settle()
    report["qt_platform"] = app.platformName()
    window.grab().save(str(args.output.with_suffix(".webp")), "WEBP")
    peak = workspace.peaks[0]
    workspace.move_peak(peak.id, (peak.positions[0], peak.positions[1] + 0.00001))
    assert not workspace.fitting.current

    def execute():
        QtTest.QTest.mouseClick(workspace.fit_button, QtCore.Qt.MouseButton.LeftButton)
        wait_until(lambda: not workspace.fit_busy, timeout=240)
        assert workspace.fitting.current, workspace.fitting.failure

    timed("native_adjusted_source_free_refit", execute)
    report["native_current_after_refit"] = workspace.fitting.current
    window._saved_key = window._analysis_key()
    window.close()
    app.processEvents()


if __name__ == "__main__":
    main()
