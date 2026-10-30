"""Measure linked/portable project workflows on real data, including descendant RSS."""

import argparse
import gc
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.gui import project
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.fit_positions import dataset_completion
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.project_process import save_project
from peakfit.gui.project_results import verify_model
from peakfit.gui.session_evidence import SessionEvidence
from peakfit.gui.spectrum import read_dataset

ROOT = Path(__file__).resolve().parents[1]


def worker(args):  # noqa: PLR0915
    """Run real project operations; also runs against merged schema-3 code as baseline."""
    report: dict[str, Any] = {
        "schema": project.VERSION,
        "platform": platform.platform(),
        "seconds": {},
    }

    def timed(name, action):
        gc.collect()
        args.phase.write_text(name)
        start = perf_counter()
        value = action()
        report["seconds"][name] = perf_counter() - start
        return value

    options: dict[str, Any] = {
        "progress": lambda _: None,
        "canceled": lambda: False,
        "reusable": True,
    }
    supports_modes = "portable" in inspect.signature(save_project).parameters
    with TemporaryDirectory(prefix="peakfit-portability-data-") as temporary, threadpool_limits(1):
        root = Path(temporary)
        source = root / "spectrum.ft2"
        data = ROOT / "examples/02-advanced-fitting/data"
        shutil.copyfile(data / "pseudo3d.ft2", source)
        dataset = timed(
            "load", lambda: read_dataset(source, data / "pseudo3d.list", data / "b1_offsets.txt")
        )
        config = PeakFitConfig()
        config.fitting.refine_iterations = 2
        prepared = prepare_dataset_fit(dataset.spectra, dataset.peaks, config)
        prepared = replace(
            prepared, snapshot=replace(prepared.snapshot, source_files=dataset.sources)
        )
        result = timed("fit", lambda: finish_dataset_fit(prepared, compute_dataset_fit(prepared)))
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
            accepted = timed("durable_publication", lambda: owner.preserve(analysis))
            path, saved, _ = timed(
                "first_save", lambda: save_project(accepted, root / "Analysis.peakfit", **options)
            )
            timed("routine_save", lambda: save_project(saved, path, **options))
            loaded = timed("verified_reopen", lambda: project.open_analysis(path))
            assert loaded.dataset is not None, loaded.reason
            verify_model(loaded.completion.result)
            other, _, _ = timed(
                "save_as", lambda: save_project(saved, root / "Other.peakfit", **options)
            )
            portable_options = dict(options, portable=True) if supports_modes else options
            portable, _, _ = timed(
                "portable_export",
                lambda: save_project(saved, root / "Portable.peakfit", **portable_options),
            )
            report["project_bytes"] = {
                label: sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
                for label, directory in (
                    ("standard", path),
                    ("second_analysis", other),
                    ("portable", portable),
                )
            }
            report["standard_embeds_cube"] = (path / "spectra").exists()
            for old, new in zip(
                result.completed_fit.samples,
                loaded.completion.result.completed_fit.samples,
                strict=True,
            ):
                for field in ("data", "neighbor_corrections"):
                    assert getattr(old, field).tobytes() == getattr(new, field).tobytes()
            report["exact_scientific_samples"] = True
            moved = root / "moved.ft2"
            source.rename(moved)
            missing = timed("missing_source_reopen", lambda: project.open_analysis(path))
            assert missing.completion is not None
            report["missing_source_has_spectrum"] = missing.dataset is not None
            if supports_modes:
                assert missing.dataset is None
            relinked = timed("explicit_relink", lambda: project.open_analysis(path, relink=moved))
            assert relinked.dataset is not None, relinked.reason
            moved.unlink()
            unlinked = timed("portable_source_free_reopen", lambda: project.open_analysis(portable))
            assert unlinked.dataset is not None, unlinked.reason
            assert unlinked.dataset.spectra.data.tobytes() == dataset.spectra.data.tobytes()
            report["portable_exact_cube"] = True
            if args.native and supports_modes:
                from validate_storage_policy import native_refit  # noqa: PLC0415

                # Reuse the existing Cocoa overlay/residual/adjust/refit exercise.
                args.output = args.report
                report["timings_seconds"] = report["seconds"]
                native_refit(unlinked, args, report, timed)
        finally:
            owner.close()
    args.report.write_text(json.dumps(report, indent=2) + "\n")


def main():
    """Sample phase-specific worker/descendant RSS, excluding this monitor process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--phase", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="peakfit-portability-monitor-") as temporary:
        root = Path(temporary)
        phase, report = root / "phase", root / "report.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--output",
            str(args.output),
            "--phase",
            str(phase),
            "--report",
            str(report),
        ]
        if args.native:
            command.append("--native")
        child = subprocess.Popen(
            command, env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        )
        peaks = {}
        while child.poll() is None:
            label = phase.read_text() if phase.exists() else "startup"
            try:
                process = psutil.Process(child.pid)
                processes = [process, *process.children(recursive=True)]
                sizes = {str(p.pid): p.memory_info().rss for p in processes if p.is_running()}
                total = sum(sizes.values())
                if total > peaks.get(label, {}).get("summed_rss_bytes", 0):
                    peaks[label] = {"summed_rss_bytes": total, "rss_by_pid": sizes}
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.025)
        if child.returncode:
            raise RuntimeError(f"Measurement worker failed: {child.returncode}")
        measured = json.loads(report.read_text())
        measured["phase_peak_memory"] = peaks
        measured["rss_method"] = (
            "25ms sampled sum of worker and descendants RSS; shared pages may count twice"
        )
        measured["numpy"] = np.__version__
        args.output.write_text(json.dumps(measured, indent=2) + "\n")
        if report.with_suffix(".webp").exists():
            shutil.copyfile(report.with_suffix(".webp"), args.output.with_suffix(".webp"))


if __name__ == "__main__":
    main()
