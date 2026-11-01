"""Real-data save/reopen timings and native Qt evidence for analysis persistence."""

import argparse
import gc
import json
import platform
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np
from PySide6 import QtCore, QtWidgets
from validate_gui_editor import settle, wait_until

from peakfit.gui.editor.window import EditorWindow
from peakfit.gui.project import verify_sources
from peakfit.gui.spectrum import read_dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/validation/gui-v2-persistence"


def retained_array_bytes(value, seen=None):  # noqa: PLR0911
    """Count unique NumPy owners reachable through scientific value containers."""
    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        owner = value
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        key = ("buffer", id(owner))
        if key in seen:
            return 0
        seen.add(key)
        return owner.nbytes
    if is_dataclass(value) and not isinstance(value, type):
        return sum(retained_array_bytes(getattr(value, f.name), seen) for f in fields(value))
    if isinstance(value, Mapping):
        return sum(retained_array_bytes(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(retained_array_bytes(v, seen) for v in value)
    return 0


def main():  # noqa: PLR0915
    """Measure actual publication and validation, including fsync and source hashes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--phase-file", type=Path)
    parser.add_argument("--collect-after-close", action="store_true")
    args = parser.parse_args()
    output = args.output
    phase = ["load"]
    window = None
    last_stage = [None]
    retained = {}

    def record_phase():
        if args.phase_file is None:
            return
        stage = ""
        if window is not None and window.workspace is not None:
            fitting = window.workspace.fitting
            stage = fitting.process.progress if fitting.process.busy else fitting.attempt
            if stage.startswith("Pass "):
                stage = "optimizer"
        value = {"phase": phase[0], "stage": stage}
        if value != last_stage[0]:
            args.phase_file.write_text(json.dumps(value))
            last_stage[0] = value

    def mark_phase(value):
        phase[0] = value
        record_phase()
        if (
            args.phase_file
            and value.endswith("_settled")
            and window is not None
            and window.workspace is not None
        ):
            workspace = window.workspace
            fitting = workspace.fitting
            retained[value] = {
                "scientific_array_bytes": retained_array_bytes(fitting.completion),
                "resident_full_runs": len(
                    {id(c) for c in (*fitting.history, fitting.completion) if hasattr(c, "result")}
                ),
                "model_cache_bytes": sum(a.nbytes for a in workspace.canvas._model_cache.values()),
                "contour_cache_bytes": workspace.canvas.cache_bytes,
            }

    record_phase()
    output.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication([])
    app.setStyle("Fusion")
    phase_timer = QtCore.QTimer()
    phase_timer.setInterval(50)
    phase_timer.timeout.connect(record_phase)
    phase_timer.start()
    data = ROOT / "examples/02-advanced-fitting/data"
    start = perf_counter()
    dataset = read_dataset(data / "pseudo3d.ft2", data / "pseudo3d.list", data / "b1_offsets.txt")
    report = {
        "platform": platform.platform(),
        "qt_platform": app.platformName(),
        "python": platform.python_version(),
        "shape": list(dataset.spectra.data.shape),
        "source_bytes": dataset.path.stat().st_size,
        "load_and_fingerprint_seconds": perf_counter() - start,
    }
    window = EditorWindow()
    window.install_dataset(dataset)
    window.show()
    w = window.workspace
    assert w is not None
    w.fitting.setup_seen = True
    mark_phase("fit")
    started = perf_counter()
    w.fitting.fit()
    wait_until(lambda: not w.fitting.busy, timeout=120)
    assert w.fitting.current, w.fitting.failure
    completion = w.fitting.completion
    assert completion is not None
    report["initial_fit_and_durable_acceptance_seconds"] = perf_counter() - started
    mark_phase("fit_settled")
    if args.phase_file:
        settle(500)
    w.set_plane(60)
    w.select_peak(w.peaks[0].id)
    beats, last = [], [perf_counter()]
    mark_phase("first_save")
    timer = QtCore.QTimer()
    timer.setInterval(10)

    def beat():
        now = perf_counter()
        beats.append((phase[0], (now - last[0]) * 1000))
        last[0] = now

    timer.timeout.connect(beat)
    timer.start()
    with TemporaryDirectory(prefix="peakfit-native-persistence-") as temporary:
        destination = Path(temporary) / "Analysis.peakfit"
        with patch.object(
            QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(destination), "")
        ):
            start = perf_counter()
            assert window.save_analysis_dialog()
            wait_until(lambda: not window._saving, timeout=120)
            report["first_save_seconds"] = perf_counter() - start
        paths = {p: p.stat().st_mtime_ns for p in (destination / "runs").rglob("*") if p.is_file()}
        mark_phase("routine_save")
        start = perf_counter()
        window.save_analysis_dialog()
        wait_until(lambda: not window._saving, timeout=120)
        report["routine_save_seconds"] = perf_counter() - start
        report["routine_unchanged_evidence_rewrites"] = sum(
            p.stat().st_mtime_ns != t for p, t in paths.items()
        )
        report["project_bytes"] = sum(
            p.stat().st_size for p in destination.rglob("*") if p.is_file()
        )
        report["routine_source_content_bytes_read"] = 0  # The routine path only stats source links.
        report["routine_bytes_written"] = (destination / "manifest.json").stat().st_size
        report["spectrum_payload_members"] = [
            str(p.relative_to(destination))
            for p in destination.rglob("*")
            if p.is_file() and p.suffix in {".ft2", ".ft3"}
        ]
        report["payloads"] = {
            str(p.relative_to(destination)): p.stat().st_size
            for p in destination.rglob("*")
            if p.is_file()
        }
        mark_phase("save_as")
        second = Path(temporary) / "Second.peakfit"
        with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(second), "")):
            start = perf_counter()
            window.save_analysis_dialog(save_as=True)
            wait_until(lambda: not window._saving, timeout=120)
            report["save_as_seconds"] = perf_counter() - start
        mark_phase("reopen")
        start = perf_counter()
        window.load_analysis(second)
        wait_until(lambda: not window._busy, timeout=120)
        report["verified_reopen_seconds"] = perf_counter() - start
        assert window.workspace is not None
        assert window.workspace.fitting.current
        mark_phase("reopen_settled")
        if args.phase_file:
            settle(500)
        w = window.workspace
        assert w.peaks == completion.working_peaks
        mark_phase("display")
        w.set_mode(1)
        wait_until(lambda: not w.canvas.contour_pending, timeout=30)
        settle(2000)
        for appearance in ("Light", "Dark"):
            window.apply_appearance(appearance)
            settle(500)
            window.grab().save(str(output / f"reopened-{appearance.lower()}.webp"), "WEBP", 95)
        w.set_mode(2)
        settle(1500)
        window.grab().save(str(output / "residual.webp"), "WEBP", 95)
        peak = w.peaks[0]
        w.move_peak(peak.id, (peak.positions[0], peak.positions[1] + 0.00001))
        assert not w.fitting.current
        report["manual_edit_outdates"] = True
        mark_phase("refit")
        assert w.fitting.result is not None
        previous_run = w.fitting.result.run_id
        w.fitting.fit()
        wait_until(lambda: not w.fitting.busy, timeout=120)
        assert w.fitting.result is not None
        assert w.fitting.result.run_id != previous_run, w.fitting.failure
        assert w.fitting.current
        report["reopened_refit_current"] = True
        mark_phase("refit_settled")
        if args.phase_file:
            settle(500)
        mark_phase("numerical_review")
        # Explicitly selected missing source demonstrates numerical mode with real evidence.
        session_root = w.fitting._recovery_path
        analysis = verify_sources(window.capture_analysis(), relink=Path(temporary) / "missing.ft2")
        assert analysis.dataset is None
        window.install_analysis(analysis)
        window.apply_appearance("Light")
        settle(500)
        window.grab().save(str(output / "numerical-only.webp"), "WEBP", 95)
        report["heartbeat_ms"] = {
            stage: {
                "p95": float(np.percentile([b for p, b in beats if p == stage], 95)),
                "max": max(b for p, b in beats if p == stage),
            }
            for stage in sorted({p for p, _ in beats})
        }
        timer.stop()
        window.close()
        if session_root:
            wait_until(lambda: not session_root.exists())
        report["session_storage_cleaned_after_numeric_close"] = True
        mark_phase("closed")
        if args.phase_file:
            settle(500)
    if args.collect_after_close:
        report["post_close_collected_objects"] = gc.collect()
        mark_phase("closed_after_gc")
        settle(500)
    phase_timer.stop()
    if args.phase_file:
        report["retained_array_snapshots"] = retained
    (output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
