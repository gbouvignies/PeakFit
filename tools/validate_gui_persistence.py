"""Real-data save/reopen timings and native Qt evidence for analysis persistence."""

import json
import platform
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np
from PySide6 import QtCore, QtWidgets
from threadpoolctl import threadpool_limits
from validate_gui_editor import settle, wait_until

from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.window import EditorWindow
from peakfit.gui.fit_positions import dataset_completion
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.project import verify_sources
from peakfit.gui.spectrum import read_dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/validation/gui-v2-persistence"


def main():  # noqa: PLR0915
    """Measure actual publication and validation, including fsync and source hashes."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication([])
    app.setStyle("Fusion")
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
    with threadpool_limits(limits=1):
        prepared = prepare_dataset_fit(dataset.spectra, dataset.peaks)
        prepared = replace(
            prepared, snapshot=replace(prepared.snapshot, source_files=dataset.sources)
        )
        result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    completion = dataset_completion(
        result, prepared.snapshot, submitted_revision=0, manual_revision=0
    )
    window = EditorWindow()
    window.install_dataset(dataset)
    window.show()
    w = window.workspace
    assert w is not None
    w.fitting._submitted_revision = 0
    w.fitting._publish(completion)
    assert w.fitting.result is not None
    w.set_plane(60)
    w.select_peak(w.peaks[0].id)
    beats, last = [], [perf_counter()]
    phase = ["first_save"]
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
        phase[0] = "routine_save"
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
        phase[0] = "save_as"
        second = Path(temporary) / "Second.peakfit"
        with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(second), "")):
            start = perf_counter()
            window.save_analysis_dialog(save_as=True)
            wait_until(lambda: not window._saving, timeout=120)
            report["save_as_seconds"] = perf_counter() - start
        phase[0] = "reopen"
        start = perf_counter()
        window.load_analysis(second)
        wait_until(lambda: not window._busy, timeout=120)
        report["verified_reopen_seconds"] = perf_counter() - start
        assert window.workspace is not None
        assert window.workspace.fitting.current
        w = window.workspace
        assert w.peaks == completion.working_peaks
        phase[0] = "display"
        w.set_mode(1)
        wait_until(lambda: not w.canvas.contour_pending, timeout=30)
        settle(2000)
        for appearance in ("Light", "Dark"):
            window.apply_appearance(appearance)
            settle(500)
            window.grab().save(str(OUTPUT / f"reopened-{appearance.lower()}.webp"), "WEBP", 95)
        w.set_mode(2)
        settle(1500)
        window.grab().save(str(OUTPUT / "residual.webp"), "WEBP", 95)
        peak = w.peaks[0]
        w.move_peak(peak.id, (peak.positions[0], peak.positions[1] + 0.00001))
        assert not w.fitting.current
        report["manual_edit_outdates"] = True
        phase[0] = "refit"
        assert w.fitting.result is not None
        previous_run = w.fitting.result.run_id
        w.fitting.fit()
        wait_until(lambda: not w.fitting.busy, timeout=120)
        assert w.fitting.result is not None
        assert w.fitting.result.run_id != previous_run, w.fitting.failure
        assert w.fitting.current
        report["reopened_refit_current"] = True
        phase[0] = "numerical_review"
        # Explicitly selected missing source demonstrates numerical mode with real evidence.
        analysis = verify_sources(window.capture_analysis(), relink=Path(temporary) / "missing.ft2")
        assert analysis.dataset is None
        window.install_analysis(analysis)
        window.apply_appearance("Light")
        settle(500)
        window.grab().save(str(OUTPUT / "numerical-only.webp"), "WEBP", 95)
        report["heartbeat_ms"] = {
            stage: {
                "p95": float(np.percentile([b for p, b in beats if p == stage], 95)),
                "max": max(b for p, b in beats if p == stage),
            }
            for stage in sorted({p for p, _ in beats})
        }
        timer.stop()
        window.close()
    (OUTPUT / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
