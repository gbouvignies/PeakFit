"""Real-data resident history and lossless scientific-JSON compression measurements."""

import argparse
import gc
import gzip
import json
import platform
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from PySide6 import QtWidgets
from validate_gui_editor import rss_mib, settle, wait_until

from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.project import Analysis, dataset_header, open_analysis, save_analysis
from peakfit.gui.spectrum import read_dataset

ROOT = Path(__file__).resolve().parents[1]


def main():  # noqa: PLR0915
    """Run fresh fits, release local references, and sample settled workspace RSS."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fits", type=int, default=12)
    parser.add_argument("--compression", action="store_true")
    parser.add_argument("--reopen", action="store_true")
    args = parser.parse_args()
    app = QtWidgets.QApplication([])
    data = ROOT / "examples/02-advanced-fitting/data"
    dataset = read_dataset(data / "pseudo3d.ft2", data / "pseudo3d.list", data / "b1_offsets.txt")
    workspace = EditorWorkspace(dataset)
    workspace.resize(1280, 800)
    workspace.show()
    workspace.fitting.setup_seen = True
    wait_until(lambda: not workspace.canvas.contour_pending)
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "qt_platform": app.platformName(),
        "fits": [],
        "baseline_rss_mib": rss_mib(),
    }
    with TemporaryDirectory(prefix="peakfit-history-benchmark-") as temp:
        for index in range(args.fits):
            started = perf_counter()
            previous = workspace.fitting.result.run_id if workspace.fitting.result else None
            workspace.fitting.fit()
            wait_until(lambda: not workspace.fitting.busy, timeout=180)
            assert workspace.fitting.current, workspace.fitting.failure
            assert workspace.fitting.result is not None
            assert workspace.fitting.result.run_id != previous
            workspace.set_mode(0)
            wait_until(lambda: not workspace.canvas.contour_pending, timeout=30)
            settle(50)
            gc.collect()
            report["fits"].append(
                {
                    "fit": index + 1,
                    "rss_mib": rss_mib(),
                    "seconds": perf_counter() - started,
                    "history_entries": len(workspace.fitting.history),
                    "resident_full_runs": len(
                        {
                            id(c)
                            for c in (*workspace.fitting.history, workspace.fitting.completion)
                            if hasattr(c, "result")
                        }
                    ),
                }
            )
            if index == 0 and args.compression:
                analysis = Analysis(
                    None,
                    dataset_header(dataset),
                    workspace.peaks,
                    workspace.fitting.configuration.model_dump_json(),
                    workspace.edits.revision,
                    tuple(workspace.fitting.history),
                    journal=tuple(workspace.fitting.journal),
                )
                path = save_analysis(analysis, Path(temp) / "Compression.peakfit")
                summary = next(path.rglob("fit.json*"))
                raw = (
                    gzip.decompress(summary.read_bytes())
                    if summary.suffix == ".gz"
                    else summary.read_bytes()
                )
                report["scientific_json_bytes"] = len(raw)
                report["compression"] = []
                for level in (1, 3, 6):
                    start = perf_counter()
                    compressed = gzip.compress(raw, compresslevel=level, mtime=0)
                    written = perf_counter() - start
                    start = perf_counter()
                    assert gzip.decompress(compressed) == raw
                    report["compression"].append(
                        {
                            "gzip_level": level,
                            "bytes": len(compressed),
                            "compression_seconds": written,
                            "decompression_seconds": perf_counter() - start,
                        }
                    )
                del analysis, raw, compressed
        if args.reopen:
            analysis = Analysis(
                None,
                dataset_header(dataset),
                workspace.peaks,
                workspace.fitting.configuration.model_dump_json(),
                workspace.edits.revision,
                tuple(workspace.fitting.history),
                journal=tuple(workspace.fitting.journal),
            )
            destination = save_analysis(analysis, Path(temp) / "Reopen.peakfit")
            assert workspace.fitting.result is not None
            expected_id = workspace.fitting.result.run_id
        recovery = workspace.fitting._recovery_path
        workspace.stop()
        workspace.close()
        settle(50)
        if recovery:
            # Normal cleanup is queued after any in-flight publication work.
            deadline = perf_counter() + 5
            while recovery.exists() and perf_counter() < deadline:
                settle(10)
        report["recovery_path_after_stop_exists"] = recovery.exists() if recovery else False
        if args.reopen:
            start = perf_counter()
            reopened = open_analysis(destination)
            assert reopened.completion.result.run_id == expected_id
            assert reopened.current
            assert reopened.dataset is not None
            report["verified_reopen_seconds"] = perf_counter() - start
            report["reopened_full_history_entries"] = len(reopened.runs)
            report["reopened_resident_full_runs"] = len(
                {id(c) for c in (*reopened.runs, reopened.completion) if hasattr(c, "result")}
            )
        # Baseline implementation has no cleanup; do not leave benchmark archives behind.
        if recovery and recovery.exists():
            shutil.rmtree(recovery.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
