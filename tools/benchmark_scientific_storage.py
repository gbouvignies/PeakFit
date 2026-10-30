"""Reproduce real-data storage measurements in a fresh process per implementation."""

import argparse
import gc
import json
import pickle
import platform
import resource
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import h5py
import numpy as np

from peakfit.gui.dataset_fit import _completed_bytes, compute_dataset_fit, finish_dataset_fit
from peakfit.gui.fit_positions import dataset_completion
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.project import (
    Analysis,
    dataset_header,
    open_analysis,
    save_analysis,
    stored_analysis,
)
from peakfit.gui.project_results import verify_model
from peakfit.gui.spectrum import read_dataset

ROOT = Path(__file__).resolve().parents[1]


def timed(action):
    """Measure the complete synchronous operation."""
    start = perf_counter()
    result = action()
    return result, perf_counter() - start


def main():  # noqa: PLR0915
    """Measure real scientific fitting, durable publication, and verified reading."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", action="store_true")
    parser.add_argument("--publication-payload", type=Path)
    args = parser.parse_args()
    if args.publication_payload:
        # Benchmark-owned trusted IPC only, never a project file format.
        analysis = pickle.loads(args.publication_payload.read_bytes())
        gc.collect()
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with TemporaryDirectory(prefix="peakfit-publication-memory-") as temporary:
            _, seconds = timed(
                lambda: save_analysis(analysis, Path(temporary) / "Evidence.peakfit")
            )
        args.output.write_text(
            json.dumps(
                {
                    "before_publication_peak_rss_bytes": before,
                    "publication_peak_rss_bytes": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                    "seconds": seconds,
                },
                indent=2,
            )
            + "\n"
        )
        return
    data = ROOT / "examples/02-advanced-fitting/data"
    dataset = read_dataset(data / "pseudo3d.ft2", data / "pseudo3d.list", data / "b1_offsets.txt")
    from peakfit.engine.domain.config import PeakFitConfig  # noqa: PLC0415

    config = PeakFitConfig()
    config.fitting.refine_iterations = 2
    prepared = prepare_dataset_fit(dataset.spectra, dataset.peaks, config)
    prepared = replace(prepared, snapshot=replace(prepared.snapshot, source_files=dataset.sources))
    result, fit_seconds = timed(lambda: finish_dataset_fit(prepared, compute_dataset_fit(prepared)))
    completion = dataset_completion(
        result, prepared.snapshot, submitted_revision=0, manual_revision=0
    )
    analysis = Analysis(
        None,
        dataset_header(dataset),
        completion.working_peaks,
        config.model_dump_json(),
        0,
        (completion,),
    )
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "shape": dataset.spectra.data.shape,
        "fit_seconds": fit_seconds,
        "peak_rss_before_publication_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    with TemporaryDirectory(prefix="peakfit-storage-benchmark-") as temporary:
        root = Path(temporary)
        payload = root / "trusted-ipc.pkl"
        payload.write_bytes(_completed_bytes(analysis))
        publication_report = root / "publication.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--publication-payload",
                str(payload),
                "--output",
                str(publication_report),
            ],
            check=True,
        )
        report["isolated_publication"] = json.loads(publication_report.read_text())
        path, report["durable_publication_seconds"] = timed(
            lambda: save_analysis(analysis, root / "Recovery.peakfit")
        )
        report["fit_to_durable_acceptance_seconds"] = (
            fit_seconds + report["durable_publication_seconds"]
        )
        saved = stored_analysis(analysis, path)
        permanent, report["first_save_seconds"] = timed(
            lambda: save_analysis(saved, root / "Analysis.peakfit")
        )
        saved = stored_analysis(saved, permanent)
        _, report["routine_save_seconds"] = timed(lambda: save_analysis(saved, permanent))
        loaded, report["verified_reopen_seconds"] = timed(lambda: open_analysis(permanent))
        assert loaded.dataset is not None, loaded.reason
        historical, report["historical_load_seconds"] = timed(lambda: loaded.runs[0].load())
        verify_model(historical.result)
        for a, b in zip(result.outcome.clusters, historical.result.outcome.clusters, strict=True):
            if a.analytical_evaluation is not None:
                for name in (
                    "shapes",
                    "amplitudes",
                    "amplitude_covariance",
                    "raw_residuals",
                    "model_values",
                    "normalized_residuals",
                ):
                    x, y = (
                        getattr(a.analytical_evaluation, name),
                        getattr(b.analytical_evaluation, name),
                    )
                    assert x.dtype == y.dtype
                    assert x.shape == y.shape
                    assert x.tobytes() == y.tobytes(), name
        report["exact_round_trip_and_reconstruction"] = True
        report["project_bytes"] = sum(p.stat().st_size for p in permanent.rglob("*") if p.is_file())
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if args.candidates:
            from peakfit.fit.storage import read_scientific, write_scientific  # noqa: PLC0415

            report["candidates"] = []
            for name, compression in (("npz_stored", ZIP_STORED), ("npz_deflate1", ZIP_DEFLATED)):
                gc.collect()
                candidate = root / (name + ".npz")
                _, write_seconds = timed(
                    lambda candidate=candidate, compression=compression: write_scientific(
                        candidate,
                        result.outcome,
                        result.completed_fit.samples,
                        compression=compression,
                    )
                )
                restored, read_seconds = timed(
                    lambda candidate=candidate: read_scientific(candidate)
                )
                assert (
                    restored[0].clusters[0].analytical_evaluation.shapes.tobytes()
                    == result.outcome.clusters[0].analytical_evaluation.shapes.tobytes()
                )
                report["candidates"].append(
                    {
                        "format": name,
                        "bytes": candidate.stat().st_size,
                        "write_seconds": write_seconds,
                        "read_seconds": read_seconds,
                    }
                )
            # Same arrays and small metadata as the ZIP candidate, no chunking/compression.
            candidate = root / "hdf5.h5"

            with np.load(root / "npz_stored.npz", allow_pickle=False) as source:
                hdf_arrays = {
                    name: source[name] for name in source.files if name != "metadata.json"
                }
            with ZipFile(root / "npz_stored.npz") as archive:
                hdf_metadata = archive.read("metadata.json")

            def write_hdf5():
                with h5py.File(candidate, "w") as target:
                    for name, array in hdf_arrays.items():
                        target.create_dataset(name, data=array)
                    target.create_dataset("metadata", data=np.void(hdf_metadata))

            _, write_seconds = timed(write_hdf5)

            def read_hdf5():
                with h5py.File(candidate) as source:

                    def visit(_name, item):
                        if isinstance(item, h5py.Dataset):
                            item[()]

                    source.visititems(visit)

            _, read_seconds = timed(read_hdf5)
            report["candidates"].append(
                {
                    "format": "hdf5_uncompressed_arrays",
                    "bytes": candidate.stat().st_size,
                    "write_seconds": write_seconds,
                    "read_seconds": read_seconds,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
