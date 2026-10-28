"""Benchmark completed-model review and candidate numerical archive formats.

Run: uv run python tools/benchmark_review_projection.py > /tmp/review-benchmark.json
Uses a real two-pass fit of the bundled 131-plane dataset, one BLAS thread and
no projection caches. Archives are temporary experiments, not a new file contract.
"""

import cProfile
import hashlib
import io
import json
import platform
import pstats
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import h5py
import numpy as np
from threadpoolctl import threadpool_limits

from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.fit.fitting import load_data
from peakfit.fit.pipeline import run_pipeline
from peakfit.fit.simulation import project_final_outcome


def measure(operation, repeats=5):
    """Measure complete calls, including validation and allocation."""
    trials = []
    for _ in range(repeats):
        started = perf_counter()
        operation()
        trials.append(perf_counter() - started)
    return {"median_seconds": float(np.median(trials)), "trials_seconds": trials}


def archives(completed):
    """Compare lossless storage of only the additional terminal numerical inputs."""
    arrays = {}
    for samples in completed.samples:
        prefix = f"cluster_{samples.cluster_id}"
        arrays[prefix + "_data"] = samples.data
        arrays[prefix + "_corrections"] = samples.neighbor_corrections
        for dimension, indices in enumerate(samples.grid_indices):
            arrays[f"{prefix}_index_{dimension}"] = indices
    results = {"array_bytes": sum(a.nbytes for a in arrays.values())}
    with TemporaryDirectory(prefix="peakfit-archive-benchmark-") as directory:
        root = Path(directory)
        for kind in ("json", "npz", "npz_compressed", "hdf5_gzip"):
            path = root / kind
            started = perf_counter()
            if kind == "json":
                path.write_text(json.dumps({k: a.tolist() for k, a in arrays.items()}))
            elif kind.startswith("npz"):
                with path.open("wb") as handle:
                    writer = np.savez_compressed if kind == "npz_compressed" else np.savez
                    writer(handle, **arrays)
            else:
                with h5py.File(path, "w") as handle:
                    for key, values in arrays.items():
                        handle.create_dataset(key, data=values, compression="gzip", shuffle=True)
            write_seconds = perf_counter() - started
            started = perf_counter()
            if kind == "json":
                loaded = {k: np.asarray(a) for k, a in json.loads(path.read_text()).items()}
            elif kind.startswith("npz"):
                with np.load(path, allow_pickle=False) as handle:
                    loaded = {key: handle[key] for key in handle.files}
            else:
                with h5py.File(path, "r") as handle:
                    loaded = {key: handle[key][()] for key in handle}
            read_seconds = perf_counter() - started
            for key, values in arrays.items():
                np.testing.assert_array_equal(loaded[key], values)
                assert loaded[key].dtype == values.dtype
            results[kind] = {
                "file_bytes": path.stat().st_size,
                "write_seconds": write_seconds,
                "read_seconds": read_seconds,
            }
    return results


def main():
    """Report scientific workload, projection costs, profile and archive comparison."""
    root = Path(__file__).resolve().parents[1] / "examples/02-advanced-fitting/data"
    config = PeakFitConfig(fitting=FitConfig(refine_iterations=2))
    with threadpool_limits(limits=1):
        started = perf_counter()
        data = load_data(
            root / "pseudo3d.ft2", root / "pseudo3d.list", root / "b1_offsets.txt", config
        )
        preparation_seconds = perf_counter() - started
        started = perf_counter()
        pipeline = run_pipeline(
            config, data.clusters, data.noise, Parameters.from_peaks(data.peaks), data.peaks
        )
        fit_seconds = perf_counter() - started
        outcome, snapshot = pipeline.final_outcome, pipeline.simulation_snapshot
        ny, nx = data.spectra.data.shape[1:]
        iy, ix = ny // 2, nx // 2
        grid = tuple(index.ravel() for index in np.indices((ny, nx)))
        operations = {
            "native_plane": lambda: project_final_outcome(outcome, snapshot, grid, planes=[0]),
            "row_all_planes": lambda: project_final_outcome(
                outcome, snapshot, [np.full(nx, iy), np.arange(nx)]
            ),
            "column_all_planes": lambda: project_final_outcome(
                outcome, snapshot, [np.arange(ny), np.full(ny, ix)]
            ),
            "crosshair_all_planes": lambda: project_final_outcome(
                outcome, snapshot, [np.array([iy]), np.array([ix])]
            ),
        }
        timings = {name: measure(operation) for name, operation in operations.items()}
        profile = cProfile.Profile()
        profile.runcall(operations["native_plane"])
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(15)
        storage = archives(pipeline.completed_fit)
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas_threads": 1,
        "spectrum_sha256": hashlib.sha256((root / "pseudo3d.ft2").read_bytes()).hexdigest(),
        "data_shape": data.spectra.data.shape,
        "peaks": len(data.peaks),
        "shape_names": data.shape_names,
        "clusters": len(outcome.clusters),
        "classifications": {
            name: sum(c.classification == name for c in outcome.clusters)
            for name in ("converged", "usable_non_converged", "unusable")
        },
        "terminal_correction_revision": outcome.terminal_correction_revision,
        "preparation_seconds": preparation_seconds,
        "fit_and_capture_seconds": fit_seconds,
        "projection": timings,
        "native_plane_profile": stream.getvalue(),
        "terminal_input_storage": storage,
    }
    print(json.dumps(report, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
