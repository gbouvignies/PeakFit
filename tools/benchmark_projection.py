"""Compare complete linear-projection work on all bundled experimental groups.

Run: uv run python tools/benchmark_projection.py > /tmp/projection-timing.json
The original path includes both amplitudes and the Jacobian pseudoinverse helper.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits

from peakfit.engine.algorithms.linear_algebra import factorize_shapes
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.fit.fitting import load_data


def measure(groups, kind):
    """Time nine sweeps of the actual shapes and plane-wise data."""
    trials = []
    for _ in range(9):
        started = perf_counter()
        for shapes, data in groups:
            if kind == "original_qr":
                q, r = np.linalg.qr(shapes.T, mode="reduced")
                solve_triangular(r, q.T @ data, check_finite=False)
                data - q @ (q.T @ data)
                solve_triangular(r, q.T, check_finite=False)
            elif kind == "rank_aware":
                projection = factorize_shapes(shapes)
                amplitudes = projection.solve(data)
                data - shapes.T @ amplitudes
            else:
                amplitudes = np.linalg.lstsq(shapes.T, data, rcond=None)[0]
                data - shapes.T @ amplitudes
                np.linalg.pinv(shapes.T)
        trials.append(perf_counter() - started)
    return {"median_seconds": float(np.median(trials)), "trials": trials}


def main():
    """Load the same production masks used by the fit command."""
    root = Path(__file__).resolve().parents[1] / "examples/02-advanced-fitting/data"
    with threadpool_limits(limits=1):
        data = load_data(
            root / "pseudo3d.ft2", root / "pseudo3d.list", root / "b1_offsets.txt", PeakFitConfig()
        )
        groups = [(c.evaluate(Parameters.from_peaks(c.peaks)), c.data) for c in data.clusters]
        results = {kind: measure(groups, kind) for kind in ("original_qr", "rank_aware", "lstsq")}
    print(json.dumps(results, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
