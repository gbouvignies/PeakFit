"""Write the bundled example's golden fit reference from a CLI fit.

Run with ``uv run python tools/write_golden_reference.py``. It fits the bundled
example with ``FIT_ARGUMENTS`` and writes ``tests/data/golden/example-fit.json``
(provenance, statistics, nonlinear parameters) and ``example-fit-amplitudes.npz``
(amplitudes and their conditional standard errors), which
``tests/integration/test_golden_regression.py`` compares against. Pass
``--run RUN_DIR`` to write from an existing fit made with the same arguments.

The reference is scientific evidence: regenerate it only for a reviewed and
explained numerical change (see "Golden fit reference" in docs/development.md).
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from peakfit.io.scientific import read_scientific

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/data/golden"
DATA = "examples/02-advanced-fitting/data"
# Paths are relative to the repository root; the fixture runs them from there.
FIT_ARGUMENTS = [
    f"{DATA}/pseudo3d.ft2",
    f"{DATA}/pseudo3d.list",
    "--z-values",
    f"{DATA}/b1_offsets.txt",
    "--refine",
    "1",
    "--format",
    "json",
    "--format",
    "csv",
    "--format",
    "txt",
]


def run_example_fit(output: Path) -> Path:
    """Fit the bundled example into ``output`` and return its run directory."""
    command = [sys.executable, "-m", "peakfit", "fit", *FIT_ARGUMENTS]
    command += ["--output", str(output), "--headless"]
    subprocess.run(command, check=True, cwd=ROOT, capture_output=True)
    (run,) = (path for path in output.iterdir() if path.is_dir())
    return run


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def write_reference(run: Path, source_commit: str) -> None:
    """Record ``run``'s outcome as the reference, naming the commit that produced it."""
    outcome, _samples, _context = read_scientific(run / "scientific.npz")
    clusters, arrays = [], {}
    for cluster in outcome.clusters:
        evaluation = cluster.analytical_evaluation
        if evaluation is None:
            raise ValueError(f"Cluster {cluster.cluster_id} has no analytical result")
        clusters.append(
            {
                "cluster_id": cluster.cluster_id,
                "peak_names": list(cluster.peak_names),
                "classification": cluster.classification.value,
                "uncertainty_warning": evaluation.uncertainty_warning is not None,
                "chi_squared": evaluation.statistics.chi_squared,
                "parameters": [
                    {
                        "name": p.name,
                        "value": p.value,
                        "standard_error": _finite_or_none(p.standard_error),
                        "min": p.min,
                        "max": p.max,
                        "vary": p.vary,
                    }
                    for p in cluster.final_nonlinear_parameters
                ],
            }
        )
        arrays[f"amplitudes_{cluster.cluster_id}"] = evaluation.amplitudes
        arrays[f"amplitude_standard_errors_{cluster.cluster_id}"] = (
            evaluation.amplitude_standard_errors
        )
    statistics = outcome.statistics
    reference = {
        "provenance": {
            "description": "peakfit fit of examples/02-advanced-fitting with fit_arguments",
            "peakfit_version": version("peakfit"),
            "source_commit": source_commit,
        },
        "fit_arguments": FIT_ARGUMENTS,
        "statistics": {
            "chi_squared": statistics.chi_squared,
            "n_observations": statistics.n_observations,
            "n_fitted_parameters": statistics.n_fitted_parameters,
            "degrees_of_freedom": statistics.degrees_of_freedom,
        },
        "clusters": clusters,
    }
    GOLDEN.mkdir(parents=True, exist_ok=True)
    (GOLDEN / "example-fit.json").write_text(json.dumps(reference, indent=1) + "\n")
    np.savez_compressed(GOLDEN / "example-fit-amplitudes.npz", **arrays)


def main() -> None:
    """Fit the example (or read ``--run``) and write the reference files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="existing run directory to record")
    parser.add_argument("--source-commit", required=True, help="commit whose code produced the fit")
    args = parser.parse_args()
    if args.run is not None:
        write_reference(args.run, args.source_commit)
        return
    with tempfile.TemporaryDirectory() as output:
        write_reference(run_example_fit(Path(output)), args.source_commit)


if __name__ == "__main__":
    main()
