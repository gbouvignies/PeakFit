"""Validate the real CLI against PR #60 and measure complete process overhead.

uv run python tools/validate_linewidth_comparison.py --directory /tmp/linewidth-61
Retains full candidate outputs, logs and measurements; requires a fresh directory.
"""

# ruff: noqa: PLR2004, T201 - fixed experimental controls and CLI results
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {269: ("36N-H", "93N-H"), 360: ("5N-H", "76N-H"), 424: ("53N-H", "147N-H")}


def main() -> None:
    """Measure separate CLI processes and verify their actual scientific outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    environment = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MPLCONFIGDIR": "/tmp/peakfit-mpl",
    }
    measurements = []
    for repeat in range(args.repeats):
        for mode in ("single", "comparison"):
            output = args.directory / f"{mode}-{repeat}"
            command = [
                sys.executable,
                "-m",
                "peakfit",
                "fit",
                str(ROOT / "examples/02-advanced-fitting/data/pseudo3d.ft2"),
                str(ROOT / "examples/02-advanced-fitting/data/pseudo3d.list"),
                "-z",
                str(ROOT / "examples/02-advanced-fitting/data/b1_offsets.txt"),
                "--headless",
                "--workers",
                "1",
                "--refine",
                "2",
                "-o",
                str(output),
            ]
            if mode == "comparison":
                command.extend(["--compare-linewidth-start", "13"])
            started = time.perf_counter()
            # Same CLI module in a fresh process; atexit captures that process's
            # own peak RSS (bytes on macOS), avoiding cumulative child high-water marks.
            wrapper = (
                "import atexit,resource,runpy,sys; "
                "atexit.register(lambda: sys.stderr.write('PEAK_RSS_BYTES=' + "
                "str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) + '\\n')); "
                "sys.argv[0]='peakfit'; runpy.run_module('peakfit',run_name='__main__')"
            )
            process = subprocess.run(
                [sys.executable, "-c", wrapper, *command[3:]],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
            seconds = time.perf_counter() - started
            (args.directory / f"{mode}-{repeat}.log").write_text(process.stdout + process.stderr)
            match = re.search(r"PEAK_RSS_BYTES=(\d+)", process.stderr)
            assert match is not None, "This measurement driver requires macOS resource.getrusage"
            measurements.append(
                {
                    "mode": mode,
                    "repeat": repeat,
                    "seconds": seconds,
                    "peak_rss_bytes": int(match[1]),
                    "command": command,
                }
            )
    analyze(args.directory, measurements)


def analyze(directory: Path, measurements: list[dict]) -> None:
    """Check retained outputs without repeating the timed processes."""
    with gzip.open(ROOT / "docs/validation/linewidth-scaling-results-fits.json.gz", "rt") as stream:
        evidence = json.load(stream)
    reference = {
        "baseline": evidence["full_runs"][0],
        "alternative-13hz": evidence["selection_controls"]["single_global_13"],
    }
    targets = []
    for repeat in range(len(measurements) // 2):
        candidate_dir = next((directory / f"comparison-{repeat}").glob("*/comparison.json")).parent
        for candidate, archived in reference.items():
            document = json.loads((candidate_dir / candidate / "summary/fit.json").read_text())
            for cluster_id, pair in PAIRS.items():
                cluster = next(c for c in document["clusters"] if c["cluster_id"] == cluster_id)
                previous = next(
                    g
                    for g in archived["groups"]
                    if g["cluster_id"] == cluster_id and g["pass"] == 2
                )
                parameters = {p["name"]: p["value"] for p in cluster["final_nonlinear_parameters"]}
                chi = cluster["analytical_evaluation"]["statistics"]["chi_squared"]
                assert abs(chi - previous["chi_squared_actual"]) < 0.02
                for name, parameter in previous["parameters"].items():
                    tolerance = 2e-5 if name.endswith(".cs") else 0.02
                    assert abs(parameters[name] - parameter["value"]) < tolerance
                warning = cluster["analytical_evaluation"]["uncertainty_warning"]
                assert bool(warning) == (candidate == "baseline")
                targets.append(
                    {
                        "repeat": repeat,
                        "candidate": candidate,
                        "cluster_id": cluster_id,
                        "pair": pair,
                        "chi_squared": chi,
                        "warning": warning,
                        "separation_ppm": {
                            axis: abs(
                                parameters[f"{pair[0]}.{axis}.cs"]
                                - parameters[f"{pair[1]}.{axis}.cs"]
                            )
                            for axis in ("F2", "F3")
                        },
                        "parameters": parameters,
                        "parameter_uncertainties": {
                            p["name"]: p["standard_error"]
                            for p in cluster["final_nonlinear_parameters"]
                        },
                        "reduced_chi_squared": cluster["analytical_evaluation"]["statistics"][
                            "reduced_chi_squared"
                        ],
                        "optimizer_provenance": cluster["optimizer_provenance"],
                    }
                )
        rows = list(csv.DictReader((candidate_dir / "comparison.csv").read_text().splitlines()))
        assert len({r["cluster_id"] for r in rows}) == 121
    medians = {
        mode: {
            key: statistics.median(m[key] for m in measurements if m["mode"] == mode)
            for key in ("seconds", "peak_rss_bytes")
        }
        for mode in ("single", "comparison")
    }
    provenance = json.loads(
        next((directory / "comparison-0").glob("*/comparison.json")).read_text()
    )
    result = {
        "measurements": measurements,
        "medians": medians,
        "targets": targets,
        "input_files": provenance["input_files"],
        "git_commit": provenance["git_commit"],
    }
    (ROOT / "docs/validation/linewidth-comparison-results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(medians, indent=2))


if __name__ == "__main__":
    main()
