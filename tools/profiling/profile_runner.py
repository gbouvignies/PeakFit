#!/usr/bin/env python3
"""
Profiling harness for PeakFit.

Features:
- Run a CLI command or call a module-level function and profile it.
- Supports cProfile (built-in), pyinstrument (if installed), and scalene (if installed).
- Saves profile outputs (pstats, svg, html) for analysis.

Usage examples (see README for full usage):
  python tools/profiling/profile_runner.py --cmd "uv run peakfit fit examples/01-basic-fitting/sample.ft2 examples/01-basic-fitting/sample_peaks.csv" --profiler cprofile --output profile.prof
  py-spy record -o profile.svg -- python tools/profiling/profile_runner.py --cmd "uv run peakfit fit ..." --profiler none

Note: For CLI profiling we shell out; for module-based profiling, pass `--module modulename:callable` and optional JSON kwargs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: str) -> int:
    print(f"Running command: {cmd}")
    return subprocess.run(cmd, shell=True).returncode


def run_module_call(module_call: str, kwargs_json: str | None) -> None:
    module_name, _, callable_name = module_call.partition(":")
    if not callable_name:
        raise RuntimeError("module-call must be MODULE:callable")
    module = importlib.import_module(module_name)
    callable_obj = getattr(module, callable_name)
    if kwargs_json:
        kwargs = json.loads(kwargs_json)
    else:
        kwargs = {}
    # Convert commonly-used path-like kwargs into pathlib.Path objects
    path_keys = [
        "spectrum",
        "peaklist",
        "z_values",
        "output",
        "spectrum_path",
        "peaklist_path",
        "z_values_path",
        "path_output",
    ]
    from pathlib import Path

    for k in list(kwargs.keys()):
        if k in path_keys and isinstance(kwargs[k], str):
            kwargs[k] = Path(kwargs[k])
    print(f"Calling {module_name}:{callable_name} with kwargs={kwargs}")
    callable_obj(**kwargs)


def cprofile_run(target_fn, output_file: str):
    import cProfile
    import pstats

    pr = cProfile.Profile()
    pr.enable()
    t0 = time.time()
    target_fn()
    t1 = time.time()
    pr.disable()
    print(f"Execution time: {t1 - t0:.3f}s")
    pr.dump_stats(output_file)
    # report top functions
    ps = pstats.Stats(pr).sort_stats("cumtime")
    print("\nTop 30 by cumulative time:")
    ps.print_stats(30)


def pyinstrument_run(target_fn, output_file: str):
    try:
        from pyinstrument import Profiler
    except Exception:
        print("pyinstrument not installed: pip install pyinstrument")
        raise
    prof = Profiler()
    prof.start()
    t0 = time.time()
    target_fn()
    t1 = time.time()
    prof.stop()
    print(f"Execution time: {t1 - t0:.3f}s")
    with open(output_file, "w") as fh:
        fh.write(prof.output_text(unicode=True))
    print(f"Wrote pyinstrument text output to: {output_file}")


def scalene_run(cmd: str, output_file: str):
    # scalene is a separate tool; we shell out to it for profiling a command
    try:
        pass  # just to detect presence
    except Exception:
        print("scalene not installed locally. Please run: pip install scalene")
        # We'll still attempt to use scalene if available as CLI
    call = f"scalene --reduced-profile --outfile {output_file} -- {cmd}"
    print(f"Running scalene: {call}")
    subprocess.run(call, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser(description="Profile PeakFit runs (CLI or module call)")
    parser.add_argument("--cmd", help="Shell command to run (e.g. 'uv run peakfit fit ...')")
    parser.add_argument(
        "--module",
        help="Module callable in the form module:callable (callable should be no-arg or accept kwargs)",
    )
    parser.add_argument("--kwargs", help="JSON string of kwargs for module callable (optional)")
    parser.add_argument(
        "--profiler",
        default="cprofile",
        choices=["cprofile", "pyinstrument", "scalene", "none"],
        help="Which profiler to use",
    )
    parser.add_argument("--output", default="./profile.prof", help="Output profile file path")
    parser.add_argument(
        "--repeat", type=int, default=1, help="How many times to repeat the workload (for timing)"
    )

    args = parser.parse_args()

    if args.cmd is None and args.module is None:
        print("Either --cmd or --module must be specified")
        sys.exit(2)

    def _target():
        if args.cmd:
            # run the command via subprocess
            rc = run_cmd(args.cmd)
            if rc != 0:
                raise RuntimeError(f"Command returned code {rc}")
        else:
            run_module_call(args.module, args.kwargs)

    # Repeat the workload to reduce variance if requested
    if args.profiler == "cprofile":
        for i in range(args.repeat):
            fpath = args.output
            if args.repeat > 1:
                fpath = str(Path(args.output).with_suffix(f".r{i}.prof"))
            print(f"Running cProfile iteration {i + 1}/{args.repeat}, saving to {fpath}")
            cprofile_run(_target, fpath)

    elif args.profiler == "pyinstrument":
        for i in range(args.repeat):
            fpath = args.output
            if args.repeat > 1:
                fpath = str(Path(args.output).with_suffix(f".r{i}.txt"))
            print(f"Running pyinstrument iteration {i + 1}/{args.repeat}, saving to {fpath}")
            pyinstrument_run(_target, fpath)

    elif args.profiler == "scalene":
        if args.module:
            raise RuntimeError("Scalene profiler only supports shell commands. Please set --cmd")
        for i in range(args.repeat):
            fpath = args.output
            if args.repeat > 1:
                fpath = str(Path(args.output).with_suffix(f".r{i}.scalene"))
            print(f"Running scalene iteration {i + 1}/{args.repeat}, saving to {fpath}")
            scalene_run(args.cmd, fpath)

    else:
        # none: just run the workload and time it
        for i in range(args.repeat):
            t0 = time.time()
            _target()
            t1 = time.time()
            print(f"Iteration {i + 1}/{args.repeat} runtime: {t1 - t0:.3f}s")


if __name__ == "__main__":
    main()
