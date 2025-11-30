#!/usr/bin/env python3
"""Summarize top CPU and memory hotspots from a Scalene JSON profile.

Usage: python tools/profiling/parse_scalene_summary.py tmp/scalene.json
"""

import json
import sys
from pathlib import Path


def summarize(path: Path, top_n: int = 10):
    with path.open("r") as f:
        data = json.load(f)

    files = data.get("files", {})

    per_file = []
    for fname, fdata in files.items():
        total_copy_mb_s = 0.0
        total_growth_mb = 0.0
        cpu_sample_count = 0
        total_python_percent = 0.0
        total_c_percent = 0.0

        for line in fdata.get("lines", []):
            total_copy_mb_s += line.get("n_copy_mb_s", 0.0)
            total_growth_mb += line.get("n_growth_mb", 0.0)
            cpu_sample_count += len(line.get("cpu_samples_list", []) or [])
            total_python_percent += line.get("n_cpu_percent_python", 0.0) or 0.0
            total_c_percent += line.get("n_cpu_percent_c", 0.0) or 0.0

        per_file.append(
            (
                fname,
                round(total_copy_mb_s, 6),
                round(total_growth_mb, 6),
                cpu_sample_count,
                round(total_python_percent, 6),
                round(total_c_percent, 6),
            )
        )

    per_line = []
    for fname, fdata in files.items():
        for line in fdata.get("lines", []):
            lineno = line.get("lineno")
            growth = line.get("n_growth_mb", 0.0) or 0.0
            copy = line.get("n_copy_mb_s", 0.0) or 0.0
            samples = len(line.get("cpu_samples_list", []) or [])
            if growth or copy or samples:
                per_line.append(
                    (fname, lineno, growth, copy, samples, line.get("line", "").strip())
                )

    def sort_key_cpu(item):
        # prefer python percent then c percent then available samples
        _, _a, _b, samples, py_percent, c_percent = item
        return (py_percent + c_percent, samples)

    def sort_key_mem(item):
        _, copy, growth, samples, py_percent, c_percent = item
        return (growth, copy)

    print("\nTop files by total memory growth (MB):")
    for i, item in enumerate(sorted(per_file, key=sort_key_mem, reverse=True)[:top_n]):
        print(
            f"{i + 1:2d}. {item[0]}  growth={item[2]:>8} MB, copy/s={item[1]:>8} MB/s, cpu_samples={item[3]}, py%={item[4]} c%={item[5]}"
        )

    print("\nTop files by CPU (sum of python+c percent):")
    for i, item in enumerate(sorted(per_file, key=sort_key_cpu, reverse=True)[:top_n]):
        print(
            f"{i + 1:2d}. {item[0]}  py%={item[4]:>6} c%={item[5]:>6}, samples={item[3]}, growth={item[2]} MB, copy/s={item[1]} MB/s"
        )

    print("\nTop lines by memory growth (MB):")
    for i, item in enumerate(sorted(per_line, key=lambda x: x[2], reverse=True)[:top_n]):
        fname, lineno, growth, copy, samples, code = item
        print(
            f"{i + 1:2d}. {fname}:{lineno} growth={growth} MB, copy/s={copy} MB/s, samples={samples}  code={code}"
        )

    print("\nTop lines by copy MB/s:")
    for i, item in enumerate(sorted(per_line, key=lambda x: x[3], reverse=True)[:top_n]):
        fname, lineno, growth, copy, samples, code = item
        print(
            f"{i + 1:2d}. {fname}:{lineno} copy/s={copy} MB/s, growth={growth} MB, samples={samples}  code={code}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: parse_scalene_summary.py <scalene.json>")
        sys.exit(1)
    summarize(Path(sys.argv[1]))
