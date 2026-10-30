"""Sample concurrent process-tree RSS, including transient archive-worker memory."""

import argparse
import json
import subprocess
from pathlib import Path
from time import sleep


def main():
    """Run a command and sample its entire descendant tree every 20 ms."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-file", type=Path)
    parser.add_argument("--max-rss-bytes", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    process = subprocess.Popen(args.command)
    peak = 0
    samples = 0
    phases = {}
    processes = {}
    while process.poll() is None:
        listing = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss=,command="], capture_output=True, text=True, check=True
        )
        rows = []
        for row in listing.stdout.splitlines():
            pid, parent, rss, command = row.strip().split(None, 3)
            rows.append((int(pid), int(parent), int(rss) * 1024, command))
        phase = "unmarked"
        if args.phase_file and args.phase_file.exists():
            try:
                event = json.loads(args.phase_file.read_text())
                phase = event["phase"] + "/" + event["stage"]
            except json.JSONDecodeError:
                pass
        tree = {process.pid}
        while True:
            found = tree | {pid for pid, parent, _rss, _command in rows if parent in tree}
            if found == tree:
                break
            tree = found
        active = []
        for pid, parent, rss, command in rows:
            if pid not in tree:
                continue
            role = (
                "gui"
                if pid == process.pid
                else ("resource_tracker" if "resource_tracker" in command else "worker")
            )
            active.append({"pid": pid, "parent": parent, "rss_bytes": rss, "role": role})
            previous = processes.get(str(pid), {"peak_rss_bytes": 0})
            if rss >= previous["peak_rss_bytes"]:
                processes[str(pid)] = {
                    "role": role,
                    "command": command,
                    "peak_rss_bytes": rss,
                    "peak_phase": phase,
                }
        total = sum(p["rss_bytes"] for p in active)
        peak = max(peak, total)
        previous = phases.get(phase, {"peak_rss_bytes": 0, "samples": 0})
        previous["samples"] += 1
        if total >= previous["peak_rss_bytes"]:
            previous.update(peak_rss_bytes=total, breakdown=active)
        phases[phase] = previous
        samples += 1
        sleep(0.02)
    args.output.write_text(
        json.dumps(
            {
                "peak_concurrent_tree_rss_bytes": peak,
                "samples": samples,
                "returncode": process.returncode,
                "phases": phases,
                "processes": processes,
            },
            indent=2,
        )
        + "\n"
    )
    exceeded = args.max_rss_bytes is not None and peak > args.max_rss_bytes
    raise SystemExit(process.returncode or (2 if exceeded else 0))


if __name__ == "__main__":
    main()
