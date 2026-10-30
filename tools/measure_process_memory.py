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
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    process = subprocess.Popen(args.command)
    peak = 0
    samples = 0
    while process.poll() is None:
        listing = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True, check=True
        )
        rows = [tuple(map(int, row.split())) for row in listing.stdout.splitlines()]
        tree = {process.pid}
        while True:
            found = tree | {pid for pid, parent, _rss in rows if parent in tree}
            if found == tree:
                break
            tree = found
        peak = max(peak, sum(rss * 1024 for pid, _parent, rss in rows if pid in tree))
        samples += 1
        sleep(0.02)
    args.output.write_text(
        json.dumps(
            {
                "peak_concurrent_tree_rss_bytes": peak,
                "samples": samples,
                "returncode": process.returncode,
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
