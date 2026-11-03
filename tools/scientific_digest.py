"""Print an exact digest of a fit's scientific content, ignoring run identity and timing.

Run with ``uv run python tools/scientific_digest.py`` to fit the bundled example
(with the golden reference's arguments) and digest it, or pass a run directory or
``scientific.npz``. Run it on the base branch and on a change that must not alter
numerical behavior, in the same environment: the digests must be identical.

Every array member is hashed byte for byte. In ``metadata.json`` only the run ID,
timestamp, output directory and per-cluster fit times are removed; they differ
between identical runs. The digest is only meaningful within one environment
(platform, BLAS and dependency versions).
"""

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from write_golden_reference import run_example_fit

VOLATILE = {"run_id", "timestamp", "fit_time"}


def _without_volatile(value, path=()):
    if isinstance(value, dict):
        return {
            key: _without_volatile(item, (*path, key))
            for key, item in value.items()
            if key not in VOLATILE and (*path[-2:], key) != ("output", "mapping", "directory")
        }
    if isinstance(value, list):
        return [_without_volatile(item, path) for item in value]
    return value


def digests(archive: Path) -> dict[str, str]:
    """SHA-256 per top-level member group, plus the scrubbed metadata."""
    groups = {}
    with zipfile.ZipFile(archive) as members:
        for name in sorted(members.namelist()):
            payload = members.read(name)
            if name == "metadata.json":
                scrubbed = _without_volatile(json.loads(payload))
                payload = json.dumps(scrubbed, sort_keys=True).encode()
            group = groups.setdefault(name.split("/")[0], hashlib.sha256())
            group.update(name.encode() + b"\0" + payload)
    return {group: value.hexdigest() for group, value in groups.items()}


def main() -> None:
    """Digest the given fit, or fit the bundled example first."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="run directory or scientific.npz")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as output:
        path = args.path or run_example_fit(Path(output))
        archive = path / "scientific.npz" if path.is_dir() else path
        result = digests(archive)
    total = hashlib.sha256("".join(f"{k}:{v}" for k, v in sorted(result.items())).encode())
    for group, value in sorted(result.items()):
        print(f"{group:16} {value}")  # noqa: T201
    print(f"{'total':16} {total.hexdigest()}")  # noqa: T201


if __name__ == "__main__":
    main()
