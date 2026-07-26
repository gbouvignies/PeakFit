"""Operational metadata for durable completed-fit projections."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

try:
    __version__ = metadata.version("peakfit")
except metadata.PackageNotFoundError:
    __version__ = "unknown"


@dataclass
class RunMetadata:
    """Reproducibility metadata that is separate from scientific fit truth."""

    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    software_version: str = __version__
    git_commit: str | None = None
    python_version: str = sys.version
    platform: str = field(default_factory=platform.platform)
    input_files: dict[str, dict[str, str]] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    command_line: str = ""
    run_duration_seconds: float | None = None


def capture_output_metadata(config: dict[str, Any], input_files: dict[str, Path]) -> RunMetadata:
    """Capture operational metadata without evaluating a fitted model."""
    run_metadata = RunMetadata(
        git_commit=_current_git_commit(),
        configuration=config,
    )
    for name, path in input_files.items():
        if path.exists():
            run_metadata.input_files[name] = {
                "path": path.name,
                "checksum_sha256": _compute_file_checksum(path),
            }
    return run_metadata


def _current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return result.stdout.strip()[:12] if result.returncode == 0 else None


def _compute_file_checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["RunMetadata", "capture_output_metadata"]
