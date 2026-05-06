"""IO utilities."""

import os
from pathlib import Path


def format_path(path: Path | str) -> str:
    """Format paths for user-facing output using a CWD-relative representation."""
    p = Path(path)
    if not p.is_absolute():
        return str(p)

    try:
        return os.path.relpath(p.resolve(), Path.cwd().resolve())
    except Exception:
        return str(p)
