"""Path resolution utilities."""

import datetime
from pathlib import Path


def resolve_output_path(base_dir: str | Path, include_timestamp: bool = False) -> Path:
    """Resolve the output directory.

    If `include_timestamp` is True, appends a timestamp to the directory name.
    Creates the directory (and parents) if it does not exist.

    Args:
        base_dir: The base directory path.
        include_timestamp: Whether to append a timestamp (YYYYMMDD_HHMMSS).

    Returns:
    -------
        The resolved Path object.

    Raises:
    ------
        ValueError: If base_dir is invalid (e.g. None or empty string).
        OSError: If directory creation fails.
    """
    if not base_dir:
        msg = "Base directory cannot be empty or None"
        raise ValueError(msg)

    output_path = Path(base_dir)

    if include_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if str(output_path) == ".":
            output_path = Path(f"output_{timestamp}")
        else:
            output_path = output_path / timestamp

    # Create directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path
