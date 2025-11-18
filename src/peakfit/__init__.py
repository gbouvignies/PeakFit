import contextlib
from importlib import metadata

__version__ = "0.3.0"

with contextlib.suppress(metadata.PackageNotFoundError):
    __version__ = metadata.version(__name__)
