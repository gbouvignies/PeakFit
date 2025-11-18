from importlib import metadata

__version__ = "0.3.0"

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    pass  # Use the hardcoded version above
