"""Package metadata behavior."""

import importlib.util
from pathlib import Path


def test_source_only_version_fallback_is_explicit_unknown() -> None:
    """Source execution does not claim the last released package version."""
    source = Path(__file__).parents[1] / "src" / "peakfit" / "__init__.py"
    spec = importlib.util.spec_from_file_location("_peakfit_source_only", source)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.__version__ == "0+unknown"
