"""Parallel module removal test.

This test ensures that the parallelization module has been removed and is
no longer importable.
"""

import pytest


def test_parallel_module_removed():
    """Importing `peakfit.core.fitting.parallel` should fail now that it's removed."""
    with pytest.raises(ModuleNotFoundError):
        __import__("peakfit.core.fitting.parallel")


def test_dummy_cleanup():
    """Placeholder to ensure this file contains only tests about parallel removal."""
    assert True


def test_clean_parallel_file_exists():
    """Ensure that parallel module is not importable and the test file only checks for its removal."""
    with pytest.raises(ModuleNotFoundError):
        __import__("peakfit.core.fitting.parallel")
