"""Fixtures shared across the test suite."""

import pytest


@pytest.fixture(scope="session")
def qapp():
    """The process's single QApplication; Qt is imported only by tests that request it."""
    from PySide6 import QtWidgets  # noqa: PLC0415

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
