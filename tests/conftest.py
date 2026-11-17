"""Fixtures shared across the test suite."""

import pytest


@pytest.fixture(scope="session")
def qapp():
    """The process's single QApplication; Qt is imported only by tests that request it."""
    from PySide6 import QtWidgets  # noqa: PLC0415

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def settings(tmp_path):
    """Keep application preferences out of the developer's own settings."""
    from PySide6 import QtCore  # noqa: PLC0415

    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat, QtCore.QSettings.Scope.UserScope, str(tmp_path)
    )
    QtCore.QSettings().clear()
    yield QtCore.QSettings()
    QtCore.QSettings().clear()
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.NativeFormat)
