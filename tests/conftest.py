"""Fixtures shared across the test suite."""

import sys

import pytest

_DEFAULT_LOOK: list = []  # the application's own font and palette, once it exists


@pytest.fixture(scope="session")
def qapp():
    """The process's single QApplication; Qt is imported only by tests that request it."""
    from PySide6 import QtWidgets  # noqa: PLC0415

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    if not _DEFAULT_LOOK:
        _DEFAULT_LOOK.extend((QtWidgets.QApplication.font(), QtWidgets.QApplication.palette()))
    return app


@pytest.fixture(autouse=True)
def _default_look():
    """Give every test the application's own font and palette.

    A window's theme sets them for the whole application, so without this a test
    would lay out with the font of whichever test ran before it on the same worker.
    """
    yield
    widgets = sys.modules.get("PySide6.QtWidgets")
    if widgets is not None and widgets.QApplication.instance() is not None and _DEFAULT_LOOK:
        font, palette = _DEFAULT_LOOK
        widgets.QApplication.setFont(font)
        widgets.QApplication.setPalette(palette)


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
