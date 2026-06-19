"""Qt Abstraction Layer (Shim).

This module provides a central point for importing Qt classes.
It currently relies on PySide6 but can be switched to PyQt6 or PyQt5 if needed.
"""

import os

# Force the Qt binding for Matplotlib and other tools if not set
if "QT_API" not in os.environ:
    os.environ["QT_API"] = "pyside6"

try:
    from PySide6.QtCore import QEventLoop, QObject, QRect, QSize, Qt, Signal, Slot
    from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSlider,
        QSpinBox,
        QSplitter,
        QStatusBar,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )

    # Aliases to match PyQt5 usage (if any specifically needed, though mostly aligned)
    # PySide6 uses connection.exec() instead of exec_() but we handle that in code.

except ImportError as exc:
    msg = "PySide6 is required for PeakFit plotting. Install project dependencies with `uv sync`."
    raise ImportError(msg) from exc

__all__ = [
    "QAction",
    "QActionGroup",
    "QApplication",
    "QCheckBox",
    "QComboBox",
    "QDoubleSpinBox",
    "QEventLoop",
    "QFileDialog",
    "QGroupBox",
    "QHBoxLayout",
    "QIcon",
    "QKeySequence",
    "QLabel",
    "QMainWindow",
    "QMessageBox",
    "QObject",
    "QPushButton",
    "QRect",
    "QScrollArea",
    "QSize",
    "QSlider",
    "QSpinBox",
    "QSplitter",
    "QStatusBar",
    "QToolBar",
    "QVBoxLayout",
    "QWidget",
    "Qt",
    "Signal",
    "Slot",
]
