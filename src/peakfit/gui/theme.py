"""One visual system for the desktop application: tokens, palette, stylesheet and fonts.

Every colour the GUI draws comes from a `Theme`. Chrome stays neutral; colour is
reserved for scientific layers, selection and state (specification §12).
"""

from dataclasses import dataclass

from PySide6 import QtGui, QtWidgets

# Logical-pixel rhythm shared by layouts.
SPACE_XS, SPACE_S, SPACE_M, SPACE_L = 4, 8, 12, 16
CONTROL_HEIGHT = 26
ROW_HEIGHT = 28
FONT_BODY, FONT_SMALL, FONT_CAPTION, FONT_TITLE = 13, 12, 11, 15


@dataclass(frozen=True, slots=True)
class Theme:
    """Named colours for one appearance; hex strings for QSS and pens alike."""

    dark: bool
    window: str
    chrome: str
    panel: str
    plot: str
    divider: str
    border: str
    text: str
    secondary: str
    faint: str
    accent: str
    on_accent: str
    selection: str
    hover: str
    warning: str
    warning_soft: str
    error: str
    success: str
    measured: str
    measured_negative: str
    model: str
    residual_positive: str
    residual_negative: str
    pin: str
    members: tuple[str, ...]


LIGHT = Theme(
    dark=False,
    window="#eef0f3",
    chrome="#f7f8fa",
    panel="#fbfbfc",
    plot="#ffffff",
    divider="#e1e5ea",
    border="#c9d0d8",
    text="#1d232b",
    secondary="#5f6a76",
    faint="#8e98a3",
    accent="#2f5da8",
    on_accent="#ffffff",
    selection="#dbe6f6",
    hover="#eef3fa",
    warning="#9a5b08",
    warning_soft="#fbefd9",
    error="#b23a2a",
    success="#2c7a5a",
    measured="#1f4f99",
    measured_negative="#8e3b8e",
    model="#c98512",
    residual_positive="#d2492e",
    residual_negative="#1f8a8a",
    pin="#c98512",
    members=("#2f5da8", "#7a5cc7", "#2e9c7a", "#c2577a"),
)

DARK = Theme(
    dark=True,
    window="#0f1318",
    chrome="#171c22",
    panel="#1a2027",
    plot="#13171c",
    divider="#29313a",
    border="#3a4552",
    text="#e1e6ec",
    secondary="#a3aeba",
    faint="#6f7a86",
    accent="#7ea6e8",
    on_accent="#0f1318",
    selection="#22324a",
    hover="#1d2733",
    warning="#e2a24a",
    warning_soft="#3a2c14",
    error="#f08a76",
    success="#6cc39b",
    measured="#8fb3ea",
    measured_negative="#d08ad0",
    model="#e9a83c",
    residual_positive="#f07a5e",
    residual_negative="#4cc2c2",
    pin="#e9a83c",
    members=("#8fb3ea", "#b39cf0", "#5fcaa6", "#ec8fb0"),
)


def color(value: str, alpha: float = 1.0) -> QtGui.QColor:
    """A QColor from a token, optionally translucent."""
    result = QtGui.QColor(value)
    result.setAlphaF(alpha)
    return result


def ui_font() -> QtGui.QFont:
    """System UI family at the body size, with tabular figures so numbers align."""
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.GeneralFont)
    font.setPixelSize(FONT_BODY)
    font.setFeature(QtGui.QFont.Tag("tnum"), 1)
    return font


def palette(theme: Theme) -> QtGui.QPalette:
    """Native widgets, dialogs and tooltips follow the same tokens as the stylesheet."""
    result = QtGui.QPalette()
    roles = QtGui.QPalette.ColorRole
    for role, value in {
        roles.Window: theme.chrome,
        roles.Base: theme.plot,
        roles.AlternateBase: theme.panel,
        roles.WindowText: theme.text,
        roles.Text: theme.text,
        roles.Button: theme.panel,
        roles.ButtonText: theme.text,
        roles.BrightText: theme.on_accent,
        roles.Highlight: theme.selection,
        roles.HighlightedText: theme.text,
        roles.PlaceholderText: theme.faint,
        roles.ToolTipBase: theme.panel,
        roles.ToolTipText: theme.text,
        roles.Link: theme.accent,
        roles.LinkVisited: theme.accent,
        roles.Mid: theme.divider,
        roles.Dark: theme.border,
        roles.Light: theme.plot,
        roles.Midlight: theme.panel,
        roles.Shadow: theme.border,
        roles.Accent: theme.accent,
    }.items():
        result.setColor(role, QtGui.QColor(value))
    for role in (roles.Text, roles.ButtonText, roles.WindowText):
        result.setColor(QtGui.QPalette.ColorGroup.Disabled, role, QtGui.QColor(theme.faint))
    return result


def stylesheet(t: Theme) -> str:
    """Scoped rules for states QPalette cannot express: hover, checked, focus, roles."""
    return f"""
QWidget {{ font-size: {FONT_BODY}px; }}
QMainWindow, QDialog {{ background: {t.chrome}; }}
QToolTip {{ background: {t.panel}; color: {t.text}; border: 1px solid {t.border};
    padding: 4px 6px; }}
QStatusBar {{ background: {t.chrome}; color: {t.secondary}; border-top: 1px solid {t.divider};
    font-size: {FONT_CAPTION}px; }}
QStatusBar QLabel {{ color: {t.secondary}; font-size: {FONT_CAPTION}px; }}
QStatusBar::item {{ border: 0; }}

QWidget#toolbar, QWidget#planeStrip {{ background: {t.chrome}; }}
QWidget#toolbar {{ border-bottom: 1px solid {t.divider}; }}
QWidget#planeStrip {{ border-top: 1px solid {t.divider}; }}
QWidget#navigator {{ background: {t.panel}; }}
QScrollArea#inspector, QWidget#inspectorBody {{ background: {t.panel}; border: 0; }}
QPushButton#tab {{ border: 0; border-bottom: 2px solid transparent; border-radius: 0;
    background: transparent; padding: 4px 0; min-height: 20px; color: {t.secondary};
    font-weight: 600; }}
QPushButton#tab:checked {{ color: {t.text}; border-bottom-color: {t.accent}; }}
QPushButton#tab:disabled {{ color: {t.faint}; }}
QPushButton#tab:hover:enabled {{ background: transparent; color: {t.text}; }}
QTreeWidget::item {{ height: {ROW_HEIGHT}px; border-bottom: 1px solid {t.divider}; }}
QTreeWidget::item:selected {{ background: {t.selection}; color: {t.text}; }}
QTreeWidget::item:hover:!selected {{ background: {t.hover}; }}

QLabel#title {{ font-size: {FONT_TITLE}px; font-weight: 600; }}
QLabel#welcome {{ font-size: 30px; font-weight: 600; }}
QLabel#section {{ font-size: {FONT_CAPTION}px; color: {t.faint}; letter-spacing: 0.5px; }}
QLabel#muted {{ color: {t.secondary}; font-size: {FONT_SMALL}px; }}
QLabel#caption {{ color: {t.faint}; font-size: {FONT_CAPTION}px; }}
QLabel#value {{ font-weight: 600; }}
QLabel#error {{ color: {t.error}; }}
QLabel#warning {{ color: {t.warning}; font-size: {FONT_SMALL}px; }}
QLabel#runChip {{ background: {t.selection}; color: {t.text}; border-radius: 11px;
    padding: 2px 10px; font-size: {FONT_SMALL}px; }}

QPushButton {{ min-height: {CONTROL_HEIGHT - 2}px; padding: 0 10px; border: 1px solid {t.border};
    border-radius: 5px; background: {t.panel}; color: {t.text}; }}
QPushButton:hover:enabled {{ background: {t.hover}; }}
QPushButton:pressed {{ background: {t.selection}; }}
QPushButton:checked {{ background: {t.selection}; border-color: {t.accent}; }}
QPushButton:disabled {{ color: {t.faint}; border-color: {t.divider}; }}
QPushButton:focus {{ border-color: {t.accent}; }}
QPushButton#primary {{ background: {t.accent}; border-color: {t.accent}; color: {t.on_accent};
    font-weight: 600; padding: 0 14px; }}
QPushButton#primary:hover:enabled {{ background: {t.accent}; }}
QPushButton#primary:disabled {{ background: {t.divider}; border-color: {t.divider};
    color: {t.faint}; }}
QPushButton#chip {{ border-radius: 11px; min-height: 20px; padding: 0 10px;
    color: {t.secondary}; font-size: {FONT_SMALL}px; background: transparent; }}
QPushButton#chip:hover:enabled {{ color: {t.text}; background: {t.hover}; }}
QPushButton#segment {{ border-radius: 0; border-right-width: 0; padding: 0 9px;
    color: {t.secondary}; background: {t.panel}; }}
QPushButton#segment[edge="first"] {{ border-top-left-radius: 5px;
    border-bottom-left-radius: 5px; }}
QPushButton#segment[edge="last"] {{ border-top-right-radius: 5px;
    border-bottom-right-radius: 5px; border-right-width: 1px; }}
QPushButton#segment:checked {{ background: {t.selection}; color: {t.text}; font-weight: 600;
    border-color: {t.border}; }}
QPushButton#segment:disabled {{ color: {t.faint}; }}

QToolButton {{ border: 1px solid transparent; border-radius: 5px; padding: 3px;
    background: transparent; color: {t.secondary}; }}
QToolButton:hover:enabled {{ background: {t.hover}; }}
QToolButton:checked {{ background: {t.selection}; }}
QToolButton:focus {{ border-color: {t.accent}; }}
QToolButton#text {{ padding: 3px 8px; }}

QLineEdit, QAbstractSpinBox, QComboBox {{ min-height: {CONTROL_HEIGHT - 4}px; padding: 0 6px;
    border: 1px solid {t.border}; border-radius: 5px; background: {t.plot}; color: {t.text};
    selection-background-color: {t.selection}; selection-color: {t.text}; }}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus {{ border: 1px solid {t.accent}; }}
QLineEdit:disabled, QAbstractSpinBox:disabled {{ color: {t.faint}; background: {t.panel};
    border-color: {t.divider}; }}

QListWidget, QListView, QTableView, QTreeView {{ border: 0; background: {t.panel};
    outline: 0; alternate-background-color: {t.panel}; }}
QListWidget::item {{ border-bottom: 1px solid {t.divider}; }}
QListWidget::item:selected {{ background: {t.selection}; color: {t.text}; }}
QListWidget::item:hover:!selected {{ background: {t.hover}; }}
QHeaderView::section {{ background: {t.panel}; color: {t.faint}; border: 0;
    border-bottom: 1px solid {t.divider}; padding: 4px 6px; font-size: {FONT_CAPTION}px; }}
QTableView {{ gridline-color: {t.divider}; }}

QGroupBox {{ border: 0; border-top: 1px solid {t.divider}; margin-top: 14px;
    padding-top: 14px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 0; padding: 0; }}
QScrollArea#inspector QGroupBox {{ border: 0; margin-top: 0; padding-top: 0; }}
QScrollArea {{ border: 0; background: transparent; }}
QTextBrowser {{ border: 0; background: {t.panel}; }}

QSplitter::handle {{ background: {t.divider}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

QSlider::groove:horizontal {{ height: 4px; border-radius: 2px; background: {t.divider}; }}
QSlider::sub-page:horizontal {{ height: 4px; border-radius: 2px; background: {t.accent}; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -6px 0; border-radius: 8px;
    background: {t.plot}; border: 2px solid {t.accent}; }}

QScrollBar:vertical {{ width: 10px; background: transparent; margin: 0; }}
QScrollBar:horizontal {{ height: 10px; background: transparent; margin: 0; }}
QScrollBar::handle {{ background: {t.border}; border-radius: 4px; min-height: 24px;
    min-width: 24px; margin: 2px; }}
QScrollBar::handle:hover {{ background: {t.faint}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QCheckBox {{ spacing: 6px; }}
"""


def apply(window: QtWidgets.QWidget, theme: Theme) -> None:
    """Install the palette application-wide (dialogs, tooltips) and the window stylesheet."""
    app = QtWidgets.QApplication.instance()
    assert isinstance(app, QtWidgets.QApplication)
    app.setFont(ui_font())
    app.setPalette(palette(theme))
    window.setStyleSheet(stylesheet(theme))
