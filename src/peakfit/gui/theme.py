"""One visual system for the desktop application: tokens, palette, stylesheet, fonts, focus.

Every colour the GUI draws comes from a `Theme`. Chrome stays neutral; colour is
reserved for scientific layers, selection and state (specification §12).
"""

from dataclasses import dataclass

import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

# Logical-pixel rhythm shared by layouts.
SPACE_XS, SPACE_S, SPACE_M, SPACE_L = 4, 8, 12, 16
CONTROL_HEIGHT = 26
ROW_HEIGHT = 28
# Keyboard focus: a ring this wide, this far outside a control's own border.
FOCUS_WIDTH, FOCUS_GAP = 2, 1
# A widget with this property set draws its own focus frame (the 2D canvas frames
# its plot area); the shared ring leaves it alone.
OWN_FOCUS_FRAME = "ownFocusFrame"


@dataclass(frozen=True, slots=True)
class Theme:
    """Named colours for one appearance; hex strings for QSS and pens alike.

    Text roles: ``secondary`` and the quieter ``faint`` (captions, headers, placeholders)
    both reach 4.5:1 on every surface; ``disabled`` is weaker and marks only unavailable
    controls. ``outline`` bounds editable controls at 3:1 on every surface; ``border``
    stays a quiet edge for buttons and tooltips, ``divider`` a separator.
    """

    dark: bool
    window: str
    chrome: str
    panel: str
    plot: str
    divider: str
    border: str
    outline: str
    text: str
    secondary: str
    faint: str
    disabled: str
    accent: str
    on_accent: str
    focus: str
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
    outline="#7d8792",
    text="#1d232b",
    secondary="#5f6a76",
    faint="#636d79",
    disabled="#8e98a3",
    accent="#2f5da8",
    on_accent="#ffffff",
    focus="#1a56c8",
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
    outline="#66727f",
    text="#e1e6ec",
    secondary="#a3aeba",
    faint="#86919d",
    disabled="#6f7a86",
    accent="#7ea6e8",
    on_accent="#0f1318",
    focus="#8ab4ff",
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


# Specification §12 sizes at its reference 13 px body: body and controls, secondary
# text and plot labels, dense captions, selected identity, and the welcome heading.
_REFERENCE_BODY = 13
_RATIOS = {"small": 12, "caption": 11, "title": 15, "display": 30}


@dataclass(frozen=True, slots=True)
class TypeScale:
    """Every text size the GUI uses, in logical pixels, derived from one body size.

    The system's general font sets the body size; the others keep the specification's
    ratios to it, so a larger system font enlarges all of PeakFit's text alike. A
    system font smaller than the specification's 13 px body keeps the specification's
    sizes, so dense captions never fall below 11 px.
    """

    body: int
    small: int
    caption: int
    title: int
    display: int

    @classmethod
    def for_body(cls, body: int) -> TypeScale:
        """Sizes in the specification's ratios to ``body`` (no smaller than the reference)."""
        body = max(_REFERENCE_BODY, body)
        scaled = {name: round(body * size / _REFERENCE_BODY) for name, size in _RATIOS.items()}
        return cls(body=body, **scaled)

    @classmethod
    def from_font(cls, font: QtGui.QFont) -> TypeScale:
        """Sizes for a system font given in points or pixels (points follow screen DPI)."""
        return cls.for_body(QtGui.QFontInfo(font).pixelSize())


REFERENCE_TYPE = TypeScale.for_body(_REFERENCE_BODY)


class _Installed:
    scale: TypeScale | None = None


def type_scale() -> TypeScale:
    """The application's sizes, derived once from the system general font."""
    if _Installed.scale is None:
        system = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.GeneralFont)
        _Installed.scale = TypeScale.from_font(system)
    return _Installed.scale


def use_type_scale(scale: TypeScale | None) -> None:
    """Replace the derived sizes (``None`` derives them again); before widgets are built."""
    _Installed.scale = scale


def ui_font(sizes: TypeScale) -> QtGui.QFont:
    """System UI family at the body size, with tabular figures so numbers align."""
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.GeneralFont)
    font.setPixelSize(sizes.body)
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
        result.setColor(QtGui.QPalette.ColorGroup.Disabled, role, QtGui.QColor(theme.disabled))
    return result


def stylesheet(t: Theme, s: TypeScale) -> str:
    """Scoped rules for states QPalette cannot express: hover, checked, roles.

    Keyboard focus is not a stylesheet state: `FocusRing` draws it, once, for every
    control.
    """
    return f"""
QWidget {{ font-size: {s.body}px; }}
QMainWindow, QDialog {{ background: {t.chrome}; }}
QToolTip {{ background: {t.panel}; color: {t.text}; border: 1px solid {t.border};
    padding: 4px 6px; }}
QStatusBar {{ background: {t.chrome}; color: {t.secondary}; border-top: 1px solid {t.divider};
    font-size: {s.caption}px; }}
QStatusBar QLabel {{ color: {t.secondary}; font-size: {s.caption}px; }}
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
QPushButton#tab:disabled {{ color: {t.disabled}; }}
QPushButton#tab:hover:enabled {{ background: transparent; color: {t.text}; }}
QTreeWidget::item {{ height: {ROW_HEIGHT}px; border-bottom: 1px solid {t.divider}; }}
QTreeWidget::item:selected {{ background: {t.selection}; color: {t.text}; }}
QTreeWidget::item:hover:!selected {{ background: {t.hover}; }}

QLabel#title {{ font-size: {s.title}px; font-weight: 600; }}
QLabel#welcome {{ font-size: {s.display}px; font-weight: 600; }}
QLabel#section {{ font-size: {s.caption}px; color: {t.faint}; letter-spacing: 0.5px; }}
QLabel#muted {{ color: {t.secondary}; font-size: {s.small}px; }}
QLabel#caption {{ color: {t.faint}; font-size: {s.caption}px; }}
QLabel#value {{ font-weight: 600; }}
QLabel#error {{ color: {t.error}; }}
QLabel#muted[tone="error"] {{ color: {t.error}; }}
QLabel#warning {{ color: {t.warning}; font-size: {s.small}px; }}
QLabel#runChip {{ background: {t.selection}; color: {t.text}; border-radius: 11px;
    padding: 2px 10px; font-size: {s.small}px; }}

QPushButton {{ min-height: {CONTROL_HEIGHT - 2}px; padding: 0 10px; border: 1px solid {t.border};
    border-radius: 5px; background: {t.panel}; color: {t.text}; }}
QPushButton:hover:enabled {{ background: {t.hover}; }}
QPushButton:pressed {{ background: {t.selection}; }}
QPushButton:checked {{ background: {t.selection}; border-color: {t.accent}; }}
QPushButton:disabled {{ color: {t.disabled}; border-color: {t.divider}; }}
QPushButton#primary {{ background: {t.accent}; border-color: {t.accent}; color: {t.on_accent};
    font-weight: 600; padding: 0 14px; }}
QPushButton#primary:hover:enabled {{ background: {t.accent}; }}
QPushButton#primary:disabled {{ background: {t.divider}; border-color: {t.divider};
    color: {t.disabled}; }}
QPushButton#chip {{ border-radius: 11px; min-height: 20px; padding: 0 10px;
    color: {t.secondary}; font-size: {s.small}px; background: transparent; }}
QPushButton#chip:hover:enabled {{ color: {t.text}; background: {t.hover}; }}
QPushButton#segment {{ border-radius: 0; border-right-width: 0; padding: 0 9px;
    color: {t.secondary}; background: {t.panel}; }}
QPushButton#segment[edge="first"] {{ border-top-left-radius: 5px;
    border-bottom-left-radius: 5px; }}
QPushButton#segment[edge="last"] {{ border-top-right-radius: 5px;
    border-bottom-right-radius: 5px; border-right-width: 1px; }}
QPushButton#segment:checked {{ background: {t.selection}; color: {t.text}; font-weight: 600;
    border-color: {t.border}; }}
QPushButton#segment:disabled {{ color: {t.disabled}; }}

QToolButton {{ border: 1px solid transparent; border-radius: 5px; padding: 3px;
    background: transparent; color: {t.secondary}; }}
QToolButton:hover:enabled {{ background: {t.hover}; }}
QToolButton:checked {{ background: {t.selection}; }}
/* A disclosure: a quiet line on the content's left edge; its arrow shows it is open. */
QToolButton#text {{ padding: 1px 2px 1px 0; font-size: {s.small}px; }}
QToolButton#text:checked, QToolButton#text:hover:enabled {{ background: transparent; }}
QToolButton#text:hover:enabled {{ color: {t.text}; }}
QPushButton#link {{ border: 0; background: transparent; color: {t.accent}; padding: 1px 0;
  min-height: 0; text-align: left; }}
QPushButton#link:hover {{ text-decoration: underline; }}

QLineEdit, QAbstractSpinBox, QComboBox {{ min-height: {CONTROL_HEIGHT - 4}px; padding: 0 6px;
    border: 1px solid {t.outline}; border-radius: 5px; background: {t.plot}; color: {t.text};
    selection-background-color: {t.selection}; selection-color: {t.text}; }}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {{ color: {t.disabled};
    background: {t.panel}; border-color: {t.divider}; }}

QListWidget, QListView, QTableView, QTreeView {{ border: 0; background: {t.panel};
    outline: 0; alternate-background-color: {t.panel}; }}
QListWidget::item {{ border-bottom: 1px solid {t.divider}; }}
QListWidget::item:selected {{ background: {t.selection}; color: {t.text}; }}
QListWidget::item:hover:!selected {{ background: {t.hover}; }}
QHeaderView::section {{ background: {t.panel}; color: {t.faint}; border: 0;
    border-bottom: 1px solid {t.divider}; padding: 4px 6px; font-size: {s.caption}px; }}
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

QSlider:horizontal {{ min-height: 20px; }}
QSlider::groove:horizontal {{ height: 4px; border-radius: 2px; background: {t.divider}; }}
QSlider::sub-page:horizontal {{ height: 4px; border-radius: 2px; background: {t.accent}; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -7px 0; border-radius: 9px;
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


class ControlStyle(QtWidgets.QProxyStyle):
    """Fusion with token-coloured control boundaries the stylesheet cannot reach.

    Fusion derives the checkbox outline from the window colour (about 1.5:1); the
    stylesheet cannot restyle it without replacing the check mark, so the outline is
    redrawn here in the ``outline`` token. Fusion's own focus marks are not drawn:
    `FocusRing` is the one keyboard-focus treatment (specification §12).
    """

    def __init__(self, theme: Theme) -> None:
        super().__init__("Fusion")
        self.theme = theme

    # PySide6's stubs declare QCommonStyle and QProxyStyle overloads that conflict
    # with each other and omit QStyleOption's ``state`` and ``rect`` members.
    def drawPrimitive(  # noqa: N802  # ty: ignore[invalid-method-override]
        self,
        element: QtWidgets.QStyle.PrimitiveElement,
        option: QtWidgets.QStyleOption,
        painter: QtGui.QPainter,
        /,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        """Fusion's drawing without its focus marks, then the checkbox outline in the token."""
        elements = QtWidgets.QStyle.PrimitiveElement
        if element == elements.PE_FrameFocusRect:
            return
        if element != elements.PE_IndicatorCheckBox:
            super().drawPrimitive(element, option, painter, widget)
            return
        unfocused = QtWidgets.QStyleOptionButton(option)  # ty: ignore[no-matching-overload]
        unfocused.state &= ~QtWidgets.QStyle.StateFlag.State_HasFocus  # ty: ignore[unresolved-attribute]
        super().drawPrimitive(element, unfocused, painter, widget)
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.translate(0.5, 0.5)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(self.theme.outline)))
        painter.drawRect(option.rect.adjusted(0, 0, -1, -1))  # ty: ignore[unresolved-attribute]
        painter.restore()


def apply(window: QtWidgets.QWidget, theme: Theme) -> None:
    """Install the style and palette application-wide (dialogs, tooltips) and the stylesheet."""
    app = QtWidgets.QApplication.instance()
    assert isinstance(app, QtWidgets.QApplication)
    style = app.style()
    if isinstance(style, ControlStyle):
        style.theme = theme
    else:
        app.setStyle(ControlStyle(theme))
    app.setFont(ui_font(type_scale()))
    app.setPalette(palette(theme))
    window.setStyleSheet(stylesheet(theme, type_scale()))
    FocusRing.install(app, theme)


class _Ring(QtWidgets.QWidget):
    """The painted ring: a transparent, mouse-transparent sibling over the focused widget."""

    def __init__(self) -> None:
        super().__init__()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.color = QtGui.QColor()
        self.radius = 0.0

    def paintEvent(self, _event) -> None:  # noqa: N802
        """A FOCUS_WIDTH stroke along the widget's edge, square inside views."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, self.radius > 0)
        pen = QtGui.QPen(self.color, FOCUS_WIDTH)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        half = FOCUS_WIDTH / 2
        painter.drawRoundedRect(
            QtCore.QRectF(self.rect()).adjusted(half, half, -half, -half), self.radius, self.radius
        )


class FocusRing(QtCore.QObject):
    """The one keyboard-focus indicator of the application (specification §12, UX-4).

    It follows the application's focus widget and draws a FOCUS_WIDTH ring in the
    focus token. Around a control the ring sits FOCUS_GAP outside the control's own
    border, so it touches only the surface and never an accent fill (the Fit
    button); it differs from checked and selected states, which are fills and
    one-pixel borders. A list, tree or other scroll area fills its panel, so the
    ring is drawn on the view's own edge instead. Widgets marked OWN_FOCUS_FRAME
    draw their own frame.

    Like CSS ``:focus-visible``, the ring marks keyboard focus only: it shows once
    focus arrives by Tab, Backtab or a shortcut, or a key is pressed, and hides
    while focus came from, or follows, a mouse press.
    """

    _instance: FocusRing | None = None

    def __init__(self, app: QtWidgets.QApplication) -> None:
        super().__init__(app)
        self.ring: _Ring | None = None
        self.target: QtWidgets.QWidget | None = None
        self.watched: list[QtWidgets.QWidget] = []
        self.color = QtGui.QColor(LIGHT.focus)
        self.keyboard = False
        self.modality = _Modality(self)
        app.installEventFilter(self.modality)
        app.focusChanged.connect(self._follow)

    @classmethod
    def install(cls, app: QtWidgets.QApplication, theme: Theme) -> FocusRing:
        """The application's ring, created once, in the theme's focus colour."""
        if cls._instance is None or not shiboken6.isValid(cls._instance):
            cls._instance = cls(app)
        ring = cls._instance
        ring.color = QtGui.QColor(theme.focus)
        ring._follow(None, app.focusWidget())
        return ring

    def _painter(self) -> _Ring:
        # Deleting the focused widget's parent deletes the ring with it.
        if self.ring is None or not shiboken6.isValid(self.ring):
            self.ring = _Ring()
        return self.ring

    def _follow(self, _old, new: QtWidgets.QWidget | None) -> None:
        for widget in self.watched:
            if shiboken6.isValid(widget):
                widget.removeEventFilter(self)
        self.target, self.watched = None, []
        ring = self._painter()
        ring.hide()
        if (
            new is None
            or new.isWindow()
            or new.parentWidget() is None
            # Menus and combo-box popups show their own current item.
            or new.window().windowType() == QtCore.Qt.WindowType.Popup
            or new.property(OWN_FOCUS_FRAME)
        ):
            return
        host = self._host(new)
        ring.color = self.color
        # Reparent before watching anything: a move during the reparent must not
        # place and raise the ring mid-reparent (a native crash on macOS).
        ring.setParent(host)
        self.target = new
        # The target and its ancestors below the host: any of them moving moves the ring.
        widget: QtWidgets.QWidget | None = new
        while widget is not None and widget is not host:
            widget.installEventFilter(self)
            self.watched.append(widget)
            widget = widget.parentWidget()
        self._place()

    def _rect(self, target: QtWidgets.QWidget, host: QtWidgets.QWidget) -> QtCore.QRect:
        """The ring's rectangle in `host` coordinates."""
        inside = isinstance(target, QtWidgets.QAbstractScrollArea)
        margin = 0 if inside else FOCUS_WIDTH + FOCUS_GAP
        origin = target.mapTo(host, QtCore.QPoint(0, 0))
        return QtCore.QRect(origin, target.size()).adjusted(-margin, -margin, margin, margin)

    def _host(self, target: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """The nearest ancestor with room for the ring, never past scrolled content.

        A control flush with its parent's edge would lose part of its ring; climbing
        to an ancestor that contains it keeps the ring whole. A scroll area's
        content widget is the last stop, so the ring scrolls with the control.
        """
        host = target.parentWidget()
        assert host is not None
        while not host.rect().contains(self._rect(target, host)):
            parent = host.parentWidget()
            if (
                parent is None
                or host.isWindow()
                or isinstance(parent, QtWidgets.QAbstractScrollArea)
                or isinstance(parent.parentWidget(), QtWidgets.QAbstractScrollArea)
            ):
                break
            host = parent
        return host

    def _place(self) -> None:
        target, ring = self.target, self._painter()
        host = ring.parentWidget()
        if target is None or host is None or not shiboken6.isValid(target):
            ring.hide()
            return
        ring.radius = 0.0 if isinstance(target, QtWidgets.QAbstractScrollArea) else 6.0
        ring.setGeometry(self._rect(target, host))
        ring.setVisible(self.keyboard and target.isVisibleTo(host))
        ring.raise_()
        ring.update()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        """Keep the ring on the focused widget as it or an ancestor moves, resizes or hides."""
        kind = event.type()
        if watched is self.target and kind == QtCore.QEvent.Type.ParentChange:
            self._follow(None, self.target)
        elif watched in self.watched and kind in (
            QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Show,
            QtCore.QEvent.Type.Hide,
        ):
            self._place()
        return False

    def set_keyboard(self, keyboard: bool) -> None:
        """Record the input modality and show or hide the ring to match."""
        if keyboard != self.keyboard:
            self.keyboard = keyboard
            self._place()


_KEYBOARD_REASONS = (
    QtCore.Qt.FocusReason.TabFocusReason,
    QtCore.Qt.FocusReason.BacktabFocusReason,
    QtCore.Qt.FocusReason.ShortcutFocusReason,
)
_MODIFIERS = (
    QtCore.Qt.Key.Key_Shift,
    QtCore.Qt.Key.Key_Control,
    QtCore.Qt.Key.Key_Meta,
    QtCore.Qt.Key.Key_Alt,
    QtCore.Qt.Key.Key_AltGr,
    QtCore.Qt.Key.Key_CapsLock,
)


class _Modality(QtCore.QObject):
    """Whether the user is working by keyboard or by pointer, as `FocusRing` needs it.

    An application-wide filter: key presses and keyboard focus reasons mean
    keyboard, mouse presses and mouse focus mean pointer. Programmatic focus keeps
    the current modality.
    """

    def __init__(self, ring: FocusRing) -> None:
        super().__init__(ring)
        self.ring = ring

    def eventFilter(self, _watched, event) -> bool:  # noqa: N802
        """Update the ring's modality from input and focus events; never consume them."""
        kind = event.type()
        if kind == QtCore.QEvent.Type.KeyPress:
            if event.key() not in _MODIFIERS:
                self.ring.set_keyboard(True)
        elif kind in (
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.MouseButtonDblClick,
        ):
            self.ring.set_keyboard(False)
        elif kind == QtCore.QEvent.Type.FocusIn:
            reason = event.reason()
            if reason in _KEYBOARD_REASONS:
                self.ring.set_keyboard(True)
            elif reason == QtCore.Qt.FocusReason.MouseFocusReason:
                self.ring.set_keyboard(False)
        return False
