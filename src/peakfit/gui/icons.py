"""Small stroke icons drawn for PeakFit and tinted from theme tokens."""

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtSvg

if TYPE_CHECKING:
    from peakfit.gui.theme import Theme

# 20×20 view box, drawn with a round 1.6 px stroke.
_SHAPES = {
    "panel-left": '<rect x="2.5" y="3.5" width="15" height="13" rx="2"/><path d="M7.5 3.5v13"/>',
    "add-peak": '<circle cx="10" cy="10" r="7"/><path d="M10 6.5v7M6.5 10h7"/>',
    "pin": (
        '<circle cx="10" cy="10" r="2.2"/>'
        '<path d="M10 2.5v4.3M10 13.2v4.3M2.5 10h4.3M13.2 10h4.3"/>'
    ),
    "full": '<path d="M3 7.5V3h4.5M12.5 3H17v4.5M17 12.5V17h-4.5M7.5 17H3v-4.5"/>',
    "sliders": (
        '<path d="M3 6h9M15.5 6H17M3 14h2.5M9 14h8"/>'
        '<circle cx="13.7" cy="6" r="1.8"/><circle cx="7.2" cy="14" r="1.8"/>'
    ),
    "review": '<path d="M4 5h8M4 10h8M4 15h5"/><path d="M12.5 14.5l2 2 3.5-4"/>',
    "chevron-left": '<path d="M12 4.5L6.5 10l5.5 5.5"/>',
    "chevron-right": '<path d="M8 4.5l5.5 5.5L8 15.5"/>',
}


def _pixmap(name: str, color: str, size: int, ratio: float) -> QtGui.QPixmap:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="none" '
        f'stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
        f"{_SHAPES[name]}</svg>"
    )
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode()))
    pixmap = QtGui.QPixmap(round(size * ratio), round(size * ratio))
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def icon(name: str, theme: Theme, size: int = 18) -> QtGui.QIcon:
    """Secondary tint at rest, accent when checked, faint when disabled."""
    ratio = 2.0
    result = QtGui.QIcon()
    for mode, state, value in (
        (QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off, theme.secondary),
        (QtGui.QIcon.Mode.Active, QtGui.QIcon.State.Off, theme.text),
        (QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On, theme.accent),
        (QtGui.QIcon.Mode.Active, QtGui.QIcon.State.On, theme.accent),
        (QtGui.QIcon.Mode.Disabled, QtGui.QIcon.State.Off, theme.faint),
        (QtGui.QIcon.Mode.Disabled, QtGui.QIcon.State.On, theme.faint),
    ):
        result.addPixmap(_pixmap(name, value, size, ratio), mode, state)
    return result
