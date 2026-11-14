"""Readable chrome: text and control contrast, and type sizes from the system font (spec §12)."""

import re

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui import theme as themes

APPEARANCES = [pytest.param(themes.LIGHT, id="light"), pytest.param(themes.DARK, id="dark")]
SURFACES = ("window", "chrome", "panel", "plot")


def _luminance(value: QtGui.QColor) -> float:
    """WCAG 2.x relative luminance of an sRGB colour."""

    def linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(c) for c in (value.redF(), value.greenF(), value.blueF()))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: QtGui.QColor | str, b: QtGui.QColor | str) -> float:
    high, low = sorted((_luminance(QtGui.QColor(a)), _luminance(QtGui.QColor(b))), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_contrast_matches_known_wcag_values():
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast("#767676", "#ffffff") == pytest.approx(4.54, abs=0.01)


@pytest.mark.parametrize("theme", APPEARANCES)
@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("role", ["secondary", "faint"])
def test_secondary_text_is_readable_on_every_surface(theme, surface, role):
    assert contrast(getattr(theme, role), getattr(theme, surface)) >= 4.5


@pytest.mark.parametrize("theme", APPEARANCES)
@pytest.mark.parametrize("surface", SURFACES)
def test_control_boundaries_are_visible_on_every_surface(theme, surface):
    assert contrast(theme.outline, getattr(theme, surface)) >= 3.0


@pytest.mark.parametrize("theme", APPEARANCES)
def test_placeholders_are_readable_in_inputs(theme):
    palette = themes.palette(theme)
    roles = QtGui.QPalette.ColorRole
    assert contrast(palette.color(roles.PlaceholderText), palette.color(roles.Base)) >= 4.5


@pytest.mark.parametrize("theme", APPEARANCES)
def test_disabled_text_keeps_a_weaker_role_of_its_own(theme):
    assert contrast(theme.disabled, theme.panel) < contrast(theme.faint, theme.panel)


def _boundary_contrast(qapp, theme, make) -> float:
    """Render one unlabelled control on the panel surface; strongest pixel against it."""
    host = QtWidgets.QWidget()
    host.setObjectName("navigator")  # a panel surface
    host.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
    layout = QtWidgets.QVBoxLayout(host)
    control = make()
    control.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)  # the focus ring is not the boundary
    layout.addWidget(control)
    themes.apply(host, theme)
    host.resize(160, 60)
    host.show()
    qapp.processEvents()
    image = control.grab().toImage()
    host.close()
    pixels = (image.pixelColor(x, y) for y in range(image.height()) for x in range(image.width()))
    return max(contrast(pixel, theme.panel) for pixel in pixels)


@pytest.mark.parametrize("theme", APPEARANCES)
@pytest.mark.parametrize(
    "make",
    [QtWidgets.QLineEdit, QtWidgets.QCheckBox],
    ids=["line-edit", "checkbox"],
)
def test_rendered_controls_show_their_boundary(qapp, theme, make):
    assert _boundary_contrast(qapp, theme, make) >= 3.0


def _system_font(qapp, pixels: int) -> QtGui.QFont:
    font = QtGui.QFont(qapp.font())
    font.setPixelSize(pixels)
    return font


def test_the_specification_sizes_hold_at_the_default_system_font(qapp):
    sizes = themes.TypeScale.from_font(_system_font(qapp, 13))
    assert (sizes.body, sizes.small, sizes.caption, sizes.title) == (13, 12, 11, 15)


def test_a_larger_system_font_enlarges_every_size_in_proportion(qapp):
    # 17 px body: 17·12/13 ≈ 15.7, 17·11/13 ≈ 14.4, 17·15/13 ≈ 19.6, 17·30/13 ≈ 39.2.
    sizes = themes.TypeScale.from_font(_system_font(qapp, 17))
    assert (sizes.body, sizes.small, sizes.caption, sizes.title, sizes.display) == (
        17,
        16,
        14,
        20,
        39,
    )


def test_a_system_font_in_points_is_read_in_logical_pixels(qapp):
    font = QtGui.QFont(qapp.font())
    font.setPointSizeF(20.0)
    expected = QtGui.QFontInfo(font).pixelSize()
    assert expected > 13
    assert themes.TypeScale.from_font(font).body == expected


def test_a_smaller_system_font_keeps_the_specification_minimums(qapp):
    sizes = themes.TypeScale.from_font(_system_font(qapp, 9))
    assert (sizes.body, sizes.caption) == (13, 11)


@pytest.mark.parametrize("theme", APPEARANCES)
def test_the_stylesheet_sizes_text_only_from_the_derived_sizes(qapp, theme):
    sizes = themes.TypeScale.from_font(_system_font(qapp, 17))
    sheet = themes.stylesheet(theme, sizes)
    used = {int(px) for px in re.findall(r"font-size:\s*(\d+)px", sheet)}
    assert used == {sizes.body, sizes.small, sizes.caption, sizes.title, sizes.display}


def test_the_application_font_follows_the_derived_body_size(qapp):
    sizes = themes.TypeScale.from_font(_system_font(qapp, 17))
    assert themes.ui_font(sizes).pixelSize() == 17
