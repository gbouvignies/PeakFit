"""The inspector's Run page fits the inspector without sideways scrolling."""

import pytest
from PySide6 import QtWidgets

from peakfit.gui import theme as themes
from peakfit.gui.editor.labels import Row
from peakfit.gui.editor.run_page import RunPage

INSPECTOR = 320  # logical px at the 13 px body; it widens with the body size


@pytest.fixture(params=[13, 17], ids=["13px", "17px"])
def body(request, qapp):
    """Sizes as derived from a system font; the application font restored afterwards."""
    font, palette = qapp.font(), qapp.palette()
    themes.use_type_scale(themes.TypeScale.for_body(request.param))
    yield request.param
    themes.use_type_scale(None)
    qapp.setFont(font)
    qapp.setPalette(palette)


@pytest.mark.usefixtures("body")
def test_the_run_page_with_every_section_fits_the_inspector(qapp):
    window = QtWidgets.QWidget()
    themes.apply(window, themes.LIGHT)
    page = RunPage()
    page.setParent(window)
    page.show_run(
        [
            Row("Run", "f714e22e · Outdated"),
            Row("Warnings", "▲ 12 clusters", style="warning"),
            Row(
                "Noise check",
                "▲ ¹⁵N F2 · errors may be miscalibrated",
                style="warning",
                info="What was found and what to do.",
            ),
        ],
        fitted=True,
        warned=True,
    )
    page.show_unexplained(
        "Find on plane 131",
        True,
        "",
        [],
        "Diagnostic only · residual beyond ±5 σ_stat",
        about="What the list is.",
    )
    window.show()
    qapp.processEvents()
    assert not page.unexplained_info.isHidden()
    assert page.minimumSizeHint().width() <= themes.text_room(INSPECTOR) - 2 * themes.SPACE_L
