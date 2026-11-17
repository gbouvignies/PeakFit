"""The Model page's clustering, plane and constraint settings (W4, issue #216).

Each control commits the engine configuration of the equivalent TOML, refuses an
invalid value with its reason until it is corrected, and marks a setting that
differs from the active run's.
"""

import math
from typing import TYPE_CHECKING

import pytest
from builders import two_cluster_dataset
from PySide6 import QtWidgets

from peakfit.engine.domain.constraints import ParameterConstraint
from peakfit.gui import theme as themes
from peakfit.gui.editor.axis_table import DIFFERS, AxisInfo
from peakfit.gui.editor.constraints import rule_text
from peakfit.gui.editor.model_page import ModelPage, RunSettings, Setting
from peakfit.io.config import load_config

if TYPE_CHECKING:
    from peakfit.engine.domain.config import PeakFitConfig

pytestmark = pytest.mark.usefixtures("qapp", "windows")

AXES = [AxisInfo("F2", "¹⁵N F2", "sp2"), AxisInfo("F3", "¹H F3", "sp2")]
INSPECTOR_CONTENT = 288  # logical px: a 320 px inspector less its margins at 1280×800
_WINDOWS: list[QtWidgets.QWidget] = []


@pytest.fixture
def windows():
    yield
    _WINDOWS.clear()


@pytest.fixture(scope="module")
def preview():
    """A real preparation of the synthetic dataset: three planes, Gaussian on both axes."""
    _spectra, _peaks, _config, prepared = two_cluster_dataset()
    return prepared


def _toml(tmp_path, text: str = "") -> PeakFitConfig:
    path = tmp_path / f"peakfit-{len(list(tmp_path.iterdir()))}.toml"
    path.write_text(text + "\n")
    return load_config(path)


def _page(config: PeakFitConfig, preview=None, run: RunSettings | None = None):
    """The page in a themed window, following its own commits as the workspace does."""
    window = QtWidgets.QWidget()
    themes.apply(window, themes.LIGHT)
    layout = QtWidgets.QVBoxLayout(window)
    page = ModelPage()
    layout.addWidget(page)
    page.set_axes(AXES, planes=3)
    emitted: list[PeakFitConfig] = []

    def follow(settings):
        emitted.append(settings)
        page.show_state(settings, preview, "", run)

    page.changed.connect(follow)
    page.show_state(config, preview, "", run)
    window.resize(320, 900)
    window.show()
    _WINDOWS.append(window)
    return page, emitted


def _type(edit: QtWidgets.QLineEdit, text: str) -> None:
    edit.setText(text)
    edit.editingFinished.emit()


def _choose(combo: QtWidgets.QComboBox, data: str) -> None:
    combo.setCurrentIndex(combo.findData(data))


def _add_doublet(page, axis: int, j: str) -> None:
    row = page.axes.rows[axis]
    row.add.click()
    _type(row.j, j)


def _set_bounds(page, axis: int, low: str, high: str) -> None:
    row = page.axes.rows[axis]
    _type(row.low, low)
    _type(row.high, high)


def _hold(page, axis: int) -> None:
    page.axes.rows[axis].fit.setChecked(False)


def _phase(page, axis: int) -> None:
    page.axes.rows[axis].phase.setChecked(True)


def _family(page, axis: int, family: str) -> None:
    row = page.axes.rows[axis]
    row.family.setCurrentIndex(row.family.findData(family))


def _manual_noise(page, text):
    _choose(page.noise_mode, "manual")
    _type(page.noise_value, text)


def _level(page, text):
    _choose(page.threshold_mode, "level")
    _type(page.level_value, text)


def _factor(page, value):
    page.threshold.setValue(value)


def _exclude(page, text):
    _type(page.excluded, text)


def _window(page, axis, text):
    page.constraints_toggle.setChecked(True)
    _type(page.constraints.windows[axis], text)


def _rule(page, axis, kind, start="", low="", high="", hold=False):
    page.constraints_toggle.setChecked(True)
    editor = page.constraints
    editor.add_rule.click()
    row = editor.rules[-1]
    _choose(row.axis, axis)
    _choose(row.kind, kind)
    for edit, text in ((row.start, start), (row.low, low), (row.high, high)):
        if text:
            _type(edit, text)
    if hold:
        row.hold.setChecked(True)


@pytest.mark.parametrize(
    ("edits", "toml"),
    [
        # The per-axis table (W3).
        pytest.param(
            [(_add_doublet, 0, "92")], "[fitting.axes.F2]\ndoublet = { j = 92.0 }", id="doublet"
        ),
        pytest.param(
            [(_add_doublet, 0, "92"), (_set_bounds, 0, "80", "100")],
            "[fitting.axes.F2]\ndoublet = { j = 92.0, min = 80.0, max = 100.0 }",
            id="bounds",
        ),
        pytest.param(
            [(_add_doublet, 1, "7"), (_hold, 1)],
            "[fitting.axes.F3]\ndoublet = { j = 7.0, vary = false }",
            id="held",
        ),
        pytest.param([(_phase, 1)], "[fitting.axes.F3]\nphase = true", id="phase"),
        pytest.param(
            [(_family, 0, "sp1"), (_family, 1, "sp1")], '[fitting]\nfamily = "sp1"', id="common"
        ),
        pytest.param(
            [(_family, 1, "lorentzian")],
            '[fitting.axes.F3]\nfamily = "lorentzian"',
            id="one-axis-family",
        ),
        # Clustering, planes and constraints (W4).
        pytest.param([(_manual_noise, "0.0031")], "noise_level = 0.0031", id="manual-noise"),
        pytest.param(
            [(_manual_noise, "3.1 × 10⁻³")], "noise_level = 0.0031", id="manual-noise-scaled"
        ),
        pytest.param([(_level, "0.35")], "[clustering]\ncontour_level = 0.35", id="level"),
        pytest.param(
            [(_level, "1.5e5")], "[clustering]\ncontour_level = 150000.0", id="level-scientific"
        ),
        pytest.param([(_factor, 8.0)], "[clustering]\ncontour_factor = 8.0", id="factor"),
        pytest.param([(_exclude, "2, 3")], "exclude_planes = [1, 2]", id="exclude"),
        pytest.param([(_exclude, "1–2")], "exclude_planes = [0, 1]", id="exclude-range"),
        pytest.param(
            [(_window, "F2", "0.5")],
            "[parameters.position_windows]\nF2 = 0.5",
            id="position-window",
        ),
        pytest.param(
            [(_rule, "F2", "lw", "", "5", "100")],
            '[parameters.defaults]\n"*.F2.lw" = { min = 5.0, max = 100.0 }',
            id="bounds-rule",
        ),
        pytest.param(
            [(_rule, "*", "lw", "20", "", "", True)],
            '[parameters.defaults]\n"*.*.lw" = { value = 20.0, vary = false }',
            id="held-rule",
        ),
        pytest.param(
            [(_rule, "*", "lw", "", "5", "80"), (_rule, "F3", "lw", "", "", "", True)],
            '[parameters.defaults]\n"*.*.lw" = { min = 5.0, max = 80.0 }\n'
            '"*.F3.lw" = { vary = false }',
            id="ordered-rules",
        ),
    ],
)
def test_each_model_edit_gives_the_engine_config_of_the_equivalent_toml(edits, toml, tmp_path):
    page, emitted = _page(_toml(tmp_path))
    for edit, *arguments in edits:
        edit(page, *arguments)
    assert not page.invalid
    assert emitted[-1] == _toml(tmp_path, toml)


def test_the_estimate_stays_visible_beside_a_manual_noise(tmp_path, preview):
    config = _toml(tmp_path)
    page, emitted = _page(config, preview)
    estimate = preview.estimated_noise
    assert page.noise_mode.currentData() == "estimated"
    assert not page.noise_value.isVisible()
    # Choosing Manual starts from the estimate as shown: one commit of that value.
    _choose(page.noise_mode, "manual")
    assert page.noise_value.isVisible()
    assert emitted[-1].noise_level == pytest.approx(estimate, rel=1e-3)
    assert page.noise_value.text() == page.noise_estimate.text().removeprefix("estimate ")
    assert page.noise_estimate.isVisible()
    # Back to the estimate clears the value.
    _choose(page.noise_mode, "estimated")
    assert emitted[-1].noise_level is None


@pytest.mark.parametrize(
    ("edit", "text", "setting", "reason"),
    [
        (_manual_noise, "0", Setting.NOISE, "positive"),
        (_manual_noise, "-1", Setting.NOISE, "positive"),
        (_manual_noise, "abc", Setting.NOISE, "positive"),
        (_level, "0", Setting.THRESHOLD, "positive"),
        (_level, "nan", Setting.THRESHOLD, "positive"),
        (_exclude, "1-3", Setting.PLANES, "at least one plane"),
        (_exclude, "4", Setting.PLANES, "between 1 and 3"),
        (_exclude, "two", Setting.PLANES, "not a plane number"),
    ],
)
def test_an_invalid_value_is_explained_and_blocks_fit_until_corrected(
    tmp_path, edit, text, setting, reason
):
    config = _toml(tmp_path)
    page, emitted = _page(config)
    before = len(emitted)
    invalid: list[str] = []
    page.invalid_changed.connect(invalid.append)
    edit(page, text)
    # A mode switch may commit; the invalid text itself never does.
    assert all(settings == emitted[-1] for settings in emitted[before:])
    message = page.errors[setting]
    assert message.isVisible()
    assert reason in message.text()
    assert reason in page.invalid
    assert invalid[-1] == page.invalid
    field = {
        Setting.NOISE: page.noise_value,
        Setting.THRESHOLD: page.level_value,
        Setting.PLANES: page.excluded,
    }[setting]
    assert field.property("invalid") is True
    # A valid value commits and clears the reason.
    _type(field, "2")
    assert not page.invalid
    assert not message.isVisible()
    assert invalid[-1] == ""


@pytest.mark.parametrize(
    ("rule", "reason"),
    [
        (("F2", "lw", "120", "5", "100"), "outside its range"),
        (("F2", "lw", "", "100", "5"), "empty"),
        (("F2", "lw", "x"), "number"),
    ],
)
def test_a_rule_whose_bounds_do_not_contain_its_value_is_refused(tmp_path, rule, reason):
    page, emitted = _page(_toml(tmp_path))
    _rule(page, *rule)
    row = page.constraints.rules[-1]
    assert row.message.isVisible()
    assert reason in row.message.text()
    assert reason in page.invalid
    # Valid steps on the way may commit; the invalid combination never does.
    for settings in emitted:
        for kept in settings.parameters.defaults.patterns.values():
            low = -math.inf if kept.min is None else kept.min
            high = math.inf if kept.max is None else kept.max
            assert low < high
            assert kept.value is None or low <= kept.value <= high


def test_a_rule_on_a_parameter_the_model_lacks_is_refused(tmp_path, preview):
    page, _emitted = _page(_toml(tmp_path), preview)
    _rule(page, "F2", "eta", hold=True)  # Gaussian: no η
    row = page.constraints.rules[-1]
    assert row.unit.text() == ""  # η has no unit; the new rule's LW had Hz
    assert "matches no parameter" in row.message.text()
    assert "matches no parameter" in page.invalid


def test_an_imported_rule_for_named_peaks_is_shown_and_removable(tmp_path, preview):
    config = _toml(tmp_path, '[parameters.defaults]\n"2N-H.*.*" = { vary = false }')
    page, emitted = _page(config, preview)
    page.constraints_toggle.setChecked(True)
    (row,) = page.constraints.rules
    assert "2N-H.*.*" in row.pattern.text()
    assert "named peaks" in page.invalid
    row.remove.click()
    assert emitted[-1].parameters.defaults.patterns == {}
    assert not page.invalid


def test_a_new_rule_waits_for_a_value_with_a_hint(tmp_path):
    page, emitted = _page(_toml(tmp_path))
    page.constraints_toggle.setChecked(True)
    page.constraints.add_rule.click()
    row = page.constraints.rules[-1]
    assert row.message.isVisible()
    assert row.message.objectName() == "caption"  # a hint, not an error
    assert page.invalid  # Fit waits for the rule
    assert not emitted
    row.remove.click()
    assert not page.constraints.rules
    assert not page.invalid


def test_each_new_setting_is_marked_when_it_differs_from_the_run(tmp_path, preview):
    recorded = _toml(tmp_path)
    run = RunSettings("5d7b4768", recorded)
    page, _emitted = _page(recorded, preview, run)
    assert not page.differs.isVisible()
    _manual_noise(page, "0.0031")
    _level(page, "0.35")
    _exclude(page, "2")
    _window(page, "F2", "0.5")
    for setting, used in (
        (Setting.NOISE, "the estimate"),
        (Setting.THRESHOLD, "5 × noise"),
        (Setting.PLANES, "every plane"),
    ):
        assert page.titles[setting].text().endswith(DIFFERS)
        assert page.titles[setting].toolTip() == f"Run 5d7b4768 used {used}"
    assert page.constraints_differs.isVisible()
    assert page.constraints_differs.toolTip() == "Run 5d7b4768 used no constraints"
    assert "5d7b4768" in page.differs.text()
    # The run's own settings carry no mark.
    page.show_state(recorded, preview, "", run)
    assert not any(title.text().endswith(DIFFERS) for title in page.titles.values())
    assert not page.constraints_differs.isVisible()


def test_excluded_planes_read_as_plane_numbers(tmp_path):
    page, _emitted = _page(_toml(tmp_path, "exclude_planes = [0, 1]"))
    assert page.excluded.text() == "1–2"
    assert page.planes.text() == "1 of 3 fitted"


def test_every_new_setting_in_its_widest_state_fits_the_inspector(tmp_path, preview):
    config = _toml(tmp_path, '[parameters.defaults]\n"2N-H.*.*" = { vary = false }')
    page, _emitted = _page(config, preview)
    _manual_noise(page, "1.234 × 10⁻⁵")
    _level(page, "1.234 × 10⁵")
    _exclude(page, "1, 41–42")  # invalid here: three planes; its message wraps
    _rule(page, "F3", "eta", "0.5", "0.25", "0.75", hold=True)
    page.constraints.add_rule.click()
    assert page.minimumSizeHint().width() <= INSPECTOR_CONTENT


@pytest.mark.parametrize("pattern", ["*.F2.cs", "*.*.phase", "*"])
def test_an_imported_rule_the_desktop_fit_supports_stays_valid_and_editable(tmp_path, pattern):
    config = _toml(tmp_path, f'[parameters.defaults]\n"{pattern}" = {{ vary = false }}')
    page, emitted = _page(config)
    page.constraints_toggle.setChecked(True)
    (row,) = page.constraints.rules
    assert row.pattern.text() == pattern
    assert not page.invalid
    row.hold.setChecked(False)
    _type(row.start, "1")
    assert emitted[-1].parameters.defaults.patterns[pattern].value == 1.0
    assert emitted[-1].parameters.defaults.patterns[pattern].vary is None


def test_the_constraints_mark_names_what_the_run_used(tmp_path):
    recorded = _toml(tmp_path)
    page, _emitted = _page(recorded, run=RunSettings("5d7b4768", recorded))
    _window(page, "F2", "0.5")
    assert page.constraints_differs.toolTip() == "Run 5d7b4768 used no constraints"


def test_a_run_records_each_rule_as_submitted():
    rule = ParameterConstraint(value=20.0, min=5.0, max=80.25, vary=False)
    assert rule_text("*.F2.lw", rule) == "*.F2.lw: start 20 · range 5–80.25 · held"
    assert rule_text("*.*.eta", ParameterConstraint(max=0.5)) == "*.*.eta: range −∞–0.5"
