"""`peakfit gui` argument validation and routing, without importing Qt."""

import subprocess
import sys
import types
from pathlib import Path

import pytest
from click.utils import strip_ansi
from typer.testing import CliRunner

from peakfit.cli.app import app

DATA = Path(__file__).resolve().parents[1] / "examples" / "02-advanced-fitting" / "data"
SPECTRUM = DATA / "pseudo3d.ft2"
PEAKS = DATA / "pseudo3d.list"
VALUES = DATA / "b1_offsets.txt"


@pytest.fixture
def launches(monkeypatch):
    """Replace the Qt window module so routing is tested without starting Qt."""
    calls = []
    fake = types.ModuleType("peakfit.gui.editor.window")
    fake.launch = lambda *args, **kwargs: calls.append((args, kwargs))  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "peakfit.gui.editor.window", fake)
    return calls


def invoke(*args: str | Path):
    return CliRunner().invoke(app, ["gui", *map(str, args)])


def error(result) -> str:
    """Rich wraps errors in a box; compare words, not layout."""
    return " ".join(strip_ansi(result.output).replace("│", " ").split())


def test_empty_window(launches) -> None:
    result = invoke()
    assert result.exit_code == 0, result.output
    assert launches == [((None, None, None), {"swap_peak_axes": False})]


@pytest.mark.parametrize(("order", "swap"), [("indirect-direct", False), ("direct-indirect", True)])
def test_dataset_with_peaks_values_and_explicit_mapping(launches, order, swap) -> None:
    result = invoke(SPECTRUM, "--peaks", PEAKS, "-z", VALUES, "--peak-axis-order", order)
    assert result.exit_code == 0, result.output
    assert launches == [((SPECTRUM, PEAKS, VALUES), {"swap_peak_axes": swap})]


def test_saved_analysis_directory(launches, tmp_path) -> None:
    analysis = tmp_path / "Study.peakfit"
    analysis.mkdir()
    result = invoke(analysis)
    assert result.exit_code == 0, result.output
    assert launches == [((analysis, None, None), {"swap_peak_axes": False})]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--peaks", PEAKS, "--peak-axis-order", "indirect-direct"), "Provide a spectrum"),
        (("--z-values", VALUES), "Provide a spectrum"),
        (("--peak-axis-order", "direct-indirect"), "Provide a spectrum"),
        ((SPECTRUM, "--peaks", PEAKS), "requires --peak-axis-order"),
        ((SPECTRUM, "--peak-axis-order", "indirect-direct"), "only with --peaks"),
        ((SPECTRUM, "--peak-axis-order", "diagonal"), "diagonal"),
        ((DATA, "--z-values", VALUES), "is a directory"),
    ],
)
def test_invalid_combinations_are_rejected_before_launch(launches, args, message) -> None:
    result = invoke(*args)
    assert result.exit_code == 2
    assert message in error(result)
    assert launches == []


@pytest.mark.parametrize(
    "extra",
    [
        ("--peaks", PEAKS, "--peak-axis-order", "indirect-direct"),
        ("--z-values", VALUES),
        ("--peak-axis-order", "indirect-direct"),
    ],
)
def test_saved_analysis_rejects_dataset_companions(launches, tmp_path, extra) -> None:
    analysis = tmp_path / "Study.peakfit"
    analysis.mkdir()
    result = invoke(analysis, *extra)
    assert result.exit_code == 2
    assert "records its own inputs" in error(result)
    assert launches == []


def test_saved_analysis_must_be_a_directory(launches, tmp_path) -> None:
    analysis = tmp_path / "Study.peakfit"
    analysis.write_text("not an analysis")
    result = invoke(analysis)
    assert result.exit_code == 2
    assert "not a .peakfit analysis directory" in error(result)
    assert launches == []


def test_missing_input_is_rejected(launches, tmp_path) -> None:
    result = invoke(tmp_path / "missing.ft2")
    assert result.exit_code == 2
    assert launches == []


def test_help_documents_the_unified_entry_point() -> None:
    result = CliRunner().invoke(app, ["gui", "--help"])
    text = strip_ansi(result.output)
    assert ".peakfit" in text
    assert "--peak-axis-order" in text
    assert "--z-values" in text


def test_ordinary_cli_does_not_import_qt() -> None:
    script = (
        "import sys\n"
        "from typer.testing import CliRunner\n"
        "from peakfit.cli.app import app\n"
        "for args in (['--help'], ['gui', '--help'], ['fit', '--help']):\n"
        "    assert CliRunner().invoke(app, args).exit_code == 0\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in {'PySide6', 'pyqtgraph'}"
        " or m.startswith('peakfit.gui')]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
