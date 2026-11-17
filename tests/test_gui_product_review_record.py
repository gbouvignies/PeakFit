"""The GUI product review quotes its record and cites exactly the figures it keeps."""

import importlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "docs/validation/gui-product-review.md"
FOLDER = ROOT / "docs/validation/gui-product-review"
RECORD = FOLDER / "measurements.json"
REVIEWED = "6a5dc18"  # main on 17 November 2026


@pytest.fixture(scope="module")
def tool():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "tools"))
        yield importlib.import_module("capture_gui_review")


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def note():
    return NOTE.read_text(encoding="utf-8")


def webp_size(path: Path) -> list[int]:
    """Pixel width and height from a WebP file's header (lossy, lossless or extended)."""
    head = path.read_bytes()[:30]
    assert head[:4] == b"RIFF", path
    assert head[8:12] == b"WEBP", path
    chunk = head[12:16]
    if chunk == b"VP8X":
        return [
            1 + int.from_bytes(head[24:27], "little"),
            1 + int.from_bytes(head[27:30], "little"),
        ]
    if chunk == b"VP8 ":
        return [
            int.from_bytes(head[26:28], "little") & 0x3FFF,
            int.from_bytes(head[28:30], "little") & 0x3FFF,
        ]
    bits = int.from_bytes(head[21:25], "little")
    return [(bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1]


def test_the_record_describes_the_reviewed_commit_on_a_native_window(record, note) -> None:
    assert record["commit"].startswith(REVIEWED)
    assert record["modified"] is False
    assert record["qt_platform"] == "cocoa"
    assert f"`{REVIEWED}`" in note


def test_the_note_quotes_the_record(tool, record, note) -> None:
    assert tool.table(record) in note


def test_the_note_cites_every_kept_figure_and_nothing_else(tool, record, note) -> None:
    cited = set(re.findall(r"\(gui-product-review/([\w.-]+\.webp)\)", note))
    kept = {path.name for path in FOLDER.iterdir()}
    assert cited == set(tool.FIGURES)
    assert kept == {*tool.FIGURES, RECORD.name}
    assert list(record["figures"]) == list(tool.FIGURES)


def test_each_figure_is_the_image_its_record_describes(record) -> None:
    for name, figure in record["figures"].items():
        assert webp_size(FOLDER / name) == figure["image"], name


def test_the_record_covers_the_reviewed_window_sizes(tool, record) -> None:
    assert [size["requested"] for size in record["sizes"]] == list(tool.SIZES)
    for size in record["sizes"]:
        assert size["window"]["logical"][0] == int(size["requested"].split("x")[0])
