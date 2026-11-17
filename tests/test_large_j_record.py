"""The large-J validation records follow their protocol, and the note quotes them."""

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs/validation/large-j.json"
CONFIRMATION = ROOT / "docs/validation/large-j-confirmation.json"
NOTE = ROOT / "docs/validation/large-j.md"


@pytest.fixture(scope="module")
def tool():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "tools"))
        yield importlib.import_module("validate_large_j")


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def confirmation():
    return json.loads(CONFIRMATION.read_text(encoding="utf-8"))


def test_the_record_used_the_fixed_protocol(tool, record) -> None:
    protocol = record["protocol"]
    assert protocol["seed"] == tool.SEED
    assert protocol["realizations"] == tool.REALIZATIONS
    assert protocol["blocks"] == tool.BLOCKS
    assert protocol["gate_se"] == tool.GATE_SE
    assert protocol["nominal_coverage"] == tool.NOMINAL
    assert protocol["misleading_limit"] == tool.MISS_LIMIT
    assert protocol["qualified_at_least"] == tool.QUALIFIED_AT_LEAST
    assert protocol["qualified_at_most"] == tool.QUALIFIED_AT_MOST
    assert protocol["contour"] == tool.CONTOUR
    assert protocol["contour_noise"] == tool.CONTOUR_NOISE
    assert [case["case"] for case in record["cases"]] == tool.NAMES


def test_the_verdict_follows_the_gates(tool, record) -> None:
    for case in record["cases"]:
        gates = [*case["gates"].values(), *case["blocks"]["replication"].values()]
        assert case["passes"] == all(gate["passes"] for gate in gates)
        assert case["calibration_gated"] == (case["case"] in tool.CALIBRATED)
        assert case["gates"]["region"]["rate"] == 1.0
    assert record["passes"] == all(case["passes"] for case in record["cases"])


def test_the_confirmation_takes_every_near_limit_gate(tool, record, confirmation) -> None:
    near = [
        [case["case"], name]
        for case in record["cases"]
        for name, gate in case["gates"].items()
        if gate.get("near_limit")
    ]
    protocol = confirmation["protocol"]
    assert protocol["confirmed"] == near
    assert protocol["seed"] == tool.SEED
    assert protocol["first_index"] == tool.CONFIRMATION_FIRST
    assert protocol["realizations"] == tool.CONFIRMATION_REALIZATIONS
    assert confirmation["passes"] == all(g["confirmation"]["passes"] for g in confirmation["gates"])


def test_the_note_quotes_the_records(tool, record, confirmation) -> None:
    note = NOTE.read_text(encoding="utf-8")
    assert tool.case_table(record) in note
    assert tool.recovery_table(record) in note
    assert tool.confirmation_table(confirmation) in note
