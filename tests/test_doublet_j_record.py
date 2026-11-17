"""The doublet-J validation record follows its protocol, and the note quotes it."""

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs/validation/doublet-j.json"
NOTE = ROOT / "docs/validation/doublet-j.md"


@pytest.fixture(scope="module")
def tool():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "tools"))
        yield importlib.import_module("validate_doublet_j")


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_the_record_used_the_fixed_protocol(tool, record) -> None:
    protocol = record["protocol"]
    assert protocol["seed"] == tool.SEED
    assert protocol["realizations"] == tool.REALIZATIONS
    assert protocol["blocks"] == tool.BLOCKS
    assert protocol["gate_se"] == tool.GATE_SE
    assert protocol["nominal_coverage"] == tool.NOMINAL
    assert protocol["misleading_limit"] == tool.MISS_LIMIT
    assert [(c["lineshape"], c["regime"]) for c in record["cases"]] == [
        tuple(case) for case in tool.CASES
    ]


def test_the_verdict_follows_the_gates(record) -> None:
    for case in record["cases"]:
        gates = [*case["gates"].values(), *case["blocks"]["replication"].values()]
        assert case["passes"] == all(gate["passes"] for gate in gates)
        assert ("coverage_1sigma" in case["gates"]) == case["calibration_gated"]
    assert record["passes"] == all(case["passes"] for case in record["cases"])


def test_the_note_quotes_the_record(tool, record) -> None:
    assert tool.table(record) in NOTE.read_text(encoding="utf-8")
