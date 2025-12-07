"""Tests for refactored IO writers."""

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

from peakfit.io.writers.base import flatten_diagnostics, get_peak_name
from peakfit.io.writers.csv_writer import CSVWriter


# Mocks
class ConvergenceStatus(str, Enum):
    GOOD = "Good"
    POOR = "Poor"


@dataclass
class MockParam:
    name: str
    param_id: Any = None


@dataclass
class MockParamId:
    peak_name: str


@dataclass
class MockParamDiag:
    name: str
    rhat: float | None = 1.0
    ess_bulk: float | None = 1000.0
    ess_tail: float | None = 800.0
    status: ConvergenceStatus = ConvergenceStatus.GOOD


@dataclass
class MockDiag:
    parameter_diagnostics: list[MockParamDiag]


@dataclass
class MockCluster:
    cluster_id: int
    peak_names: list[str]


@dataclass
class MockMetadata:
    timestamp: str = "2023-01-01"


@dataclass
class MockMethod:
    value: str = "method"


@dataclass
class MockResults:
    clusters: list[MockCluster]
    mcmc_diagnostics: list[MockDiag] | None = None
    metadata: MockMetadata = field(default_factory=MockMetadata)
    method: MockMethod = field(default_factory=MockMethod)


# Tests
def test_get_peak_name_legacy():
    # Legacy format: name_param
    assert get_peak_name(MockParam(name="2N-H_pos"), ["2N-H"]) == "2N-H"
    # Legacy format: simple prefix matching
    assert get_peak_name(MockParam(name="Peak1_w"), ["Peak1", "Peak2"]) == "Peak1"


def test_get_peak_name_modern():
    # Modern format: peak.axis.type
    assert get_peak_name(MockParam(name="2N-H.F1.pos"), ["2N-H"]) == "2N-H"


def test_get_peak_name_with_id():
    # Using param_id
    param = MockParam(name="irrelevant", param_id=MockParamId(peak_name="RealPeak"))
    assert get_peak_name(param, ["Other"]) == "RealPeak"


def test_get_peak_name_fallback():
    # Fallback
    assert get_peak_name(MockParam(name="unknown"), ["First", "Second"]) == "First"
    assert get_peak_name(MockParam(name="unknown"), []) == ""


def test_flatten_diagnostics():
    # Setup
    cluster1 = MockCluster(cluster_id=1, peak_names=["P1"])
    cluster2 = MockCluster(cluster_id=2, peak_names=["P2", "P3"])

    diag1 = MockDiag(
        parameter_diagnostics=[
            MockParamDiag(name="p1_param", rhat=1.01, status=ConvergenceStatus.GOOD)
        ]
    )
    diag2 = MockDiag(
        parameter_diagnostics=[
            MockParamDiag(name="p2_param", rhat=1.1, status=ConvergenceStatus.POOR)
        ]
    )

    results = MockResults(clusters=[cluster1, cluster2], mcmc_diagnostics=[diag1, diag2])

    # Execute
    flattened = list(flatten_diagnostics(results))

    # Verify
    assert len(flattened) == 2

    # Row 1
    assert flattened[0][0] == 1  # cluster_id
    assert flattened[0][1] == ["P1"]  # peak_names
    assert flattened[0][2] == "p1_param"  # param_name
    assert flattened[0][3] == 1.01  # rhat
    assert flattened[0][6] == "Good"  # status

    # Row 2
    assert flattened[1][0] == 2
    assert flattened[1][1] == ["P2", "P3"]
    assert flattened[1][6] == "Poor"


def test_csv_writer_diagnostics(tmp_path):
    # Setup
    cluster1 = MockCluster(cluster_id=1, peak_names=["P1"])
    diag1 = MockDiag(
        parameter_diagnostics=[
            MockParamDiag(
                name="p1_param",
                rhat=1.01,
                ess_bulk=100,
                ess_tail=200,
                status=ConvergenceStatus.GOOD,
            )
        ]
    )
    results = MockResults(clusters=[cluster1], mcmc_diagnostics=[diag1])

    writer = CSVWriter()
    output_path = tmp_path / "diagnostics.csv"

    # Execute
    writer.write_diagnostics(results, output_path)

    # Verify
    assert output_path.exists()
    content = output_path.read_text()

    # Parse CSV
    reader = csv.reader(content.splitlines())
    rows = list(reader)

    # Check header
    header_idx = -1
    for i, row in enumerate(rows):
        if row and row[0] == "cluster_id":
            header_idx = i
            break

    assert header_idx != -1
    data_row = rows[header_idx + 1]

    # cluster_id, peak_names, parameter, rhat, ess_bulk, ess_tail, status
    assert data_row[0] == "1"
    assert data_row[1] == "P1"
    assert data_row[2] == "p1_param"
    assert "1.0100" in data_row[3]
    assert data_row[4] == "100"
    assert data_row[5] == "200"
    assert data_row[6] == "Good"
