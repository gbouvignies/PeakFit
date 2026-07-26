from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from peakfit.io.readers.results import ResultsLoader

if TYPE_CHECKING:
    from pathlib import Path


def test_results_loader_uses_canonical_summary_path(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    (summary_dir / "fit.json").write_text("{}\n", encoding="utf-8")

    loader = ResultsLoader(tmp_path)

    assert loader.summary_path == tmp_path / "summary" / "fit.json"


def test_results_loader_rejects_summary_subdirectory(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    (summary_dir / "fit.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        ResultsLoader(summary_dir)
