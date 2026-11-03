from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from peakfit.io.readers.results import ResultsLoader

if TYPE_CHECKING:
    from pathlib import Path


def test_results_loader_reads_only_the_scientific_archive(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    (summary_dir / "fit.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=r"No scientific\.npz"):
        ResultsLoader(tmp_path)
