"""Genuine fixtures emitted by PR #71 at 1051ceb: earlier formats are rejected explicitly."""

from pathlib import Path

import pytest

from peakfit.gui.project import open_analysis
from peakfit.io.scientific import read_scientific

DATA = Path(__file__).parent / "data/storage-compat"


@pytest.mark.parametrize("name", ["science-v1.npz", "science-v1-initial-context.npz"])
def test_earlier_scientific_versions_are_explicitly_rejected(name):
    with pytest.raises(ValueError, match="Unsupported scientific result version"):
        read_scientific(DATA / name)


@pytest.mark.parametrize("name", ["Legacy.peakfit", "Milestone3.peakfit"])
def test_obsolete_project_formats_are_explicitly_rejected(name):
    with pytest.raises(ValueError, match="Unsupported analysis version"):
        open_analysis(DATA / name)


def test_legacy_json_only_is_readable_but_cannot_invent_mcmc_parent(tmp_path):
    import gzip  # noqa: PLC0415

    from peakfit.io.readers.results import ResultsLoader  # noqa: PLC0415
    from peakfit.mcmc.analysis import run_mcmc_analysis  # noqa: PLC0415

    source = next((DATA / "Milestone3.peakfit").rglob("fit.json.gz"))
    (tmp_path / "summary").mkdir()
    with gzip.open(source, "rb") as stream:
        (tmp_path / "summary/fit.json").write_bytes(stream.read())
    assert ResultsLoader(tmp_path).load_summary().clusters
    with pytest.raises(ValueError, match="lacks validated terminal samples"):
        run_mcmc_analysis(tmp_path, n_steps=2)
