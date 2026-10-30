"""Read genuine fixtures emitted by PR #71 at 1051ceb; migration is explicit."""

from pathlib import Path

import numpy as np
import pytest

from peakfit.gui.project import open_analysis
from peakfit.io.scientific import read_scientific, write_scientific

DATA = Path(__file__).parent / "data/storage-compat"


def test_original_scientific_v1_remains_readable_and_rewrites_v2(tmp_path):
    old, samples, context = read_scientific(DATA / "science-v1.npz")
    path = write_scientific(tmp_path / "migrated.npz", old, samples, context=context)
    new, restored, _ = read_scientific(path)
    for a, b in zip(old.clusters, new.clusters, strict=True):
        np.testing.assert_array_equal(
            a.sample_evaluation.model_values, b.sample_evaluation.model_values
        )
        np.testing.assert_array_equal(
            a.sample_evaluation.raw_residuals, b.sample_evaluation.raw_residuals
        )
    for a, b in zip(samples, restored, strict=True):
        assert a.neighbor_corrections.tobytes() == b.neighbor_corrections.tobytes()


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


def test_first_pr_scientific_context_remains_readable_without_invented_adoption():
    from peakfit.io.readers.continuation import load_completed  # noqa: PLC0415

    path = DATA / "science-v1-initial-context.npz"
    outcome, samples, context = read_scientific(path)
    assert "coordinate_table" not in context
    assert "export_state" not in context
    assert context["peak_identities"]
    assert outcome.n_optimizer_passes == 2
    assert samples
    restored = load_completed(path)
    assert restored.outcome.n_optimizer_passes == 2
