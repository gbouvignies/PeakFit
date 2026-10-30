"""Read genuine fixtures emitted by PR #71 at 1051ceb; migration is explicit."""

from pathlib import Path

import numpy as np
import pytest

from peakfit.gui.project import open_analysis, save_analysis
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
def test_original_linked_project_remains_numerically_readable(tmp_path, name):
    analysis = open_analysis(DATA / name)
    assert analysis.dataset is None  # Original source deliberately unavailable.
    assert analysis.completion.result.outcome.n_optimizer_passes == 2
    assert analysis.completion.result.completed_fit.samples
    saved = save_analysis(analysis, tmp_path / "Migrated.peakfit")
    reopened = open_analysis(saved)
    assert reopened.dataset is None
    assert reopened.completion.result.run_id == analysis.completion.result.run_id
    assert not (saved / "spectra").exists()
    with pytest.raises(ValueError, match="Portable export requires"):
        save_analysis(analysis, tmp_path / "Portable.peakfit", portable=True)


@pytest.mark.parametrize("name", ["Legacy.peakfit", "Milestone3.peakfit"])
def test_portable_export_of_old_project_requires_verified_source(tmp_path, monkeypatch, name):
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    from test_gui_dataset import _dataset  # noqa: PLC0415

    from peakfit.gui import project  # noqa: PLC0415
    from peakfit.gui.spectrum import Dataset  # noqa: PLC0415

    spectrum, _, _, _ = _dataset()
    # Exact source verification must not compare a regenerated libm-dependent
    # synthetic spectrum with bytes recorded on another platform.
    spectrum.data = np.load(DATA / "source.npy", allow_pickle=False)
    source = tmp_path / "verified.ft2"
    source.write_bytes(b"legacy")
    record = json.dumps(
        {"path": str(source), "checksum_sha256": hashlib.sha256(b"legacy").hexdigest()}
    )
    dataset = Dataset(source, spectrum, (), (("spectrum", record),), True, 5.0, 0.01)
    monkeypatch.setattr(project, "read_dataset", lambda *_args, **_kw: dataset)
    linked = open_analysis(DATA / name, relink=source)
    assert linked.dataset is not None, linked.reason
    saved = save_analysis(linked, tmp_path / "Migrated.peakfit", portable=True)
    source.unlink()
    reopened = open_analysis(saved)
    assert reopened.dataset is not None, reopened.reason
    np.testing.assert_array_equal(reopened.dataset.spectra.data, spectrum.data)
    assert reopened.completion.result.run_id == linked.completion.result.run_id


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


def test_plain_json_project_v1_layout_remains_readable(tmp_path):
    import gzip  # noqa: PLC0415
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415
    import shutil  # noqa: PLC0415

    path = tmp_path / "Plain.peakfit"
    shutil.copytree(DATA / "Milestone3.peakfit", path)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["schema_version"] = "1.0.0"
    for entry in manifest["runs"]:
        directory = path / "runs" / entry["run_id"]
        with gzip.open(directory / "summary/fit.json.gz", "rb") as stream:
            payload = stream.read()
        (directory / "summary/fit.json").write_bytes(payload)
        del entry["files"]["summary/fit.json.gz"]
        entry["files"]["summary/fit.json"] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        (directory / "summary/fit.json.gz").unlink()
    (path / "manifest.json").write_text(json.dumps(manifest))
    assert open_analysis(path).completion.result.outcome.n_optimizer_passes == 2


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
