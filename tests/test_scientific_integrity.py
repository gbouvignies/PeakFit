"""Integrity checks raise one error type, with numpy's tolerance rule and pinned tolerances."""

import json
import struct
from zipfile import ZipFile

import numpy as np
import pytest
from builders import completed_result

from peakfit.gui.fit_export import export_fit_result
from peakfit.io import scientific
from peakfit.io.readers import continuation
from peakfit.io.readers.continuation import restore_completed
from peakfit.shared import exceptions
from peakfit.shared.exceptions import ScientificIntegrityError, require_close


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (1.0, 1.0),
        (1.0 + 0.9e-10, 1.0),
        (1.0 + 1.1e-9, 1.0),
        (1e-13, 0.0),
        (1e-11, 0.0),
        (np.nan, np.nan),
        (np.nan, 1.0),
        (np.inf, np.inf),
        (np.inf, -np.inf),
        ([1.0, 2.0], 1.0),
        ([1.0, 1.0], 1.0),
        ([1.0, 2.0], [1.0, 2.0, 3.0]),
        ([[1.0, 2.0]], [1.0, 2.0]),
    ],
)
def test_require_close_decides_as_numpy_assert_allclose(actual, expected):
    try:
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    except AssertionError:
        with pytest.raises(ScientificIntegrityError, match="differs"):
            require_close(actual, expected, rtol=1e-10, atol=1e-12, message="differs")
    else:
        require_close(actual, expected, rtol=1e-10, atol=1e-12, message="differs")


@pytest.mark.parametrize("detailed", [False, True])
def test_integrity_tolerances_are_pinned(tmp_path, monkeypatch, detailed):
    spectra, _, _, result = completed_result([], False, detailed)
    directory = export_fit_result(result, tmp_path / "results")
    used = {}

    def recording(module):
        def check(actual, expected, *, rtol, atol, message):
            used.setdefault(module, set()).add((rtol, atol))
            exceptions.require_close(actual, expected, rtol=rtol, atol=atol, message=message)

        return check

    monkeypatch.setattr(scientific, "require_close", recording("archive"))
    monkeypatch.setattr(continuation, "require_close", recording("continuation"))
    outcome, samples, context = scientific.read_scientific(directory / "scientific.npz")
    restore_completed(outcome, samples, context, source_data=spectra.data)

    evaluator, model, terminal = (1e-10, 1e-12), (1e-10, 1e-9), (1e-8, 1e-9)
    if detailed:
        assert used == {"archive": {model}, "continuation": {evaluator, model}}
    else:
        assert used == {"continuation": {evaluator, terminal}}


def _rewrite(path, change):
    with ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    document = json.loads(members["metadata.json"])
    change(document)
    members["metadata.json"] = json.dumps(document).encode()
    with ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


@pytest.mark.parametrize(
    ("damage", "detail"),
    [
        ("truncated", "File is not a zip file"),
        ("missing", "missing field 'statistics'"),
        ("mistyped", "must be a mapping"),
        ("inconsistent", "Recorded chi-squared disagrees"),
    ],
)
def test_damaged_archive_names_the_file_and_the_next_step(tmp_path, damage, detail):
    _, _, _, result = completed_result([], False, True)
    archive = export_fit_result(result, tmp_path / "results") / "scientific.npz"
    if damage == "truncated":
        archive.write_bytes(archive.read_bytes()[:100])
    elif damage == "missing":
        _rewrite(archive, lambda d: d["outcome"].pop("statistics"))
    elif damage == "mistyped":
        _rewrite(archive, lambda d: d["outcome"].update(statistics="none"))
    else:

        def chi_squared(document):
            statistics = document["outcome"]["clusters"][0]["analytical_evaluation"]["statistics"]
            statistics["chi_squared"] = {"float": struct.pack(">d", 2.0).hex()}

        _rewrite(archive, chi_squared)
    with pytest.raises(ScientificIntegrityError) as error:
        scientific.read_scientific(archive)
    message = str(error.value)
    assert message.startswith("Cannot read scientific archive ")
    assert "scientific.npz" in message
    assert detail in message
    assert message.endswith("Refit the dataset, or restore an intact copy of the file.")
