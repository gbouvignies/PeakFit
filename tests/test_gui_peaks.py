"""Working-coordinate invariants without a Qt application."""

from pathlib import Path

import pytest

from peakfit.gui.peaks import PeakEdits, WorkingPeak, export_peaks
from peakfit.gui.spectrum import load_peaks, load_spectrum

DATA = Path(__file__).resolve().parents[1] / "examples" / "02-advanced-fitting" / "data"


def test_edit_history_keeps_identity_validates_atomically_and_tracks_saved_state() -> None:
    peak = WorkingPeak.new("duplicate", (120.1234567890123, 8.23456789012345))
    other = WorkingPeak.new(peak.name, peak.positions)
    edits = PeakEdits((peak, other))
    with pytest.raises(ValueError, match="finite"):
        edits.update(peak.id, "renamed", (float("nan"), 8.0))
    assert edits.peaks == (peak, other)
    assert not edits.dirty
    edits.update(peak.id, "renamed", (121.2345678901234, 8.12345678901234))
    assert edits.peaks[0].id == peak.id
    assert edits.peaks[1] == other
    edits.mark_saved()
    edits.delete(peak.id)
    edits.undo()
    assert not edits.dirty
    edits.undo()
    assert edits.peaks == (peak, other)
    assert edits.dirty
    edits.redo()
    assert not edits.dirty
    edits.add((122.0, 8.0))
    assert not edits.can_redo
    edits.undo()
    assert not edits.dirty


def test_export_round_trips_unfolded_coordinates_assignments_and_empty_list(tmp_path: Path) -> None:
    spectra = load_spectrum(DATA / "pseudo3d.ft2")
    peaks = load_peaks(DATA / "pseudo3d.list", spectra)
    assert len(peaks) == 166
    special = WorkingPeak.new('A, "quoted" assignment', (350.1234567890123, 8.123456789012345))
    special_names = tuple(WorkingPeak.new(name, special.positions) for name in ("NA", "001"))
    exported = (*peaks, special, *special_names)
    path = tmp_path / "edited.csv"
    export_peaks(path, exported)
    imported = load_peaks(path, spectra)
    for original, restored in zip(exported, imported, strict=True):
        assert restored.name == original.name
        assert restored.positions == original.positions
    export_peaks(path, ())
    assert load_peaks(path, spectra) == []
