"""Legacy state must retain the actual terminal likelihood inputs."""

import pickle

import numpy as np
import pytest
from test_scientific_storage import completed_result

from peakfit.io.state import load_state, save_state


def test_pickle_round_trip_retains_terminal_corrections(tmp_path):
    _, _, pipeline, _ = completed_result([], False)
    state = pipeline.continuation_state
    assert any(np.any(c.corrections) for c in state.clusters)
    path = save_state(tmp_path / "state.pkl", state)
    loaded = load_state(path)
    for old, new in zip(state.clusters, loaded.clusters, strict=True):
        np.testing.assert_array_equal(old.corrections, new.corrections)
        np.testing.assert_array_equal(old.corrected_data, new.corrected_data)


def test_legacy_state_without_corrections_remains_readable(tmp_path):
    _, _, pipeline, _ = completed_result([], False)
    path = tmp_path / "legacy.pkl"
    path.write_bytes(pickle.dumps(pipeline.continuation_state.model_dump()))
    with pytest.warns(RuntimeWarning, match="omitted terminal corrections"):
        loaded = load_state(path)
    assert len(loaded.clusters) == len(pipeline.continuation_state.clusters)
