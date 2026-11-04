"""One bounded cache of total-model planes, and one normalized residual."""

import numpy as np

from peakfit.gui.fit_display import ModelPlane, ModelPlanes, normalized_residual


def _plane(size: int = 4) -> ModelPlane:
    values = np.ones((size, size))
    return ModelPlane(np.arange(size), np.arange(size), values, edge_visible=False)


def test_the_most_recent_eight_planes_are_retained():
    planes = ModelPlanes()
    for plane in range(9):
        planes.put("run", plane, _plane())
    assert planes.get("run", 0) is None
    assert planes.get("run", 1) is not None  # reading makes it most recent
    planes.put("run", 9, _plane())
    assert planes.get("run", 2) is None
    assert ("run", 1) in planes.retained()
    assert len(planes.retained()) == 8


def test_the_byte_bound_evicts_the_oldest_planes():
    planes = ModelPlanes()
    big = _plane(2048)  # 32 MiB of values
    for plane in range(5):
        planes.put("run", plane, big)
    assert planes.nbytes <= 128 * 1024**2
    assert planes.get("run", 0) is None
    assert planes.get("run", 4) is big


def test_keep_only_drops_other_runs():
    planes = ModelPlanes()
    planes.put("old", 0, _plane())
    planes.put("new", 0, _plane())
    planes.keep_only("new")
    assert planes.retained() == [("new", 0)]
    planes.keep_only(None)
    assert planes.retained() == []


def test_normalized_residual_is_measured_minus_model_over_noise():
    rng = np.random.default_rng(0)
    observed, model = rng.normal(size=(2, 5, 7))
    np.testing.assert_array_equal(
        normalized_residual(observed, model, 0.37), (observed - model) / 0.37
    )
