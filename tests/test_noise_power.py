"""The noise-power check: processing the header does not record (correlated-noise §2.6, T5).

The NMRPipe fixtures are made by ``tools/make_nmrpipe_noise_fixtures.py``. Each row of
an impulse fixture is NMRPipe's response to one time-domain impulse, so random
combinations of the rows are NMRPipe-processed complex white noise, exactly.
"""

from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest

from peakfit.engine.algorithms.noise_correlation import NoisePowerStatus
from peakfit.engine.algorithms.noise_power import check_noise_power
from peakfit.io.readers.spectrum import read_noise_model

NMRPIPE = Path(__file__).parent / "data/nmrpipe"
CEST = Path(__file__).parents[1] / "examples/04-uncertainty-analysis/data/pseudo3d.ft2"
LINES = 2000


def _processed_noise(name: str, seed: int = 0) -> tuple[dict, np.ndarray]:
    """NMRPipe-processed noise lines: random combinations of the impulse responses."""
    dic, responses = ng.pipe.read(str(NMRPIPE / f"{name}.ft1"))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((LINES, len(responses))) @ np.asarray(responses, dtype=float)
    return dic, noise.astype(np.float32)


def _fixture(name: str) -> tuple[dict, np.ndarray]:
    """A committed NMRPipe-processed noise fixture: one processed noise FID per row."""
    dic, data = ng.pipe.read(str(NMRPIPE / f"{name}.ft1"))
    return dic, np.asarray(data, dtype=np.float32)


@pytest.mark.parametrize("name", ["sine_bell_p1_180", "digital_filter", "digital_filter_off"])
def test_correctly_recorded_processing_raises_no_alarm(name):
    dic, noise = _processed_noise(name)
    model = check_noise_power(read_noise_model(dic, noise.shape), noise)
    (axis,) = model.power_check
    assert axis.status is NoisePowerStatus.CONSISTENT
    assert axis.late_power == pytest.approx(1.0, abs=0.05)
    assert axis.lines == LINES
    assert model.power_warning is None


def test_the_same_noise_without_linear_prediction_raises_no_alarm():
    dic, noise = _fixture("noise_without_lp")
    model = check_noise_power(read_noise_model(dic, noise.shape), noise)
    (axis,) = model.power_check
    assert axis.status is NoisePowerStatus.CONSISTENT
    assert model.power_warning is None


@pytest.mark.parametrize(
    ("name", "loader"),
    [
        ("linear_prediction", _fixture),  # 32 acquired points extended to 64
        ("earlier_window", _processed_noise),  # EM -lb 100, then the recorded SP
        ("sp_size", _processed_noise),  # SP -size 16 over 32 acquired points
    ],
)
def test_processing_the_header_does_not_record_is_detected(name, loader):
    dic, noise = loader(name)
    header = read_noise_model(dic, noise.shape)
    assert header.modelled, header.reason  # the header alone looks like ordinary processing
    model = check_noise_power(header, noise)
    (axis,) = model.power_check
    assert axis.status is NoisePowerStatus.MISMATCH
    assert axis.late_power is not None
    assert axis.late_power < 0.85  # less late-FID noise than the recorded window predicts
    assert model.axes == header.axes  # a warning only: the model itself is unchanged
    warning = model.power_warning
    assert warning is not None
    assert "F2" in warning
    assert "check how" in warning.lower()
    assert "errors may be miscalibrated" in warning


def test_the_bundled_cest_example_s_nitrogen_axis_shows_linear_prediction():
    # About 47 of the 92 recorded ¹⁵N points were acquired; LP made the rest
    # (profile-uncertainty §2.2). The ¹H axis is processed as recorded.
    dic, data = ng.pipe.read(str(CEST))
    model = check_noise_power(read_noise_model(dic, data.shape), data)
    nitrogen, proton = model.power_check
    assert nitrogen.status is NoisePowerStatus.MISMATCH
    assert nitrogen.acquired_estimate is not None
    assert 44 <= nitrogen.acquired_estimate <= 50
    assert proton.status is NoisePowerStatus.CONSISTENT
    warning = model.power_warning
    assert warning is not None
    assert f"about {nitrogen.acquired_estimate} of the 92 recorded points" in warning
    assert "F3" not in warning


def test_one_plane_still_shows_a_large_mismatch():
    dic, data = ng.pipe.read(str(CEST))
    plane = data[:1]
    nitrogen, proton = check_noise_power(read_noise_model(dic, plane.shape), plane).power_check
    assert nitrogen.status is NoisePowerStatus.MISMATCH
    assert proton.status is NoisePowerStatus.INCONCLUSIVE  # 15 noise-only ¹H rows cannot decide
    assert proton.late_power is None


def test_too_few_lines_are_inconclusive_and_never_warn():
    dic, noise = _fixture("linear_prediction")
    few = noise[:20]
    model = check_noise_power(read_noise_model(dic, few.shape), few)
    (axis,) = model.power_check
    assert axis.status is NoisePowerStatus.INCONCLUSIVE
    assert model.power_warning is None


def test_a_model_without_a_prediction_is_not_checked():
    _dic, noise = _fixture("linear_prediction")
    header = read_noise_model({}, noise.shape)
    assert check_noise_power(header, noise) == header
    assert check_noise_power(header, noise).power_check == ()
