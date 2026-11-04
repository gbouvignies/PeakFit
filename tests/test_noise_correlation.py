"""Processing-derived noise correlation: docs/correlated-noise.md §2 and validation C1."""

from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest

from peakfit.engine.algorithms.noise_correlation import (
    AxisProcessing,
    NoiseModel,
    cluster_correlation,
    not_modelled,
)
from peakfit.io.readers.spectrum import read_noise_model

CEST = Path(__file__).parents[1] / "examples/04-uncertainty-analysis/data/pseudo3d.ft2"


def _processed_covariance(axis: AxisProcessing) -> np.ndarray:
    """Covariance of the real processed points for unit complex white noise in the FID.

    The processing is linear: apodize, zero fill, Fourier transform, apply the
    first-order phase and keep the real part. Its covariance is Re[A Aᴴ].
    """
    lags = np.arange(axis.transform)
    fourier = np.exp(-2j * np.pi * np.outer(lags, np.arange(axis.acquired)) / axis.transform)
    phase = np.exp(1j * np.deg2rad(axis.phase) * lags / axis.transform)
    operator = phase[:, None] * fourier * axis.weights()[None, :]
    return np.real(operator @ operator.conj().T)


@pytest.mark.parametrize(
    "axis",
    [
        AxisProcessing(12, 40, "sine_bell", offset=0.35, end=0.98, power=2.0, first_point=0.5),
        AxisProcessing(10, 32, "sine_bell", offset=0.4, end=0.98, power=1.0, phase=180.0),
        AxisProcessing(16, 64, "none", first_point=0.5, phase=-90.0),
    ],
)
def test_c1_the_correlation_is_that_of_the_processing_operator(axis):
    covariance = _processed_covariance(axis)
    sd = np.sqrt(np.diag(covariance))
    expected = covariance / np.outer(sd, sd)
    lags = np.abs(np.subtract.outer(np.arange(axis.transform), np.arange(axis.transform)))
    np.testing.assert_allclose(axis.correlation()[lags], expected, atol=1e-12)


def test_without_zero_fill_or_apodization_the_points_are_independent():
    rho = AxisProcessing(32, 32, "none").correlation()
    assert rho[0] == 1.0
    np.testing.assert_allclose(rho[1:], 0.0, atol=1e-12)


def test_c3_twofold_zero_fill_without_apodization_leaves_a_small_correlation():
    # Odd lags keep 1/N: small, but not zero.
    n = 92
    rho = AxisProcessing(n, 2 * n, "none").correlation()
    np.testing.assert_allclose(rho[1::2], 1.0 / n, atol=1e-12)
    np.testing.assert_allclose(rho[2::2], 0.0, atol=1e-12)


def test_a_cluster_correlation_is_the_separable_product_of_its_axes():
    axes = (
        AxisProcessing(20, 64, "sine_bell", offset=0.4, end=0.98, power=2.0),
        AxisProcessing(30, 80, "sine_bell", offset=0.3, end=0.98, power=2.0, first_point=0.5),
    )
    first, second = np.meshgrid(np.arange(3, 8), np.arange(10, 14), indexing="ij")
    grid = [first.ravel(), second.ravel()]
    correlation = cluster_correlation(NoiseModel(axes=axes), grid)

    def per_axis(axis, index):
        return axis.correlation()[np.abs(np.subtract.outer(index, index))]

    expected = np.kron(per_axis(axes[0], np.arange(3, 8)), per_axis(axes[1], np.arange(10, 14)))
    np.testing.assert_allclose(correlation, expected, atol=1e-15)
    np.testing.assert_array_equal(np.diag(correlation), 1.0)


def test_no_correlation_is_formed_when_it_is_not_modelled():
    with pytest.raises(ValueError, match="not modelled: no header"):
        cluster_correlation(not_modelled("no header"), [np.arange(3)])


@pytest.fixture(scope="module")
def cest_header():
    dic, data = ng.pipe.read(str(CEST))
    return dic, data.shape


def test_the_bundled_cest_header_is_modelled_from_its_processing(cest_header):
    dic, shape = cest_header
    model = read_noise_model(dic, shape)
    assert model.modelled, model.reason
    nitrogen, proton = model.axes
    assert (nitrogen.acquired, nitrogen.transform, nitrogen.window) == (92, 256, "sine_bell")
    assert (nitrogen.first_point, nitrogen.phase, nitrogen.power) == (1.0, 180.0, 2.0)
    assert nitrogen.offset == pytest.approx(0.4)
    assert (proton.acquired, proton.transform, proton.window) == (768, 2048, "sine_bell")
    assert (proton.first_point, proton.phase, proton.power) == (0.5, 0.0, 2.0)
    assert proton.offset == pytest.approx(0.3)
    # The lag-1 correlations the specification reports (§2.4).
    assert nitrogen.correlation()[1] == pytest.approx(0.764, abs=5e-4)
    assert proton.correlation()[1] == pytest.approx(0.656, abs=5e-4)


@pytest.mark.parametrize(
    ("changes", "modelled"),
    [
        # ¹H is extracted: APOD 204 for N·S/M = 768 × 546/2048 = 204.75.
        ({"FDF2APOD": 205.0}, True),  # rounded to nearest
        ({"FDF2APOD": 203.0}, False),
        ({"FDF2APOD": 300.0}, False),  # a window shorter or longer than the acquisition
        # ¹⁵N is not extracted: APOD must equal TDSIZE exactly.
        ({"FDF3APOD": 91.0}, False),
    ],
)
def test_the_apodized_length_must_agree_within_one_rounded_point(cest_header, changes, modelled):
    dic, shape = cest_header
    model = read_noise_model({**dic, **changes}, shape)
    assert model.modelled is modelled
    if not modelled:
        assert "APOD" in str(model.reason)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"FDF2APODCODE": 2.0}, "unsupported apodization"),
        ({"FDF2TDSIZE": 0.0}, "TDSIZE"),
        ({"FDF2FTSIZE": 512.0}, "FTSIZE"),
        ({"FDF3FTFLAG": 0.0}, "not Fourier transformed"),
        ({"FDF2APODQ3": float("nan")}, "sine-bell parameters"),
        ({"FDF2C1": None}, "first-point scale"),
    ],
)
def test_insufficient_or_unsupported_headers_fall_back_explicitly(cest_header, changes, message):
    dic, shape = cest_header
    model = read_noise_model({**dic, **changes}, shape)
    assert not model.modelled
    assert model.axes == ()
    assert str(model.reason).startswith("axis ")
    assert message in str(model.reason)


def test_a_spectrum_without_an_nmrpipe_header_is_not_modelled():
    model = read_noise_model({}, (4, 101))
    assert not model.modelled
    assert model.reason == "no NMRPipe processing header"


@pytest.mark.parametrize(
    "order",
    [
        None,
        [],
        [2.0],  # too short for three axes
        [2.0, "x", 3.0, 4.0],
        [2.0, float("nan"), 3.0, 4.0],
        [2.0, 9.0, 3.0, 4.0],  # not an NMRPipe axis code
        [2.0, 1.5, 3.0, 4.0],
        2.0,  # not a sequence
    ],
)
def test_a_malformed_axis_order_is_not_modelled(cest_header, order):
    dic, shape = cest_header
    model = read_noise_model({**dic, "FDDIMORDER": order}, shape)
    assert not model.modelled
    assert model.axes == ()
    assert "FDDIMORDER" in str(model.reason) or model.reason == "no NMRPipe processing header"


@pytest.mark.parametrize(
    "window",
    [
        {"FDF2APODQ1": 0.0, "FDF2APODQ2": 0.0},  # sin(0) everywhere: zero variance
        {"FDF2APODQ2": 1.6, "FDF2APODQ3": 2.5},  # negative sines to a fractional power
        {"FDF2APODQ2": 0.4, "FDF2APODQ3": 1e6},  # sin ≤ 0.95, to the 10⁶: underflows to 0
    ],
)
def test_a_window_with_zero_or_non_finite_variance_is_not_modelled(cest_header, window):
    dic, shape = cest_header
    with np.errstate(all="raise"):
        model = read_noise_model({**dic, **window}, shape)
    assert not model.modelled
    assert "noise variance" in str(model.reason)
