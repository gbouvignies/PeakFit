"""Display formatting follows the data's resolution and never alters values."""

import math
from types import SimpleNamespace

import pytest

from peakfit.gui import formatting


def axis(ppm_per_point: float):
    return SimpleNamespace(pts2ppm=lambda points: 10.0 - points * ppm_per_point)


@pytest.mark.parametrize(
    ("step", "decimals"),
    [(4.0 / 546, 4), (18.64 / 256, 3), (0.5, 2), (2e-5, 6), (float("nan"), 4)],
)
def test_ppm_decimals_add_one_place_beyond_digital_resolution(step, decimals):
    assert formatting.ppm_decimals(axis(step)) == decimals


def test_sigma_sets_the_displayed_precision():
    assert formatting.with_sigma(7.985847416453826, 5.21e-5) == ("7.985847", "0.000052")
    assert formatting.with_sigma(12.336918772753798, 0.0655) == ("12.337", "0.066")
    assert formatting.with_sigma(1234.5, 21.0) == ("1234", "21")
    assert formatting.with_sigma(8.35, math.nan) == ("8.3500", "σ unavailable")


def test_axis_exponent_only_for_large_or_small_ranges():
    assert formatting.exponent(2.1e6) == 6
    assert formatting.exponent(8.0) == 0
    assert formatting.exponent(9999.0) == 0
    assert formatting.exponent(2e-4) == -4
    assert formatting.exponent(0.0) == 0


def test_nucleus_raises_mass_number_without_inferring_one():
    assert formatting.nucleus("15N", "F2") == "¹⁵N"
    assert formatting.nucleus("H", "F3") == "H"
    assert formatting.nucleus(None, "F1") == "F1"


def test_display_numbers_use_typographic_minus():
    assert formatting.number(-0.5, 2) == "−0.50"
    assert formatting.intensity(-12000.0) == "−12000"
    assert formatting.number(math.inf, 2) == "–"
