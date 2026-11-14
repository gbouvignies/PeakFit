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


def test_scaled_values_never_use_e_notation():
    assert formatting.scaled(11923.4) == "1.192 ×10⁴"
    assert formatting.scaled(59612.0) == "5.961 ×10⁴"
    assert formatting.scaled(-2.5e-5) == "−2.5 ×10⁻⁵"
    assert formatting.scaled(123.456) == "123.5"
    assert formatting.scaled(math.nan) == "–"


@pytest.mark.parametrize(
    ("text", "value"),
    [("2500", 2500.0), ("1.5e5", 1.5e5), ("6.035 ×10⁴", 60350.0), ("−2.5 ×10⁻³", -2.5e-3)],
)
def test_typed_numbers_read_plain_scientific_and_scaled(text, value):
    assert formatting.parse_scaled(text) == pytest.approx(value, rel=1e-12)


@pytest.mark.parametrize("text", ["1,5", "", "×10⁴", "abc"])
def test_other_text_is_not_a_number(text):
    with pytest.raises(ValueError, match="could not convert"):
        formatting.parse_scaled(text)
