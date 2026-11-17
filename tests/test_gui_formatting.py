"""One formatter for every number, time and axis name the GUI shows (specification §5, §12)."""

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from peakfit.engine.domain.spectrum import SpectralParameters, get_dimension_label
from peakfit.gui import formatting
from peakfit.gui.formatting import NOT_ASSESSED, measured


def shown_value(text: str, power: int) -> float:
    return formatting.parse_scaled(text) * 10.0**power


def last_digit(text: str, power: int) -> float:
    """The place value of the last digit shown (trailing zeros of integers count)."""
    mantissa = text.replace(formatting.MINUS, "-")
    decimals = len(mantissa.partition(".")[2])
    return 10.0 ** (power - decimals)


# ------------------------------------------------------------------ value ± σ
def test_value_and_sigma_share_one_power_of_ten():
    shown = measured(3103917717.0, 132348379.0)
    assert shown == formatting.Measured("3.10", "0.13", 9)
    assert shown.text == "(3.10 ± 0.13) × 10⁹"
    assert shown.value_text == "3.10 × 10⁹"
    assert shown.sigma_text == "0.13 × 10⁹"


def test_sigma_sets_the_displayed_precision_in_plain_range():
    assert measured(7.985847416453826, 5.21e-5).text == "7.985847 ± 0.000052"
    assert measured(12.336918772753798, 0.0655).text == "12.337 ± 0.066"
    assert measured(1234.5, 21.0).text == "1234 ± 21"
    # A σ of three digits rounds the value to the same place: tens.
    assert measured(1234.5, 213.0).text == "1230 ± 210"


@pytest.mark.parametrize("magnitude", [10.0**k for k in range(-3, 13)])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_magnitudes_from_milli_to_tera(magnitude, sign):
    value, sigma = sign * 3.14159265 * magnitude, 0.0271828 * magnitude
    shown = measured(value, sigma)
    assert "e" not in shown.text
    # Two significant digits of σ, in the shared mantissa.
    digits = shown.sigma.replace(".", "").lstrip("0")
    assert len(digits) == formatting.SIGMA_DIGITS
    assert shown.value.startswith(formatting.MINUS) == (sign < 0)
    # Plain digits from 0.01 up to 10⁴, a power of ten beyond.
    assert (shown.power == 0) == (1e-2 <= abs(value) < 1e4)


def test_sigma_rounding_into_the_next_decade_keeps_two_digits():
    assert measured(5.0, 0.0996).text == "5.00 ± 0.10"
    assert measured(9.9996e5, 2.0).text == "(9.999600 ± 0.000020) × 10⁵"
    # The value rounds up into the next decade: the shared power moves with it.
    assert measured(9.99996e5, 0.4e5).text == "(1.000 ± 0.040) × 10⁶"
    # Rounding to the σ's place carries 9999.97 to 10⁴: the power follows.
    assert measured(9999.97, 30.0).text == "(1.0000 ± 0.0030) × 10⁴"


def test_sigma_larger_than_the_value_sets_the_power():
    assert measured(0.003, 0.02).text == "0.003 ± 0.020"
    assert measured(5e6, 8e6).text == "(5.0 ± 8.0) × 10⁶"
    assert measured(-2e6, 3e7).text == "(−0.2 ± 3.0) × 10⁷"
    assert measured(-2e5, 3e7).text == "(0.0 ± 3.0) × 10⁷"


def test_zero_sigma_is_an_exact_value():
    shown = measured(1.0, 0.0)
    assert shown.assessed
    assert shown.text == "1 ± 0"
    assert measured(2.5e9, 0.0).text == "(2.5 ± 0) × 10⁹"


@pytest.mark.parametrize("sigma", [math.nan, math.inf, -1.0])
def test_unavailable_sigma_is_not_assessed(sigma):
    shown = measured(3103917717.0, sigma)
    assert not shown.assessed
    assert shown.sigma_text == NOT_ASSESSED
    assert shown.text == "3.104 × 10⁹ (σ not assessed)"
    # A quantity with fixed decimals keeps them without a σ.
    assert measured(7.98586, sigma, decimals=3).text == "7.986 (σ not assessed)"


def test_non_finite_value_shows_a_dash():
    assert measured(math.nan, 1.0).value == "–"


def test_displayed_values_round_to_half_their_last_digit():
    """Every value differs from the stored one by less than half its last shown digit."""
    rng = np.random.default_rng(20261117)
    for _ in range(20000):
        value = float(rng.choice([-1, 1]) * 10.0 ** rng.uniform(-3.5, 12.5))
        sigma = float(abs(value) * 10.0 ** rng.uniform(-6, 1.5))
        shown = measured(value, sigma)
        # The σ's second significant digit is the last place shown, for both.
        place = 10.0 ** (math.floor(math.log10(shown_value(shown.sigma, shown.power))) - 1)
        assert len(shown.sigma.replace(".", "").lstrip("0").rstrip("0")) <= 2
        for text, stored in ((shown.value, value), (shown.sigma, sigma)):
            error = abs(shown_value(text, shown.power) - stored)
            assert error <= 0.5 * place * (1 + 1e-9), (value, sigma)
        bare = formatting.scaled(value)
        mantissa, _, _ = bare.partition(" ×")
        power = formatting.exponent(abs(formatting.parse_scaled(bare)))
        error = abs(formatting.parse_scaled(bare) - value)
        assert error <= 0.5 * last_digit(mantissa, power) * (1 + 1e-9), value


# ------------------------------------------------------------------ columns
def test_a_column_of_one_magnitude_carries_its_factor_in_the_header():
    amplitudes = formatting.column([3.10e9, 1.21e9, 2.05e9], [1.3e8, 4.0e7, 9.9e7])
    assert amplitudes.factor == "× 10⁹"
    assert amplitudes.header("Amplitude") == "Amplitude (× 10⁹)"
    assert [c.text for c in amplitudes.cells] == ["3.10 ± 0.13", "1.210 ± 0.040", "2.050 ± 0.099"]


def test_a_column_of_mixed_magnitudes_keeps_a_factor_per_cell():
    mixed = formatting.column([3.1e9, 2.0e8])
    assert mixed.factor == ""
    assert [c.value_text for c in mixed.cells] == ["3.1 × 10⁹", "2 × 10⁸"]
    assert formatting.column([7.98, 8.03]).header("δ") == "δ"


# ------------------------------------------------------------------ quantities
def test_each_quantity_has_fixed_decimals():
    assert formatting.chi2(3.0) == "3.00"
    assert formatting.ppm(120.20348) == "120.203"
    assert formatting.hz(15.73) == "15.7"
    assert formatting.sigma_multiple(-0.66, signed=True) == "−0.7"
    assert formatting.sigma_multiple(2.24, signed=True) == "+2.2"
    assert formatting.correlation(-0.4149) == "−0.41"
    assert formatting.correlation(-0.001) == "0.00"
    assert formatting.seconds(3.81) == "3.8 s"


def test_display_numbers_use_typographic_minus():
    assert formatting.number(-0.5, 2) == "−0.50"
    assert formatting.intensity(-12000.0) == "−1.2 × 10⁴"
    assert formatting.number(math.inf, 2) == "–"


def test_scaled_values_never_use_e_notation():
    assert formatting.scaled(11922.5) == "1.192 × 10⁴"
    assert formatting.scaled(59612.0) == "5.961 × 10⁴"
    assert formatting.scaled(-2.5e-5) == "−2.5 × 10⁻⁵"
    assert formatting.scaled(1e-8) == "1 × 10⁻⁸"
    assert formatting.scaled(123.456) == "123.5"
    assert formatting.scaled(9999.7) == "1 × 10⁴"
    assert formatting.scaled(math.nan) == "–"


def test_settings_show_their_shortest_exact_form():
    assert formatting.setting(5.0) == "5"
    assert formatting.setting(0.3) == "0.3"
    assert formatting.setting(1e-8) == "1 × 10⁻⁸"


def test_counts_are_grouped_and_pluralised():
    assert formatting.count(1, "cluster") == "1 cluster"
    assert formatting.count(121, "cluster") == "121 clusters"
    assert formatting.count(6158, "sample") == "6 158 samples"
    assert formatting.count(2, "analysis", "analyses") == "2 analyses"


def test_axis_exponent_only_for_large_or_small_ranges():
    assert formatting.exponent(2.1e6) == 6
    assert formatting.exponent(8.0) == 0
    assert formatting.exponent(9999.0) == 0
    assert formatting.exponent(2e-4) == -4
    assert formatting.exponent(0.0) == 0


# ------------------------------------------------------------------ times
def test_times_are_local_with_their_zone():
    recorded = datetime(2026, 11, 17, 11, 0, tzinfo=UTC)
    local = recorded.astimezone()
    expected = f"{local:%Y-%m-%d %H:%M} {local.tzname()}"
    assert formatting.local_time(recorded.isoformat()) == expected
    # A time recorded without a zone was recorded in UTC.
    assert formatting.local_time("2026-11-17T11:00:00") == expected
    later = (recorded + timedelta(minutes=1)).isoformat()
    assert formatting.local_time(later) != expected
    assert formatting.local_time("not a time") == "not a time"


# ------------------------------------------------------------------ axis names
def spectral_axis(index: int, nucleus: str | None) -> SpectralParameters:
    return SpectralParameters(
        size=64,
        sw=1000.0,
        obs=60.0,
        car=100.0,
        aq_time=0.05,
        apocode=1.0,
        apodq1=0.5,
        apodq2=0.98,
        apodq3=1.0,
        p180=False,
        direct=False,
        ft=True,
        label=get_dimension_label(index),
        nucleus=nucleus,
    )


@pytest.mark.parametrize(
    ("nuclei", "names"),
    [
        (("15N", "1H"), ("¹⁵N F2", "¹H F3")),
        (("13C", "1H"), ("¹³C F2", "¹H F3")),
        (("1H", "1H"), ("¹H F2", "¹H F3")),
        (("13C", "15N", "1H"), ("¹³C F2", "¹⁵N F3", "¹H F4")),
        (("N", "H"), ("N F2", "H F3")),
        ((None, None), ("F2", "F3")),
    ],
    ids=["HN", "HC", "homonuclear", "three-axis", "bare-element", "unlabelled"],
)
def test_axis_names_are_nucleus_and_f_dimension_and_distinct(nuclei, names):
    axes = [spectral_axis(i, n) for i, n in enumerate(nuclei)]
    assert tuple(formatting.axis_name(a) for a in axes) == names
    assert len(set(names)) == len(names)
    assert formatting.axis_title(axes[0]) == f"{names[0]} (ppm)"


def test_nucleus_raises_mass_number_without_inferring_one():
    assert formatting.nucleus("15N", "F2") == "¹⁵N"
    assert formatting.nucleus("H", "F3") == "H"
    assert formatting.nucleus(None, "F1") == "F1"


def test_plane_axis_uses_its_label_and_unit_when_known():
    assert formatting.plane_axis_name(values=True, label="B₁ offset", unit="Hz") == (
        "B₁ offset (Hz)"
    )
    assert formatting.plane_axis_name(values=True) == "Plane value"
    assert formatting.plane_axis_name(values=False) == "Plane"


# ------------------------------------------------------------------ typed input
@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("2500", 2500.0),
        ("1.5e5", 1.5e5),
        ("6.035 × 10⁴", 60350.0),
        ("6.035 ×10⁴", 60350.0),
        ("−2.5 × 10⁻³", -2.5e-3),
        ("1 × 10⁻⁸", 1e-8),
    ],
)
def test_typed_numbers_read_plain_scientific_and_scaled(text, value):
    assert formatting.parse_scaled(text) == pytest.approx(value, rel=1e-12)


@pytest.mark.parametrize("text", ["1,5", "", "× 10⁴", "abc"])
def test_other_text_is_not_a_number(text):
    with pytest.raises(ValueError, match="could not convert"):
        formatting.parse_scaled(text)


def test_plane_values_show_as_supplied():
    assert formatting.plane_value(-12000.0) == "−12000"
    assert formatting.plane_value(0.0125) == "0.0125"
    assert formatting.plane_value(0.0) == "0"
    assert formatting.plane_value(2.5e-5) == "2.5 × 10⁻⁵"
