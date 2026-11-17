"""The one route from a number, time or axis to text in the GUI (specification §5, §12).

Every label, list, table, field, axis and tooltip that shows a float formats it
here; a static test finds any other. Formatting is presentation only: stored
float64 values are never rounded, and editable fields reveal full round-trip
precision on focus (:func:`exact`).

The rules:

- A value with its standard error: the σ keeps two significant digits, the value is
  rounded to the same decimal place, and both share one power of ten,
  ``(3.10 ± 0.13) × 10⁹`` (:func:`measured`).
- A column of values of one order of magnitude carries a single ``× 10ⁿ`` in its
  header (:func:`column`).
- Each quantity has fixed decimals: ppm 3, Hz 1, χ²ᵣ 2, multiples of σ 1, correlation
  2, seconds 1 (:func:`ppm`, :func:`hz` …).
- Plain digits between 10⁻³ and 10⁴; beyond that a power of ten, never ``e`` notation.
- Times are local, with the time zone (:func:`local_time`).
- An axis is named by its nucleus and its F-dimension, ``¹⁵N F2`` (:func:`axis_name`).
"""

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_PLAIN = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
# Plain digits read well between 10⁻³ and 10⁴; beyond that values share a power of ten.
_LARGE, _SMALL = 4, -3
MINUS = "−"
DASH = "–"  # a value that cannot be shown (non-finite)
NOT_ASSESSED = "not assessed"
SIGMA_DIGITS = 2  # significant digits of a standard error
SIGNIFICANT = 4  # significant digits of a value without a standard error
NARROW_SPACE = " "  # thousands grouping in counts

# Fixed decimals of each quantity, wherever it is shown.
PPM_DECIMALS = 3
HZ_DECIMALS = 1
CHI2_DECIMALS = 2
SIGMA_MULTIPLE_DECIMALS = 1  # residuals and |r|max in units of σ_stat
CORRELATION_DECIMALS = 2
SECONDS_DECIMALS = 1


def superscript(text: str) -> str:
    """Digits and minus as superscripts, for isotopes and exponents."""
    return text.translate(_SUPERSCRIPT)


def nucleus(name: str | None, label: str) -> str:
    """The file's nucleus with its mass number raised (15N → ¹⁵N); never inferred."""
    if not name:
        return label
    digits = len(name) - len(name.lstrip("0123456789"))
    return superscript(name[:digits]) + name[digits:]


# ------------------------------------------------------------------ axis names
def axis_name(axis) -> str:
    """Nucleus and F-dimension, ``¹⁵N F2``: the one name of a spectral axis.

    The F-number is always present, so homonuclear axes stay distinct. Direct and
    indirect appear only in help text, never in a name.
    """
    return " ".join(part for part in (nucleus(axis.nucleus, ""), axis.label) if part)


def axis_title(axis, unit: str = "ppm") -> str:
    """An axis name with its unit, ``¹⁵N F2 (ppm)``, for plot axes and column headers."""
    return f"{axis_name(axis)} ({unit})"


def plane_axis_name(*, values: bool, label: str = "", unit: str = "") -> str:
    """The plane axis: its label and unit when known, else Plane value or Plane.

    PeakFit never infers what the planes vary (offsets, delays, temperatures …):
    imported plane values without a label are a "Plane value", and planes without
    values are numbered.
    """
    if label:
        return f"{label} ({unit})" if unit else label
    return "Plane value" if values else "Plane"


# ------------------------------------------------------------------ numbers
def _fixed(value: float, decimals: int) -> str:
    """``value`` rounded to ``decimals`` (negative: tens, hundreds…), typographic minus."""
    if decimals < 0:
        value = round(value, decimals)
        decimals = 0
    text = f"{value:.{decimals}f}"
    if text.lstrip("-").strip("0.") == "":  # −0.00 is 0.00
        text = text.lstrip("-")
    return text.replace("-", MINUS)


def number(value: float, decimals: int) -> str:
    """Fixed decimals with a typographic minus, for read-only display."""
    if not math.isfinite(value):
        return DASH
    return _fixed(value, decimals)


def exponent(limit: float) -> int:
    """Power of ten for a magnitude that would otherwise need ``2e+06`` notation."""
    if not math.isfinite(limit) or limit <= 0:
        return 0
    power = math.floor(math.log10(limit))
    return power if power >= _LARGE or power <= _SMALL else 0


def factor(power: int) -> str:
    """``× 10ⁿ``, or nothing for plain digits."""
    return f"× 10{superscript(str(power))}" if power else ""


def _with_factor(text: str, power: int) -> str:
    return f"{text} {factor(power)}" if power else text


def _significant_decimals(value: float, digits: int) -> int:
    """Decimals that show ``digits`` significant digits of a nonzero ``value``."""
    rounded = float(f"{value:.{digits - 1}e}")  # 9.996 to three digits is 10.0
    return digits - 1 - math.floor(math.log10(abs(rounded)))


def scaled(value: float, digits: int = SIGNIFICANT) -> str:
    """A value to ``digits`` significant figures, with ``× 10ⁿ`` in place of e notation.

    Trailing zeros are dropped (2.50 → 2.5): the digits shown are the value's, not a
    claim about its precision.
    """
    if not math.isfinite(value):
        return DASH
    if value == 0:
        return "0"
    power = exponent(abs(float(f"{value:.{digits - 1}e}")))
    mantissa = value * 10.0**-power
    decimals = _significant_decimals(mantissa, digits)
    text = _fixed(mantissa, decimals)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return _with_factor(text, power)


def shared(values: Sequence[float], digits: int = SIGNIFICANT) -> list[str]:
    """Values read side by side: one power of ten and one decimal place, set by the largest.

    The largest shows ``digits`` significant digits; the others the same decimals.
    """
    finite = [abs(v) for v in values if math.isfinite(v) and v]
    if not finite:
        return [DASH if not math.isfinite(v) else "0" for v in values]
    largest = max(finite)
    power = exponent(abs(float(f"{largest:.{digits - 1}e}")))
    decimals = max(_significant_decimals(largest * 10.0**-power, digits), 0)
    return [
        _with_factor(_fixed(v * 10.0**-power, decimals), power) if math.isfinite(v) else DASH
        for v in values
    ]


def intensity(value: float) -> str:
    """An intensity, noise level or plane value: four significant digits, × 10ⁿ beyond 10⁴."""
    return scaled(value)


def plane_value(value: float) -> str:
    """A plane value (offset, delay, temperature…) as supplied: up to six significant digits.

    Plane values are labels the user typed, so integers up to 10⁶ stay plain (−12000).
    """
    if not math.isfinite(value):
        return DASH
    if value == 0 or 1e-3 <= abs(value) < 1e6:  # noqa: PLR2004
        decimals = max(_significant_decimals(value, 6), 0) if value else 0
        text = _fixed(value, decimals)
        return text.rstrip("0").rstrip(".") if "." in text else text
    return scaled(value, 6)


def exact(value: float) -> str:
    """Full round-trip precision, for an editable field that has focus or for copying."""
    return repr(float(value))


def sigma_available(sigma: float) -> bool:
    """Whether a standard error is a σ to show: finite and positive."""
    return math.isfinite(sigma) and sigma > 0


@dataclass(frozen=True, slots=True)
class Measured:
    """A value and its σ as shown: mantissas at one shared power of ten.

    ``sigma`` is the σ mantissa, ``"0"`` for an exact value, or ``NOT_ASSESSED``.
    """

    value: str
    sigma: str
    power: int

    @property
    def assessed(self) -> bool:
        """Whether a σ is shown (zero counts: the value is exact)."""
        return self.sigma != NOT_ASSESSED

    @property
    def factor(self) -> str:
        """The shared ``× 10ⁿ``, or nothing."""
        return factor(self.power)

    @property
    def value_text(self) -> str:
        """The value alone, with its power of ten."""
        return _with_factor(self.value, self.power)

    @property
    def sigma_text(self) -> str:
        """The σ alone, with its power of ten, or "not assessed"."""
        return _with_factor(self.sigma, self.power) if self.assessed else NOT_ASSESSED

    @property
    def text(self) -> str:
        """``(3.10 ± 0.13) × 10⁹``, ``7.98586 ± 0.00009``, or ``3.104 × 10⁹ (σ not assessed)``."""
        if not self.assessed:
            return f"{self.value_text} (σ {NOT_ASSESSED})"
        pair = f"{self.value} ± {self.sigma}"
        return f"({pair}) {self.factor}" if self.power else pair


def measured(
    value: float, sigma: float, *, decimals: int | None = None, power: int | None = None
) -> Measured:
    """A value and its standard error to the σ's precision, sharing one power of ten.

    The σ keeps two significant digits and the value is rounded to the same decimal
    place. A zero σ shows ``0`` and a missing, non-finite or negative one
    ``not assessed``; the value then has ``decimals`` places when the quantity fixes
    them, else four significant digits. ``power`` forces the shared power of ten (a
    column's); by default it follows the larger of |value| and σ.
    """
    if not math.isfinite(value):
        return Measured(DASH, DASH if math.isfinite(sigma) else NOT_ASSESSED, 0)
    assessed = math.isfinite(sigma) and sigma >= 0
    if not (assessed and sigma > 0):
        error = "0" if assessed else NOT_ASSESSED
        if decimals is not None and power is None:
            return Measured(number(value, decimals), error, 0)
        if power is None:
            power = exponent(abs(float(f"{value:.{SIGNIFICANT - 1}e}"))) if value else 0
        return Measured(scaled(value * 10.0**-power), error, power)
    if power is None:
        power = exponent(max(abs(value), sigma))
        # The value or σ that rounds up into the next decade moves the shared power.
        places = _significant_decimals(sigma * 10.0**-power, SIGMA_DIGITS)
        largest = max(abs(value), sigma) * 10.0**-power
        if power and abs(round(largest, places)) >= 10:  # noqa: PLR2004
            power += 1
    error = sigma * 10.0**-power
    places = _significant_decimals(error, SIGMA_DIGITS)
    return Measured(_fixed(value * 10.0**-power, places), _fixed(error, places), power)


@dataclass(frozen=True, slots=True)
class Column:
    """Cells of one table column and the ``× 10ⁿ`` its header carries, if shared."""

    factor: str
    cells: tuple[Measured, ...]

    def header(self, title: str) -> str:
        """``title`` with the column's shared factor: ``Amplitude (× 10⁹)``."""
        return f"{title} ({self.factor})" if self.factor else title


def column(values: Sequence[float], sigmas: Sequence[float] | None = None) -> Column:
    """A column whose values share one order of magnitude gets one factor in its header.

    Otherwise each cell keeps its own power of ten.
    """
    errors = list(sigmas) if sigmas is not None else [math.nan] * len(values)
    pairs = list(zip(values, errors, strict=True))

    powers = {measured(v, s).power for v, s in pairs if math.isfinite(v)}
    if len(powers) == 1 and (power := powers.pop()):
        cells = tuple(measured(v, s, power=power) for v, s in pairs)
        return Column(factor(power), tuple(Measured(c.value, c.sigma, 0) for c in cells))
    return Column("", tuple(measured(v, s) for v, s in pairs))


# ------------------------------------------------------------------ quantities
UNIT_DECIMALS = {"ppm": PPM_DECIMALS, "Hz": HZ_DECIMALS}


def quantity(value: float, unit: str) -> str:
    """A value in ``unit`` without a σ: the unit's fixed decimals, else four significant digits."""
    decimals = UNIT_DECIMALS.get(unit)
    return number(value, decimals) if decimals is not None else scaled(value)


def ppm(value: float) -> str:
    """A chemical shift or position, 3 decimals."""
    return number(value, PPM_DECIMALS)


def hz(value: float) -> str:
    """A linewidth or coupling, 1 decimal."""
    return number(value, HZ_DECIMALS)


def chi2(value: float) -> str:
    """A reduced χ², 2 decimals."""
    return number(value, CHI2_DECIMALS)


def sigma_multiple(value: float, *, signed: bool = False) -> str:
    """A residual in units of σ_stat, 1 decimal; ``signed`` shows + on positives."""
    text = number(value, SIGMA_MULTIPLE_DECIMALS)
    return f"+{text}" if signed and math.isfinite(value) and value > 0 else text


def correlation(value: float) -> str:
    """A correlation coefficient, 2 decimals."""
    return number(value, CORRELATION_DECIMALS)


def seconds(value: float) -> str:
    """A duration, ``3.8 s``."""
    return f"{number(value, SECONDS_DECIMALS)} s"


def mebibytes(value: float) -> str:
    """A size in MiB, 1 decimal."""
    return f"{number(value, 1)} MiB"


def percent(fraction: float) -> str:
    """A fraction as a whole percentage, ``20 %``."""
    return f"{number(100 * fraction, 0)} %"


def setting(value: float) -> str:
    """A configured threshold or factor (5σ, ±8 σ_stat, 0.3 c): its shortest exact form."""
    if not math.isfinite(value):
        return DASH
    if value == int(value) and abs(value) < 10**_LARGE:
        return _fixed(value, 0)
    return scaled(value, 6)


def count(n: int, singular: str, plural: str | None = None) -> str:
    """``1 cluster``, ``121 clusters``, ``6 158 samples``: grouped, pluralised."""
    noun = singular if n == 1 else (plural or f"{singular}s")
    return f"{grouped(n)} {noun}"


def grouped(n: int) -> str:
    """An integer with narrow-space thousands grouping, ``6 158``."""
    return f"{n:,}".replace(",", NARROW_SPACE).replace("-", MINUS)


def unbroken_name(name: str) -> str:
    """An assignment that never wraps inside itself (159N-H): non-breaking hyphens, spaces."""
    return name.replace("-", "\u2011").replace(" ", "\u00a0")


# ------------------------------------------------------------------ times
def local_time(iso: str) -> str:
    """A recorded ISO time in local time with its zone, ``2026-11-17 12:00 CET``.

    Times without a zone were recorded in UTC. Text that is not an ISO time is
    returned as it is.
    """
    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    local = moment.astimezone()
    return f"{local:%Y-%m-%d %H:%M} {local.tzname() or format(local, '%z')}"


# ------------------------------------------------------------------ typed input
def parse_scaled(text: str) -> float:
    """A typed number: plain, scientific (1.5e5) or as :func:`scaled` shows it.

    Raises ``ValueError`` for anything else, including a comma decimal.
    """
    plain = text.strip().replace(MINUS, "-").replace(" ", "").replace(NARROW_SPACE, "")
    mantissa, times, power = plain.partition("×10")
    if not times:
        return float(plain)
    return float(mantissa) * 10.0 ** int(power.translate(_PLAIN))
