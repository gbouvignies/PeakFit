"""Display formatting shared by every label, list, field and axis.

Formatting is presentation only: stored float64 values are never rounded, and
editable fields reveal full round-trip precision on focus (specification §5).
"""

import math

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
# Plain digits read well between 10⁻³ and 10⁴; beyond that axes share a power of ten.
_LARGE, _SMALL = 4, -3


def superscript(text: str) -> str:
    """Digits and minus as superscripts, for isotopes and axis exponents."""
    return text.translate(_SUPERSCRIPT)


def nucleus(name: str | None, label: str) -> str:
    """The file's nucleus with its mass number raised (15N → ¹⁵N); never inferred."""
    if not name:
        return label
    digits = len(name) - len(name.lstrip("0123456789"))
    return superscript(name[:digits]) + name[digits:]


def ppm_decimals(axis) -> int:
    """One decimal place beyond the axis digital resolution (ppm per point)."""
    step = abs(float(axis.pts2ppm(1.0)) - float(axis.pts2ppm(0.0)))
    if not math.isfinite(step) or step <= 0:
        return 4
    return max(1, math.ceil(-math.log10(step)) + 1)


def number(value: float, decimals: int) -> str:
    """Fixed decimals with a typographic minus, for read-only display."""
    if not math.isfinite(value):
        return "–"
    return f"{value:.{decimals}f}".replace("-", "−")


def intensity(value: float) -> str:
    """Raw intensity or residual readout with six significant digits."""
    if not math.isfinite(value):
        return "–"
    return f"{value:.6g}".replace("-", "−")


def with_sigma(value: float, sigma: float) -> tuple[str, str]:
    """Value and σ to the same decimal position, σ to two significant figures."""
    if not math.isfinite(sigma) or sigma <= 0:
        return number(value, 4), "σ unavailable"
    decimals = max(0, 1 - math.floor(math.log10(sigma)))
    return number(value, decimals), number(sigma, decimals)


def exponent(limit: float) -> int:
    """Power of ten for an axis whose values would otherwise need `2e+06` notation."""
    if not math.isfinite(limit) or limit <= 0:
        return 0
    power = math.floor(math.log10(limit))
    return power if power >= _LARGE or power <= _SMALL else 0
