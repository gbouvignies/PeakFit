"""Spectrum readers for NMRPipe and related formats."""

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from nmrglue.fileio.pipe import guess_udic, read
from numpy.typing import NDArray

from peakfit.engine.algorithms.noise_correlation import AxisProcessing, NoiseModel, not_modelled
from peakfit.engine.domain.param_id import PSEUDO_AXIS
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters, get_dimension_label

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


T = TypeVar("T", float, NDArray[Any])

P1_MIN = 175.0
P1_MAX = 185.0

# NMRPipe nucleus label mapping from header codes
NUCLEUS_LABELS: dict[str, str] = {
    "1": "1H",
    "2": "2H",
    "13": "13C",
    "15": "15N",
    "19": "19F",
    "31": "31P",
}


def _positive_int(value: Any) -> int | None:
    """Parse a positive integer from header value."""
    try:
        parsed = round(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def read_spectral_parameters(dic: dict[str, Any], data: NDArray[Any]) -> list[Any]:
    """Read spectral parameters from an NMRPipe dictionary."""
    spec_params: list[SpectralParameters] = []

    for i in range(data.ndim):
        size = data.shape[i]
        fdf = f"FDF{int(dic['FDDIMORDER'][data.ndim - 1 - i])}"
        is_direct = i == data.ndim - 1
        ft = dic.get(f"{fdf}FTFLAG", 0.0) == 1.0

        if i == 0:
            dim_label = PSEUDO_AXIS
            nucleus = None
            sw = obs = car = aq_time = 1.0
            p180 = False
        else:
            spectral_index = i - 1
            dim_label = get_dimension_label(spectral_index)
            nucleus_code = str(int(dic.get(f"{fdf}OBS", 0) % 100))
            label_key = f"{fdf}LABEL"
            if dic.get(label_key):
                nucleus = str(dic[label_key]).strip()
            else:
                nucleus = NUCLEUS_LABELS.get(nucleus_code)

            if ft:
                sw = dic.get(f"{fdf}SW", 1.0)
                orig = dic.get(f"{fdf}ORIG", 0.0)
                obs = dic.get(f"{fdf}OBS", 1.0)
                car = orig + sw / 2.0 - sw / size
                aq_time = dic.get(f"{fdf}APOD", 0.0) / max(sw, 1e-6)
                p180 = P1_MIN <= abs(dic.get(f"{fdf}P1", 0.0)) <= P1_MAX
            else:
                sw = obs = car = aq_time = 1.0
                p180 = False

        spec_params.append(
            SpectralParameters(
                size=size,
                sw=sw,
                obs=obs,
                car=car,
                aq_time=aq_time,
                apocode=dic.get(f"{fdf}APODCODE", 0.0),
                apodq1=dic.get(f"{fdf}APODQ1", 0.0),
                apodq2=dic.get(f"{fdf}APODQ2", 0.0),
                apodq3=dic.get(f"{fdf}APODQ3", 0.0),
                p180=p180,
                direct=is_direct,
                ft=ft,
                label=dim_label,
                nucleus=nucleus,
                td_size=_positive_int(dic.get(f"{fdf}TDSIZE")),
                ft_size=_positive_int(dic.get(f"{fdf}FTSIZE")),
            )
        )

    return spec_params


# NMRPipe APODCODE values the noise model supports: none, and the sine bell (SP).
_WINDOWS: dict[int, Literal["none", "sine_bell"]] = {0: "none", 1: "sine_bell"}


def _axis_processing(  # noqa: PLR0911 - one return per check of §2.3
    dic: dict[str, Any], fdf: str, stored: int
) -> AxisProcessing | str:
    """One axis's processing, or why it cannot be modelled (docs/correlated-noise.md §2.3).

    NMRPipe records the apodized length ``APOD`` rescaled by any extraction and
    stored as a whole number, so it must lie within one point of N·S/M.
    """

    def number(key: str) -> float | None:
        raw = dic.get(f"{fdf}{key}")
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    if number("FTFLAG") != 1.0:
        return "not Fourier transformed"
    acquired, transform, apod = number("TDSIZE"), number("FTSIZE"), number("APOD")
    if acquired is None or acquired < 1 or acquired != round(acquired):
        return "no acquired size (TDSIZE)"
    if transform is None or transform != round(transform) or transform < max(acquired, stored):
        return f"Fourier-transform size (FTSIZE {transform}) below the acquired or stored size"
    expected = acquired * stored / transform
    if apod is None or abs(apod - expected) >= 1.0:
        return (
            f"apodized length APOD {apod} disagrees with TDSIZE x stored/FTSIZE = "
            f"{expected:g}: the window does not cover the acquired points"
        )
    code = number("APODCODE")
    window = _WINDOWS.get(int(code)) if code is not None and code == round(code) else None
    if window is None:
        return f"unsupported apodization (APODCODE {code})"
    shape = {}
    if window == "sine_bell":
        q1, q2, q3 = number("APODQ1"), number("APODQ2"), number("APODQ3")
        if q1 is None or q2 is None or q3 is None or q3 <= 0:
            return "sine-bell parameters (APODQ1-3) missing or invalid"
        shape = {"offset": q1, "end": q2, "power": q3}
    first, phase = number("C1"), number("P1")
    if first is None or phase is None:
        return "first-point scale (C1) or first-order phase (P1) missing"
    return AxisProcessing(
        acquired=int(acquired),
        transform=int(transform),
        window=window,
        first_point=first + 1.0,
        phase=phase,
        **shape,
    )


def read_noise_model(dic: dict[str, Any], shape: Sequence[int]) -> NoiseModel:
    """The spectrum's processing-derived noise correlation, or why it is not modelled.

    Every spectral axis (all but the first, pseudo axis) must pass the checks of
    ``docs/correlated-noise.md`` §2.3. One failing axis leaves the whole spectrum not
    modelled: there is no partial model.
    """
    ndim = len(shape)
    order = dic.get("FDDIMORDER")
    if not dic or order is None or ndim < 2:  # noqa: PLR2004
        return not_modelled("no NMRPipe processing header")
    axes = []
    for i in range(1, ndim):
        fdf = f"FDF{int(order[ndim - 1 - i])}"
        axis = _axis_processing(dic, fdf, int(shape[i]))
        if isinstance(axis, str):
            return not_modelled(f"axis {get_dimension_label(i - 1)}: {axis}")
        axes.append(axis)
    return NoiseModel(axes=tuple(axes))


def read_spectra(
    path_spectra: Path,
    path_z_values: Path | None = None,
    exclude_list: Sequence[int] | None = None,
) -> Spectra:
    """Read an NMRPipe spectrum and optional plane values file."""
    dic, data = read(path_spectra)
    data = data.astype(np.float32)

    udic = guess_udic(dic, data)

    first_is_freq = bool(udic.get(0, {}).get("freq", True))

    if first_is_freq:
        data = np.expand_dims(data, axis=0)

    if path_z_values is not None:
        z_values = np.genfromtxt(path_z_values, dtype=None, encoding="utf-8")
    else:
        z_values = np.arange(data.shape[0], dtype=int)

    params = read_spectral_parameters(dic, data)

    spectra = Spectra(
        dic=dic,
        data=data,
        z_values=z_values,
        params=params,
    )
    spectra.exclude_planes(exclude_list)

    return spectra


__all__ = ["NUCLEUS_LABELS", "read_noise_model", "read_spectra", "read_spectral_parameters"]
