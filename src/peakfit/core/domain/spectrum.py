"""Domain models and helpers for spectra and spectral parameters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from nmrglue.fileio.pipe import guess_udic, read
from numpy.typing import NDArray

from peakfit.core.shared.typing import FittingOptions, FloatArray

ArrayInt = NDArray[np.int_]
T = TypeVar("T", float, FloatArray)

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


def get_dimension_label(n_spectral_dims: int, dim_index: int) -> str:
    """Get the NMRPipe-style dimension label (F1, F2, F3, F4).

    NMRPipe convention:
    - F1 is the first indirect dimension (lowest frequency, first acquired)
    - Fn (highest n) is the direct/acquisition dimension

    For a 2D spectrum: F1 (indirect), F2 (direct)
    For a 3D spectrum: F1, F2 (indirect), F3 (direct)
    For a 4D spectrum: F1, F2, F3 (indirect), F4 (direct)

    Args:
        n_spectral_dims: Total number of spectral dimensions (excluding pseudo)
        dim_index: 0-based index of the dimension (0 = first spectral dim)

    Returns:
        Dimension label like "F1", "F2", "F3", "F4"
    """
    # dim_index 0 corresponds to F1, dim_index 1 to F2, etc.
    return f"F{dim_index + 1}"


@dataclass
class DimensionInfo:
    """Metadata for a single spectral dimension.

    Follows NMRPipe convention where F1 is first indirect, Fn is direct.
    """

    index: int  # 0-based index within spectral dimensions
    label: str  # "F1", "F2", "F3", "F4"
    nucleus: str | None  # "1H", "15N", "13C", etc. (from header)
    size: int  # Number of points
    sw_hz: float  # Spectral width in Hz
    sf_mhz: float  # Spectrometer frequency in MHz
    is_direct: bool  # True for acquisition dimension
    is_pseudo: bool = False  # True for the series dimension (CEST offsets, etc.)


@dataclass
class SpectralParameters:
    """Parameters for a single spectral dimension.

    Contains both NMRPipe header information and derived values
    for unit conversions.
    """

    size: int
    sw: float  # Spectral width in Hz
    obs: float  # Spectrometer frequency in MHz
    car: float  # Carrier frequency
    aq_time: float
    apocode: float
    apodq1: float
    apodq2: float
    apodq3: float
    p180: bool
    direct: bool
    ft: bool
    label: str = ""  # Dimension label: "F1", "F2", etc.
    nucleus: str | None = None  # Nucleus label: "1H", "15N", etc.
    delta: float = field(init=False)
    first: float = field(init=False)

    def __post_init__(self) -> None:
        self.delta = -self.sw / (self.size * self.obs) if self.size * self.obs != 0.0 else 0.0
        self.first = self.car / self.obs - self.delta * self.size / 2.0 if self.obs != 0.0 else 0.0

    def hz2pts_delta(self, hz: T) -> T:
        return hz / (self.obs * self.delta)

    def pts2hz_delta(self, pts: T) -> T:
        return pts * self.obs * self.delta

    def hz2pts(self, hz: T) -> T:
        return ((hz / self.obs) - self.first) / self.delta

    def hz2pt_i(self, hz: float) -> int:
        return round(self.hz2pts(hz)) % self.size

    def pts2hz(self, pts: T) -> T:
        return (pts * self.delta + self.first) * self.obs

    def ppm2pts(self, ppm: T) -> T:
        return (ppm - self.first) / self.delta

    def ppm2pt_i(self, ppm: float) -> int:
        return round(self.ppm2pts(ppm)) % self.size

    def pts2ppm(self, pts: T) -> T:
        return (pts * self.delta) + self.first

    def hz2ppm(self, hz: T) -> T:
        return hz / self.obs


def read_spectral_parameters(
    dic: dict[str, Any], data: FloatArray, *, has_pseudo_dim: bool = False
) -> list[SpectralParameters]:
    """Read spectral parameters from NMRPipe dictionary.

    Args:
        dic: NMRPipe header dictionary
        data: Spectrum data array
        has_pseudo_dim: If True, first dimension is pseudo (not a spectral dim)

    Returns:
        List of SpectralParameters, one per dimension
    """
    spec_params: list[SpectralParameters] = []

    # Count spectral dimensions (excluding pseudo if present)
    n_spectral_dims = data.ndim - 1 if has_pseudo_dim else data.ndim

    for i in range(data.ndim):
        size = data.shape[i]
        fdf = f"FDF{int(dic['FDDIMORDER'][data.ndim - 1 - i])}"
        is_direct = i == data.ndim - 1
        ft = dic.get(f"{fdf}FTFLAG", 0.0) == 1.0

        # Determine dimension label
        if has_pseudo_dim and i == 0:
            # First dimension is pseudo (CEST offsets, relaxation delays, etc.)
            dim_label = "pseudo"
            nucleus = None
        else:
            # Spectral dimension - use F1, F2, F3, F4 convention
            spectral_index = i - 1 if has_pseudo_dim else i
            dim_label = get_dimension_label(n_spectral_dims, spectral_index)
            # Try to get nucleus from header
            nucleus_code = str(int(dic.get(f"{fdf}OBS", 0) % 100))  # Rough heuristic
            # Better: use FDLABEL if available
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
            )
        )

    return spec_params


@dataclass
class Spectra:
    """Container for NMR spectrum data with metadata.

    Handles pseudo-ND experiments where the first dimension represents
    a series (CEST offsets, relaxation delays, etc.) rather than a
    spectral dimension.
    """

    dic: dict
    data: FloatArray
    z_values: np.ndarray
    pseudo_dim_added: bool = False

    def __post_init__(self) -> None:
        udic = guess_udic(self.dic, self.data)
        no_pseudo_dim = udic[0]["freq"]
        if no_pseudo_dim:
            self.data = np.expand_dims(self.data, axis=0)
            self.pseudo_dim_added = True
        if self.z_values.size == 0:
            self.z_values = np.arange(self.data.shape[0])

    @cached_property
    def params(self) -> list[SpectralParameters]:
        """Get spectral parameters for all dimensions."""
        return read_spectral_parameters(self.dic, self.data, has_pseudo_dim=True)

    @property
    def n_spectral_dims(self) -> int:
        """Number of spectral dimensions (excluding pseudo dimension)."""
        return self.data.ndim - 1

    @property
    def spectral_params(self) -> list[SpectralParameters]:
        """Get spectral parameters for spectral dimensions only (excluding pseudo)."""
        return self.params[1:]

    @cached_property
    def dimensions(self) -> list[DimensionInfo]:
        """Get dimension info for all spectral dimensions.

        Returns list ordered from F1 (first indirect) to Fn (direct).
        """
        dims = []
        for i, param in enumerate(self.spectral_params):
            dims.append(
                DimensionInfo(
                    index=i,
                    label=param.label,
                    nucleus=param.nucleus,
                    size=param.size,
                    sw_hz=param.sw,
                    sf_mhz=param.obs,
                    is_direct=param.direct,
                    is_pseudo=False,
                )
            )
        return dims

    def get_dimension(self, identifier: str | int) -> DimensionInfo:
        """Get dimension info by label or index.

        Args:
            identifier: Either a label ("F1", "F2") or 0-based index

        Returns:
            DimensionInfo for the requested dimension

        Raises:
            KeyError: If dimension not found
        """
        if isinstance(identifier, int):
            if 0 <= identifier < len(self.dimensions):
                return self.dimensions[identifier]
            msg = f"Dimension index {identifier} out of range (0-{len(self.dimensions) - 1})"
            raise KeyError(msg)

        for dim in self.dimensions:
            if dim.label == identifier:
                return dim
        msg = f"Dimension '{identifier}' not found. Available: {[d.label for d in self.dimensions]}"
        raise KeyError(msg)

    def get_dimension_labels(self) -> list[str]:
        """Get ordered list of dimension labels (e.g., ['F1', 'F2'] for 2D)."""
        return [dim.label for dim in self.dimensions]

    def exclude_planes(self, exclude_list: Sequence[int] | None) -> None:
        if exclude_list is None:
            return
        mask = ~np.isin(range(self.data.shape[0]), exclude_list)
        self.data, self.z_values = self.data[mask], self.z_values[mask]


def read_spectra(
    path_spectra: Path,
    path_z_values: Path | None = None,
    exclude_list: Sequence[int] | None = None,
) -> Spectra:
    dic, data = read(path_spectra)
    data = data.astype(np.float32)

    if path_z_values is not None:
        z_values = np.genfromtxt(path_z_values, dtype=None, encoding="utf-8")
    else:
        z_values = np.array([])

    spectra = Spectra(dic, data, z_values)
    spectra.exclude_planes(exclude_list)

    return spectra


def determine_shape_name(dim_params: SpectralParameters) -> str:
    if dim_params.apocode == 1.0:
        if dim_params.apodq3 == 1.0:
            return "sp1"
        if dim_params.apodq3 == 2.0:
            return "sp2"
    if dim_params.apocode in {0.0, 2.0}:
        return "no_apod"
    return "pvoigt"


def get_shape_names(clargs: FittingOptions, spectra: Spectra) -> list[str]:
    match (clargs.pvoigt, clargs.lorentzian, clargs.gaussian):
        case (True, _, _):
            shape = "pvoigt"
        case (_, True, _):
            shape = "lorentzian"
        case (_, _, True):
            shape = "gaussian"
        case _:
            return [determine_shape_name(param) for param in spectra.params[1:]]

    return [shape] * (spectra.data.ndim - 1)


__all__ = [
    "NUCLEUS_LABELS",
    "DimensionInfo",
    "Spectra",
    "SpectralParameters",
    "determine_shape_name",
    "get_dimension_label",
    "get_shape_names",
    "read_spectra",
    "read_spectral_parameters",
]
