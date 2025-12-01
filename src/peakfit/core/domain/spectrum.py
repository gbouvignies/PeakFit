"""Domain models and helpers for spectra and spectral parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from nmrglue.fileio.pipe import guess_udic, read
from numpy.typing import NDArray

from peakfit.core.fitting.parameters import PSEUDO_AXIS

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from peakfit.core.shared.typing import FittingOptions, FloatArray

ArrayInt = NDArray[np.int_]
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


def get_dimension_label(dim_index: int) -> str:
    """Get the dimension label using Bruker Topspin convention for pseudo-nD.

    Bruker Topspin convention for pseudo-3D experiments:
    - F1 = pseudo-dimension (intensities, CEST offsets, relaxation delays)
    - F2 = first spectral dimension (indirect, e.g., 15N)
    - F3 = second spectral dimension (direct/acquisition, e.g., 1H)

    For pseudo-3D (2 spectral dims): F2 (indirect), F3 (direct)
    For pseudo-4D (3 spectral dims): F2, F3 (indirect), F4 (direct)

    Args:
        dim_index: 0-based index of the spectral dimension (0 = first spectral dim)

    Returns
    -------
        Dimension label like "F2", "F3", "F4"
    """
    # Offset by 2: dim_index 0 → F2, dim_index 1 → F3, etc.
    # F1 is reserved for the pseudo-dimension
    return f"F{dim_index + 2}"


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
        """Compute derived conversion parameters (delta and first sample)."""
        self.delta = -self.sw / (self.size * self.obs) if self.size * self.obs != 0.0 else 0.0
        self.first = self.car / self.obs - self.delta * self.size / 2.0 if self.obs != 0.0 else 0.0

    def hz2pts_delta(self, hz: T) -> T:
        """Convert a frequency difference in Hz into point units using delta scaling."""
        return hz / (self.obs * self.delta)

    def pts2hz_delta(self, pts: T) -> T:
        """Convert a point delta count back to Hz using delta scaling."""
        return pts * self.obs * self.delta

    def hz2pts(self, hz: T) -> T:
        """Convert a frequency in Hz to fractional point coordinate."""
        return ((hz / self.obs) - self.first) / self.delta

    def hz2pt_i(self, hz: float) -> int:
        """Convert a frequency in Hz to the nearest integer point index within the dimension."""
        return round(self.hz2pts(hz)) % self.size

    def pts2hz(self, pts: T) -> T:
        """Convert a fractional point coordinate back to frequency (Hz)."""
        return (pts * self.delta + self.first) * self.obs

    def ppm2pts(self, ppm: T) -> T:
        """Convert a ppm value to fractional point coordinate using first/delta."""
        return (ppm - self.first) / self.delta

    def ppm2pt_i(self, ppm: float) -> int:
        """Convert a ppm value into an integer point index (wrapped by dimension size)."""
        return round(self.ppm2pts(ppm)) % self.size

    def pts2ppm(self, pts: T) -> T:
        """Convert fractional point coordinate to ppm using delta/first scaling."""
        return (pts * self.delta) + self.first

    def hz2ppm(self, hz: T) -> T:
        """Convert frequency (Hz) to ppm using observation frequency (MHz)."""
        return hz / self.obs


def read_spectral_parameters(
    dic: dict[str, Any], data: FloatArray, *, has_pseudo_dim: bool = False
) -> list[SpectralParameters]:
    """Read spectral parameters from NMRPipe dictionary.

    Args:
        dic: NMRPipe header dictionary
        data: Spectrum data array
        has_pseudo_dim: If True, first dimension is pseudo (not a spectral dim)

    Returns
    -------
        List of SpectralParameters, one per dimension
    """
    spec_params: list[SpectralParameters] = []

    for i in range(data.ndim):
        size = data.shape[i]
        fdf = f"FDF{int(dic['FDDIMORDER'][data.ndim - 1 - i])}"
        is_direct = i == data.ndim - 1
        ft = dic.get(f"{fdf}FTFLAG", 0.0) == 1.0

        # Determine dimension label
        if has_pseudo_dim and i == 0:
            # First dimension is pseudo (CEST offsets, relaxation delays, etc.)
            # Use F1 label following Bruker convention
            dim_label = PSEUDO_AXIS
            nucleus = None
        else:
            # Spectral dimension - use F2, F3, F4 convention
            spectral_index = i - 1 if has_pseudo_dim else i
            dim_label = get_dimension_label(spectral_index)
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
        """Post-initialization: detect pseudo-dim and ensure `z_values` are set.

        Uses `nmrglue.fileio.pipe.guess_udic` to inspect the header and
        sets `pseudo_dim_added` if a pseudo dimension was added.
        """
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

        Returns
        -------
            DimensionInfo for the requested dimension

        Raises
        ------
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
        """Remove the planes (first axis) listed in `exclude_list` from the data.

        Args:
            exclude_list: Sequence of integer plane indices to exclude (first axis).
        """
        if exclude_list is None:
            return
        mask = ~np.isin(range(self.data.shape[0]), exclude_list)
        self.data, self.z_values = self.data[mask], self.z_values[mask]


def read_spectra(
    path_spectra: Path,
    path_z_values: Path | None = None,
    exclude_list: Sequence[int] | None = None,
) -> Spectra:
    """Read an NMRPipe spectrum and optional plane/z-values file.

    Returns a `Spectra` object containing the header dictionary and data, and
    optionally a set of z-values loaded from `path_z_values`.
    """
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
    """Infer default shape name for a dimension from apodization parameters.

    - Returns `sp1` or `sp2` for apocode=1 with q=1 or 2 respectively, `no_apod`
      for apocode 0 or 2, and `pvoigt` otherwise.
    """
    if dim_params.apocode == 1.0:
        if dim_params.apodq3 == 1.0:
            return "sp1"
        if dim_params.apodq3 == 2.0:
            return "sp2"
    if dim_params.apocode in {0.0, 2.0}:
        return "no_apod"
    return "pvoigt"


def get_shape_names(clargs: FittingOptions, spectra: Spectra) -> list[str]:
    """Return a list of shape names per spectral dimension based on CLI options.

    If `pvoigt`, `lorentzian` or `gaussian` are specified, that shape is used
    for all spectral dimensions. Otherwise, the default shape is computed from
    header params per-dimension.
    """
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
