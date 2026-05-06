"""Domain models and helpers for spectra and spectral parameters."""

from typing import TYPE_CHECKING, Any, overload

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from peakfit.engine.domain.coordinates import DimensionContext
from peakfit.shared.typing import FloatArray

_MIN_SPECTRA_NDIM = 2

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.engine.domain.config import FitConfig


def get_dimension_label(dim_index: int) -> str:
    """Get dimension label using Bruker Topspin convention.

    F1 = pseudo-dimension, F2/F3/F4 = spectral dimensions.
    """
    return f"F{dim_index + 2}"


class DimensionInfo(BaseModel):
    """Metadata for a single spectral dimension.

    Follows NMRPipe convention where F1 is first indirect, Fn is direct.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int  # 0-based index within spectral dimensions
    label: str  # "F1", "F2", "F3", "F4"
    nucleus: str | None  # "1H", "15N", "13C", etc. (from header)
    size: int  # Number of points
    sw_hz: float  # Spectral width in Hz
    sf_mhz: float  # Spectrometer frequency in MHz
    is_direct: bool  # True for acquisition dimension
    is_pseudo: bool = False  # True for the series dimension (CEST offsets, etc.)


class SpectralParameters(BaseModel):
    """Parameters for a single spectral dimension.

    Contains both NMRPipe header information and derived values
    for unit conversions.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

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
    td_size: int | None = None  # Acquisition-domain points (from header TDSIZE)
    ft_size: int | None = None  # Frequency-domain processed points (from header FTSIZE)
    delta: float = Field(default=0.0, init=False)
    first: float = Field(default=0.0, init=False)

    @model_validator(mode="after")
    def compute_derived_params(self) -> SpectralParameters:
        """Compute derived conversion parameters (delta and first sample)."""
        self.delta = -self.sw / (self.size * self.obs) if self.size * self.obs != 0.0 else 0.0
        self.first = self.car / self.obs - self.delta * self.size / 2.0 if self.obs != 0.0 else 0.0
        return self

    @overload
    def hz2pts_delta(self, hz: float) -> float: ...

    @overload
    def hz2pts_delta(self, hz: FloatArray) -> FloatArray: ...

    def hz2pts_delta(self, hz: float | FloatArray) -> float | FloatArray:
        """Convert a frequency difference in Hz into point units using delta scaling."""
        return hz / (self.obs * self.delta)

    @overload
    def pts2hz_delta(self, pts: float) -> float: ...

    @overload
    def pts2hz_delta(self, pts: FloatArray) -> FloatArray: ...

    def pts2hz_delta(self, pts: float | FloatArray) -> float | FloatArray:
        """Convert a point delta count back to Hz using delta scaling."""
        return pts * self.obs * self.delta

    @overload
    def hz2pts(self, hz: float) -> float: ...

    @overload
    def hz2pts(self, hz: FloatArray) -> FloatArray: ...

    def hz2pts(self, hz: float | FloatArray) -> float | FloatArray:
        """Convert a frequency in Hz to fractional point coordinate."""
        return ((hz / self.obs) - self.first) / self.delta

    def hz2pt_i(self, hz: float) -> int:
        """Convert a frequency in Hz to the nearest integer point index within the dimension."""
        return int(round(self.hz2pts(hz)) % self.size)

    @overload
    def pts2hz(self, pts: float) -> float: ...

    @overload
    def pts2hz(self, pts: FloatArray) -> FloatArray: ...

    def pts2hz(self, pts: float | FloatArray) -> float | FloatArray:
        """Convert a fractional point coordinate back to frequency (Hz)."""
        return (pts * self.delta + self.first) * self.obs

    @overload
    def ppm2pts(self, ppm: float) -> float: ...

    @overload
    def ppm2pts(self, ppm: FloatArray) -> FloatArray: ...

    def ppm2pts(self, ppm: float | FloatArray) -> float | FloatArray:
        """Convert a ppm value to fractional point coordinate using first/delta."""
        return (ppm - self.first) / self.delta

    def ppm2pt_i(self, ppm: float) -> int:
        """Convert a ppm value into an integer point index (wrapped by dimension size)."""
        return int(round(self.ppm2pts(ppm)) % self.size)

    @overload
    def pts2ppm(self, pts: float) -> float: ...

    @overload
    def pts2ppm(self, pts: FloatArray) -> FloatArray: ...

    def pts2ppm(self, pts: float | FloatArray) -> float | FloatArray:
        """Convert fractional point coordinate to ppm using delta/first scaling."""
        return (pts * self.delta) + self.first

    @overload
    def hz2ppm(self, hz: float) -> float: ...

    @overload
    def hz2ppm(self, hz: FloatArray) -> FloatArray: ...

    def hz2ppm(self, hz: float | FloatArray) -> float | FloatArray:
        """Convert frequency (Hz) to ppm using observation frequency (MHz)."""
        return hz / self.obs

    @overload
    def ppm2hz(self, ppm: float) -> float: ...

    @overload
    def ppm2hz(self, ppm: FloatArray) -> FloatArray: ...

    def ppm2hz(self, ppm: float | FloatArray) -> float | FloatArray:
        """Convert ppm to frequency (Hz) using observation frequency (MHz)."""
        return ppm * self.obs

    @overload
    def ppm2rads(self, ppm: float) -> float: ...

    @overload
    def ppm2rads(self, ppm: FloatArray) -> FloatArray: ...

    def ppm2rads(self, ppm: float | FloatArray) -> float | FloatArray:
        """Convert ppm difference to angular frequency (rad/s)."""
        return (2.0 * np.pi) * self.ppm2hz(ppm)

    def to_dimension_context(self, label: str) -> DimensionContext:
        """Create a DimensionContext for this dimension.

        This keeps all coordinate-conversion wiring in one place.
        """
        return DimensionContext(
            label=label,
            size=self.size,
            is_direct=self.direct,
            has_p180=self.p180,
            ppm2pts=self.ppm2pts,
            ppm2pt_i=self.ppm2pt_i,
            pts2hz_delta=self.pts2hz_delta,
            ppm2hz=self.ppm2hz,
        )


class Spectra(BaseModel):
    """Container for NMR spectrum data with metadata.

    Handles pseudo-ND experiments where the first dimension represents
    a series (CEST offsets, relaxation delays, etc.) rather than a
    spectral dimension.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    dic: dict[str, Any]
    data: FloatArray = Field(description="Spectrum data (FloatArray)")
    z_values: FloatArray = Field(description="Z-values array (FloatArray)")
    pseudo_dim_added: bool = False

    @field_validator("data", mode="before")
    @classmethod
    def ensure_float_array(cls, v: Any) -> Any:
        """Convert list/tuple to ndarray before Pydantic validation."""
        if isinstance(v, (list, tuple)):
            return np.array(v, dtype=np.float64)
        return v

    @field_validator("z_values", mode="before")
    @classmethod
    def ensure_z_values_array(cls, v: Any) -> Any:
        """Convert list/tuple to ndarray before Pydantic validation."""
        if isinstance(v, (list, tuple)):
            return np.array(v, dtype=np.float64)
        return v

    @model_validator(mode="after")
    def initialize_spectra(self) -> Spectra:
        """Post-initialization: ensure `z_values` are set."""
        if self.data.ndim < _MIN_SPECTRA_NDIM:
            msg = f"Spectra data must have at least 2 dimensions, got {self.data.ndim}"
            raise ValueError(msg)
        # Initialize z_values if empty
        if self.z_values.size == 0:
            self.z_values = np.arange(self.data.shape[0], dtype=np.float64)
        return self

    params: list[SpectralParameters] = Field(description="Spectral parameters for all dimensions")

    @property
    def n_spectral_dims(self) -> int:
        """Number of spectral dimensions (excluding pseudo dimension)."""
        return len(self.params) - 1

    @property
    def spectral_params(self) -> list[SpectralParameters]:
        """Get spectral parameters for spectral dimensions only (excluding pseudo)."""
        return self.params[1:]

    @property
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
        """Get dimension info by label or index."""
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
        """Remove planes (first axis) listed in exclude_list from the data."""
        if exclude_list is None:
            return
        mask = ~np.isin(range(self.data.shape[0]), exclude_list)
        self.data, self.z_values = self.data[mask], self.z_values[mask]


def determine_shape_name(dim_params: SpectralParameters) -> str:
    """Infer default shape name for a dimension from apodization parameters."""
    sp2_apodq3 = 2.0
    if dim_params.apocode == 1.0:
        if dim_params.apodq3 == 1.0:
            return "sp1"
        if dim_params.apodq3 == sp2_apodq3:
            return "sp2"
    if dim_params.apocode in {0.0, sp2_apodq3}:
        return "no_apod"
    return "pvoigt"


def get_shape_names(config: FitConfig, spectra: Spectra) -> list[str]:
    """Determine list of shape names per dimension.

    Logic:
    - If config.lineshape is explicit, use it.
    - If "auto", inspect detected processing parameters (params).
    """
    requested = config.lineshape
    if requested != "auto":
        return [requested] * spectra.n_spectral_dims

    return [determine_shape_name(param) for param in spectra.spectral_params]


Spectrum = Spectra

__all__ = [
    "DimensionInfo",
    "Spectra",
    "SpectralParameters",
    "Spectrum",
    "determine_shape_name",
    "get_dimension_label",
    "get_shape_names",
]
