"""Domain models and helpers for spectra and spectral parameters."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from peakfit.engine.domain.config import PHASED_FAMILIES, LineshapeFamily
from peakfit.engine.domain.coordinates import DimensionContext
from peakfit.shared.typing import FloatArray

_MIN_SPECTRA_NDIM = 2

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.engine.domain.config import Doublet, FitConfig


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
    def pts2hz_delta(self, pts: float) -> float: ...

    @overload
    def pts2hz_delta(self, pts: FloatArray) -> FloatArray: ...

    def pts2hz_delta(self, pts: float | FloatArray) -> float | FloatArray:
        """Convert a point delta count back to Hz using delta scaling."""
        return pts * self.obs * self.delta

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
    source_plane_indices: tuple[int, ...] = ()

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
        if not self.source_plane_indices:
            self.source_plane_indices = tuple(range(self.data.shape[0]))
        if len(self.source_plane_indices) != self.data.shape[0]:
            raise ValueError("Source plane mapping must match the fitted series count.")
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

    def excluding_planes(self, excluded: Sequence[int]) -> Spectra:
        """The planes a fit uses: this spectrum without the explicitly ``excluded`` planes.

        ``excluded`` holds zero-based indices of this spectrum's planes. The returned
        spectrum keeps each remaining plane's ``source_plane_indices`` entry, so its
        columns stay identified; this spectrum is unchanged. With nothing excluded it is
        returned itself.
        """
        indices = fitted_plane_indices(self.data.shape[0], excluded)
        if len(indices) == self.data.shape[0]:
            return self
        return self.model_copy(
            update={
                "data": self.data[list(indices)],
                "z_values": self.z_values[list(indices)],
                "source_plane_indices": tuple(self.source_plane_indices[i] for i in indices),
            }
        )


def fitted_plane_indices(n_planes: int, excluded: Sequence[int]) -> tuple[int, ...]:
    """Zero-based indices of the planes a fit uses: every plane not explicitly excluded.

    This is the one check of an exclusion against a spectrum. Raises ``ValueError``
    for an index outside the ``n_planes`` planes, or an exclusion that leaves no
    plane to fit.
    """
    left_out = set(excluded)
    outside = sorted(i for i in left_out if not 0 <= i < n_planes)
    if outside:
        msg = (
            f"Cannot exclude plane {format_planes(outside)}: the spectrum has "
            f"{n_planes} plane{'s' if n_planes != 1 else ''}, numbered 0 to {n_planes - 1}."
        )
        raise ValueError(msg)
    included = tuple(i for i in range(n_planes) if i not in left_out)
    if not included:
        raise ValueError("Every plane is excluded; at least one plane must be fitted.")
    return included


def format_planes(planes: Sequence[int]) -> str:
    """Zero-based plane indices as user-facing text: ``1, 4``."""
    return ", ".join(str(i) for i in planes)


# The family an automatic axis falls back to when no apodization is recognized.
DEFAULT_SHAPE: LineshapeFamily = "pvoigt"


def determine_shape_name(dim_params: SpectralParameters) -> LineshapeFamily:
    """Infer default shape name for a dimension from apodization parameters."""
    sp2_apodq3 = 2.0
    if dim_params.apocode == 1.0:
        if dim_params.apodq3 == 1.0:
            return "sp1"
        if dim_params.apodq3 == sp2_apodq3:
            return "sp2"
    if dim_params.apocode in {0.0, sp2_apodq3}:
        return "no_apod"
    return DEFAULT_SHAPE


@dataclass(frozen=True)
class AxisLineshape:
    """The lineshape model on one spectral axis of a spectrum, ``auto`` resolved.

    ``detected`` says the family came from the axis's apodization (``auto``).
    """

    axis: str
    family: str
    doublet: Doublet | None = None
    phase: bool = False
    detected: bool = False

    @property
    def details(self) -> str:
        """Its splitting and phase as display text (see ``model_details``)."""
        return model_details(self.doublet, phase=self.phase)


def model_details(doublet: Doublet | None, *, phase: bool) -> str:
    """An axis model's splitting and phase: `` doublet (J 92 Hz in [46, 184]), phase fitted``.

    Empty for a singlet without phase; the family goes before it, in each interface's words.
    """
    text = ""
    if doublet is not None:
        fixed = ", fixed" if not doublet.vary else ""
        text += f" doublet (J {doublet.j:g}\u00a0Hz in [{doublet.min:g}, {doublet.max:g}]{fixed})"
    return f"{text}, phase fitted" if phase else text


def axis_lineshapes(config: FitConfig, spectra: Spectra) -> list[AxisLineshape]:
    """The lineshape model of each spectral axis, in axis order.

    ``auto`` takes the family from each axis's NMRPipe apodization. A configured
    axis the spectrum lacks, or a phase on a family that has none, is refused.
    """
    labels = [
        param.label or get_dimension_label(i) for i, param in enumerate(spectra.spectral_params)
    ]
    unknown = sorted(set(config.axes) - set(labels))
    if unknown:
        msg = (
            f"[fitting.axes] names {', '.join(unknown)}, which this spectrum lacks; "
            f"its spectral axes are {', '.join(labels)}."
        )
        raise ValueError(msg)
    resolved = []
    for label, param in zip(labels, spectra.spectral_params, strict=True):
        model = config.axis_model(label)
        detected = model.family == "auto"
        family = determine_shape_name(param) if detected else str(model.family)
        if model.phase and family not in PHASED_FAMILIES:
            how = "detected from the apodization" if detected else "configured"
            msg = (
                f"{label}: phase fitting needs a phased lineshape "
                f"({', '.join(sorted(PHASED_FAMILIES))}), "
                f"but the {how} family is {family}. Choose a phased family on {label} "
                "or turn its phase off."
            )
            raise ValueError(msg)
        resolved.append(AxisLineshape(label, family, model.doublet, model.phase, detected))
    return resolved


Spectrum = Spectra

__all__ = [
    "DEFAULT_SHAPE",
    "AxisLineshape",
    "DimensionInfo",
    "Spectra",
    "SpectralParameters",
    "Spectrum",
    "axis_lineshapes",
    "determine_shape_name",
    "fitted_plane_indices",
    "format_planes",
    "get_dimension_label",
    "model_details",
]
