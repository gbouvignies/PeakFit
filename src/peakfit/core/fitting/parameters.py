"""Parameter management for NMR peak fitting.

This module provides Parameter and Parameters classes for managing
fitting parameters with bounds, types, and metadata.

The ParameterId class provides a unified parameter identification system
using dot-notation (e.g., '2N-H.F1.cs') for consistent naming across
fitting, output, and user configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, KeysView, ValuesView

# Pseudo-dimension axis label (Bruker convention: F1 for highest indirect dimension)
PSEUDO_AXIS = "F1"


class ParameterType(str, Enum):
    """Types of NMR fitting parameters."""

    POSITION = "position"  # Peak center in ppm
    FWHM = "fwhm"  # Full width at half maximum (Hz)
    FRACTION = "fraction"  # Mixing parameter (0-1)
    PHASE = "phase"  # Phase correction (degrees)
    JCOUPLING = "jcoupling"  # J-coupling constant (Hz)
    AMPLITUDE = "amplitude"  # Peak amplitude
    GENERIC = "generic"  # Other parameters


# Default bounds for NMR parameter types
_DEFAULT_BOUNDS: dict[ParameterType, tuple[float, float]] = {
    ParameterType.POSITION: (-np.inf, np.inf),  # Set dynamically from spectrum
    ParameterType.FWHM: (0.1, 200.0),  # Typical NMR linewidths
    ParameterType.FRACTION: (0.0, 1.0),  # Mixing fractions
    ParameterType.PHASE: (-180.0, 180.0),  # Phase in degrees
    ParameterType.JCOUPLING: (0.0, 20.0),  # Typical J-couplings
    ParameterType.AMPLITUDE: (-np.inf, np.inf),  # Allow negative (CEST, anti-phase)
    ParameterType.GENERIC: (-np.inf, np.inf),
}

# Maps ParameterType to user-friendly short name for output
_PARAM_TYPE_SHORT_NAMES: dict[ParameterType, str] = {
    ParameterType.POSITION: "cs",  # chemical shift
    ParameterType.FWHM: "lw",  # linewidth
    ParameterType.FRACTION: "eta",  # eta for pseudo-Voigt
    ParameterType.PHASE: "phase",
    ParameterType.JCOUPLING: "j",  # J-coupling
    ParameterType.AMPLITUDE: "I",  # intensity
    ParameterType.GENERIC: "param",
}


@dataclass(frozen=True)
class ParameterId:
    """Structured identifier for NMR fitting parameters.

    Provides a unified naming system using dot-notation for consistent
    parameter identification across fitting, output, and configuration.

    The full name format is: `{peak_name}.{axis}.{param_type}` or `{peak_name}.{axis}.I{index}`

    Axis naming follows Bruker TopSpin convention for pseudo-3D experiments:
    - F1 = pseudo-dimension (intensities, CEST offsets, relaxation delays)
    - F2 = first spectral dimension (indirect, e.g., 15N)
    - F3 = second spectral dimension (direct/acquisition, e.g., 1H)

    Examples
    --------
    - Chemical shift: "2N-H.F3.cs" (direct/1H), "2N-H.F2.cs" (indirect/15N)
    - Linewidth: "2N-H.F3.lw", "2N-H.F2.lw"
    - Amplitude: "2N-H.F1.I0" (pseudo-dimension intensity, plane 0)
    - Phase (cluster-level): "cluster_0.F3.phase"

    Attributes
    ----------
        peak_name: Name of the peak (e.g., "2N-H", "G45N-HN")
        axis: Dimension label (e.g., "F1", "F2", "F3")
        param_type: Type of parameter (ParameterType enum)
        index: Optional index for multi-valued parameters (e.g., amplitude per plane)
        cluster_id: Optional cluster ID for cluster-level parameters (e.g., phase)
    """

    peak_name: str
    param_type: ParameterType
    axis: str | None = None
    index: int | None = None
    cluster_id: int | None = None

    def __post_init__(self) -> None:
        """Validate parameter identifier components."""
        if not self.peak_name and self.cluster_id is None:
            msg = "ParameterId requires either peak_name or cluster_id"
            raise ValueError(msg)

        # Axis is required for all parameter types except GENERIC
        requires_axis = self.param_type != ParameterType.GENERIC
        if requires_axis and self.axis is None and self.cluster_id is None:
            msg = f"ParameterId for {self.param_type.value} requires axis"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        """Full parameter name in dot-notation.

        Format: `{entity}.{axis}.{type}[{index}]`
        where entity is peak_name or cluster_id.

        Examples
        --------
        - "2N-H.F2.cs"
        - "2N-H.F3.lw"
        - "2N-H.F1.I0"
        - "cluster_0.F3.phase"
        """
        return self._build_name()

    @property
    def user_name(self) -> str:
        """User-friendly parameter name for output.

        Format: `{type}_{axis}` or `{type}{index}_{axis}`

        Examples
        --------
        - "cs_F2", "lw_F3"
        - "I0_F1", "I5_F1"
        - "phase_F2"
        """
        short_name = _PARAM_TYPE_SHORT_NAMES.get(self.param_type, self.param_type.value)

        # For indexed parameters (amplitudes), include index in the name: I0_F1, I1_F1
        base = f"{short_name}{self.index}" if self.index is not None else short_name

        return f"{base}_{self.axis}" if self.axis else base

    def _build_name(self) -> str:
        """Build the full parameter name."""
        parts: list[str] = []

        # Entity: peak name or cluster id
        if self.cluster_id is not None and self.param_type == ParameterType.PHASE:
            parts.append(f"cluster_{self.cluster_id}")
        else:
            parts.append(self.peak_name)

        # Axis (if applicable)
        if self.axis:
            parts.append(self.axis)

        # Parameter type short name
        short_name = _PARAM_TYPE_SHORT_NAMES.get(self.param_type, self.param_type.value)
        parts.append(short_name)

        # Build base name with dots
        base_name = ".".join(parts)

        # Add index suffix if present (e.g., I0, I1 for amplitudes)
        if self.index is not None:
            return f"{base_name}{self.index}"

        return base_name

    @classmethod
    def position(cls, peak_name: str, axis: str) -> ParameterId:
        """Create a position (chemical shift) parameter ID."""
        return cls(peak_name=peak_name, axis=axis, param_type=ParameterType.POSITION)

    @classmethod
    def linewidth(cls, peak_name: str, axis: str) -> ParameterId:
        """Create a linewidth (FWHM) parameter ID."""
        return cls(peak_name=peak_name, axis=axis, param_type=ParameterType.FWHM)

    @classmethod
    def fraction(cls, peak_name: str, axis: str) -> ParameterId:
        """Create a fraction (eta) parameter ID for pseudo-Voigt."""
        return cls(peak_name=peak_name, axis=axis, param_type=ParameterType.FRACTION)

    @classmethod
    def phase(cls, cluster_id: int, axis: str) -> ParameterId:
        """Create a phase parameter ID (cluster-level)."""
        return cls(
            peak_name="",
            axis=axis,
            param_type=ParameterType.PHASE,
            cluster_id=cluster_id,
        )

    @classmethod
    def jcoupling(cls, peak_name: str, axis: str) -> ParameterId:
        """Create a J-coupling parameter ID."""
        return cls(peak_name=peak_name, axis=axis, param_type=ParameterType.JCOUPLING)

    @classmethod
    def amplitude(cls, peak_name: str, axis: str, plane_index: int = 0) -> ParameterId:
        """Create an amplitude parameter ID.

        Args:
            peak_name: Name of the peak
            axis: Dimension label (e.g., "F1" for pseudo-dimension)
            plane_index: Index of the plane (0-based, default 0)
        """
        return cls(
            peak_name=peak_name,
            axis=axis,
            param_type=ParameterType.AMPLITUDE,
            index=plane_index,
        )

    @classmethod
    def from_name(cls, name: str) -> ParameterId:
        """Parse a parameter name back into a ParameterId.

        Args:
            name: Parameter name like "2N-H.F1.cs"

        Returns
        -------
            Parsed ParameterId

        Raises
        ------
            ValueError: If name cannot be parsed
        """
        return _parse_parameter_name(name)

    def __str__(self) -> str:
        """Return the full parameter name."""
        return self.name

    def __hash__(self) -> int:
        """Hash based on the full name."""
        return hash(self.name)


def _parse_parameter_name(name: str) -> ParameterId:
    """Parse a parameter name into a ParameterId.

    Args:
        name: Parameter name to parse (dot-notation format)

    Returns
    -------
        Parsed ParameterId

    Raises
    ------
        ValueError: If name cannot be parsed
    """
    import re

    # Handle amplitude format: "peak.F1.I0" (with axis)
    amp_match = re.match(r"^(.+)\.(F\d+)\.I(\d+)$", name)
    if amp_match:
        return ParameterId.amplitude(
            amp_match.group(1), amp_match.group(2), int(amp_match.group(3))
        )

    # Handle dot-notation: "peak.axis.type" or "cluster_N.axis.type"
    dot_match = re.match(r"^(.+)\.(F\d+)\.(\w+)$", name)
    if dot_match:
        entity, axis, type_name = dot_match.groups()

        # Check for cluster-level parameter
        cluster_match = re.match(r"^cluster_(\d+)$", entity)
        cluster_id = int(cluster_match.group(1)) if cluster_match else None
        peak_name = "" if cluster_id is not None else entity

        # Map type name to ParameterType
        param_type = _name_to_param_type(type_name)
        return ParameterId(
            peak_name=peak_name,
            axis=axis,
            param_type=param_type,
            cluster_id=cluster_id,
        )

    msg = f"Cannot parse parameter name: {name}"
    raise ValueError(msg)


def _name_to_param_type(type_name: str) -> ParameterType:
    """Convert a short type name to ParameterType."""
    name_map = {
        "cs": ParameterType.POSITION,
        "lw": ParameterType.FWHM,
        "eta": ParameterType.FRACTION,
        "phase": ParameterType.PHASE,
        "j": ParameterType.JCOUPLING,
        "I": ParameterType.AMPLITUDE,
    }
    return name_map.get(type_name, ParameterType.GENERIC)


@dataclass
class Parameter:
    """Single NMR fitting parameter with bounds and metadata.

    Designed specifically for NMR lineshape fitting with support for
    different parameter types (position, FWHM, phase, etc.).

    Attributes
    ----------
        name: Parameter name (string). Can be a ParameterId.name or plain string.
        value: Current parameter value.
        min: Lower bound for fitting.
        max: Upper bound for fitting.
        vary: Whether parameter varies during fitting (nonlinear optimization).
        param_type: Type of NMR parameter (affects default bounds).
        unit: Unit string (e.g., "Hz", "ppm", "deg").
        stderr: Standard error from fitting.
        computed: True for parameters computed analytically (e.g., amplitudes).
        param_id: Optional structured ParameterId for rich identification.
    """

    name: str
    value: float
    min: float = -np.inf
    max: float = np.inf
    vary: bool = True
    param_type: ParameterType = ParameterType.GENERIC
    unit: str = ""  # Optional unit string (e.g., "Hz", "ppm", "deg")
    stderr: float = 0.0  # Standard error from fitting
    computed: bool = False  # True for parameters computed analytically (e.g., amplitudes)
    param_id: ParameterId | None = None  # Optional structured identifier

    def __post_init__(self) -> None:
        """Validate parameter bounds and apply type-specific defaults."""
        # Enforce invariant: computed parameters cannot vary
        if self.computed and self.vary:
            msg = f"Parameter {self.name}: computed=True requires vary=False"
            raise ValueError(msg)

        # Apply type-specific defaults if bounds not explicitly set
        if self.min == -np.inf and self.max == np.inf and self.param_type != ParameterType.GENERIC:
            default_min, default_max = _DEFAULT_BOUNDS[self.param_type]
            self.min = default_min
            self.max = default_max

        if self.min > self.max:
            msg = f"Parameter {self.name}: min ({self.min}) > max ({self.max})"
            raise ValueError(msg)
        if not self.min <= self.value <= self.max:
            msg = (
                f"Parameter {self.name}: value ({self.value}) "
                f"outside bounds [{self.min}, {self.max}]"
            )
            raise ValueError(msg)

    def __repr__(self) -> str:
        """Return a string representation of the parameter."""
        if self.computed:
            vary_str = "computed"
        elif self.vary:
            vary_str = "vary"
        else:
            vary_str = "fixed"
        min_str = f"{self.min:.4g}" if self.min > -1e10 else "-inf"
        max_str = f"{self.max:.4g}" if self.max < 1e10 else "inf"
        unit_str = f" {self.unit}" if self.unit else ""
        return (
            f"<Parameter {self.name}={self.value:.6g}{unit_str} "
            f"[{min_str}, {max_str}] ({vary_str})>"
        )

    def is_at_boundary(self, tol: float = 1e-6) -> bool:
        """Check if parameter is at or near its boundary.

        Args:
            tol: Tolerance for boundary check

        Returns
        -------
            True if at boundary
        """
        at_min = abs(self.value - self.min) < tol * (1 + abs(self.value))
        at_max = abs(self.value - self.max) < tol * (1 + abs(self.value))
        return at_min or at_max

    def relative_position(self) -> float:
        """Get the relative position of value within bounds (0 to 1).

        Returns
        -------
            0.0 if at min, 1.0 if at max, 0.5 if centered
        """
        if self.max == self.min:
            return 0.5
        if np.isinf(self.min) or np.isinf(self.max):
            return 0.5
        return (self.value - self.min) / (self.max - self.min)


@dataclass
class Parameters:
    """Collection of fitting parameters."""

    _params: dict[str, Parameter] = field(default_factory=dict)

    def add(
        self,
        name: str | ParameterId,
        value: float = 0.0,
        min: float = -np.inf,
        max: float = np.inf,
        vary: bool = True,
        param_type: ParameterType = ParameterType.GENERIC,
        unit: str = "",
        computed: bool = False,
    ) -> None:
        """Add a parameter.

        Args:
            name: Parameter name (string or ParameterId)
            value: Initial value
            min: Lower bound
            max: Upper bound
            vary: Whether parameter varies during fitting
            param_type: Type of NMR parameter (affects default bounds)
            unit: Unit string (e.g., "Hz", "ppm", "deg")
            computed: Whether parameter is computed analytically (implies vary=False)
        """
        # Handle ParameterId input
        if isinstance(name, ParameterId):
            param_id = name
            name_str = param_id.name
            # Infer param_type from ParameterId if not explicitly set
            if param_type == ParameterType.GENERIC:
                param_type = param_id.param_type
        else:
            param_id = None
            name_str = name

        self._params[name_str] = Parameter(
            name_str,
            value,
            min,
            max,
            vary,
            param_type,
            unit,
            stderr=0.0,
            computed=computed,
            param_id=param_id,
        )

    def __getitem__(self, key: str) -> Parameter:
        """Get parameter by name."""
        return self._params[key]

    def __setitem__(self, key: str, value: Parameter) -> None:
        """Set parameter."""
        self._params[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if parameter exists."""
        return key in self._params

    def __iter__(self) -> Iterator[str]:
        """Iterate over parameter names."""
        return iter(self._params)

    def keys(self) -> KeysView[str]:
        """Get parameter names."""
        return self._params.keys()

    def values(self) -> ValuesView[Parameter]:
        """Get parameter objects."""
        return self._params.values()

    def items(self) -> ItemsView[str, Parameter]:
        """Get parameter name-value pairs."""
        return self._params.items()

    def update(self, other: Parameters) -> None:
        """Update parameters from another Parameters object."""
        for name, param in other.items():
            self._params[name] = param

    def copy(self) -> Parameters:
        """Create a copy of parameters."""
        new_params = Parameters()
        for name, param in self._params.items():
            new_params._params[name] = Parameter(
                name=name,
                value=param.value,
                min=param.min,
                max=param.max,
                vary=param.vary,
                param_type=param.param_type,
                unit=param.unit,
                stderr=param.stderr,
                computed=param.computed,
                param_id=param.param_id,
            )
        return new_params

    def get_vary_names(self) -> list[str]:
        """Get names of parameters that vary (nonlinear optimization)."""
        return [name for name, param in self._params.items() if param.vary]

    def get_computed_names(self) -> list[str]:
        """Get names of computed parameters (e.g., amplitudes)."""
        return [name for name, param in self._params.items() if param.computed]

    def get_fitted_names(self) -> list[str]:
        """Get names of all fitted parameters (vary=True or computed=True).

        This includes both nonlinearly optimized parameters and analytically
        computed parameters like amplitudes.
        """
        return [name for name, param in self._params.items() if param.vary or param.computed]

    def get_n_fitted_params(self) -> int:
        """Get total number of fitted parameters for DOF calculation.

        This counts both nonlinearly optimized (vary=True) and analytically
        computed (computed=True) parameters. Used for correct degrees of
        freedom in reduced chi-square calculation.
        """
        return sum(1 for param in self._params.values() if param.vary or param.computed)

    def get_vary_values(self) -> np.ndarray:
        """Get values of varying parameters as array."""
        return np.array([self._params[name].value for name in self.get_vary_names()])

    def get_vary_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds for varying parameters."""
        names = self.get_vary_names()
        lower = np.array([self._params[name].min for name in names])
        upper = np.array([self._params[name].max for name in names])
        return lower, upper

    def set_vary_values(self, values: np.ndarray) -> None:
        """Set values of varying parameters from array."""
        names = self.get_vary_names()
        for name, value in zip(names, values, strict=True):
            self._params[name].value = value

    def set_errors(self, errors: np.ndarray) -> None:
        """Set standard errors for varying parameters.

        Args:
            errors: Array of standard errors for varying parameters
        """
        names = self.get_vary_names()
        for name, error in zip(names, errors, strict=True):
            self._params[name].stderr = error

    def get_vary_bounds_list(self) -> list[tuple[float, float]]:
        """Get bounds for varying parameters as list of tuples.

        Returns
        -------
            List of (min, max) tuples for each varying parameter
        """
        names = self.get_vary_names()
        return [(self._params[name].min, self._params[name].max) for name in names]

    def __len__(self) -> int:
        """Return number of parameters."""
        return len(self._params)

    def __repr__(self) -> str:
        """Return a string representation of the parameters collection."""
        n_total = len(self._params)
        n_vary = len(self.get_vary_names())
        n_computed = len(self.get_computed_names())
        if n_computed > 0:
            return f"<Parameters: {n_total} total, {n_vary} varying, {n_computed} computed>"
        return f"<Parameters: {n_total} total, {n_vary} varying>"

    def summary(self) -> str:
        """Get a formatted summary of all parameters.

        Returns
        -------
            Multi-line string with parameter details
        """
        lines = ["Parameters:", "=" * 60]
        for name in self._params:
            param = self._params[name]
            if param.computed:
                vary_str = "computed"
            elif param.vary:
                vary_str = "vary"
            else:
                vary_str = "fixed"
            min_str = f"{param.min:.4g}" if param.min > -1e10 else "-inf"
            max_str = f"{param.max:.4g}" if param.max < 1e10 else "inf"
            lines.append(
                f"  {name:20s} = {param.value:12.6g} [{min_str:>10s}, {max_str:<10s}] ({vary_str})"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def get_boundary_params(self) -> list[str]:
        """Get names of parameters that are at their boundaries.

        Returns
        -------
            List of parameter names at boundaries
        """
        return [
            name for name, param in self._params.items() if param.vary and param.is_at_boundary()
        ]

    def freeze(self, names: list[str] | None = None) -> None:
        """Set parameters to not vary (freeze them).

        Args:
            names: List of parameter names to freeze. If None, freeze all.
        """
        if names is None:
            names = list(self._params.keys())
        for name in names:
            if name in self._params:
                self._params[name].vary = False

    def unfreeze(self, names: list[str] | None = None) -> None:
        """Set parameters to vary (unfreeze them).

        Args:
            names: List of parameter names to unfreeze. If None, unfreeze all.
        """
        if names is None:
            names = list(self._params.keys())
        for name in names:
            if name in self._params:
                self._params[name].vary = True

    def get_by_peak(self, peak_name: str) -> list[Parameter]:
        """Get all parameters belonging to a specific peak.

        Args:
            peak_name: Peak name to filter by

        Returns
        -------
            List of parameters for the peak
        """
        return [
            param
            for param in self._params.values()
            if param.param_id is not None and param.param_id.peak_name == peak_name
        ]

    def get_by_type(self, param_type: ParameterType) -> list[Parameter]:
        """Get all parameters of a specific type.

        Args:
            param_type: Parameter type to filter by

        Returns
        -------
            List of parameters of the specified type
        """
        return [p for p in self._params.values() if p.param_type == param_type]

    def get_by_axis(self, axis: str) -> list[Parameter]:
        """Get all parameters for a specific axis/dimension.

        Args:
            axis: Axis label (e.g., "F1", "F2")

        Returns
        -------
            List of parameters for the axis
        """
        return [
            param
            for param in self._params.values()
            if param.param_id is not None and param.param_id.axis == axis
        ]
