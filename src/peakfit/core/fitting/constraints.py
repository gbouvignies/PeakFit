"""Parameter constraints system for NMR peak fitting.

This module provides a flexible constraint system that allows users to:
- Set parameter starting values, bounds, and vary status
- Define position windows relative to peak positions
- Apply constraints using glob-style pattern matching
- Load constraints from previous fit results

The constraint priority (low to high):
1. Code defaults (from lineshape models)
2. Global defaults (parameters.defaults)
3. Per-type defaults (parameters.defaults by pattern)
4. Per-dimension position windows (parameters.position_windows)
5. Per-peak constraints (parameters.peaks.{name})
6. Per-peak position windows (parameters.peaks.{name}.position_windows)
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from peakfit.core.fitting.parameters import Parameters


class ParameterConstraint(BaseModel):
    """Constraint for a single parameter.

    All fields are optional - only specified fields will be applied.

    Attributes
    ----------
        value: Starting value for the parameter
        min: Lower bound for fitting
        max: Upper bound for fitting
        vary: Whether parameter varies during fitting (False = fixed)
    """

    value: float | None = None
    min: float | None = None
    max: float | None = None
    vary: bool | None = None

    def is_empty(self) -> bool:
        """Check if constraint has any non-None fields."""
        return all(v is None for v in [self.value, self.min, self.max, self.vary])


class PositionWindowConfig(BaseModel):
    """Position window configuration per axis.

    Defines how much a peak position can move from its starting value.
    The actual bounds are computed as: [position - window, position + window]
    """

    F1: float | None = None
    F2: float | None = None
    F3: float | None = None
    F4: float | None = None

    def get(self, axis: str) -> float | None:
        """Get window for a specific axis."""
        return getattr(self, axis, None)

    def __getitem__(self, axis: str) -> float | None:
        """Allow dict-like access."""
        return self.get(axis)


class PeakConstraints(BaseModel):
    """Constraints for a single peak.

    Attributes
    ----------
        position_window: Window (ppm) for all position parameters of this peak
        position_windows: Per-axis position windows (overrides position_window)
        parameters: Direct parameter constraints keyed by "{axis}.{type}"
            e.g., "F2.cs", "F3.lw", "F2.eta"
    """

    position_window: float | None = None
    position_windows: PositionWindowConfig = Field(default_factory=PositionWindowConfig)
    # Parameter constraints keyed by short name like "F2.cs", "F3.lw"
    # Using a dict allows flexible parameter targeting
    parameters: dict[str, ParameterConstraint] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def collect_inline_constraints(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Collect inline parameter constraints from TOML.

        Allows users to write:
            [parameters.peaks."2N-H"]
            "F2.cs" = { value = 120.5 }

        Instead of:
            [parameters.peaks."2N-H".parameters]
            "F2.cs" = { value = 120.5 }
        """
        if not isinstance(data, dict):
            return data

        parameters = data.get("parameters", {})
        known_fields = {
            "position_window",
            "position_windows",
            "parameters",
        }

        # Move any unknown keys that look like parameter specs into parameters
        for key in list(data.keys()):
            if key not in known_fields and re.match(r"^F\d+\.\w+$", key):
                value = data.pop(key)
                if isinstance(value, dict):
                    parameters[key] = value
                else:
                    # Allow shorthand: "F2.cs" = 120.5 means { value = 120.5 }
                    parameters[key] = {"value": value}

        data["parameters"] = parameters
        return data


class ParameterDefaults(BaseModel):
    """Default constraints applied by pattern matching.

    Patterns use glob-style matching:
    - "*" matches any sequence of characters
    - "?" matches any single character

    Examples
    --------
        "*.*.cs" - all chemical shift parameters
        "*.*.lw" - all linewidth parameters
        "*.F2.*" - all F2 dimension parameters
        "2N-H.*.*" - all parameters for peak 2N-H
    """

    # Pattern -> constraint mapping
    patterns: dict[str, ParameterConstraint] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def collect_patterns(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Collect pattern constraints from flat TOML structure."""
        if not isinstance(data, dict):
            return data

        patterns = data.get("patterns", {})

        # Move any keys that look like patterns into patterns dict
        for key in list(data.keys()):
            if key != "patterns" and ("*" in key or "." in key):
                value = data.pop(key)
                if isinstance(value, dict):
                    patterns[key] = value

        data["patterns"] = patterns
        return data


class ParameterConfig(BaseModel):
    """Complete parameter configuration.

    This is the main configuration class that users interact with via TOML.

    Example TOML:
        [parameters]
        position_window = 0.1  # Global default

        [parameters.position_windows]
        F2 = 0.5  # 15N dimension
        F3 = 0.05  # 1H dimension

        [parameters.defaults]
        "*.*.lw" = { min = 5.0, max = 100.0 }
        "*.*.eta" = { value = 0.5, vary = false }

        [parameters.peaks."2N-H"]
        position_window = 0.02
        "F2.cs" = { vary = false }

        [parameters.peaks."G45N-HN".position_windows]
        F2 = 1.0
        F3 = 0.03
    """

    # Global position window (ppm) for all peaks/axes
    position_window: float | None = None

    # Per-axis position windows
    position_windows: PositionWindowConfig = Field(default_factory=PositionWindowConfig)

    # Pattern-based defaults
    defaults: ParameterDefaults = Field(default_factory=ParameterDefaults)

    # Per-peak constraints
    peaks: dict[str, PeakConstraints] = Field(default_factory=dict)

    # Load starting values from previous fit
    from_file: Path | None = None

    @field_validator("from_file", mode="before")
    @classmethod
    def validate_from_file(cls, v: str | Path | None) -> Path | None:
        """Convert string to Path."""
        if v is None:
            return None
        return Path(v)


@dataclass
class ResolvedConstraint:
    """Fully resolved constraint for a specific parameter.

    This represents the final constraint after all priority rules
    have been applied.
    """

    value: float | None = None
    min: float | None = None
    max: float | None = None
    vary: bool | None = None
    source: str = "default"  # Where this constraint came from

    def merge_from(self, other: ParameterConstraint, source: str) -> None:
        """Merge another constraint, overwriting non-None values."""
        if other.value is not None:
            self.value = other.value
            self.source = source
        if other.min is not None:
            self.min = other.min
            self.source = source
        if other.max is not None:
            self.max = other.max
            self.source = source
        if other.vary is not None:
            self.vary = other.vary
            self.source = source


@dataclass
class ConstraintResolver:
    """Resolves parameter constraints from configuration.

    This class handles the priority-based resolution of constraints
    from various sources (defaults, patterns, per-peak, etc.).
    """

    config: ParameterConfig
    _from_file_values: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Load values from file if specified."""
        if self.config.from_file is not None:
            self._load_from_file(self.config.from_file)

    def _load_from_file(self, path: Path) -> None:
        """Load parameter values from a previous fit result."""
        if not path.exists():
            msg = f"Parameter file not found: {path}"
            raise FileNotFoundError(msg)

        suffix = path.suffix.lower()
        if suffix == ".json":
            self._load_from_json(path)
        else:
            msg = f"Unsupported file format: {suffix}"
            raise ValueError(msg)

    def _load_from_json(self, path: Path) -> None:
        """Load parameter values from JSON fit summary."""
        with path.open() as f:
            data = json.load(f)

        # Handle fit_summary.json format
        if "clusters" in data:
            for cluster in data["clusters"]:
                for param in cluster.get("lineshape_parameters", []):
                    name = param.get("name", "")
                    value = param.get("value")
                    if name and value is not None:
                        self._from_file_values[name] = float(value)

                for amp in cluster.get("amplitudes", []):
                    peak = amp.get("peak_name", "")
                    plane = amp.get("plane_index", 0)
                    value = amp.get("value")
                    if peak and value is not None:
                        name = f"{peak}.F1.I{plane}"
                        self._from_file_values[name] = float(value)

    def resolve(
        self,
        param_name: str,
        peak_name: str,
        axis: str,
        param_type: str,
        current_value: float,
    ) -> ResolvedConstraint:
        """Resolve all constraints for a parameter.

        Args:
            param_name: Full parameter name (e.g., "2N-H.F2.cs")
            peak_name: Peak name (e.g., "2N-H")
            axis: Axis label (e.g., "F2")
            param_type: Parameter type short name (e.g., "cs", "lw")
            current_value: Current/default value from lineshape model

        Returns
        -------
            Fully resolved constraint with all priorities applied
        """
        resolved = ResolvedConstraint(value=current_value, source="model_default")

        # 1. Apply pattern-based defaults
        self._apply_pattern_defaults(resolved, param_name)

        # 2. Apply position window if this is a position parameter
        if param_type == "cs":
            self._apply_position_window(resolved, peak_name, axis, current_value)

        # 3. Apply per-peak constraints
        self._apply_peak_constraints(resolved, peak_name, axis, param_type)

        # 4. Apply values from file (lowest priority for values, doesn't affect bounds/vary)
        if param_name in self._from_file_values and resolved.value == current_value:
            resolved.value = self._from_file_values[param_name]
            resolved.source = "from_file"

        return resolved

    def _apply_pattern_defaults(self, resolved: ResolvedConstraint, param_name: str) -> None:
        """Apply pattern-based default constraints."""
        for pattern, constraint in self.config.defaults.patterns.items():
            if self._matches_pattern(param_name, pattern):
                resolved.merge_from(constraint, f"pattern:{pattern}")

    def _apply_position_window(
        self,
        resolved: ResolvedConstraint,
        peak_name: str,
        axis: str,
        position: float,
    ) -> None:
        """Apply position window constraints for chemical shift parameters."""
        window: float | None = None
        source = "default"

        # Priority 1: Global position window
        if self.config.position_window is not None:
            window = self.config.position_window
            source = "global_position_window"

        # Priority 2: Per-axis position window
        axis_window = self.config.position_windows.get(axis)
        if axis_window is not None:
            window = axis_window
            source = f"position_windows.{axis}"

        # Priority 3: Per-peak position window
        if peak_name in self.config.peaks:
            peak_config = self.config.peaks[peak_name]

            if peak_config.position_window is not None:
                window = peak_config.position_window
                source = f"peaks.{peak_name}.position_window"

            # Priority 4: Per-peak, per-axis position window
            peak_axis_window = peak_config.position_windows.get(axis)
            if peak_axis_window is not None:
                window = peak_axis_window
                source = f"peaks.{peak_name}.position_windows.{axis}"

        # Apply window if set
        if window is not None:
            resolved.min = position - window
            resolved.max = position + window
            resolved.source = source

    def _apply_peak_constraints(
        self,
        resolved: ResolvedConstraint,
        peak_name: str,
        axis: str,
        param_type: str,
    ) -> None:
        """Apply per-peak parameter constraints."""
        if peak_name not in self.config.peaks:
            return

        peak_config = self.config.peaks[peak_name]

        # Look for constraint by short name (e.g., "F2.cs")
        short_name = f"{axis}.{param_type}"
        if short_name in peak_config.parameters:
            constraint = peak_config.parameters[short_name]
            resolved.merge_from(constraint, f"peaks.{peak_name}.{short_name}")

    @staticmethod
    def _matches_pattern(name: str, pattern: str) -> bool:
        """Check if parameter name matches a glob pattern.

        Supports:
        - "*" matches any sequence
        - "?" matches single character
        - "." is treated literally

        Examples
        --------
            "*.*.cs" matches "2N-H.F2.cs"
            "*.F2.*" matches "2N-H.F2.cs", "Peak1.F2.lw"
        """
        # Convert glob pattern to regex
        regex = fnmatch.translate(pattern)
        return bool(re.match(regex, name))


def apply_constraints(
    params: Parameters,
    config: ParameterConfig,
) -> Parameters:
    """Apply constraints from configuration to parameters.

    This is the main entry point for constraint application.

    Args:
        params: Parameters object to modify
        config: Parameter configuration with constraints

    Returns
    -------
        Modified Parameters object (same instance)
    """
    resolver = ConstraintResolver(config)

    for name, param in params.items():
        # Extract components from parameter name or param_id
        if param.param_id is not None:
            peak_name = param.param_id.peak_name
            axis = param.param_id.axis or ""
            # Get short type name
            from peakfit.core.fitting.parameters import _PARAM_TYPE_SHORT_NAMES

            param_type = _PARAM_TYPE_SHORT_NAMES.get(param.param_id.param_type, "param")
        else:
            # Fallback: parse from name
            peak_name, axis, param_type = _parse_param_name(name)

        # Skip if we couldn't parse
        if not peak_name:
            continue

        # Resolve constraints
        resolved = resolver.resolve(
            param_name=name,
            peak_name=peak_name,
            axis=axis,
            param_type=param_type,
            current_value=param.value,
        )

        # Apply resolved constraints safely
        # We calculate new values first and update via __dict__ to avoid
        # validation errors during intermediate states (e.g. setting min > current value)

        new_min = resolved.min if resolved.min is not None else param.min
        new_max = resolved.max if resolved.max is not None else param.max
        new_value = resolved.value if resolved.value is not None else param.value

        # Clamp value to new bounds
        if new_value < new_min:
            new_value = new_min
        if new_value > new_max:
            new_value = new_max

        # Update attributes directly
        param.__dict__["min"] = new_min
        param.__dict__["max"] = new_max
        param.__dict__["value"] = new_value

        if resolved.vary is not None:
            param.vary = resolved.vary

    return params


def _parse_param_name(name: str) -> tuple[str, str, str]:
    """Parse parameter name into components.

    Args:
        name: Parameter name like "2N-H.F2.cs" or "cluster_0.F2.phase"

    Returns
    -------
        Tuple of (peak_name, axis, param_type)
    """
    # Match standard format: peak.axis.type or peak.axis.typeN
    match = re.match(r"^(.+)\.(F\d+)\.(\w+?)(\d*)$", name)
    if match:
        peak_name = match.group(1)
        axis = match.group(2)
        param_type = match.group(3)
        return peak_name, axis, param_type

    return "", "", ""


def constraints_from_cli(
    fix_patterns: list[str] | None = None,
    vary_patterns: list[str] | None = None,
    position_window: float | None = None,
    position_window_f2: float | None = None,
    position_window_f3: float | None = None,
    from_file: Path | None = None,
) -> ParameterConfig:
    """Create ParameterConfig from CLI arguments.

    Args:
        fix_patterns: Patterns for parameters to fix
        vary_patterns: Patterns for parameters to vary
        position_window: Global position window (ppm)
        position_window_f2: F2 (indirect) position window
        position_window_f3: F3 (direct) position window
        from_file: Path to load starting values from

    Returns
    -------
        ParameterConfig instance
    """
    config = ParameterConfig(
        position_window=position_window,
        from_file=from_file,
    )

    # Set per-axis windows
    if position_window_f2 is not None:
        config.position_windows.F2 = position_window_f2
    if position_window_f3 is not None:
        config.position_windows.F3 = position_window_f3

    # Add fix patterns
    if fix_patterns:
        for pattern in fix_patterns:
            config.defaults.patterns[pattern] = ParameterConstraint(vary=False)

    # Add vary patterns (processed after fix, so they can override)
    if vary_patterns:
        for pattern in vary_patterns:
            config.defaults.patterns[pattern] = ParameterConstraint(vary=True)

    return config


__all__ = [
    "ConstraintResolver",
    "ParameterConfig",
    "ParameterConstraint",
    "ParameterDefaults",
    "PeakConstraints",
    "PositionWindowConfig",
    "ResolvedConstraint",
    "apply_constraints",
    "constraints_from_cli",
]
