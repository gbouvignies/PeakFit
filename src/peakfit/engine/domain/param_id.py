"""Structured identifiers for PeakFit parameters."""

from pydantic import BaseModel, ConfigDict, model_validator

# Pseudo-dimension axis label (Bruker convention: F1 for highest indirect dimension)
PSEUDO_AXIS = "F1"

_PARAMETER_NAME_PARTS = 3


class ParameterId(BaseModel):
    """Structured identifier for NMR fitting parameters.

    Full name format: {peak_name}.{axis}.{param_type} or {peak_name}.{axis}.I{index}
    Axis naming follows Bruker TopSpin convention (F1=pseudo, F2/F3=spectral).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    peak_name: str = ""
    label: str  # Short name (e.g. "cs", "lw", "eta", "I")
    axis: str | None = None
    index: int | None = None
    cluster_id: int | None = None

    @model_validator(mode="after")
    def validate_id(self) -> ParameterId:
        """Validate parameter identifier components."""
        if not self.peak_name and self.cluster_id is None:
            msg = "ParameterId requires either peak_name or cluster_id"
            raise ValueError(msg)

        # Avoid ambiguous identifiers: cluster parameters should not also carry a peak name.
        if self.cluster_id is not None and self.peak_name:
            msg = "ParameterId cannot set both peak_name and cluster_id"
            raise ValueError(msg)

        return self

    @property
    def name(self) -> str:
        """Full parameter name in dot-notation."""
        return self._build_name()

    @property
    def user_name(self) -> str:
        """User-friendly parameter name for output (e.g., 'cs_F2', 'I0_F1')."""
        short_name = self.label or "param"
        base = f"{short_name}{self.index}" if self.index is not None else short_name
        return f"{base}_{self.axis}" if self.axis else base

    def _build_name(self) -> str:
        """Build the full parameter name."""
        parts: list[str] = []

        # Entity: peak name or cluster id
        if self.cluster_id is not None:
            parts.append(f"cluster_{self.cluster_id}")
        else:
            parts.append(self.peak_name)

        # Axis (if applicable)
        if self.axis:
            parts.append(self.axis)

        # Parameter type short name (now just the label)
        parts.append(self.label)

        # Build base name with dots
        base_name = ".".join(parts)

        # Add index suffix if present (e.g., I0, I1 for amplitudes)
        if self.index is not None:
            return f"{base_name}{self.index}"

        return base_name

    @classmethod
    def from_name(cls, name: str) -> ParameterId:
        """Parse a parameter name back into a ParameterId."""
        return _parse_parameter_name(name)

    def __str__(self) -> str:
        """Return the full parameter name."""
        return self.name

    def __hash__(self) -> int:
        """Hash based on the full name."""
        return hash(self.name)


def _parse_parameter_name(name: str) -> ParameterId:
    """Parse a parameter name into a ParameterId.

    Supported formats:
    - {peak}.{axis}.{type}     (e.g., "peak1.F2.cs")
    - {peak}.{axis}.I{index}   (e.g., "peak1.F2.I0")
    - cluster_{id}.{axis}.{type} (e.g., "cluster_1.F2.phase")
    """
    parts = name.split(".")
    if len(parts) != _PARAMETER_NAME_PARTS:
        raise ValueError(f"Invalid parameter name format: {name}")

    entity, axis, suffix = parts

    # Handle amplitude (I0, I1, etc.)
    if suffix.startswith("I") and suffix[1:].isdigit():
        return ParameterId(peak_name=entity, axis=axis, label="I", index=int(suffix[1:]))

    # Cluster ID check
    cluster_id = None
    peak_name = ""

    if entity.startswith("cluster_"):
        try:
            cluster_id = int(entity.split("_")[1])
        except (IndexError, ValueError):
            # Fallback if not strictly "cluster_N"
            peak_name = entity
    else:
        peak_name = entity

    # Standard parameter types
    return ParameterId(
        peak_name=peak_name,
        axis=axis,
        cluster_id=cluster_id,
        label=suffix,
    )


__all__ = ["PSEUDO_AXIS", "ParameterId"]
