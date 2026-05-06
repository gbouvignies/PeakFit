"""Parameter name to index mapping utilities."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from peakfit.engine.domain.param_id import ParameterId

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.engine.domain.params_scalar import Parameters


@dataclass(frozen=True, slots=True)
class ParameterMap:
    """Centralized mapping between parameter names and vector indices."""

    name_to_index: dict[str, int]

    @classmethod
    def from_names(cls, names: Sequence[str]) -> ParameterMap:
        """Build mapping from an ordered list of parameter names."""
        return cls(name_to_index={name: i for i, name in enumerate(names)})

    @classmethod
    def from_peaks(cls, peaks: Sequence[Any]) -> ParameterMap:
        """Build mapping from parameter names to shape indices."""
        name_to_index: dict[str, int] = {}
        index = 0
        for peak in peaks:
            for shape in peak.shapes:
                peak_name = peak.name
                axis = shape.axis
                specs = shape.get_parameter_spec()
                pos_spec = specs[0] if specs else None
                width_spec = specs[1] if len(specs) > 1 else None

                if pos_spec is not None:
                    pos_id = ParameterId(
                        peak_name=peak_name,
                        axis=axis,
                        label=pos_spec.name,
                    ).name
                    name_to_index[pos_id] = index

                if width_spec is not None:
                    width_id = ParameterId(
                        peak_name=peak_name, axis=axis, label=width_spec.name
                    ).name
                    name_to_index[width_id] = index

                amp_id = ParameterId(
                    peak_name=peak_name,
                    axis=axis,
                    label="I",
                    index=0,
                ).name
                name_to_index[amp_id] = index

                for spec in specs[2:]:
                    p_name = ParameterId(
                        peak_name=peak_name,
                        axis=axis,
                        label=spec.name,
                    ).name
                    name_to_index[p_name] = index

                index += 1

        return cls(name_to_index=name_to_index)

    def index_of(self, name: str) -> int | None:
        """Return the vector index for a parameter name."""
        return self.name_to_index.get(name)

    def filter_existing(self, params: Parameters) -> ParameterMap:
        """Return mapping for names present in params."""
        filtered = {name: idx for name, idx in self.name_to_index.items() if name in params}
        return type(self)(name_to_index=filtered)

    def get(self, name: str, default: int | None = None) -> int | None:
        """Dictionary-style lookup with optional default."""
        return self.name_to_index.get(name, default)

    def __contains__(self, name: object) -> bool:
        """Return True if the name exists in the mapping."""
        return name in self.name_to_index


__all__ = ["ParameterMap"]
