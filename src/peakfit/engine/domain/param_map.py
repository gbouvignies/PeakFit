"""Parameter name to index mapping utilities."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ParameterMap:
    """Centralized mapping between parameter names and vector indices."""

    name_to_index: dict[str, int]

    @classmethod
    def from_names(cls, names: Sequence[str]) -> ParameterMap:
        """Build mapping from an ordered list of parameter names."""
        return cls(name_to_index={name: i for i, name in enumerate(names)})

    def index_of(self, name: str) -> int | None:
        """Return the vector index for a parameter name."""
        return self.name_to_index.get(name)

    def get(self, name: str, default: int | None = None) -> int | None:
        """Dictionary-style lookup with optional default."""
        return self.name_to_index.get(name, default)

    def __contains__(self, name: object) -> bool:
        """Return True if the name exists in the mapping."""
        return name in self.name_to_index


__all__ = ["ParameterMap"]
