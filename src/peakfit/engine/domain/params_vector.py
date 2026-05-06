"""Vectorized parameter utilities."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from peakfit.engine.domain.param_id import ParameterId
from peakfit.engine.domain.param_map import ParameterMap

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.engine.domain.params_scalar import Parameters


@dataclass(frozen=True, slots=True)
class FitParametersIndex:
    """Precomputed mapping for fast Parameters -> FitParameters conversion."""

    positions: list[tuple[str | None, float]]
    widths: list[tuple[str | None, float]]
    amplitudes: list[tuple[str, float]]
    extras: dict[str, list[tuple[str, float]]]


class FitParameters(BaseModel):
    """Vectorized parameters for high-performance fitting."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    positions: np.ndarray = Field(description="Peak positions (ppm)")
    widths: np.ndarray = Field(description="Peak widths (Hz)")
    amplitudes: np.ndarray | None = Field(default=None, description="Peak amplitudes")

    # Generic extra shape parameters (e.g. "eta", "j", "phase")
    # Mapping: param_label -> array of values (one per peak/shape)
    extras: dict[str, np.ndarray] = Field(default_factory=dict)

    # Optional map to link scalar names to vector indices
    map: ParameterMap | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def ensure_arrays(cls, data: Any) -> Any:
        """Convert list inputs to numpy arrays."""
        if isinstance(data, dict):
            for field in ["positions", "widths", "amplitudes"]:
                if (
                    field in data
                    and data[field] is not None
                    and not isinstance(data[field], np.ndarray)
                ):
                    data[field] = np.array(data[field], dtype=np.float64)

            if "extras" in data and isinstance(data["extras"], dict):
                for key, val in data["extras"].items():
                    if not isinstance(val, np.ndarray):
                        data["extras"][key] = np.array(val, dtype=np.float64)

        return data

    @model_validator(mode="after")
    def check_consistency(self) -> FitParameters:
        """Ensure all present arrays have the same length (n_peaks)."""
        n_peaks = len(self.positions)
        if len(self.widths) != n_peaks:
            raise ValueError(
                f"Widths length ({len(self.widths)}) does not match positions ({n_peaks})"
            )

        if self.amplitudes is not None and len(self.amplitudes) != n_peaks:
            raise ValueError(
                f"Amplitudes length ({len(self.amplitudes)}) does not match positions ({n_peaks})"
            )

        for name, arr in self.extras.items():
            if len(arr) != n_peaks:
                if name == "phase" and len(arr) != n_peaks:
                    pass
                raise ValueError(
                    f"Extra '{name}' length ({len(arr)}) does not match positions ({n_peaks})"
                )

        return self

    def to_array(self) -> np.ndarray:
        """Flatten all present arrays into a single 1D array."""
        arrays = [self.positions, self.widths]
        if self.amplitudes is not None:
            arrays.append(self.amplitudes)

        # Sort keys for deterministic order
        for key in sorted(self.extras.keys()):
            arrays.append(self.extras[key])

        return np.concatenate(arrays)

    def from_array(self, array: np.ndarray) -> None:
        """Update arrays from a flat 1D array."""
        n = len(self.positions)
        offset = 0

        self.positions = array[offset : offset + n]
        offset += n

        self.widths = array[offset : offset + n]
        offset += n

        if self.amplitudes is not None:
            self.amplitudes = array[offset : offset + n]
            offset += n

        for key in sorted(self.extras.keys()):
            self.extras[key] = array[offset : offset + n]
            offset += n

    @classmethod
    def from_parameters(
        cls,
        params: Parameters,
        peaks: Sequence[Any],
        *,
        index: FitParametersIndex | None = None,
    ) -> FitParameters:
        """Create FitParameters from scalar Parameters and a list of Peaks."""
        plan = index or cls.build_index(peaks)
        return cls._build_from_index(params, plan)

    @staticmethod
    def build_index(peaks: Sequence[Any]) -> FitParametersIndex:
        """Precompute parameter name/default mappings for fast conversion."""
        positions: list[tuple[str | None, float]] = []
        widths: list[tuple[str | None, float]] = []
        amplitudes: list[tuple[str, float]] = []
        extras: dict[str, list[tuple[str, float]]] = {}
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
                    positions.append((pos_id, shape.center))
                else:
                    positions.append((None, shape.center))

                if width_spec is not None:
                    width_id = ParameterId(
                        peak_name=peak_name, axis=axis, label=width_spec.name
                    ).name
                    widths.append((width_id, width_spec.default))
                else:
                    widths.append((None, getattr(shape, "fwhm", getattr(shape, "lw", 1.0))))

                amp_id = ParameterId(
                    peak_name=peak_name,
                    axis=axis,
                    label="I",
                    index=0,
                ).name
                amplitudes.append((amp_id, 1.0))

                for spec in specs[2:]:
                    if spec.name not in extras:
                        extras[spec.name] = []

                    p_name = ParameterId(
                        peak_name=peak_name,
                        axis=axis,
                        label=spec.name,
                    ).name
                    extras[spec.name].append((p_name, spec.default))

        return FitParametersIndex(
            positions=positions,
            widths=widths,
            amplitudes=amplitudes,
            extras=extras,
        )

    @classmethod
    def _build_from_index(
        cls,
        params: Parameters,
        index: FitParametersIndex,
    ) -> FitParameters:
        """Build arrays from scalar parameters using a precomputed index."""
        positions = np.array(
            [
                params[name].value if name and name in params else default
                for name, default in index.positions
            ],
            dtype=np.float64,
        )
        widths = np.array(
            [
                params[name].value if name and name in params else default
                for name, default in index.widths
            ],
            dtype=np.float64,
        )

        has_amp = any(name in params for name, _default in index.amplitudes)
        amplitudes = (
            np.array(
                [
                    params[name].value if name in params else default
                    for name, default in index.amplitudes
                ],
                dtype=np.float64,
            )
            if has_amp
            else None
        )

        extras_arrays: dict[str, np.ndarray] = {}
        for key, items in index.extras.items():
            extras_arrays[key] = np.array(
                [params[name].value if name in params else default for name, default in items],
                dtype=np.float64,
            )

        param_map = ParameterMap.from_names(params.get_vary_names())

        return cls(
            positions=positions,
            widths=widths,
            amplitudes=amplitudes,
            extras=extras_arrays,
            map=param_map,
        )


FitParameters.model_rebuild()


__all__ = ["FitParameters", "FitParametersIndex"]
