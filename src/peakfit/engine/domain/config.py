"""Domain configuration and result models for PeakFit."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from peakfit.engine.domain.constraints import ParameterConfig
from peakfit.engine.domain.fit_steps import FitStep

LineshapeFamily = Literal["auto", "gaussian", "lorentzian", "pvoigt", "sp1", "sp2", "no_apod"]
# Families evaluated as complex signals, whose phase can be fitted.
PHASED_FAMILIES = frozenset({"sp1", "sp2", "no_apod"})
# Bounds of a doublet's J when only its start is given: [J/DOUBLET_RANGE, J*DOUBLET_RANGE].
DOUBLET_RANGE = 2.0
OutputFormat = Literal["csv", "json", "txt"]
_AXIS_LABEL = re.compile(r"F[1-9][0-9]*")
# The J model every former ``*_doublet`` lineshape used, named when refusing one.
OLD_DOUBLET = {"j": 5.0, "min": 1.0, "max": 10.0}


def _around_start(factor: float):
    """The default bound ``factor`` × the validated J start (none if J itself is invalid)."""
    return lambda data: data["j"] * factor if "j" in data else 1.0


class Doublet(BaseModel):
    """A doublet on one spectral axis: two equal components split by a J coupling.

    ``j`` is the start in hertz and has no default: the user states it. Without
    explicit bounds, J may move within [j/2, 2j] (``DOUBLET_RANGE``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    j: float = Field(gt=0, allow_inf_nan=False, description="J start, in Hz.")
    min: float = Field(
        default_factory=_around_start(1 / DOUBLET_RANGE),
        gt=0,
        allow_inf_nan=False,
        description="Lower J bound, in Hz.",
    )
    max: float = Field(
        default_factory=_around_start(DOUBLET_RANGE),
        gt=0,
        allow_inf_nan=False,
        description="Upper J bound, in Hz.",
    )
    vary: bool = Field(default=True, description="Fit J; false holds it at its start.")

    @model_validator(mode="after")
    def _start_within_bounds(self) -> Doublet:
        if not self.min <= self.j <= self.max or self.min >= self.max:
            msg = (
                f"Doublet J bounds [{self.min:g}, {self.max:g}] Hz must contain its start "
                f"{self.j:g} Hz, with min < max."
            )
            raise ValueError(msg)
        return self


class AxisModel(BaseModel):
    """The lineshape model on one spectral axis; unset fields keep the defaults.

    ``family`` unset takes ``[fitting].family``. ``phase`` fits one phase per
    cluster on this axis; it needs a phased family (sp1, sp2, no_apod).
    """

    model_config = ConfigDict(extra="forbid")

    family: LineshapeFamily | None = Field(
        default=None, description="Lineshape family on this axis; unset takes fitting.family."
    )
    doublet: Doublet | None = Field(default=None, description="A J doublet on this axis.")
    phase: bool = Field(default=False, description="Fit the phase on this axis.")


def _replaced_fields(data: dict[str, Any]) -> list[str]:
    """Why a configuration written before per-axis models is refused, with its replacement."""
    reasons = []
    if "lineshape" in data:
        old = str(data["lineshape"])
        family, doublet = old.removesuffix("_doublet"), old.endswith("_doublet")
        reasons.append(
            f"'lineshape' was replaced by 'family' (every axis) and [fitting.axes.<axis>] "
            f'(one axis): write family = "{family}"'
            + (
                ", and on each split axis a doublet with its J start in Hz, for example "
                "doublet = { j = 92.0 } under [fitting.axes.F2]. The former doublets "
                f"started at {OLD_DOUBLET['j']:g} Hz within [{OLD_DOUBLET['min']:g}, "
                f"{OLD_DOUBLET['max']:g}] Hz on every axis"
                if doublet
                else ""
            )
            + "."
        )
    if "fit_phase" in data:
        axes = data["fit_phase"] if isinstance(data["fit_phase"], list) else []
        tables = " and ".join(f"[fitting.axes.{axis}]" for axis in axes)
        reasons.append(
            f"'fit_phase' was replaced by a per-axis phase: write phase = true under {tables}."
            if tables
            else "'fit_phase' was replaced by a per-axis phase: remove it (it fitted none)."
        )
    return reasons


class FitConfig(BaseModel):
    """Configuration for the fitting process.

    Supports the common one-step configuration and optional multi-step fit steps
    with parameter constraints.

    Simple usage:
        [fitting]
        family = "auto"
        refine_iterations = 2

    Per-axis lineshape model (overrides ``family`` on one axis):
        [fitting.axes.F2]
        family = "sp1"
        phase = true
        doublet = { j = 92.0, min = 80.0, max = 100.0, vary = true }

    Advanced multi-step fit:
        [[fitting.steps]]
        name = "fix_positions"
        fix = ["*.*.cs"]
        iterations = 1

        [[fitting.steps]]
        name = "full_optimization"
        vary = ["*"]
        iterations = 2

    Parameter constraints:
        [parameters]
        position_window = 0.1

        [parameters.position_windows]
        F2 = 0.5
        F3 = 0.05

        [parameters.peaks."2N-H"]
        position_window = 0.02
        "F2.cs" = { vary = false }
    """

    model_config = ConfigDict(extra="forbid")

    family: LineshapeFamily = Field(
        default="auto",
        description="Lineshape family of every axis; 'auto' detects it from the apodization.",
    )
    axes: dict[str, AxisModel] = Field(
        default_factory=dict,
        description="Per-axis lineshape models by axis label (F2, F3, ...).",
    )
    refine_iterations: Annotated[int, Field(le=20)] = Field(
        default=1,
        description="Number of optimizer passes; corrections update between passes.",
    )
    max_iterations: Annotated[int, Field(gt=0)] = Field(
        default=1000,
        description="Maximum iterations for optimizer.",
    )
    tolerance: Annotated[float, Field(gt=0)] = Field(
        default=1e-8,
        description="Convergence tolerance for optimizer.",
    )

    # Multi-step fitting steps
    steps: list[FitStep] = Field(
        default_factory=list,
        description="Multi-step fitting steps. If empty, uses refine_iterations.",
    )

    @model_validator(mode="before")
    @classmethod
    def _refuse_replaced_fields(cls, data: Any) -> Any:
        if isinstance(data, dict) and (reasons := _replaced_fields(data)):
            raise ValueError(" ".join(reasons))
        return data

    @field_validator("axes")
    @classmethod
    def _axis_labels(cls, value: dict[str, AxisModel]) -> dict[str, AxisModel]:
        invalid = [label for label in value if not _AXIS_LABEL.fullmatch(label)]
        if invalid:
            msg = f"Axis labels must look like F2, F3: got {', '.join(invalid)}."
            raise ValueError(msg)
        return value

    def axis_model(self, label: str) -> AxisModel:
        """The model configured on axis ``label``, with its family resolved to a name."""
        model = self.axes.get(label, AxisModel())
        return model.model_copy(update={"family": model.family or self.family})

    @field_validator("refine_iterations")
    @classmethod
    def _require_positive_refine_iterations(cls, value: int) -> int:
        if value < 1:
            raise ValueError("refine_iterations must be at least 1")
        return value


class ClusterConfig(BaseModel):
    """Configuration for peak clustering and segmentation.

    Controls contour thresholds and other clustering parameters used during
    segmentation of peaks into clusters prior to fitting.
    """

    model_config = ConfigDict(extra="forbid")

    contour_factor: Annotated[float, Field(gt=0)] = Field(
        default=5.0,
        description="Factor multiplied by noise level for contour threshold.",
    )
    contour_level: float | None = Field(
        default=None,
        description="Explicit contour level (overrides contour_factor if set).",
    )


class OutputConfig(BaseModel):
    """Configuration for output file generation.

    Keep this model limited to fit-output controls that are actually
    implemented by the writer path.
    """

    model_config = ConfigDict(extra="forbid")

    directory: Path = Field(default=Path("Fits"), description="Output directory for results.")
    formats: list[OutputFormat] = Field(
        default=["csv", "txt"],
        description="Presentation formats: CSV and a short report by default; JSON is explicit.",
    )
    detailed_evidence: bool = Field(
        default=False,
        description="Include observations and predictions for portable review/continuation.",
    )
    save_simulated: bool = Field(
        default=False,
        description="Export simulated.npy in float64, with calibration in scientific.npz.",
    )
    include_timestamp: bool = Field(
        default=True,
        description="Include timestamp in output directory name.",
    )
    headless: bool = Field(
        default=False,
        description="Disable interactive/live display (use reporter-only output).",
    )


class PeakFitConfig(BaseModel):
    """Top-level PeakFit configuration for fitting, clustering, and output.

    Example TOML configuration:
        [fitting]
        family = "auto"
        refine_iterations = 2

        [fitting.axes.F2]  # optional per-axis model
        doublet = { j = 92.0 }

        # Optional: multi-step fit
        [[fitting.steps]]
        name = "fix_positions"
        fix = ["*.*.cs"]
        iterations = 1

        [[fitting.steps]]
        name = "full_optimization"
        vary = ["*"]
        iterations = 2

        [clustering]
        contour_factor = 5.0

        [output]
        directory = "Fits"
        formats = ["csv", "txt"]

        # Parameter constraints
        [parameters]
        position_window = 0.1

        [parameters.position_windows]
        F2 = 0.5   # 15N dimension
        F3 = 0.05  # 1H dimension

        [parameters.defaults]
        "*.*.lw" = { min = 5.0, max = 100.0 }

        [parameters.peaks."2N-H"]
        position_window = 0.02
        "F2.cs" = { vary = false }
    """

    model_config = ConfigDict(extra="forbid")

    fitting: FitConfig = Field(default_factory=FitConfig)
    clustering: ClusterConfig = Field(default_factory=ClusterConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    parameters: ParameterConfig = Field(
        default_factory=ParameterConfig,
        description="Parameter constraints and position windows.",
    )
    noise_level: float | None = Field(
        default=None,
        description="Manual noise level. If None, estimated automatically.",
        gt=0,
    )
    exclude_planes: list[int] = Field(
        default_factory=list,
        description="List of plane indices to exclude from fitting.",
    )

    @field_validator("exclude_planes")
    @classmethod
    def validate_exclude_planes(cls, v: list[int]) -> list[int]:
        """Validate list of plane indices to ensure no negative indices are provided."""
        if any(idx < 0 for idx in v):
            msg = "Plane indices must be non-negative"
            raise ValueError(msg)
        return sorted(set(v))


# =============================================================================
# Optimizer Configuration
# =============================================================================


@dataclass(frozen=True)
class VarProConfig:
    """Configuration for Variable Projection optimizer."""

    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    max_nfev: int = 1000
    verbose: int = 0


__all__ = [
    "DOUBLET_RANGE",
    "PHASED_FAMILIES",
    "AxisModel",
    "ClusterConfig",
    "Doublet",
    "FitConfig",
    "FitStep",
    "LineshapeFamily",
    "OutputConfig",
    "OutputFormat",
    "ParameterConfig",
    "PeakFitConfig",
    "VarProConfig",
]
