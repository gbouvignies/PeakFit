"""Prepare an isolated all-plane fit from immutable editable peak records."""

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.clustering import segment_data
from peakfit.engine.algorithms.noise import prepare_noise_level
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.fit_steps import build_fit_steps
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import get_shape_names
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.fit.pipeline import prepare_cluster_parameters

if TYPE_CHECKING:
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.gui.peaks import WorkingPeak
    from peakfit.shared.typing import IntArray


_SPECTRAL_DIMENSIONS = 2
# The preview is not a dataset-wide fit. Bound its two dense float64 point/plane
# arrays before allocation (a low-threshold bundled group alone needs 279 MiB).
LOCAL_ARRAY_BUDGET = 128 * 1024**2


@dataclass(frozen=True, slots=True)
class FitInputSnapshot:
    """Scientific inputs identifying this preparation, independent of navigation.

    Spectrum identity is local to the loaded in-memory dataset. The viewer never
    edits its samples; replacing the dataset creates a new identity. Mutable
    metadata, plane coordinates, configuration, and working peaks are copied.
    """

    spectrum_id: int
    data_shape: tuple[int, ...]
    spectral_parameters: tuple[str, ...]
    plane_values: tuple[float, ...]
    peaks: tuple[WorkingPeak, ...]
    configuration: str
    contour_level: float
    source_files: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class PeakIdentity:
    """One safe scientific identifier and its original editable assignment."""

    engine_name: str
    working_id: str
    assignment: str


@dataclass(frozen=True, slots=True)
class LocalFitScope:
    """Exact selected segmentation and members without dense numerical data."""

    cluster_id: int
    peaks: tuple[Peak, ...]
    grid_indices: tuple[IntArray, ...]
    n_series: int

    @property
    def n_points(self) -> int:
        """Number of sampled spectral points in the unmodified segmentation."""
        return len(self.grid_indices[0])


@dataclass(frozen=True, slots=True)
class PreparedLocalFit:
    """Exact scope and optional numerical inputs, never a completed fit or peak edit."""

    snapshot: FitInputSnapshot
    identities: tuple[PeakIdentity, ...]
    shape_names: tuple[str, ...]
    noise: float
    contour_level: float
    scope: LocalFitScope
    cluster: Cluster | None
    parameters: Parameters | None
    limitation: str | None = None


def prepare_local_fit(
    spectra: Spectra,
    peaks: tuple[WorkingPeak, ...],
    selected_id: str,
    config: PeakFitConfig | None = None,
    *,
    contour_level: float,
) -> PreparedLocalFit:
    """Use the regular scientific path to prepare one selected all-plane cluster.

    All working peaks participate in segmentation. The returned region uses the
    absolute lowest displayed contour intensity supplied by the viewer. No
    optimizer, auto-picking, correction updates, or file bridge is involved.
    """
    settings = config.model_copy(deep=True) if config is not None else PeakFitConfig()
    if not np.isfinite(contour_level) or contour_level <= 0:
        raise ValueError("The clustering threshold must be positive and finite.")
    settings.clustering.contour_level = float(contour_level)
    _validate_constraint_targets(settings)
    _validate_inputs(spectra, peaks, selected_id, settings)
    steps = build_fit_steps(settings.fitting.steps, settings.fitting.refine_iterations)
    if len(steps) != 1 or steps[0].iterations != 1:
        raise ValueError("Local preview supports one isolated optimizer pass, without refinement.")
    snapshot = FitInputSnapshot(
        spectrum_id=id(spectra),
        data_shape=tuple(spectra.data.shape),
        spectral_parameters=tuple(axis.model_dump_json() for axis in spectra.params),
        plane_values=tuple(float(value) for value in spectra.z_values),
        peaks=tuple(peaks),
        configuration=settings.model_dump_json(),
        contour_level=float(contour_level),
    )
    # Fresh metadata prevents edits to plane labels or axes changing this model.
    scientific_spectra = spectra.model_copy(
        update={
            "params": [axis.model_copy(deep=True) for axis in spectra.params],
            "z_values": spectra.z_values.copy(),
        }
    )
    noise = float(prepare_noise_level(settings.noise_level, scientific_spectra))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("Cannot estimate positive finite noise for this spectrum.")
    shape_names = get_shape_names(settings.fitting, scientific_spectra)
    # Enumeration is safe even for unusual imported IDs; the explicit mapping,
    # never names or list order alone, relates scientific and working identities.
    identities = tuple(
        PeakIdentity(f"p_{index}", peak.id, peak.name) for index, peak in enumerate(peaks)
    )
    scientific_peaks = [
        Peak(
            identity.engine_name,
            np.asarray(peak.positions),
            create_shapes(
                scientific_spectra,
                settings.fitting,
                identity.engine_name,
                peak.positions,
                shape_names,
            ),
        )
        for peak, identity in zip(peaks, identities, strict=True)
    ]
    segments = segment_data(scientific_spectra.data, contour_level, scientific_peaks)
    selected_index = next(i for i, item in enumerate(identities) if item.working_id == selected_id)
    segment_id = int(segments[tuple(scientific_peaks[selected_index].positions_i)])
    members = [
        peak for peak in scientific_peaks if int(segments[tuple(peak.positions_i)]) == segment_id
    ]
    n_points = int(np.count_nonzero(segments == segment_id))
    for peak in members:
        peak.set_cluster_id(segment_id)
    grid_indices = tuple(np.where(segments == segment_id))
    for indices in grid_indices:
        indices.flags.writeable = False
    scope = LocalFitScope(segment_id, tuple(members), grid_indices, spectra.data.shape[0])
    member_names = {peak.name for peak in members}
    result = PreparedLocalFit(
        snapshot,
        tuple(item for item in identities if item.engine_name in member_names),
        tuple(shape_names),
        noise,
        float(contour_level),
        scope,
        None,
        None,
    )
    array_bytes = n_points * scope.n_series * np.dtype(float).itemsize * 2
    if array_bytes > LOCAL_ARRAY_BUDGET:
        return replace(
            result,
            limitation=(
                f"This scope contains {len(members)} peaks and {n_points:,} samples across "
                f"{scope.n_series} planes. Numerical data/correction arrays would require "
                f"{array_bytes / 1024**2:.0f} MiB "
                f"(interactive preparation budget {LOCAL_ARRAY_BUDGET / 1024**2:.0f} MiB), "
                "before model allocations. The exact mask and all members remain visible. "
                "Use a higher contour threshold or the CLI for numerical preparation. "
                "No samples have been removed."
            ),
        )
    # Materialize only the selected group; preserve the engine's exact point order.
    cluster = Cluster(
        cluster_id=segment_id,
        peaks=members,
        grid_indices=list(grid_indices),
        data=np.asarray(scientific_spectra.data[(slice(None), *grid_indices)].T, dtype=float),
    )
    parameters = prepare_cluster_parameters(cluster, settings, steps[0])
    if any(
        not np.isfinite(parameter.value) or not parameter.min <= parameter.value <= parameter.max
        for parameter in parameters.values()
    ):
        raise ValueError("The model produced invalid starting values or bounds.")
    return replace(result, cluster=cluster, parameters=parameters)


def _validate_constraint_targets(settings: PeakFitConfig) -> None:
    """Reject assignment-dependent rules before replacing names with safe IDs.

    Only whole-model '*' or '*.<axis>.<parameter>' patterns are supported here.
    Apply the same restriction to fit-step rules; previous-fit files also contain
    name-based identities and cannot safely be reused without explicit mapping.
    """
    patterns = list(settings.parameters.defaults.patterns)
    for step in settings.fitting.steps:
        patterns.extend((*step.fix, *step.vary))
    if (
        settings.parameters.peaks
        or settings.parameters.from_file is not None
        or any(
            pattern != "*" and re.fullmatch(r"\*\.[^.]+\.[^.]+", pattern) is None
            for pattern in patterns
        )
    ):
        raise ValueError(
            "Name/assignment-targeted constraints and previous-fit parameter files are not "
            "supported in local preview. Remove parameters.peaks/from_file and use global "
            "or axis/parameter-wide wildcard rules (e.g. '*.*.lw', '*.F2.cs'); use the CLI "
            "for assignment-specific constraints."
        )


def _validate_inputs(
    spectra: Spectra, peaks: tuple[WorkingPeak, ...], selected_id: str, settings: PeakFitConfig
) -> None:
    """Reject inputs whose indexing or configuration would misrepresent the local fit."""
    if settings.exclude_planes:
        raise ValueError(
            "Local preview currently includes every plane; exclusions are unsupported."
        )
    if settings.fitting.fix_positions or settings.fitting.fit_j_coupling:
        raise ValueError(
            "Legacy fix_positions/fit_j_coupling flags are not supported by preparation. "
            "Use explicit fit-step rules or explicit doublet models instead."
        )
    if selected_id not in {peak.id for peak in peaks}:
        raise ValueError("Select an existing working peak before preparing a fitting group.")
    if len({peak.id for peak in peaks}) != len(peaks):
        raise ValueError("Working peaks must have unique internal identities.")
    if (
        spectra.n_spectral_dims != _SPECTRAL_DIMENSIONS
        or spectra.data.ndim != _SPECTRAL_DIMENSIONS + 1
    ):
        raise ValueError("Local preview supports 2D and pseudo-3D spectra with two spectral axes.")
    if spectra.z_values.shape != (spectra.data.shape[0],) or not np.all(
        np.isfinite(spectra.z_values)
    ):
        raise ValueError("Every experimental plane must have a finite plane coordinate.")
    # Index conversion wraps every axis for segmentation, whereas model evaluation
    # wraps only indirect axes. Reject invalid direct coordinates before indexing.
    for peak in peaks:
        for ppm, axis in zip(peak.positions, spectra.spectral_params, strict=True):
            point = float(axis.ppm2pts(ppm))
            if axis.direct and not 0 <= point <= axis.size - 1:
                raise ValueError(
                    f"Peak {peak.name!r}: direct-axis position is outside sampled coordinates. "
                    "Move or remove it before preparing a fit; direct coordinates do not wrap."
                )
