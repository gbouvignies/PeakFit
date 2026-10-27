"""Prepare an isolated all-plane fit from immutable editable peak records."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.clustering import create_clusters
from peakfit.engine.algorithms.noise import prepare_noise_level
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.fit_steps import build_fit_steps
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import get_shape_names
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.fit.pipeline import prepare_cluster_parameters

if TYPE_CHECKING:
    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.gui.peaks import WorkingPeak


_SPECTRAL_DIMENSIONS = 2


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


@dataclass(frozen=True, slots=True)
class PeakIdentity:
    """One safe scientific identifier and its original editable assignment."""

    engine_name: str
    working_id: str
    assignment: str


@dataclass(frozen=True, slots=True)
class PreparedLocalFit:
    """A prepared cluster, not a fit result or an edit of working coordinates."""

    snapshot: FitInputSnapshot
    identities: tuple[PeakIdentity, ...]
    shape_names: tuple[str, ...]
    noise: float
    contour_level: float
    cluster: Cluster
    parameters: Parameters


def prepare_local_fit(
    spectra: Spectra,
    peaks: tuple[WorkingPeak, ...],
    selected_id: str,
    config: PeakFitConfig | None = None,
) -> PreparedLocalFit:
    """Use the regular scientific path to prepare one selected all-plane cluster.

    All working peaks participate in segmentation. The returned region uses the
    scientific clustering threshold, never the viewer's display contours. No
    optimizer, auto-picking, correction updates, or file bridge is involved.
    """
    settings = config.model_copy(deep=True) if config is not None else PeakFitConfig()
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
    contour = settings.clustering.contour_level
    if contour is None:
        contour = settings.clustering.contour_factor * noise
    if not np.isfinite(contour) or contour <= 0:
        raise ValueError("The scientific clustering threshold must be positive and finite.")
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
    clusters = create_clusters(scientific_spectra, scientific_peaks, contour)
    selected_name = next(item.engine_name for item in identities if item.working_id == selected_id)
    cluster = next(
        (group for group in clusters if any(peak.name == selected_name for peak in group.peaks)),
        None,
    )
    if cluster is None:
        raise ValueError("No sampled fitting region contains the selected peak.")
    parameters = prepare_cluster_parameters(cluster, settings, steps[0])
    if any(
        not np.isfinite(parameter.value) or not parameter.min <= parameter.value <= parameter.max
        for parameter in parameters.values()
    ):
        raise ValueError("The model produced invalid starting values or bounds.")
    members = {peak.name for peak in cluster.peaks}
    return PreparedLocalFit(
        snapshot,
        tuple(item for item in identities if item.engine_name in members),
        tuple(shape_names),
        noise,
        float(contour),
        cluster,
        parameters,
    )


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
