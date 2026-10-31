"""Prepare complete or isolated fitting problems from committed working peaks."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.clustering import create_clusters
from peakfit.engine.algorithms.noise import prepare_noise_level
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.domain.fit_steps import build_fit_steps
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import get_shape_names
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.fit.pipeline import prepare_cluster_parameters

if TYPE_CHECKING:
    from peakfit.engine.domain.cluster import Cluster
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
    contour_level: float
    source_files: tuple[tuple[str, str], ...] = ()
    source_plane_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PeakIdentity:
    """One safe scientific identifier and its original editable assignment."""

    engine_name: str
    working_id: str
    assignment: str


@dataclass(frozen=True, slots=True)
class PreparedDatasetFit:
    """Reviewed complete fitting problem; no relation to display contour settings."""

    snapshot: FitInputSnapshot
    identities: tuple[PeakIdentity, ...]
    shape_names: tuple[str, ...]
    noise: float
    contour_level: float
    clusters: tuple[Cluster, ...]
    parameters: Parameters


def prepare_dataset_fit(
    spectra: Spectra,
    peaks: tuple[WorkingPeak, ...],
    config: PeakFitConfig | None = None,
) -> PreparedDatasetFit:
    """Resolve the regular whole-dataset problem from committed working records."""
    if not peaks:
        raise ValueError("Load or add peaks before fitting the dataset.")
    settings = (
        config.model_copy(deep=True)
        if config is not None
        else PeakFitConfig(fitting=FitConfig(refine_iterations=2))
    )
    _validate_constraint_targets(settings)
    _validate_inputs(spectra, peaks, settings)
    noise = float(prepare_noise_level(settings.noise_level, spectra))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("Cannot estimate positive finite noise for this spectrum.")
    threshold = settings.clustering.contour_level
    if threshold is None:
        threshold = settings.clustering.contour_factor * noise
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("The clustering threshold must be positive and finite.")
    shape_names = get_shape_names(settings.fitting, spectra)
    identities, scientific_peaks = _working_model(spectra, peaks, settings, shape_names)
    clusters = create_clusters(spectra, scientific_peaks, threshold)
    included = [peak.name for cluster in clusters for peak in cluster.peaks]
    if len(included) != len(identities) or set(included) != {i.engine_name for i in identities}:
        missing = [i.assignment for i in identities if i.engine_name not in included]
        raise ValueError(
            "Every working peak must be included exactly once. Review positions "
            "and clustering threshold. Missing: " + ", ".join(missing)
        )
    steps = build_fit_steps(settings.fitting.steps, settings.fitting.refine_iterations)
    if not steps or sum(step.iterations for step in steps) < 1:
        raise ValueError("Fitting requires at least one optimizer pass.")
    parameters = Parameters()
    for cluster in clusters:
        initial = prepare_cluster_parameters(cluster, settings, steps[0])
        if not initial.get_vary_names():
            raise ValueError(f"Cluster {cluster.cluster_id} has no variable nonlinear parameters.")
        for parameter in initial.values():
            if (
                not np.isfinite(parameter.value)
                or not parameter.min <= parameter.value <= parameter.max
            ):
                raise ValueError("The model produced invalid starting values or bounds.")
        parameters.update(initial)
    return PreparedDatasetFit(
        FitInputSnapshot(
            id(spectra),
            tuple(spectra.data.shape),
            tuple(axis.model_dump_json() for axis in spectra.params),
            tuple(float(value) for value in spectra.z_values),
            peaks,
            settings.model_dump_json(),
            float(threshold),
            source_plane_indices=spectra.source_plane_indices,
        ),
        identities,
        tuple(shape_names),
        noise,
        float(threshold),
        tuple(clusters),
        parameters,
    )


def _working_model(spectra, peaks, settings, shape_names):
    """Own model metadata and map arbitrary/duplicate assignments to safe scientific IDs."""
    owned = spectra.model_copy(
        update={
            "params": [axis.model_copy(deep=True) for axis in spectra.params],
            "z_values": spectra.z_values.copy(),
        }
    )
    identities = tuple(PeakIdentity(f"p_{i}", peak.id, peak.name) for i, peak in enumerate(peaks))
    scientific = [
        Peak(
            identity.engine_name,
            np.asarray(peak.positions),
            create_shapes(
                owned,
                settings.fitting,
                identity.engine_name,
                peak.positions,
                shape_names,
            ),
        )
        for peak, identity in zip(peaks, identities, strict=True)
    ]
    return identities, scientific


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
            "supported in the desktop dataset fit. Remove parameters.peaks/from_file and "
            "use global or axis/parameter-wide wildcard rules (e.g. '*.*.lw', '*.F2.cs'); "
            "use the CLI for assignment-specific constraints."
        )


def _validate_inputs(
    spectra: Spectra, peaks: tuple[WorkingPeak, ...], settings: PeakFitConfig
) -> None:
    """Reject inputs whose indexing or configuration would misrepresent the dataset fit."""
    if settings.exclude_planes:
        raise ValueError(
            "The desktop dataset fit includes every plane; exclusions are unsupported."
        )
    if settings.fitting.fix_positions or settings.fitting.fit_j_coupling:
        raise ValueError(
            "Legacy fix_positions/fit_j_coupling flags are not supported by preparation. "
            "Use explicit fit-step rules or explicit doublet models instead."
        )
    if len({peak.id for peak in peaks}) != len(peaks):
        raise ValueError("Working peaks must have unique internal identities.")
    if (
        spectra.n_spectral_dims != _SPECTRAL_DIMENSIONS
        or spectra.data.ndim != _SPECTRAL_DIMENSIONS + 1
    ):
        raise ValueError(
            "The desktop dataset fit supports 2D and pseudo-3D spectra with two spectral axes."
        )
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
