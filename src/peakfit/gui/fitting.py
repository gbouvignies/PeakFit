"""Prepare complete or isolated fitting problems from committed working peaks."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.clustering import prepare_clusters
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.domain.constraints import apply_constraints
from peakfit.engine.domain.fit_steps import build_fit_steps
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import get_shape_names
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.fit.pipeline import prepare_cluster_parameters

if TYPE_CHECKING:
    from collections.abc import Mapping

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
class ContinuationSource:
    """A completed run's fitted nonlinear state, keyed by stable working ID.

    ``positions`` are that run's post-fit working positions (adopted or retained);
    ``values`` hold every fitted nonlinear value by parameter suffix (``F2.cs``,
    ``F3.lw`` …), only for peaks whose fitted positions were adopted.
    """

    run_id: str
    positions: Mapping[str, tuple[float, ...]]
    values: Mapping[str, Mapping[str, float]]
    shape_names: tuple[str, ...]
    configuration: str
    spectral_parameters: tuple[str, ...]
    retained: Mapping[str, str]  # fitted but not adopted by that run, with the reason


@dataclass(frozen=True, slots=True)
class Continuation:
    """Where the submitted starting values came from, by stable working ID.

    ``continued`` peaks start from run ``run_id``'s fitted nonlinear state; the
    ``initialized`` ones from the configured defaults. ``reason`` says why no peak
    could continue when a source run was present but incompatible; ``reasons``
    says why each other initialized peak did not (added, moved, not adopted …).
    """

    run_id: str | None
    continued: tuple[str, ...]
    initialized: tuple[str, ...]
    reason: str = ""
    reasons: tuple[tuple[str, str], ...] = ()


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
    # None only for runs recorded before continuation provenance was retained.
    continuation: Continuation | None = None


def prepare_dataset_fit(
    spectra: Spectra,
    peaks: tuple[WorkingPeak, ...],
    config: PeakFitConfig | None = None,
    source: ContinuationSource | None = None,
) -> PreparedDatasetFit:
    """Resolve the regular whole-dataset problem from committed working records.

    With a ``source`` run, every peak whose working position is still the one that
    run adopted continues from its complete fitted nonlinear state; new and edited
    peaks start from the configured defaults. The submitted ``parameters`` record
    the exact starts, and ``continuation`` records where they came from.
    """
    if not peaks:
        raise ValueError("Load or add peaks before fitting the dataset.")
    settings = (
        config.model_copy(deep=True)
        if config is not None
        else PeakFitConfig(fitting=FitConfig(refine_iterations=2))
    )
    _validate_constraint_targets(settings)
    _validate_inputs(spectra, peaks, settings)
    shape_names = get_shape_names(settings.fitting, spectra)
    identities, scientific_peaks = _working_model(spectra, peaks, settings, shape_names)
    noise, threshold, clusters = prepare_clusters(spectra, scientific_peaks, settings)
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
    incompatible = _incompatible(source, settings, shape_names, spectra) if source else ""
    seeding = source if not incompatible else None
    working = {i.engine_name: i.working_id for i in identities}
    positions = {p.id: tuple(p.positions) for p in peaks}
    continued: set[str] = set()
    reasons: dict[str, str] = {}
    for cluster in clusters:
        initial = prepare_cluster_parameters(cluster, settings, steps[0])
        if seeding is not None:
            continued |= _continue(initial, cluster, settings, working, positions, seeding, reasons)
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
        Continuation(
            source.run_id if source else None,
            tuple(p.id for p in peaks if p.id in continued),
            tuple(p.id for p in peaks if p.id not in continued),
            incompatible,
            tuple((p.id, reasons[p.id]) for p in peaks if p.id in reasons),
        ),
    )


def _incompatible(source, settings, shape_names, spectra) -> str:
    """Why a run's fitted values cannot seed this model at all, or ""."""
    previous = PeakFitConfig.model_validate_json(source.configuration)
    if tuple(source.shape_names) != tuple(shape_names):
        return "lineshape models differ from that run"
    if previous.parameters != settings.parameters:
        return "parameter constraints differ from that run"
    if source.spectral_parameters != tuple(axis.model_dump_json() for axis in spectra.params):
        return "spectral axes differ from that run"
    return ""


def _continue(initial, cluster, settings, working, positions, source, reasons) -> set[str]:
    """Seed each unchanged peak of one cluster from its fitted values; return its IDs.

    A peak continues only if its working position is the one the run adopted and
    every parameter the configuration lets vary (in any fit step) has a fitted value
    within the new bounds; values are never clamped. Parameters the configuration
    fixes keep their configured values, and so does any parameter given an explicit
    configured start: the pipeline re-applies the constraints to the submitted
    starts, so they are applied here too and the recorded starts are the real ones.
    """
    configured = apply_constraints(Parameters.from_peaks(cluster.peaks), settings.parameters)
    seeded: list[str] = []
    ids: set[str] = set()
    run = source.run_id[:8]
    for peak in cluster.peaks:
        identity = working[peak.name]
        fitted = source.values.get(identity)
        if identity not in source.positions:
            reasons[identity] = f"added since run {run}"
            continue
        if tuple(source.positions[identity]) != positions[identity]:
            reasons[identity] = f"moved since run {run}"
            continue
        if identity in source.retained or fitted is None:
            why = source.retained.get(identity, "no fitted values")
            reasons[identity] = f"not adopted by run {run}: {why}"
            continue
        names = [n for n, p in configured.items() if p.vary and n.partition(".")[0] == peak.name]
        seeds = {n: fitted.get(n.partition(".")[2]) for n in names}
        if not names or any(
            v is None or not initial[n].min <= v <= initial[n].max for n, v in seeds.items()
        ):
            reasons[identity] = f"fitted values of run {run} outside the current bounds"
            continue
        for name, value in seeds.items():
            initial[name].value = value
        seeded.extend(seeds)
        ids.add(identity)
    resolved = apply_constraints(initial.copy(deep=True), settings.parameters)
    for name in seeded:
        initial[name].value = resolved[name].value
    return ids


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
    Apply the same restriction to fit-step rules.
    """
    patterns = list(settings.parameters.defaults.patterns)
    for step in settings.fitting.steps:
        patterns.extend((*step.fix, *step.vary))
    if settings.parameters.peaks or any(
        pattern != "*" and re.fullmatch(r"\*\.[^.]+\.[^.]+", pattern) is None
        for pattern in patterns
    ):
        raise ValueError(
            "Name/assignment-targeted constraints are not supported in the desktop dataset "
            "fit. Remove parameters.peaks and "
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
