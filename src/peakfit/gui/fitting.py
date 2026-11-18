"""Prepare complete or isolated fitting problems from committed working peaks."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.clustering import prepare_clusters
from peakfit.engine.algorithms.noise import prepare_noise_level
from peakfit.engine.algorithms.noise_power import check_noise_power
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.domain.constraints import (
    ParameterConstraint,
    PeakConstraints,
    PositionWindowConfig,
    apply_constraints,
)
from peakfit.engine.domain.fit_steps import build_fit_steps
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import axis_lineshapes, get_dimension_label
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.fit.pipeline import prepare_cluster_parameters
from peakfit.gui import formatting
from peakfit.io.readers.spectrum import read_noise_model

if TYPE_CHECKING:
    from collections.abc import Mapping

    from peakfit.engine.algorithms.noise_correlation import NoiseModel
    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.spectrum import AxisLineshape, Spectra
    from peakfit.gui.peaks import PeakLimits, WorkingPeak


_SPECTRAL_DIMENSIONS = 2
# How a continuation records a peak edited by hand since its source run; the run
# comparison reads these prefixes back (``editor.changes.edited_since``).
ADDED_SINCE, MOVED_SINCE = "added since run", "moved since run"
CONSTRAINED_SINCE = "constraints changed since run"


@dataclass(frozen=True, slots=True)
class FitInputSnapshot:
    """Scientific inputs identifying this preparation, independent of navigation.

    Spectrum identity is local to the loaded in-memory dataset. The viewer never
    edits its samples; replacing the dataset creates a new identity. Mutable
    metadata, plane coordinates, configuration, and working peaks are copied.
    ``data_shape`` and ``plane_values`` describe the whole measured spectrum;
    ``source_plane_indices`` are the planes fitted, column for column with the
    amplitudes (every plane but those the configuration excludes). The desktop
    dataset is the whole source spectrum, so its plane i is source plane i: views
    look a plane up here directly, and a reopened project is refused otherwise.
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

    @property
    def fitted_shape(self) -> tuple[int, ...]:
        """The shape of the fitted planes: one per amplitude column."""
        return (len(self.source_plane_indices), *self.data_shape[1:])

    @property
    def fitted_plane_values(self) -> tuple[float, ...]:
        """The value of each fitted plane, column for column with the amplitudes."""
        return tuple(self.plane_values[i] for i in self.source_plane_indices)


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
    configuration: str
    spectral_parameters: tuple[str, ...]
    retained: Mapping[str, str]  # fitted but not adopted by that run, with the reason
    limits: Mapping[str, PeakLimits]  # each peak's own parameter limits in that run


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
class ConstraintProblem:
    """Why one of a peak's own limits cannot be fitted as given, by stable working ID."""

    working_id: str
    key: str  # the parameter, ``F3.lw``
    title: str  # ``LW ¹H F3``
    reason: str  # ``Start 300 Hz is outside its range 5–100 Hz. …``


@dataclass(frozen=True, slots=True)
class PreparedDatasetFit:
    """Reviewed complete fitting problem; no relation to display contour settings."""

    snapshot: FitInputSnapshot
    identities: tuple[PeakIdentity, ...]
    lineshapes: tuple[AxisLineshape, ...]  # each spectral axis's resolved model
    noise: float  # the statistical noise the fit uses: the configured value, else the estimate
    noise_model: NoiseModel  # the processing-derived noise correlation, or why not
    contour_level: float
    clusters: tuple[Cluster, ...]
    parameters: Parameters
    # None only for runs recorded before continuation provenance was retained.
    continuation: Continuation | None = None
    # The noise estimated from the fitted planes, also when a configured value replaces it.
    estimated_noise: float | None = None
    # Peaks' own limits that cannot be fitted as given; any one blocks the fit.
    problems: tuple[ConstraintProblem, ...] = ()

    @property
    def engine_configuration(self) -> PeakFitConfig:
        """The configuration the engine fits: the settings with each peak's own limits."""
        return engine_configuration(
            PeakFitConfig.model_validate_json(self.snapshot.configuration),
            self.snapshot.peaks,
            self.identities,
        )

    def problem_text(self) -> str:
        """The first problem as Fit's reason, naming its peak; "" when there is none."""
        if not self.problems:
            return ""
        problem = self.problems[0]
        name = next(i.assignment for i in self.identities if i.working_id == problem.working_id)
        return f"{name}: {problem.title}: {problem.reason}"


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
    _validate_inputs(spectra, peaks)
    # The fit sees only the planes not excluded, exactly as ``peakfit fit`` does.
    whole, spectra = spectra, spectra.excluding_planes(settings.exclude_planes)
    lineshapes = axis_lineshapes(settings.fitting, spectra)
    identities, scientific_peaks = _working_model(spectra, peaks, lineshapes)
    engine = engine_configuration(settings, peaks, identities)
    # Own bounds reach the clustering too: a doublet's footprint follows its J bound.
    noise, threshold, clusters = prepare_clusters(spectra, scientific_peaks, engine)
    estimated = noise if settings.noise_level is None else _estimate(spectra)
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
    incompatible = _incompatible(source, settings, lineshapes, spectra) if source else ""
    seeding = source if not incompatible else None
    working = {i.engine_name: i.working_id for i in identities}
    by_id = {p.id: p for p in peaks}
    names = {
        get_dimension_label(index): formatting.axis_name(axis)
        for index, axis in enumerate(spectra.spectral_params)
    }
    continued: set[str] = set()
    reasons: dict[str, str] = {}
    problems: list[ConstraintProblem] = []
    for cluster in clusters:
        initial = prepare_cluster_parameters(cluster, engine, steps[0])
        for peak in cluster.peaks:
            problems += limit_problems(by_id[working[peak.name]], peak.name, initial, names)
        if seeding is not None:
            continued |= _continue(initial, cluster, engine, working, by_id, seeding, reasons)
        if problems:
            parameters.update(initial)
            continue  # explained by its problem; the fit is blocked until it is corrected
        if not initial.get_vary_names():
            raise ValueError(f"Cluster {cluster.cluster_id} has no variable nonlinear parameters.")
        for parameter in initial.values():
            if (
                not np.isfinite(parameter.value)
                or not parameter.min <= parameter.value <= parameter.max
            ):
                raise ValueError("The model produced invalid starting values or bounds.")
        parameters.update(initial)
    order = {p.id: index for index, p in enumerate(peaks)}
    return PreparedDatasetFit(
        FitInputSnapshot(
            id(whole),
            tuple(whole.data.shape),
            tuple(axis.model_dump_json() for axis in whole.params),
            tuple(float(value) for value in whole.z_values),
            peaks,
            settings.model_dump_json(),
            float(threshold),
            source_plane_indices=spectra.source_plane_indices,
        ),
        identities,
        tuple(lineshapes),
        noise,
        check_noise_power(read_noise_model(spectra.dic, spectra.data.shape), spectra.data),
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
        estimated,
        tuple(sorted(problems, key=lambda p: (order[p.working_id], p.key))),
    )


def _estimate(spectra: Spectra) -> float | None:
    """The noise the fitted planes would give, shown beside a configured value; None if none."""
    try:
        return prepare_noise_level(None, spectra)
    except ValueError:
        return None


def _incompatible(source, settings, lineshapes, spectra) -> str:
    """Why a run's fitted values cannot seed this model at all, or ""."""
    previous = PeakFitConfig.model_validate_json(source.configuration)
    if source.spectral_parameters != tuple(axis.model_dump_json() for axis in spectra.params):
        return "spectral axes differ from that run"
    if axis_lineshapes(previous.fitting, spectra) != list(lineshapes):
        return "lineshape models differ from that run"
    if previous.parameters != settings.parameters:
        return "parameter constraints differ from that run"
    return ""


def _continue(initial, cluster, settings, working, peaks, source, reasons) -> set[str]:
    """Seed each unchanged peak of one cluster from its fitted values; return its IDs.

    A peak continues only if its working position and its own limits are the ones
    the run adopted and used, and every parameter the configuration lets vary (in any
    fit step) has a fitted value within the new bounds; values are never clamped.
    Parameters the configuration fixes keep their configured values, and so does any
    parameter given an explicit configured start: the pipeline re-applies the
    constraints to the submitted starts, so they are applied here too and the
    recorded starts are the real ones. ``settings`` is the engine configuration.
    """
    configured = apply_constraints(Parameters.from_peaks(cluster.peaks), settings.parameters)
    seeded: list[str] = []
    ids: set[str] = set()
    run = source.run_id[:8]
    for peak in cluster.peaks:
        identity = working[peak.name]
        fitted = source.values.get(identity)
        if identity not in source.positions:
            reasons[identity] = f"{ADDED_SINCE} {run}"
            continue
        if tuple(source.positions[identity]) != tuple(peaks[identity].positions):
            reasons[identity] = f"{MOVED_SINCE} {run}"
            continue
        if source.limits.get(identity, ()) != peaks[identity].limits:
            reasons[identity] = f"{CONSTRAINED_SINCE} {run}"
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


def _working_model(spectra, peaks, lineshapes):
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
            create_shapes(owned, lineshapes, identity.engine_name, peak.positions),
        )
        for peak, identity in zip(peaks, identities, strict=True)
    ]
    return identities, scientific


def peak_constraints(peak: WorkingPeak) -> PeakConstraints | None:
    """A working peak's own limits as the engine's per-peak constraints; None if it has none.

    A window on a position is that axis's per-peak position window; the rest are the
    parameter's start, bounds and vary, as ``[parameters.peaks."<name>"]`` gives them.
    """
    windows = {key.partition(".")[0]: limit.window for key, limit in peak.limits if limit.window}
    parameters = {
        key: ParameterConstraint(
            value=limit.start, min=limit.lower, max=limit.upper, vary=limit.vary
        )
        for key, limit in peak.limits
        if (limit.start, limit.lower, limit.upper, limit.vary) != (None, None, None, None)
    }
    if not windows and not parameters:
        return None
    return PeakConstraints(position_windows=PositionWindowConfig(**windows), parameters=parameters)


def engine_configuration(
    settings: PeakFitConfig,
    peaks: tuple[WorkingPeak, ...],
    identities: tuple[PeakIdentity, ...],
) -> PeakFitConfig:
    """The working settings with each peak's own limits under its engine name.

    The working settings never name peaks: own limits live on the working peak, by
    stable ID, so they follow it through renames and two peaks sharing an
    assignment keep their own. The engine sees them as ``peakfit fit`` sees
    ``[parameters.peaks."<assignment>"]``, keyed by the safe name the desktop gives
    the peak in place of its assignment.
    """
    engine_names = {i.working_id: i.engine_name for i in identities}
    own = {engine_names[p.id]: c for p in peaks if (c := peak_constraints(p)) is not None}
    if not own:
        return settings
    return settings.model_copy(
        update={"parameters": settings.parameters.model_copy(update={"peaks": own})}
    )


def limit_problems(
    peak: WorkingPeak, engine_name: str, initial: Parameters, names: Mapping[str, str]
) -> list[ConstraintProblem]:
    """Why each of a peak's own limits cannot be fitted as given, from its resolved values.

    A limit must name a parameter of the current model, leave a non-empty range and,
    when it sets a start, keep it inside the range the settings and the limit give
    together: ``peakfit fit`` would move it to the nearest bound without saying so.
    """
    problems = []
    for key, limit in peak.limits:
        axis, _, kind = key.partition(".")
        title = parameter_title(kind, names.get(axis) or axis)
        name = f"{engine_name}.{key}"
        reason = ""
        if name not in initial:
            reason = "The current model has no such parameter. Reset it, or change the lineshape."
        else:
            parameter = initial[name]
            unit = f" {parameter.unit}" if parameter.unit else ""
            show = formatting.setting
            span = f"{show(parameter.min)}–{show(parameter.max)}{unit}"
            if not parameter.min < parameter.max:
                reason = f"The range {span} is empty. Widen it."
            elif limit.start is not None and parameter.value != limit.start:
                reason = (
                    f"Start {show(limit.start)}{unit} is outside its range {span}. "
                    "Change the start or the range."
                )
        if reason:
            problems.append(ConstraintProblem(peak.id, key, title, reason))
    return problems


# How a parameter kind reads in tables and messages, before its axis (``LW ¹H F3``).
PARAMETER_KINDS = {"cs": "δ", "lw": "LW", "eta": "η", "j": "J", "phase": "φ"}


def parameter_title(kind: str, axis: str) -> str:
    """``LW ¹H F3``: a parameter kind on an axis, as the inspector names it."""
    return f"{PARAMETER_KINDS.get(kind, kind)} {axis}"


def desktop_rule(pattern: str) -> bool:
    """Whether the desktop fit applies a rule: ``*`` or ``*.<axis>.<parameter>``.

    Peaks take safe IDs in the desktop fit, so a rule naming a peak would match none.
    """
    return pattern == "*" or re.fullmatch(r"\*\.[^.]+\.[^.]+", pattern) is not None


def _validate_constraint_targets(settings: PeakFitConfig) -> None:
    """Reject assignment-dependent rules before replacing names with safe IDs.

    Only whole-model '*' or '*.<axis>.<parameter>' patterns are supported here, also
    in fit-step rules. A peak's own constraints are its working peak's limits, never
    ``parameters.peaks`` of the working settings.
    """
    if settings.parameters.peaks:
        raise ValueError(
            "The working settings name peaks. A peak's own constraints belong to the "
            "working peak: set them on its Parameters."
        )
    patterns = list(settings.parameters.defaults.patterns)
    for step in settings.fitting.steps:
        patterns.extend((*step.fix, *step.vary))
    if not all(map(desktop_rule, patterns)):
        raise ValueError(
            "Rules naming peaks are not supported in the desktop dataset fit. Use rules "
            "on every peak ('*.*.lw', '*.F2.cs') and set one peak's own constraints on "
            "its Parameters."
        )


def _validate_inputs(spectra: Spectra, peaks: tuple[WorkingPeak, ...]) -> None:
    """Reject inputs whose indexing would misrepresent the dataset fit."""
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
