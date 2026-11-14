"""One immutable record per MCMC run, bound to the fit it sampled.

See ``docs/mcmc-formulation.md`` §8. A run is written once, atomically, to
``<fit run>/mcmc/<record id>.npz``: ``metadata.json`` and NPY members, like
``scientific.npz``. It holds each cluster's whole chain (burn-in included),
log-probabilities and identifiability blobs losslessly, with the complete diagnosis
and amplitude summaries, so the results read back without the source spectrum. Record
IDs sort chronologically.
"""

import hashlib
import os
import secrets
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zipfile import ZipFile

import numpy as np

from peakfit.engine.scientific.amplitudes import (
    AmplitudeSummaries,
    AmplitudeSummary,
    Correlation,
)
from peakfit.engine.scientific.completed import FinalClusterSamples
from peakfit.engine.scientific.sampler import (
    Diagnosis,
    EnsembleRun,
    Exchange,
    ParameterSummary,
    PosteriorSample,
    SamplerSettings,
)
from peakfit.io.output_metadata import __version__
from peakfit.io.scientific import (
    MALFORMED,
    decode_document,
    encode_value,
    read_metadata,
    read_scientific,
    unreadable,
    write_archive,
)

if TYPE_CHECKING:
    from collections.abc import Callable

KIND = "peakfit_mcmc_record"
VERSION = "1.1.0"
DIRECTORY = "mcmc"


@dataclass(frozen=True)
class ParentFit:
    """The fit a run sampled, identified by content rather than location."""

    run_id: str
    sha256: str  # of its scientific.npz
    spectrum_sha256: str | None  # the verified source; None: observations retained
    correction_revision: int  # terminal
    noise: float


@dataclass(frozen=True)
class MCMCRecord:
    """Everything one ``peakfit mcmc`` run produced and what reproduces it (§8)."""

    created: str  # ISO 8601, UTC
    parent: ParentFit
    formulation: str  # of docs/mcmc-formulation.md
    entropy: int  # cluster i's run uses SeedSequence(entropy, spawn_key=(i,))
    streams: dict[str, str]  # bit generator of each spawned stream
    amplitude_samples: int  # at most this many kept θ samples, balanced across walkers
    software: dict[str, str]  # PeakFit, NumPy and emcee versions
    posteriors: tuple[PosteriorSample, ...]


def parent_fit(archive: Path) -> ParentFit:
    """The identity of a results directory's ``scientific.npz``."""
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    outcome, samples, context = read_scientific(archive)
    info = context["metadata"]
    spectrum = None
    if not all(isinstance(s, FinalClusterSamples) for s in samples):
        spectrum = info["input_files"].get("spectrum", {}).get("checksum_sha256")
        if spectrum is None:
            raise FileNotFoundError(
                f"{archive} neither retains its observations nor records a verified "
                "spectrum; refit and export detailed evidence."
            )
    return ParentFit(
        run_id=info["run_id"],
        sha256=digest,
        spectrum_sha256=spectrum,
        correction_revision=outcome.terminal_correction_revision,
        noise=outcome.noise,
    )


def software_versions() -> dict[str, str]:
    """The versions regeneration is bit-identical under (§8)."""
    return {
        "peakfit": __version__,
        "numpy": np.__version__,
        "emcee": metadata.version("emcee"),
    }


def write_record(results_dir: Path, record: MCMCRecord) -> Path:
    """Publish ``record`` atomically under ``results_dir/mcmc``; never overwrite one.

    The parent ``scientific.npz`` must still be the one the run sampled.
    """
    archive = Path(results_dir) / "scientific.npz"
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != record.parent.sha256:
            raise ValueError(f"{archive} changed while MCMC was sampling it; rerun peakfit mcmc.")
    directory = Path(results_dir) / DIRECTORY
    directory.mkdir(exist_ok=True)
    stamp = datetime.fromisoformat(record.created).astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{stamp}-{secrets.token_hex(4)}.npz"
    arrays: dict[str, np.ndarray] = {}
    document = {
        "schema_version": VERSION,
        "kind": KIND,
        "record": encode_value(record, "record", arrays),
    }
    partial = directory / f".{path.name}.{secrets.token_hex(8)}.partial"
    try:
        write_archive(partial, document, arrays)
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        # A hard link publishes the complete file and fails if the name exists, in one
        # atomic step; a rename would silently replace an existing record.
        try:
            os.link(partial, path)
        except FileExistsError:
            raise FileExistsError(
                f"MCMC record {path} already exists; records are never overwritten."
            ) from None
    finally:
        partial.unlink(missing_ok=True)
    if os.name != "nt":
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return path


def latest_record(results_dir: Path) -> Path | None:
    """The most recent record of a results directory, or None."""
    records = sorted((Path(results_dir) / DIRECTORY).glob("*.npz"))
    return records[-1] if records else None


def read_record(path: Path) -> MCMCRecord:
    """Read and validate one record; a damaged or other-version file is refused."""
    try:
        with ZipFile(path) as archive:
            document = read_metadata(archive)
            if document.get("schema_version") != VERSION or document.get("kind") != KIND:
                raise ValueError(
                    f"Unsupported MCMC record version {document.get('schema_version')!r} "
                    f"(this PeakFit reads {VERSION})"
                )
            record = _record(decode_document(archive, document)["record"])
        _validate(record)
    except MALFORMED as error:
        raise unreadable(
            path,
            error,
            "MCMC record",
            "Rerun peakfit mcmc, or restore an intact copy of the file.",
        ) from error
    return record


# Strict decoding: every field present, nothing extra, each value of its recorded type.


def _build(cls: type, value: Any, **convert: Callable[[Any], Any]) -> Any:
    if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
        raise ValueError(f"Malformed {cls.__name__} record")
    return cls(**{name: convert[name](item) for name, item in value.items()})


def _typed(kind: type) -> Callable[[Any], Any]:
    def check(value: Any) -> Any:
        if type(value) is not kind:
            raise TypeError(f"expected {kind.__name__}, got {type(value).__name__}")
        return value

    return check


_str, _int, _float, _bool = _typed(str), _typed(int), _typed(float), _typed(bool)


def _optional(convert: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return lambda value: None if value is None else convert(value)


def _tuple(convert: Callable[[Any], Any]) -> Callable[[Any], tuple]:
    def build(value: Any) -> tuple:
        if not isinstance(value, list):
            raise TypeError(f"expected a list, got {type(value).__name__}")
        return tuple(convert(v) for v in value)

    return build


def _mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping, got {type(value).__name__}")
    return {_str(k): _str(v) for k, v in value.items()}


def _array(dtype: str, ndim: int) -> Callable[[Any], np.ndarray]:
    def check(value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray) or value.dtype != dtype or value.ndim != ndim:
            raise ValueError(f"Malformed {dtype} array of {ndim} dimensions")
        return value

    return check


def _blobs(value: Any) -> np.ndarray:
    # emcee squeezes a one-peak cluster's blobs to (steps, walkers).
    if not isinstance(value, np.ndarray) or value.dtype != bool or value.ndim not in {2, 3}:
        raise ValueError("Malformed identifiability blobs")
    return value


def _entropy(value: Any) -> int | tuple[int, ...]:
    return _tuple(_int)(value) if isinstance(value, list) else _int(value)


def _withheld(value: Any) -> tuple[str, float]:
    peak, fraction = _tuple(lambda x: x)(value)
    return _str(peak), _float(fraction)


def _amplitude(value: Any) -> AmplitudeSummary:
    return _build(
        AmplitudeSummary,
        value,
        peaks=_tuple(_str),
        plane=_int,
        mean=_float,
        sd=_float,
        quantiles=_tuple(_float),
        mean_error=_float,
        quantile_errors=_tuple(_float),
    )


def _posterior(value: Any) -> PosteriorSample:
    return _build(
        PosteriorSample,
        value,
        cluster_id=_int,
        names=_tuple(_str),
        peak_names=_tuple(_str),
        settings=lambda v: _build(
            SamplerSettings, v, walkers=_int, steps=_int, burn_in=_optional(_int)
        ),
        run=lambda v: _build(
            EnsembleRun,
            v,
            chain=_array("float64", 3),
            log_probability=_array("float64", 2),
            blobs=_blobs,
            acceptance=_array("float64", 1),
            entropy=_entropy,
            spawn_key=_tuple(_int),
        ),
        diagnosis=lambda v: _build(
            Diagnosis,
            v,
            burn_in=_int,
            tau=_array("float64", 1),
            tau_reliable=_bool,
            parameters=_tuple(
                lambda p: _build(
                    ParameterSummary,
                    p,
                    name=_str,
                    mean=_float,
                    sd=_float,
                    quantiles=_tuple(_float),
                    mean_error=_float,
                    quantile_errors=_tuple(_float),
                    tau=_float,
                    ess=_float,
                    stationary=_bool,
                    at_lower=_float,
                    at_upper=_float,
                )
            ),
            correlation=_array("float64", 2),
            stuck=_tuple(_int),
            reasons=_tuple(_str),
        ),
        exchanges=_tuple(
            lambda e: _build(Exchange, e, first=_str, second=_str, axis=_str, fraction=_float)
        ),
        determined=_array("float64", 1),
        amplitudes=lambda v: _build(
            AmplitudeSummaries,
            v,
            samples=_int,
            effective=_float,
            amplitudes=_tuple(_amplitude),
            undetermined=_tuple(_withheld),
            correlated=_tuple(
                lambda c: _build(Correlation, c, first=_str, second=_str, plane=_int, rho=_float)
            ),
            sums=_tuple(_amplitude),
        ),
        reduced_chi_squared=_float,
    )


def _record(value: Any) -> MCMCRecord:
    return _build(
        MCMCRecord,
        value,
        created=_str,
        parent=lambda v: _build(
            ParentFit,
            v,
            run_id=_str,
            sha256=_str,
            spectrum_sha256=_optional(_str),
            correction_revision=_int,
            noise=_float,
        ),
        formulation=_str,
        entropy=_int,
        streams=_mapping,
        amplitude_samples=_int,
        software=_mapping,
        posteriors=_tuple(_posterior),
    )


def _validate(record: MCMCRecord) -> None:
    """Shapes agree with the settings and names, and streams with the record's seed."""
    if len({p.cluster_id for p in record.posteriors}) != len(record.posteriors):
        raise ValueError("Duplicate MCMC cluster identity")
    for index, p in enumerate(record.posteriors):
        steps, walkers = p.settings.steps, p.settings.walkers
        n, peaks = len(p.names), len(p.peak_names)
        run, diagnosis = p.run, p.diagnosis
        blobs = (steps, walkers) if peaks == 1 else (steps, walkers, peaks)
        if (
            run.chain.shape != (steps, walkers, n)
            or run.log_probability.shape != (steps, walkers)
            or run.blobs is None
            or run.blobs.shape != blobs
            or run.acceptance.shape != (walkers,)
            or diagnosis.tau.shape != (n,)
            or diagnosis.correlation.shape != (n, n)
            or tuple(s.name for s in diagnosis.parameters) != p.names
            or p.determined.shape != (peaks,)
        ):
            raise ValueError(f"Inconsistent MCMC arrays for cluster {p.cluster_id}")
        if not (np.isfinite(p.reduced_chi_squared) and p.reduced_chi_squared >= 0):
            raise ValueError(f"Invalid reduced chi-squared for cluster {p.cluster_id}")
        if not 0 <= diagnosis.burn_in < steps:
            raise ValueError(f"Burn-in outside the chain for cluster {p.cluster_id}")
        if (run.entropy, run.spawn_key) != (record.entropy, (index,)):
            raise ValueError(f"Cluster {p.cluster_id} was not seeded by this run")


__all__ = [
    "DIRECTORY",
    "MCMCRecord",
    "ParentFit",
    "latest_record",
    "parent_fit",
    "read_record",
    "software_versions",
    "write_record",
]
