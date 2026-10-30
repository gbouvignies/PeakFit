"""Linked and portable analyses sharing atomic publication and verified evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from concurrent.futures import CancelledError
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.gui.fit_positions import DatasetCompletion, dataset_completion
from peakfit.gui.project_results import (
    journal_entry,
    read_peaks,
    restore_result,
    run_record,
    verify_model,
)
from peakfit.gui.spectrum import Dataset, read_dataset
from peakfit.io.readers.legacy_project import read_legacy_project
from peakfit.io.scientific import read_scientific, write_scientific

VERSION = "4.0.0"
MAX_RUNS = 10
MAX_EVIDENCE_BYTES = 1024**3
MAX_ARCHIVE_BYTES = 8 * 1024**3
MAX_JSON_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class Analysis:
    """Committed state, independent of source availability and Qt widgets."""

    dataset: Dataset | None
    header: dict[str, Any]
    peaks: tuple
    configuration: str
    revision: int
    runs: tuple[DatasetCompletion | StoredRun, ...] = ()
    view: dict[str, Any] | None = None
    reason: str = ""
    path: Path | None = None
    journal: tuple[dict[str, Any], ...] = ()
    protected: frozenset[str] = frozenset()
    active: DatasetCompletion | None = None

    @property
    def current(self):
        """Historical scientific currentness is independent of source availability."""
        c = self.completion
        if c is None:
            return False
        snapshot = c.result.inputs.snapshot
        return (
            c.manual_revision == self.revision
            and c.working_peaks == self.peaks
            and snapshot.configuration == self.configuration
            and snapshot.spectral_parameters == tuple(self.header["axes"])
            and snapshot.plane_values == tuple(self.header["planes"])
            and snapshot.source_plane_indices == tuple(self.header["source_plane_indices"])
            and snapshot.source_files == tuple(tuple(s) for s in self.header["sources"])
        )

    @property
    def completion(self):
        """The latest accepted run; older runs retain their own immutable inputs."""
        if self.active is not None:
            return self.active
        return self.runs[-1] if self.runs and isinstance(self.runs[-1], DatasetCompletion) else None


@dataclass(frozen=True)
class StoredRun:
    """Immutable evidence location; reading a history list allocates no scientific arrays."""

    root: Path
    entry: dict[str, Any]

    @property
    def run_id(self):
        """Stable immutable run identity."""
        return self.entry["run_id"]

    def load(self):
        """Validate and materialize one explicitly requested historical record."""
        return _load_run(self.root, self.entry)


def run_id(run):
    """Read identity without implicitly loading archived numerical evidence."""
    return run.run_id if isinstance(run, StoredRun) else run.result.run_id


def stored_analysis(analysis, path):
    """Read published small metadata; retain at most the already active completion."""
    path = path.resolve()
    manifest = _read_json(path / "manifest.json")
    return replace(
        analysis,
        path=path,
        header=manifest["dataset"],
        runs=tuple(StoredRun(path, e) for e in manifest["runs"]),
        journal=tuple(manifest["journal"]),
        active=analysis.completion,
    )


def dataset_header(dataset):
    """Retain calibration, plane identities and source provenance without arrays."""
    spectra = dataset.spectra
    source = json.loads(dict(dataset.sources)["spectrum"])
    return {
        "source": source,
        "sources": dataset.sources,
        "shape": spectra.data.shape,
        "dtype": spectra.data.dtype.str,
        "spectrum_header": spectra.dic,
        "axes": [axis.model_dump_json() for axis in spectra.params],
        "planes": spectra.z_values.tolist(),
        "source_plane_indices": spectra.source_plane_indices,
        "has_plane_values": dataset.has_plane_values,
        "trace_limit": dataset.trace_limit,
        "noise": dataset.noise,
    }


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        stream.flush()
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("Project JSON exceeds the supported 512 MiB limit.")
        os.fsync(stream.fileno())


def _sync(path):
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _sync_directory(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _read_json(path):
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Project JSON exceeds the supported 512 MiB limit.")
    with path.open("rb") as stream:
        content = stream.read(MAX_JSON_BYTES + 1)
    if len(content) > MAX_JSON_BYTES:
        raise ValueError("Expanded project JSON exceeds the supported 512 MiB limit.")
    return json.loads(content)


def _safe_run_id(value):
    if re.fullmatch(r"[a-f0-9]{32}", value) is None:
        raise ValueError("Invalid immutable run identifier.")
    return value


def _write_run(directory, completion):
    result = completion.result
    directory.mkdir()
    write_scientific(directory / "result.npz", result.outcome, result.completed_fit.samples)
    record = run_record(result)
    record["post_fit_peaks"] = [asdict(p) for p in completion.working_peaks]
    record["manual_revision"] = completion.manual_revision
    record["completion_warnings"] = completion.warnings
    _json(directory / "review.json", record)
    files = ["result.npz", "review.json"]
    for name in files:
        _sync(directory / name)
    _sync_directory(directory)
    return {
        name: {"sha256": _digest(directory / name), "bytes": (directory / name).stat().st_size}
        for name in files
    }


def save_analysis(
    analysis: Analysis,
    destination: Path,
    *,
    progress=lambda _: None,
    canceled=lambda: False,
    preserve_input_runs=False,
    portable=False,
) -> Path:
    """Publish last, retaining the previous manifest and payloads on any failure.

    A sibling lock excludes concurrent PeakFit writers. New projects are staged
    beside the destination. Routine Save reuses immutable run files verbatim.
    """
    destination = destination.resolve()
    if destination.suffix != ".peakfit":
        raise ValueError("Analysis directories must have a .peakfit suffix.")
    lock = destination.with_name("." + destination.name + ".save-lock")
    try:
        lock.mkdir()
    except FileExistsError:
        if os.name == "nt":
            raise OSError(
                f"Save lock exists: {lock}. "
                "If no save is running, remove this lock directory and retry."
            ) from None
        owner = lock / "pid"
        try:
            pid = int(owner.read_text())
            if pid <= 0:
                raise ValueError("Invalid save-lock owner")
            os.kill(pid, 0)
        except ProcessLookupError:
            shutil.rmtree(lock)
            lock.mkdir()
        except (OSError, ValueError) as error:
            raise OSError(f"Cannot inspect previous save lock {lock}: {error}") from error
        else:
            raise OSError(
                "Another PeakFit process is saving this analysis. Retry after it finishes."
            ) from None
    (lock / "pid").write_text(str(os.getpid()))
    try:
        if destination.exists():
            if analysis.path is None or analysis.path.absolute() != destination:
                raise FileExistsError("Save as requires a new analysis directory.")
            _save_generation(
                analysis,
                destination,
                destination,
                progress,
                canceled,
                preserve_input_runs,
                portable,
            )
        else:
            with TemporaryDirectory(prefix=".peakfit-save-", dir=destination.parent) as temp:
                staged = Path(temp) / destination.name
                staged.mkdir()
                _save_generation(
                    analysis, staged, destination, progress, canceled, preserve_input_runs, portable
                )
                staged.rename(destination)
                _sync_directory(destination.parent)
        return destination
    finally:
        (lock / "pid").unlink(missing_ok=True)
        lock.rmdir()


def spectrum_path(header):
    """Stable shared cube identity; plane values/view state do not duplicate it."""
    identity = hashlib.sha256(
        json.dumps(
            [
                header["source"]["checksum_sha256"],
                header["shape"],
                header["dtype"],
                header.get("source_plane_indices"),
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return "spectra/" + identity + ".npz"


def _save_spectrum(analysis, root, old, origin, progress, canceled):
    """Share one lossless experimental cube across all runs and routine saves."""
    name = spectrum_path(analysis.header)
    previous = next(
        (m.get("spectrum") for m in (old, origin) if m.get("spectrum", {}).get("path") == name),
        None,
    )
    target = root / name
    if previous and target.is_file():
        return previous
    target.parent.mkdir(exist_ok=True)
    if previous and analysis.path:
        shutil.copyfile(analysis.path / name, target)
        if _digest(target) != previous["sha256"]:
            raise ValueError("Corrupt embedded experimental spectrum.")
    else:
        if analysis.dataset is None:
            raise ValueError(
                "Portable export requires the verified experimental spectrum. "
                "Relink the source before exporting a portable analysis."
            )
        progress("Embedding experimental spectrum")
        if canceled():
            raise CancelledError
        with TemporaryDirectory(prefix=".spectrum-", dir=target.parent) as temporary:
            staged = Path(temporary) / "spectrum.npz"
            with (
                ZipFile(staged, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive,
                archive.open("data.npy", "w", force_zip64=True) as stream,
            ):
                np.lib.format.write_array(stream, analysis.dataset.spectra.data, allow_pickle=False)
            _sync(staged)
            staged.replace(target)
    _sync(target)
    _sync_directory(target.parent)
    return {"path": name, "sha256": _digest(target), "bytes": target.stat().st_size}


def _embedded_dataset(path, manifest, *, canceled=lambda: False):
    """Validate size, identity and NPY header before allocating the full cube."""
    entry = manifest["spectrum"]
    name = entry["path"]
    if re.fullmatch(r"spectra/[a-f0-9]{64}\.npz", name) is None:
        raise ValueError("Invalid embedded spectrum path.")
    payload = path / name
    if payload.is_symlink() or payload.parent.is_symlink():
        raise ValueError("Embedded spectrum must not be a symbolic link.")
    if payload.stat().st_size != entry["bytes"] or _digest(payload) != entry["sha256"]:
        raise ValueError("Corrupt embedded experimental spectrum.")
    header = manifest["dataset"]
    expected_shape, expected_dtype = tuple(header["shape"]), np.dtype(header["dtype"])
    with ZipFile(payload) as archive:
        if archive.namelist() != ["data.npy"]:
            raise ValueError("Invalid embedded spectrum members.")
        with archive.open("data.npy") as stream:
            version = np.lib.format.read_magic(stream)
            reader = {
                (1, 0): np.lib.format.read_array_header_1_0,
                (2, 0): np.lib.format.read_array_header_2_0,
            }.get(version)
            if reader is None:
                raise ValueError("Unsupported embedded spectrum header.")
            shape, _, dtype = reader(stream)
            size = math.prod(shape) * dtype.itemsize
            if (
                shape != expected_shape
                or dtype != expected_dtype
                or dtype.kind not in "fiu"
                or size > MAX_ARCHIVE_BYTES
                or archive.getinfo("data.npy").file_size != stream.tell() + size
            ):
                raise ValueError("Invalid embedded spectrum shape, dtype or payload size.")
        if canceled():
            raise CancelledError
        with archive.open("data.npy") as stream:
            data = np.lib.format.read_array(stream, allow_pickle=False)
    values = np.asarray(header["planes"], dtype=float)
    if (
        values.shape != (data.shape[0],)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(data))
    ):
        raise ValueError("Invalid embedded experimental values.")
    data.flags.writeable = values.flags.writeable = False
    spectra = Spectra(
        dic=header.get("spectrum_header", {}),
        data=data,
        z_values=values,
        params=[SpectralParameters.model_validate_json(p) for p in header["axes"]],
        source_plane_indices=tuple(header.get("source_plane_indices", range(len(values)))),
    )
    return Dataset(
        Path(header["source"]["path"]),
        spectra,
        read_peaks(manifest["working_peaks"]),
        tuple(tuple(s) for s in header["sources"]),
        header["has_plane_values"],
        header["trace_limit"],
        header["noise"],
    )


def _save_generation(  # noqa: PLR0912, PLR0915
    analysis, root, destination, progress, canceled, preserve_input_runs, portable
):
    evidence = root / "runs"
    evidence.mkdir(exist_ok=True)
    retained = tuple(
        c
        for index, c in enumerate(analysis.runs)
        if index >= len(analysis.runs) - MAX_RUNS or run_id(c) in analysis.protected
    )
    old = _read_json(root / "manifest.json") if (root / "manifest.json").exists() else {}
    origin = {}
    if (
        analysis.path
        and analysis.path != destination
        and (analysis.path / "manifest.json").exists()
    ):
        origin = _read_json(analysis.path / "manifest.json")
    spectrum = _save_spectrum(analysis, root, old, origin, progress, canceled) if portable else {}
    records = []
    for completion in retained:
        if canceled():
            raise CancelledError
        identity = _safe_run_id(run_id(completion))
        progress(f"Saving fit evidence · run {identity[:8]}")
        target = evidence / identity
        previous = next((r for r in old.get("runs", []) if r["run_id"] == identity), None)
        if previous and target.is_dir():
            files = previous["files"]
        else:
            # Interrupted unreferenced evidence never replaces a published run.
            with TemporaryDirectory(prefix=".run-", dir=evidence) as temp:
                staged = Path(temp) / identity
                original = (
                    completion.entry
                    if isinstance(completion, StoredRun)
                    else next((r for r in origin.get("runs", []) if r["run_id"] == identity), None)
                )
                source_root = (
                    completion.root if isinstance(completion, StoredRun) else analysis.path
                )
                if original is not None:
                    staged.mkdir()
                    files = original["files"]
                    for name in files:
                        copy = staged / name
                        copy.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source_root / "runs" / identity / name, copy)
                        if _digest(copy) != files[name]["sha256"]:
                            raise ValueError("Save as encountered corrupt immutable evidence.")
                        _sync(copy)
                    _sync_directory(staged)
                else:
                    files = _write_run(staged, completion)
                if target.exists():
                    shutil.rmtree(target)
                staged.rename(target)
        records.append({"run_id": identity, "files": files})
    historical_bytes = sum(
        f["bytes"]
        for r in records[:-1]
        if r["run_id"] not in analysis.protected
        for f in r["files"].values()
    )
    evicted = set()
    for record in records[:-1]:
        if historical_bytes <= MAX_EVIDENCE_BYTES:
            break
        if record["run_id"] not in analysis.protected:
            historical_bytes -= sum(f["bytes"] for f in record["files"].values())
            evicted.add(record["run_id"])
    records = [r for r in records if r["run_id"] not in evicted]
    if (
        sum(f["bytes"] for r in records for f in r["files"].values()) + spectrum.get("bytes", 0)
        > MAX_ARCHIVE_BYTES
    ):
        raise ValueError("Retained evidence exceeds the supported 8 GiB project reader limit.")
    header = dict(analysis.header)
    header["embedded"] = portable
    if analysis.dataset is not None:
        header["spectrum_header"] = analysis.dataset.spectra.dic
    source = dict(header["source"])
    location = source.get("linked_path", source["path"])
    source["relative_path"] = os.path.relpath(location, destination)
    header["source"] = source
    full_ids = {r["run_id"] for r in records}
    journal = list(analysis.journal) or [
        journal_entry(c.load() if isinstance(c, StoredRun) else c) for c in analysis.runs
    ]
    available = {run_id(c): c for c in analysis.runs}
    for root_path, manifest in ((root, old), (analysis.path, origin)):
        for entry in manifest.get("runs", []):
            available.setdefault(entry["run_id"], StoredRun(root_path, entry))
    summaries = []
    for original_entry in journal:
        entry = original_entry
        if entry["run_id"] in full_ids:
            summaries.append(dict(entry, evidence="Full", parameters=None))
            continue
        if entry.get("status") == "completed" and entry.get("parameters") is None:
            source_run = available.get(entry["run_id"])
            if source_run is None:
                raise ValueError(
                    "Cannot prune full evidence without its promised scientific summary."
                )
            completion = source_run.load() if isinstance(source_run, StoredRun) else source_run
            # Materialize the entire contract before any old payload is removed.
            entry = dict(entry, **journal_entry(completion))
        summaries.append(dict(entry, evidence="Summary only"))
    journal = summaries
    manifest = {
        "schema_version": VERSION,
        "kind": "peakfit_analysis",
        "spectrum": spectrum,
        "dataset": header,
        "working_peaks": [asdict(p) for p in analysis.peaks],
        "configuration": analysis.configuration,
        "manual_revision": analysis.revision,
        "runs": records,
        "journal": journal,
        "protected": sorted(analysis.protected),
        "storage_bytes": {
            "spectrum": spectrum.get("bytes", 0),
            "active": sum(f["bytes"] for f in records[-1]["files"].values()) if records else 0,
            "historical": historical_bytes,
            "protected": sum(
                f["bytes"]
                for r in records[:-1]
                if r["run_id"] in analysis.protected
                for f in r["files"].values()
            ),
        },
        "view": analysis.view or {},
    }
    if not portable:
        del manifest["spectrum"]
    if canceled():
        raise CancelledError
    progress("Publishing analysis")
    temporary = root / (".manifest-" + uuid4().hex)
    try:
        _json(temporary, manifest)
        _sync_directory(evidence)
        temporary.replace(root / "manifest.json")
        _sync_directory(root)
    finally:
        temporary.unlink(missing_ok=True)
    # Only now can obsolete evidence be removed; the last good generation stayed intact.
    keep = {r["run_id"] for r in records}
    if preserve_input_runs:
        # A session checkpoint can precede Qt adoption. Keep files still leased
        # by the previous UI history, even if this checkpoint demotes their entries.
        keep.update(run_id(c) for c in analysis.runs)
    for path in evidence.iterdir():
        if path.is_dir() and path.name not in keep:
            shutil.rmtree(path)

    if not portable and (root / "spectra").exists():
        shutil.rmtree(root / "spectra")


def _validate_run_files(root, entry):
    run_id = _safe_run_id(entry["run_id"])
    directory = root / "runs" / run_id
    if any(p.is_symlink() for p in (root / "runs", directory, directory / "summary")):
        raise ValueError("Project evidence directories must not be symbolic links.")
    if set(entry["files"]) not in (
        {"review.json", "result.npz"},
        {"review.json", "samples.npz", "summary/fit.json"},
        {"review.json", "samples.npz", "summary/fit.json.gz"},
    ):
        raise ValueError("Invalid project run payload list.")
    for name, info in entry["files"].items():
        path = directory / name
        if (
            path.is_symlink()
            or path.stat().st_size != info["bytes"]
            or _digest(path) != info["sha256"]
        ):
            raise ValueError(f"Corrupt project evidence: {run_id}/{name}")
    return directory


def _load_run(root, entry):
    directory = _validate_run_files(root, entry)
    run_id = entry["run_id"]
    record = _read_json(directory / "review.json")
    if record["run_id"] != run_id:
        raise ValueError("Run identifier disagrees with immutable evidence.")
    if "result.npz" in entry["files"]:
        outcome, samples, _ = read_scientific(directory / "result.npz")
    else:
        outcome, samples = read_legacy_project(directory, entry, record)
    result = restore_result(record, outcome, samples)
    post = read_peaks(record["post_fit_peaks"])
    # Validate automatic adoption again rather than trusting a saved Current flag.
    completion = dataset_completion(
        result,
        result.inputs.snapshot,
        submitted_revision=record["manual_revision"],
        manual_revision=record["manual_revision"],
    )
    if completion.working_peaks != post or completion.warnings != tuple(
        record["completion_warnings"]
    ):
        raise ValueError("Saved working transition disagrees with the fitted outcome.")
    return completion


def open_analysis(
    path: Path, *, relink: Path | None = None, progress=lambda _: None, canceled=lambda: False
) -> Analysis:
    """Verify retained evidence and either the linked source or embedded spectrum."""
    path = path.resolve()
    manifest = _read_json(path / "manifest.json")
    if (
        manifest.get("schema_version") not in {"1.0.0", "1.1.0", "2.0.0", "3.0.0", VERSION}
        or manifest.get("kind") != "peakfit_analysis"
    ):
        raise ValueError(f"Unsupported analysis version: {manifest.get('schema_version')!r}")
    allowed = {
        "schema_version",
        "kind",
        "dataset",
        "working_peaks",
        "configuration",
        "manual_revision",
        "runs",
        "view",
        "journal",
        "protected",
        "storage_bytes",
    }
    if "spectrum" in manifest and manifest.get("schema_version") in {"3.0.0", VERSION}:
        allowed.add("spectrum")
    if set(manifest) != allowed:
        raise ValueError("Invalid analysis manifest fields.")
    entries = manifest["runs"]
    if len({r["run_id"] for r in entries}) != len(entries):
        raise ValueError("Invalid retained run list.")
    if (
        sum(f["bytes"] for r in entries for f in r["files"].values())
        + manifest.get("spectrum", {}).get("bytes", 0)
        > MAX_ARCHIVE_BYTES
    ):
        raise ValueError("Project evidence exceeds the 8 GiB reader limit.")
    # Check every immutable payload, but decode numerical arrays only for the active run.
    for entry in entries:
        progress("Verifying retained evidence checksums")
        _validate_run_files(path, entry)
    runs = tuple(StoredRun(path, e) for e in entries)
    active = runs[-1].load() if runs else None
    config = PeakFitConfig.model_validate_json(manifest["configuration"])
    if active:
        snapshot = active.result.inputs.snapshot
        fingerprint = json.loads(dict(snapshot.source_files)["spectrum"])["checksum_sha256"]
        if fingerprint != manifest["dataset"]["source"]["checksum_sha256"]:
            raise ValueError("Run source identity disagrees with the analysis source.")
    manifest["dataset"].setdefault(
        "source_plane_indices", list(range(manifest["dataset"]["shape"][0]))
    )
    analysis = Analysis(
        None,
        manifest["dataset"],
        read_peaks(manifest["working_peaks"]),
        config.model_dump_json(),
        manifest["manual_revision"],
        runs,
        manifest["view"],
        path=path,
        journal=tuple(manifest.get("journal", [])),
        protected=frozenset(manifest.get("protected", [])),
        active=active,
    )
    if "spectrum" in manifest and relink is None:
        if canceled():
            return replace(
                analysis, reason="Spectrum verification canceled; numerical evidence is available."
            )
        dataset = _embedded_dataset(path, manifest, canceled=canceled)
        if analysis.completion:
            try:
                _verify_dataset(analysis.completion, dataset)
            except (ValueError, AssertionError, KeyError) as error:
                return replace(analysis, reason=f"Model verification failed: {error}")
        return replace(analysis, dataset=dataset, reason="")
    return verify_sources(analysis, relink=relink, progress=progress, canceled=canceled)


def _verify_dataset(completion, dataset):
    """Use identical model and exact paired-sample checks for both source modes."""
    verify_model(completion.result)
    planes = completion.result.inputs.snapshot.source_plane_indices
    lookup = {p: i for i, p in enumerate(dataset.spectra.source_plane_indices)}
    selected = [lookup[p] for p in planes]
    for sample in completion.result.completed_fit.samples:
        observed = dataset.spectra.data[(slice(None), *sample.grid_indices)].T[:, selected]
        if not np.array_equal(observed, sample.data):
            raise ValueError("Verified source samples disagree with retained fit evidence")


def verify_sources(analysis, *, relink=None, progress=lambda _: None, canceled=lambda: False):
    """Try only recorded candidates or the explicitly selected replacement."""
    analysis = replace(analysis, dataset=None)
    source = analysis.header["source"]
    candidates = (
        [relink]
        if relink
        else [
            analysis.path / source["relative_path"],
            Path(source.get("linked_path", source["path"])),
            Path(source["path"]),
        ]
    )
    reasons = []
    for candidate in dict.fromkeys(candidates):
        if canceled():
            return replace(
                analysis,
                reason="Source verification canceled. Relink before comparison or fitting.",
            )
        progress(f"Verifying spectrum content · {candidate.name}")
        try:
            dataset = read_dataset(candidate, canceled=canceled, progress=progress)
            identity = json.loads(dict(dataset.sources)["spectrum"])
            if identity["checksum_sha256"] != source["checksum_sha256"]:
                raise ValueError("SHA-256 differs from the spectrum used by this analysis")
            h = analysis.header
            if (
                list(dataset.spectra.data.shape) != list(h["shape"])
                or dataset.spectra.data.dtype.str != h["dtype"]
                or [a.model_dump_json() for a in dataset.spectra.params] != h["axes"]
            ):
                raise ValueError("Scientific calibration, processing, shape or dtype differs")
            values = np.asarray(h["planes"], dtype=float)
            if values.shape != (dataset.spectra.data.shape[0],) or not np.all(np.isfinite(values)):
                raise ValueError("Invalid saved ordered plane mapping")
            values.flags.writeable = False
            dataset = replace(
                dataset,
                peaks=analysis.peaks,
                spectra=dataset.spectra.model_copy(update={"z_values": values}),
                sources=tuple(tuple(s) for s in h["sources"]),
                has_plane_values=h["has_plane_values"],
            )
            progress("Checking reconstructed model against retained samples")
            for completion in (analysis.completion,) if analysis.completion else ():
                _verify_dataset(completion, dataset)
            if canceled():
                return replace(
                    analysis,
                    reason="Source verification canceled; numerical evidence is available.",
                )
            header = dict(h)
            stat = candidate.stat()
            if any(
                str(getattr(stat, field)) != identity[key]
                for field, key in (
                    ("st_size", "size"),
                    ("st_mtime_ns", "mtime_ns"),
                    ("st_ctime_ns", "ctime_ns"),
                    ("st_ino", "inode"),
                    ("st_dev", "device"),
                )
                if key in identity
            ):
                raise ValueError("Source changed during scientific verification. Verify it again.")
            header["source"] = dict(
                source,
                linked_path=str(candidate.resolve()),
                verified_at=identity.get("verified_at", source.get("verified_at", "")),
                size=str(stat.st_size),
                mtime_ns=str(stat.st_mtime_ns),
                ctime_ns=str(stat.st_ctime_ns),
                device=str(stat.st_dev),
                inode=str(stat.st_ino),
            )
            return replace(analysis, dataset=dataset, header=header, reason="")
        except Exception as error:  # Source readers can also raise format-specific decoding errors.
            reasons.append(f"{candidate}: {error}")
    return replace(analysis, reason="\n".join(reasons) or "Spectrum source unavailable.")


def source_link_unchanged(header):
    """Cheap change detection for routine Save; never hash source content here."""
    if header.get("embedded"):
        return True
    source = header["source"]
    try:
        stat = Path(source.get("linked_path", source["path"])).stat()
        return all(
            str(getattr(stat, field)) == source[key]
            for field, key in (
                ("st_size", "size"),
                ("st_mtime_ns", "mtime_ns"),
                ("st_ctime_ns", "ctime_ns"),
                ("st_dev", "device"),
                ("st_ino", "inode"),
            )
        )
    except (OSError, KeyError):
        return False
