"""Versioned linked-source analyses with atomic publication and bounded evidence."""

import hashlib
import json
import os
import re
import shutil
from concurrent.futures import CancelledError
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4
from zipfile import ZipFile

import numpy as np

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.fit.output_metadata import RunMetadata
from peakfit.gui.fit_positions import DatasetCompletion, dataset_completion
from peakfit.gui.project_results import (
    journal_entry,
    read_peaks,
    restore_result,
    run_record,
    verify_model,
)
from peakfit.gui.spectrum import Dataset, read_dataset
from peakfit.io.writers.json import write_final_outcome_summary

VERSION = "1.0.0"
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
    runs: tuple[DatasetCompletion, ...] = ()
    view: dict[str, Any] | None = None
    reason: str = ""
    path: Path | None = None
    journal: tuple[dict[str, Any], ...] = ()
    protected: frozenset[str] = frozenset()

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
            and snapshot.source_files == tuple(tuple(s) for s in self.header["sources"])
        )

    @property
    def completion(self):
        """The latest accepted run; older runs retain their own immutable inputs."""
        return self.runs[-1] if self.runs else None


def dataset_header(dataset):
    """Retain calibration, plane identities and source provenance without arrays."""
    spectra = dataset.spectra
    source = json.loads(dict(dataset.sources)["spectrum"])
    return {
        "source": source,
        "sources": dataset.sources,
        "shape": spectra.data.shape,
        "dtype": spectra.data.dtype.str,
        "axes": [axis.model_dump_json() for axis in spectra.params],
        "planes": spectra.z_values.tolist(),
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
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_run_id(value):
    if re.fullmatch(r"[a-f0-9]{32}", value) is None:
        raise ValueError("Invalid immutable run identifier.")
    return value


def _write_run(directory, completion):
    result = completion.result
    directory.mkdir()
    summary = directory / "summary" / "fit.json"
    write_final_outcome_summary(
        result.outcome,
        metadata=RunMetadata(
            timestamp=result.completed_at,
            software_version=result.software_version,
            input_files={k: json.loads(v) for k, v in result.inputs.snapshot.source_files},
            configuration=json.loads(result.inputs.snapshot.configuration),
            run_duration_seconds=result.execution_seconds,
        ),
        z_values=np.array(result.inputs.snapshot.plane_values),
        path=summary,
    )
    arrays = {}
    for s in result.completed_fit.samples:
        prefix = f"c{s.cluster_id}_"
        arrays[prefix + "data"] = s.data
        arrays[prefix + "corrections"] = s.neighbor_corrections
        arrays.update({prefix + f"index{dim}": a for dim, a in enumerate(s.grid_indices)})
    np.savez_compressed(directory / "samples.npz", **arrays)
    record = run_record(result)
    record["post_fit_peaks"] = [asdict(p) for p in completion.working_peaks]
    record["manual_revision"] = completion.manual_revision
    record["completion_warnings"] = completion.warnings
    record["arrays"] = {k: {"shape": a.shape, "dtype": a.dtype.str} for k, a in arrays.items()}
    _json(directory / "review.json", record)
    files = ["summary/fit.json", "samples.npz", "review.json"]
    if sum(a.nbytes for a in arrays.values()) > MAX_ARCHIVE_BYTES:
        raise ValueError("Sample evidence exceeds the supported 8 GiB expanded payload limit.")
    for name in files:
        if name.endswith(".json") and (directory / name).stat().st_size > MAX_JSON_BYTES:
            raise ValueError(
                "Scientific JSON exceeds the 512 MiB document limit. Export the run separately."
            )
        _sync(directory / name)
    _sync_directory(directory / "summary")
    _sync_directory(directory)
    return {
        name: {"sha256": _digest(directory / name), "bytes": (directory / name).stat().st_size}
        for name in files
    }


def save_analysis(
    analysis: Analysis, destination: Path, *, progress=lambda _: None, canceled=lambda: False
) -> Path:
    """Publish last, retaining the previous manifest and payloads on any failure.

    A sibling lock excludes concurrent PeakFit writers. New projects are staged
    beside the destination. Routine Save reuses immutable run files verbatim.
    """
    destination = destination.absolute()
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
            _save_generation(analysis, destination, destination, progress, canceled)
        else:
            with TemporaryDirectory(prefix=".peakfit-save-", dir=destination.parent) as temp:
                staged = Path(temp) / destination.name
                staged.mkdir()
                _save_generation(analysis, staged, destination, progress, canceled)
                staged.rename(destination)
                _sync_directory(destination.parent)
        return destination
    finally:
        (lock / "pid").unlink(missing_ok=True)
        lock.rmdir()


def _save_generation(analysis, root, destination, progress, canceled):  # noqa: PLR0912, PLR0915
    evidence = root / "runs"
    evidence.mkdir(exist_ok=True)
    retained = tuple(
        c
        for index, c in enumerate(analysis.runs)
        if index >= len(analysis.runs) - MAX_RUNS or c.result.run_id in analysis.protected
    )
    old = _read_json(root / "manifest.json") if (root / "manifest.json").exists() else {}
    origin = {}
    if (
        analysis.path
        and analysis.path != destination
        and (analysis.path / "manifest.json").exists()
    ):
        origin = _read_json(analysis.path / "manifest.json")
    records = []
    for completion in retained:
        if canceled():
            raise CancelledError
        run_id = _safe_run_id(completion.result.run_id)
        progress(f"Saving fit evidence · run {run_id[:8]}")
        target = evidence / run_id
        previous = next((r for r in old.get("runs", []) if r["run_id"] == run_id), None)
        if previous and target.is_dir():
            files = previous["files"]
        else:
            # Interrupted unreferenced evidence never replaces a published run.
            with TemporaryDirectory(prefix=".run-", dir=evidence) as temp:
                staged = Path(temp) / run_id
                original = next((r for r in origin.get("runs", []) if r["run_id"] == run_id), None)
                if original is not None:
                    staged.mkdir()
                    files = original["files"]
                    for name in files:
                        copy = staged / name
                        copy.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(analysis.path / "runs" / run_id / name, copy)
                        if _digest(copy) != files[name]["sha256"]:
                            raise ValueError("Save as encountered corrupt immutable evidence.")
                        _sync(copy)
                    _sync_directory(staged / "summary")
                    _sync_directory(staged)
                else:
                    files = _write_run(staged, completion)
                if target.exists():
                    shutil.rmtree(target)
                staged.rename(target)
        records.append({"run_id": run_id, "files": files})
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
    if sum(f["bytes"] for r in records for f in r["files"].values()) > MAX_ARCHIVE_BYTES:
        raise ValueError("Retained evidence exceeds the supported 8 GiB project reader limit.")
    header = dict(analysis.header)
    source = dict(header["source"])
    location = source.get("linked_path", source["path"])
    source["relative_path"] = os.path.relpath(location, destination)
    header["source"] = source
    full_ids = {r["run_id"] for r in records}
    journal = list(analysis.journal) or [journal_entry(c) for c in analysis.runs]
    # Full-run parameters have exactly one on-disk authority: summary/fit.json.
    journal = [
        dict(entry, evidence="Full", parameters=None)
        if entry["run_id"] in full_ids
        else dict(entry, evidence="Summary only")
        for entry in journal
    ]
    manifest = {
        "schema_version": VERSION,
        "kind": "peakfit_analysis",
        "dataset": header,
        "working_peaks": [asdict(p) for p in analysis.peaks],
        "configuration": analysis.configuration,
        "manual_revision": analysis.revision,
        "runs": records,
        "journal": journal,
        "protected": sorted(analysis.protected),
        "storage_bytes": {
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
    for path in evidence.iterdir():
        if path.is_dir() and path.name not in keep:
            shutil.rmtree(path)


def _load_run(root, entry):
    run_id = _safe_run_id(entry["run_id"])
    directory = root / "runs" / run_id
    if any(p.is_symlink() for p in (root / "runs", directory, directory / "summary")):
        raise ValueError("Project evidence directories must not be symbolic links.")
    expected = {"summary/fit.json", "review.json", "samples.npz"}
    if set(entry["files"]) != expected:
        raise ValueError("Invalid project run payload list.")
    for name, info in entry["files"].items():
        path = directory / name
        if (
            path.is_symlink()
            or path.stat().st_size != info["bytes"]
            or _digest(path) != info["sha256"]
        ):
            raise ValueError(f"Corrupt project evidence: {run_id}/{name}")
    record = _read_json(directory / "review.json")
    if record["run_id"] != run_id:
        raise ValueError("Run identifier disagrees with immutable evidence.")
    with ZipFile(directory / "samples.npz") as archive:
        if sum(i.file_size for i in archive.infolist()) > MAX_ARCHIVE_BYTES:
            raise ValueError("Expanded sample evidence exceeds the 8 GiB reader limit.")
    with np.load(directory / "samples.npz", allow_pickle=False) as payload:
        if set(payload.files) != set(record["arrays"]):
            raise ValueError("Sample array names disagree with manifest.")
        arrays = {}
        for name, descriptor in record["arrays"].items():
            a = payload[name]
            if (
                list(a.shape) != descriptor["shape"]
                or a.dtype.str != descriptor["dtype"]
                or a.dtype.kind not in "fiu"
            ):
                raise ValueError("Sample array shape/dtype mismatch.")
            a.flags.writeable = False
            arrays[name] = a
    result = restore_result(record, _read_json(directory / "summary" / "fit.json"), arrays)
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
    """Validate evidence before source verification; unavailable sources retain review."""
    path = path.absolute()
    manifest = _read_json(path / "manifest.json")
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
    if set(manifest) != allowed:
        raise ValueError("Invalid analysis manifest fields.")
    if manifest.get("schema_version") != VERSION or manifest.get("kind") != "peakfit_analysis":
        raise ValueError(f"Unsupported analysis version: {manifest.get('schema_version')!r}")
    entries = manifest["runs"]
    if len({r["run_id"] for r in entries}) != len(entries):
        raise ValueError("Invalid retained run list.")
    if sum(f["bytes"] for r in entries for f in r["files"].values()) > MAX_ARCHIVE_BYTES:
        raise ValueError("Project evidence exceeds the 8 GiB reader limit.")
    runs = []
    for entry in entries:
        progress("Validating numerical fit evidence")
        runs.append(_load_run(path, entry))
    config = PeakFitConfig.model_validate_json(manifest["configuration"])
    for completion in runs:
        snapshot = completion.result.inputs.snapshot
        fingerprint = json.loads(dict(snapshot.source_files)["spectrum"])["checksum_sha256"]
        if fingerprint != manifest["dataset"]["source"]["checksum_sha256"]:
            raise ValueError("Run source identity disagrees with the analysis source.")
    analysis = Analysis(
        None,
        manifest["dataset"],
        read_peaks(manifest["working_peaks"]),
        config.model_dump_json(),
        manifest["manual_revision"],
        tuple(runs),
        manifest["view"],
        path=path,
        journal=tuple(manifest.get("journal", [])),
        protected=frozenset(manifest.get("protected", [])),
    )
    return verify_sources(analysis, relink=relink, progress=progress, canceled=canceled)


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
            for completion in analysis.runs:
                verify_model(completion.result)
                # Equality of full sources is necessary but sample ordering must agree too.
                for s in completion.result.completed_fit.samples:
                    if not np.array_equal(
                        dataset.spectra.data[(slice(None), *s.grid_indices)].T, s.data
                    ):
                        raise ValueError(
                            "Verified source samples disagree with retained fit evidence"
                        )
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
