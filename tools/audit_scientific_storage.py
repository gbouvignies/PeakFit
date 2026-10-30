"""Measure real scientific outputs and retention-only size estimates; never alter writers."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import pickle
import platform
import re
import sys
import warnings
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from io import BytesIO, TextIOWrapper
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import MethodType
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from threadpoolctl import threadpool_limits

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.fit.fitting import load_data, run_fit, write_fit_run_outputs
from peakfit.fit.pipeline import run_pipeline
from peakfit.fit.run_models import FitRun
from peakfit.gui.dataset_fit import DatasetFitAttempt, finish_dataset_fit
from peakfit.gui.fit_export import export_fit_result
from peakfit.gui.fit_positions import dataset_completion
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.project import Analysis, dataset_header, save_analysis
from peakfit.gui.spectrum import read_dataset
from peakfit.io.state import load_state

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ARRAYS = (
    "shapes",
    "amplitudes",
    "amplitude_standard_errors",
    "amplitude_covariance",
    "scaled_amplitude_standard_errors",
    "model_values",
    "raw_residuals",
    "normalized_residuals",
)
DENSE_DERIVED = {"shapes", "model_values", "raw_residuals", "normalized_residuals"}


def purpose(path):  # noqa: PLR0911
    """Group physical arrays by scientific role, not compression behavior."""
    name = path[-1]
    if path[0] in {"params", "scalar_params"}:
        return "continuation parameter representations"
    if any("_grid" in part or part == "context" for part in path) and "clusters" in path:
        return "reconstructible model implementation state"
    if name == "amplitudes":
        return "fitted amplitudes"
    if name in {
        "amplitude_standard_errors",
        "amplitude_covariance",
        "scaled_amplitude_standard_errors",
    }:
        return "published amplitude uncertainties"
    if name in DENSE_DERIVED:
        return {
            "shapes": "derived sampled shapes",
            "model_values": "derived sampled models",
            "raw_residuals": "derived raw residuals",
            "normalized_residuals": "derived normalized residuals",
        }[name]
    if name == "data":
        return "retained observations"
    if name in {"neighbor_corrections", "corrections"}:
        return "terminal neighbor corrections"
    if "grid_indices" in path:
        return "exact sample mappings"
    if "coordinate_table" in path or "models" in path:
        return "coordinate provenance and geometry"
    if name in {"plane_values", "source_plane_indices"}:
        return "plane mapping"
    return "other scientific/provenance arrays"


def references(value, path=()):
    """Yield typed-array descriptors with human-readable scientific paths."""
    if isinstance(value, dict):
        if set(value) == {"array", "dtype", "shape"}:
            yield path, value
        elif set(value) == {"mapping"}:
            yield from references(value["mapping"], path)
        else:
            for name, item in value.items():
                yield from references(item, (*path, str(name)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from references(item, (*path, str(index)))


def archive_inventory(path, artifact):
    """Attribute compressed member sizes exactly, separating ZIP/NPY overhead."""
    rows = []
    with ZipFile(path) as archive:
        document = json.loads(archive.read("metadata.json"))
        duplicates = {}
        for logical, descriptor in references(document):
            name = descriptor["array"] + ".npy"
            info = archive.getinfo(name)
            payload = archive.read(name)
            array = np.load(BytesIO(payload), allow_pickle=False)
            digest = hashlib.sha256(payload).hexdigest()
            row = {
                "artifact": artifact,
                "group": purpose(logical),
                "name": name,
                "scientific_path": ".".join(logical),
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "array_bytes": array.nbytes,
                "stored_uncompressed_bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "duplicate_of": duplicates.get(digest, ""),
                "compression": "deflate-1 NPY member (includes NPY header)",
            }
            duplicates.setdefault(digest, name)
            rows.append(row)
        metadata = archive.getinfo("metadata.json")
        summary = {
            "bytes": path.stat().st_size,
            "array_bytes": sum(r["array_bytes"] for r in rows),
            "n_arrays": len(rows),
            "compressed_array_members": sum(r["compressed_bytes"] for r in rows),
            "metadata_bytes": metadata.file_size,
            "compressed_metadata_bytes": metadata.compress_size,
            "zip_overhead_bytes": path.stat().st_size
            - sum(i.compress_size for i in archive.infolist()),
        }
        assert (
            summary["compressed_array_members"]
            + summary["compressed_metadata_bytes"]
            + summary["zip_overhead_bytes"]
            == summary["bytes"]
        )
    return rows, summary


def state_inventory(path, artifact):
    """Inspect only the trusted continuation file produced by this audit itself."""
    with path.open("rb") as stream:
        state = pickle.load(stream)
    arrays, seen, rows = {}, set(), []

    def walk(value, logical):  # noqa: PLR0912
        if isinstance(value, np.ndarray):
            if id(value) not in arrays:
                arrays[id(value)] = ".".join(logical)
                rows.append(
                    {
                        "artifact": artifact,
                        "group": purpose(logical),
                        "name": ".".join(logical),
                        "scientific_path": ".".join(logical),
                        "shape": list(value.shape),
                        "dtype": value.dtype.str,
                        "array_bytes": value.nbytes,
                        "stored_uncompressed_bytes": value.nbytes,
                        "compressed_bytes": value.nbytes,
                        "duplicate_of": "",
                        "compression": "uncompressed ndarray; pickle framing accounted separately",
                    }
                )
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                walk(getattr(value, field.name), (*logical, field.name))
        elif isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, (*logical, str(key)))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, (*logical, str(index)))

        elif isinstance(value, MethodType):
            walk(value.__self__, (*logical, "__self__"))
        elif type(value).__module__.startswith("peakfit."):
            attributes = dict(getattr(value, "__dict__", {}))
            for cls in type(value).__mro__:
                slots = getattr(cls, "__slots__", ())
                if isinstance(slots, str):
                    slots = (slots,)
                for slot in slots:
                    if slot not in {"__dict__", "__weakref__"} and hasattr(value, slot):
                        attributes.setdefault(slot, getattr(value, slot))
            for key, item in attributes.items():
                walk(item, (*logical, key))

    walk(state, ())
    scalar_records = state.get("scalar_params", {}).get("params", {})
    return rows, {
        "scalar_parameter_records": len(scalar_records),
        "computed_scalar_amplitude_records": sum(
            bool(p.get("computed")) for p in scalar_records.values()
        ),
        "separately_pickled_top_level_bytes_not_additive": {
            k: len(pickle.dumps(v)) for k, v in state.items()
        },
        "bytes": path.stat().st_size,
        "n_unique_arrays": len(rows),
        "raw_array_bytes": sum(r["array_bytes"] for r in rows),
        "pickle_framing_and_nonarray_bytes": path.stat().st_size
        - sum(r["array_bytes"] for r in rows),
    }


def json_inventory(path, outcome, artifact):
    """Measure actual text spans, including whitespace, without pretending JSON has a dtype."""
    raw = path.read_bytes()
    spans, stack = {}, []
    for token in re.finditer(rb'"(?:[^"\\]|\\.)*"|\[|\]', raw):
        if token[0] == b"[":
            stack.append(token.start())
        elif token[0] == b"]":
            spans[stack.pop()] = token.end()
    expression = (
        rb'"('
        + b"|".join(name.encode() for name in (*EVALUATION_ARRAYS, "values"))
        + rb')"\s*:\s*(\[)'
    )
    counters, rows = defaultdict(int), []
    clusters = [c for c in outcome.clusters if c.analytical_evaluation is not None]
    for match in re.finditer(expression, raw):
        name = match[1].decode()
        counters[name] += 1
        size = spans[match.start(2)] - match.start(2)
        if name == "values":
            array = np.asarray(json.loads(raw[match.start(2) : spans[match.start(2)]]))
            logical = ("z_axis", "plane_values")
        else:
            cluster = clusters[counters[name] - 1]
            array = getattr(cluster.analytical_evaluation, name)
            logical = ("clusters", str(cluster.cluster_id), "analytical_evaluation", name)
        rows.append(
            {
                "artifact": artifact,
                "group": purpose(logical),
                "name": ".".join(logical),
                "scientific_path": ".".join(logical),
                "shape": list(array.shape),
                "dtype": "JSON number/null (no stored dtype); source " + array.dtype.str,
                "array_bytes": array.nbytes,
                "stored_uncompressed_bytes": size,
                "compressed_bytes": size,
                "duplicate_of": "corresponding scientific.npz array",
                "compression": "none; exact numeric-array JSON text span, including indentation",
            }
        )
    assert all(counters[name] == len(clusters) for name in EVALUATION_ARRAYS)
    return rows, {
        "bytes": len(raw),
        "numeric_array_text_bytes": sum(r["stored_uncompressed_bytes"] for r in rows),
        "metadata_and_parameter_text_bytes": len(raw)
        - sum(r["stored_uncompressed_bytes"] for r in rows),
    }


def directory_sizes(path):
    """Count logical file bytes, excluding filesystem allocation blocks and directory entries."""
    files = {
        str(p.relative_to(path)): p.stat().st_size for p in sorted(path.rglob("*")) if p.is_file()
    }
    return {"bytes": sum(files.values()), "files": files}


def subset_size(source, target, removed):
    """Size an audit-only retained-field projection at unchanged compression settings.

    This is NOT a supported PeakFit schema or a new production writer/reader.
    """
    with ZipFile(source) as archive:
        document = json.loads(archive.read("metadata.json"))
        document["kind"] = "audit_only_retention_estimate_not_peakfit"
        for cluster in document["outcome"]["clusters"]:
            evaluation = cluster["analytical_evaluation"]
            if evaluation:
                for name in removed:
                    evaluation.pop(name, None)
        for sample in document["samples"]:
            for name in removed:
                sample.pop(name, None)
        names = {d["array"] + ".npy" for _, d in references(document)}
        with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=1) as out:
            out.writestr("metadata.json", json.dumps(document, separators=(",", ":")))
            for name in sorted(names):
                with out.open(name, "w", force_zip64=True) as stream:
                    stream.write(archive.read(name))
    return {
        "bytes": target.stat().st_size,
        "retained_array_members": len(names),
        "removed_fields": sorted(removed),
        "supported_production_format": False,
    }


def reconstruction_checks(result, spectra):
    """Test algebra and regenerated arrays without refitting or re-solving amplitudes."""
    parameters = Parameters()
    for parameter in result.outcome.final_nonlinear_parameters:
        parameters.add(
            parameter.name,
            value=parameter.value,
            min_value=parameter.min,
            max_value=parameter.max,
            vary=parameter.vary,
        )
    checks = {}
    all_peaks = [p for c in result.simulation_snapshot.clusters for p in c.peaks]
    amplitude_rows = np.concatenate(
        [c.analytical_evaluation.amplitudes for c in result.outcome.clusters]
    )
    for sample, geometry, terminal in zip(
        result.completed_fit.samples,
        result.simulation_snapshot.clusters,
        result.outcome.clusters,
        strict=True,
    ):
        e = terminal.analytical_evaluation
        assert e is not None
        shape = Cluster.evaluate_peaks(geometry.peaks, parameters, list(sample.grid_indices))
        source_data = np.asarray(
            spectra.data[(slice(None), *sample.grid_indices)].T, dtype=sample.data.dtype
        )
        other = [i for i, p in enumerate(all_peaks) if p.name not in terminal.peak_names]
        neighbor = (
            Cluster.evaluate_peaks(
                [all_peaks[i] for i in other], parameters, list(sample.grid_indices)
            ).T
            @ amplitude_rows[other]
        )
        reconstructed = {
            "observations_from_verified_source": (sample.data, source_data),
            "shapes_from_descriptors_and_final_parameters": (e.shapes, shape),
            "model_from_stored_shapes_and_amplitudes": (e.model_values, e.shapes.T @ e.amplitudes),
            "model_from_regenerated_shapes_and_amplitudes": (
                e.model_values,
                shape.T @ e.amplitudes,
            ),
            "raw_residuals_from_inputs_and_stored_model": (
                e.raw_residuals,
                sample.data - sample.neighbor_corrections - e.model_values,
            ),
            "normalized_residuals_from_stored_raw_and_noise": (
                e.normalized_residuals,
                e.raw_residuals.ravel() / result.outcome.noise,
            ),
            "conditional_errors_from_covariance": (
                e.amplitude_standard_errors,
                np.sqrt(np.diag(e.amplitude_covariance)),
            ),
            "scaled_errors_from_scale": (
                e.scaled_amplitude_standard_errors,
                e.amplitude_standard_errors * e.statistics.amplitude_uncertainty_scale,
            ),
            "corrections_from_final_neighbor_models_not_equivalent_parent": (
                sample.neighbor_corrections,
                neighbor,
            ),
            "source_float32_round_trip_to_retained_dtype": (
                sample.data,
                sample.data.astype(np.float32).astype(sample.data.dtype),
            ),
        }
        for name, (old, new) in reconstructed.items():
            entry = checks.setdefault(
                name,
                {
                    "clusters": 0,
                    "bit_exact_clusters": 0,
                    "max_abs_difference": 0.0,
                    "squared_difference": 0.0,
                    "squared_reference": 0.0,
                },
            )
            entry["clusters"] += 1
            # Compare canonical C-order bytes, not incidental array strides.
            entry["bit_exact_clusters"] += int(
                old.dtype == new.dtype and old.shape == new.shape and old.tobytes() == new.tobytes()
            )
            finite = np.isfinite(old) & np.isfinite(new)
            if np.any(finite):
                delta = old[finite] - new[finite]
                entry["max_abs_difference"] = max(
                    entry["max_abs_difference"], float(np.max(np.abs(delta)))
                )
                entry["squared_difference"] += float(np.sum(delta**2))
                entry["squared_reference"] += float(np.sum(old[finite] ** 2))
    for entry in checks.values():
        entry["relative_l2_difference"] = math.sqrt(
            entry.pop("squared_difference") / max(entry.pop("squared_reference"), 1e-300)
        )
    return checks


def main():  # noqa: PLR0915
    """Audit real writers and metadata/array payloads without modifying scientific code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from peakfit.io.scientific import VERSION  # noqa: PLC0415

    if VERSION != "1.0.0":
        raise SystemExit(
            "This is the historical audit driver. Run it at 1051ceb to reproduce that audit; "
            "use tools/validate_storage_policy.py for the implemented retention policy."
        )
    data = ROOT / "examples/02-advanced-fitting/data"
    input_paths = {
        "spectrum": data / "pseudo3d.ft2",
        "peaklist": data / "pseudo3d.list",
        "z_values": data / "b1_offsets.txt",
    }
    dataset = read_dataset(*input_paths.values())
    config = PeakFitConfig()
    config.fitting.refine_iterations = 2
    prepared = prepare_dataset_fit(dataset.spectra, dataset.peaks, config)
    prepared = replace(prepared, snapshot=replace(prepared.snapshot, source_files=dataset.sources))
    report: dict[str, Any] = {
        "revision": "335f569",
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "spectrum_sha256": json.loads(dict(dataset.sources)["spectrum"])["checksum_sha256"],
        "units": "bytes; MB means 1000000 bytes",
        "source_bytes": dataset.path.stat().st_size,
        "shape": list(dataset.spectra.data.shape),
        "scenarios": {},
        "archives": {},
    }
    inventory: list[dict[str, Any]] = []
    with (
        TemporaryDirectory(prefix="peakfit-storage-audit-") as temporary,
        threadpool_limits(limits=1),
    ):
        root = Path(temporary)
        start = perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            pipeline = run_pipeline(
                config,
                prepared.clusters,
                prepared.noise,
                prepared.parameters,
                [p for c in prepared.clusters for p in c.peaks],
            )
        result = finish_dataset_fit(
            prepared,
            DatasetFitAttempt(
                pipeline.completed_fit,
                tuple(dict.fromkeys(str(w.message) for w in captured)),
                perf_counter() - start,
            ),
        )
        completion = dataset_completion(
            result, prepared.snapshot, submitted_revision=0, manual_revision=0
        )
        report["fitted_spectral_sample_points"] = sum(
            s.data.shape[0] for s in result.completed_fit.samples
        )
        report["fitted_scalar_observations"] = sum(
            s.data.size for s in result.completed_fit.samples
        )
        report["clusters"] = len(result.outcome.clusters)
        report["peaks"] = sum(len(c.peak_names) for c in result.outcome.clusters)
        report["fitted_nonlinear_parameters"] = len(result.outcome.final_nonlinear_parameters)
        report["shapes"] = prepared.shape_names
        project = save_analysis(
            Analysis(
                None,
                dataset_header(dataset),
                completion.working_peaks,
                config.model_dump_json(),
                0,
                (completion,),
            ),
            root / "Analysis.peakfit",
        )
        report["scenarios"]["project_gui_two_pass"] = directory_sizes(project)
        identities = tuple(
            {"engine_name": i.engine_name, "peak_id": i.working_id, "assignment": i.assignment}
            for i in prepared.identities
        )
        for label, formats, simulated in (
            ("default", ["json", "csv"], False),
            ("no_json", ["csv"], False),
            ("binary_only", [], False),
            ("all_optional", ["json", "csv", "txt"], True),
        ):
            settings = config.model_copy(deep=True)
            settings.output = settings.output.model_validate(
                dict(settings.output.model_dump(), formats=formats, save_simulated=simulated)
            )
            changed = replace(
                result,
                inputs=replace(
                    prepared,
                    snapshot=replace(prepared.snapshot, configuration=settings.model_dump_json()),
                ),
            )
            gui = export_fit_result(
                changed,
                root / ("gui_" + label),
                completion=replace(completion, result=changed),
                current=True,
            )
            cli = root / ("cli_matched_" + label)
            fit_run = FitRun(
                result.outcome,
                pipeline.continuation_state,
                cli,
                dataset.spectra,
                result.simulation_snapshot,
                result.completed_fit,
                identities,
            )
            write_fit_run_outputs(fit_run, dataset.spectra, settings, input_paths)
            report["scenarios"]["gui_" + label] = directory_sizes(gui)
            report["scenarios"]["cli_matched_two_pass_" + label] = directory_sizes(cli)
        # Actual CLI defaults differ from GUI default: one pass, hence zero neighbor corrections.
        cli_config = PeakFitConfig()
        loaded = load_data(
            input_paths["spectrum"], input_paths["peaklist"], input_paths["z_values"], cli_config
        )
        default_run = run_fit(loaded, cli_config, root / "cli_actual_default", workers=1)
        write_fit_run_outputs(default_run, loaded.spectra, cli_config, input_paths)
        report["scenarios"]["cli_actual_default_one_pass"] = directory_sizes(default_run.output_dir)
        artifacts = {
            "project": next(project.rglob("result.npz")),
            "gui_export": root / "gui_default/scientific.npz",
            "cli_matched_export": root / "cli_matched_default/scientific.npz",
            "cli_actual_default": default_run.output_dir / "scientific.npz",
        }
        for label, path in artifacts.items():
            rows, summary = archive_inventory(path, label)
            inventory.extend(rows)
            report["archives"][label] = summary
        for label, path, outcome in (
            ("gui_json", root / "gui_default/summary/fit.json", result.outcome),
            ("cli_matched_json", root / "cli_matched_default/summary/fit.json", result.outcome),
            ("cli_default_json", default_run.output_dir / "summary/fit.json", default_run.outcome),
        ):
            rows, summary = json_inventory(path, outcome, label)
            inventory.extend(rows)
            report["archives"][label] = summary
        for label, path in (
            ("cli_matched_pickle", root / "cli_matched_default/metadata/fitting_state.pkl"),
            ("cli_default_pickle", default_run.output_dir / "metadata/fitting_state.pkl"),
        ):
            rows, summary = state_inventory(path, label)
            inventory.extend(rows)
            report["archives"][label] = summary
        simulation_path = root / "gui_all_optional/simulated.npy"
        simulated = np.load(simulation_path, mmap_mode="r", allow_pickle=False)
        inventory.append(
            {
                "artifact": "simulation",
                "group": "derived full-grid simulation",
                "name": "simulated.npy",
                "scientific_path": "full_grid_model",
                "shape": list(simulated.shape),
                "dtype": simulated.dtype.str,
                "array_bytes": simulated.nbytes,
                "stored_uncompressed_bytes": simulation_path.stat().st_size,
                "compressed_bytes": simulation_path.stat().st_size,
                "duplicate_of": "forward evaluation from fitted parameters/amplitudes/descriptors",
                "compression": "none; includes NPY header",
            }
        )
        report["reconstruction_checks"] = reconstruction_checks(result, dataset.spectra)
        replayed = load_state(root / "cli_matched_default/metadata/fitting_state.pkl")
        report["legacy_continuation_replay"] = {
            "before_nonzero_corrections": sum(
                int(np.count_nonzero(c.corrections)) for c in pipeline.continuation_state.clusters
            ),
            "after_nonzero_corrections": sum(
                int(np.count_nonzero(c.corrections)) for c in replayed.clusters
            ),
            "max_corrected_data_difference": max(
                float(np.max(np.abs(a.corrected_data - b.corrected_data)))
                for a, b in zip(
                    pipeline.continuation_state.clusters, replayed.clusters, strict=True
                )
            ),
        }
        report["retention_estimates"] = {}
        report["project_retention_estimates"] = {}
        for label, removed in {
            "compact_results": DENSE_DERIVED | {"data", "neighbor_corrections"},
            "linked_exact_problem": DENSE_DERIVED | {"data"},
            "portable_exact_problem": DENSE_DERIVED,
            "linked_problem_with_shape_reference": (DENSE_DERIVED - {"shapes"}) | {"data"},
            "portable_problem_with_shape_reference": DENSE_DERIVED - {"shapes"},
            "exact_sample_models": {"raw_residuals", "normalized_residuals"},
            "all_evidence": set(),
        }.items():
            report["retention_estimates"][label] = subset_size(
                artifacts["gui_export"], root / (label + ".npz"), removed
            )
            project_subset = subset_size(
                artifacts["project"], root / ("project_" + label + ".npz"), removed
            )
            project_subset["total_project_bytes_with_current_manifest_review_budget"] = (
                project_subset["bytes"]
                + report["scenarios"]["project_gui_two_pass"]["bytes"]
                - report["archives"]["project"]["bytes"]
            )
            report["project_retention_estimates"][label] = project_subset
        for artifact in ("gui_export", "cli_matched_export", "cli_actual_default"):
            scientific = artifacts[artifact]
            report.setdefault("compact_by_interface", {})[artifact] = subset_size(
                scientific,
                root / (artifact + "_compact.npz"),
                DENSE_DERIVED | {"data", "neighbor_corrections"},
            )
        with (
            ZipFile(root / "gui_default/scientific.npz") as gui_archive,
            ZipFile(root / "cli_matched_default/scientific.npz") as cli_archive,
        ):
            members = {n for n in gui_archive.namelist() if n.startswith(("outcome/", "samples/"))}
            assert members == {
                n for n in cli_archive.namelist() if n.startswith(("outcome/", "samples/"))
            }
            assert all(gui_archive.read(n) == cli_archive.read(n) for n in members)
            assert (
                json.loads(gui_archive.read("metadata.json"))["outcome"]
                == json.loads(cli_archive.read("metadata.json"))["outcome"]
            )
        csv_names = ["parameters", "intensities", "clusters", "shifts"]
        assert all(
            (root / f"gui_default/tables/{name}.csv").read_bytes()
            == (root / f"cli_matched_default/tables/{name}.csv").read_bytes()
            for name in csv_names
        )

        def digest(path):
            with path.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest()

        assert digest(root / "gui_all_optional/simulated.npy") == digest(
            root / "cli_matched_all_optional/simulated.npy"
        )
        report["parity_checks"] = {
            "same_completed_two_pass_result": True,
            "outcome_and_samples_npy_members_byte_equal": len(members),
            "outcome_metadata_equal": True,
            "csv_byte_equal": csv_names,
            "optional_simulation_byte_equal": True,
        }
        report["temporary_payloads_removed_on_exit"] = True
    groups = {}
    for row in inventory:
        key = row["artifact"] + " / " + row["group"]
        entry = groups.setdefault(
            key,
            {"arrays": 0, "array_bytes": 0, "stored_uncompressed_bytes": 0, "compressed_bytes": 0},
        )
        for name in entry:
            entry[name] += 1 if name == "arrays" else row[name]
    report["array_groups"] = groups
    (args.output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    # Keep the full per-array inventory without adding multi-MB text copies to the repo.
    with (
        (args.output / "arrays.csv.gz").open("wb") as target,
        gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as zipped,
        TextIOWrapper(zipped, encoding="utf-8", newline="") as stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(dict(row, shape=json.dumps(row["shape"])) for row in inventory)


if __name__ == "__main__":
    main()
