"""Two independent production fits; neither candidate is selected or ranked."""

from __future__ import annotations

import csv
import json
import math
import time
from copy import deepcopy
from dataclasses import replace
from itertools import combinations
from typing import TYPE_CHECKING, Any

from peakfit.engine.domain.constraints import (
    ConstraintResolver,
    ParameterConstraint,
    PeakConstraints,
)
from peakfit.engine.domain.fit_steps import build_fit_steps
from peakfit.fit.fitting import run_fit, write_fit_run_outputs
from peakfit.fit.output_metadata import capture_output_metadata
from peakfit.fit.pipeline import prepare_cluster_parameters

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.domain.config import PeakFitConfig
    from peakfit.fit.final_outcome import FinalClusterOutcome
    from peakfit.fit.run_models import FitRun, LoadedData
    from peakfit.shared.reporter import Reporter


CAUTION = (
    "Both candidates are diagnostic results; no best fit or assignment is selected. "
    "Lower chi-squared, smaller uncertainties or greater separation do not establish a correct "
    "assignment. Complete multi-pass runs may have different neighboring-cluster corrections; "
    "they are not necessarily the same isolated corrected-data objective. Signed amplitude "
    "profiles remain in each candidate's normal outputs."
)


def initial_parameters(data: LoadedData, config: PeakFitConfig) -> dict[str, Any]:
    """Capture exact first-pass task parameters, including step freedom and bounds."""
    step = build_fit_steps(config.fitting.steps, config.fitting.refine_iterations)[0]
    return {
        name: parameter
        for cluster in data.clusters
        for name, parameter in prepare_cluster_parameters(cluster, config, step).items()
    }


def alternative_config(
    data: LoadedData, config: PeakFitConfig, linewidth: float
) -> tuple[PeakFitConfig, list[dict[str, Any]]]:
    """Express eligible starts using existing per-peak constraints and precedence."""
    if not math.isfinite(linewidth) or linewidth <= 0:
        raise ValueError("Alternative linewidth start must be finite and positive (Hz).")
    alternative = config.model_copy(deep=True)
    resolver = ConstraintResolver(config.parameters)
    records = []
    for name, parameter in initial_parameters(data, config).items():
        identity = parameter.param_id
        if identity is None or identity.label != "lw":
            continue
        # A sentinel current value distinguishes any explicit value (even one
        # equal to the model default) using the existing resolver, including files.
        explicit = (
            resolver.resolve(name, identity.peak_name, identity.axis or "", "lw", -math.inf).value
            != -math.inf
        )
        reason = (
            "explicit start"
            if explicit
            else "fixed on first pass"
            if not parameter.vary
            else "alternative"
        )
        if reason == "alternative":
            if not parameter.min <= linewidth <= parameter.max:
                raise ValueError(
                    f"Alternative start {linewidth:g} Hz is outside {name} bounds "
                    f"[{parameter.min:g}, {parameter.max:g}]; choose a start within bounds."
                )
            peak = alternative.parameters.peaks.setdefault(identity.peak_name, PeakConstraints())
            key = f"{identity.axis}.lw"
            existing = peak.parameters.get(key, ParameterConstraint())
            peak.parameters[key] = existing.model_copy(update={"value": linewidth})
        records.append(
            {
                "parameter": name,
                "baseline_hz": parameter.value,
                "alternative_hz": linewidth if reason == "alternative" else parameter.value,
                "reason": reason,
                "min_hz": parameter.min if math.isfinite(parameter.min) else None,
                "max_hz": parameter.max if math.isfinite(parameter.max) else None,
            }
        )
    if not any(r["reason"] == "alternative" and r["baseline_hz"] != linewidth for r in records):
        raise ValueError(
            "Alternative start is ineffective: no freely initialized first-pass linewidth "
            "would change. Inspect explicit starts, fixed parameters and fitting steps."
        )
    return alternative, records


def _candidate_rows(run: FitRun, candidate: str, baseline: FitRun | None) -> list[dict[str, Any]]:
    if baseline is not None and set(run.outcome.by_cluster_id) != set(
        baseline.outcome.by_cluster_id
    ):
        raise ValueError("Candidate cluster identities disagree")
    rows = []
    for cluster in sorted(run.outcome.clusters, key=lambda item: item.cluster_id):
        evaluation = cluster.analytical_evaluation
        reference = baseline.outcome.by_cluster_id.get(cluster.cluster_id) if baseline else None
        if reference is not None and set(reference.peak_names) != set(cluster.peak_names):
            raise ValueError(f"Peak identities disagree for cluster {cluster.cluster_id}")
        previous = (
            {p.name: p.value for p in reference.final_nonlinear_parameters} if reference else {}
        )
        parameters = [
            p for p in cluster.final_nonlinear_parameters if p.name.endswith((".cs", ".lw"))
        ]
        # An unusable cluster must still have a visible row, without invented values.
        for parameter in parameters or [None]:
            rows.append(  # noqa: PERF401 - each row represents one retained parameter
                {
                    "cluster_id": cluster.cluster_id,
                    "peak_identities": "; ".join(sorted(cluster.peak_names)),
                    "candidate": candidate,
                    "classification": cluster.classification.value,
                    "termination": cluster.optimizer_provenance.termination_message,
                    "unusable_reason": cluster.unusable_reason,
                    "chi_squared": evaluation.statistics.chi_squared if evaluation else None,
                    "reduced_chi_squared": evaluation.statistics.reduced_chi_squared
                    if evaluation
                    else None,
                    "optimality": cluster.optimizer_provenance.optimality,
                    "uncertainty_warning": evaluation.uncertainty_warning if evaluation else None,
                    "unavailable_parameter_uncertainties": any(
                        p.vary and not math.isfinite(p.standard_error)
                        for p in cluster.final_nonlinear_parameters
                    )
                    if evaluation
                    else None,
                    "parameter": parameter.name if parameter else None,
                    "value": parameter.value if parameter else None,
                    "unit": parameter.unit if parameter else None,
                    "standard_error": parameter.standard_error
                    if parameter and math.isfinite(parameter.standard_error)
                    else None,
                    "pair_separations_ppm": _pair_separations(cluster),
                    "position_delta_ppm": (
                        parameter.value - previous[parameter.name]
                        if parameter
                        and parameter.name.endswith(".cs")
                        and parameter.name in previous
                        else None
                    ),
                }
            )
    return rows


def _pair_separations(cluster: FinalClusterOutcome) -> str:
    """Display measured separations with pair and axis identities; no thresholds."""
    shifts = {p.name: p.value for p in cluster.final_nonlinear_parameters if p.name.endswith(".cs")}
    axes = sorted({name.rsplit(".", 2)[-2] for name in shifts})
    pairs = []
    for first, second in combinations(sorted(cluster.peak_names), 2):
        values = [
            f"{axis}={abs(shifts[f'{first}.{axis}.cs'] - shifts[f'{second}.{axis}.cs']):.6g}"
            for axis in axes
            if f"{first}.{axis}.cs" in shifts and f"{second}.{axis}.cs" in shifts
        ]
        if values:
            pairs.append(f"{first}/{second}: {', '.join(values)}")
    return "; ".join(pairs)


def run_linewidth_comparison(
    data: LoadedData,
    config: PeakFitConfig,
    output_dir: Path,
    linewidth: float,
    input_paths: dict[str, Path],
    *,
    optimizer: str = "varpro",
    workers: int = -1,
    reporter: Reporter | None = None,
) -> bool:
    """Retain independent runs and failure provenance in a new, reserved directory.

    Returns false if either run or its output failed. Completed scientific outputs
    of the other candidate remain available. A started manifest marks interrupted runs.
    """
    alternative, initialization = alternative_config(data, config, linewidth)
    if reporter:
        changed = sum(r["reason"] == "alternative" for r in initialization)
        reporter.info(
            f"Comparing {linewidth:g} Hz starts for {changed} linewidths; "
            f"preserving {len(initialization) - changed} explicit or fixed starts."
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Comparison output must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = dict(input_paths)
    if config.parameters.from_file is not None:
        input_paths["parameter_starts"] = config.parameters.from_file
    metadata = capture_output_metadata(config.model_dump(), input_paths)
    manifest: dict[str, Any] = {
        "input_files": metadata.input_files,
        "git_commit": metadata.git_commit,
        "status": "started",
        "requested_alternative_hz": linewidth,
        "initialization": initialization,
        "caution": CAUTION,
        "candidates": {},
    }
    manifest_path = output_dir / "comparison.json"

    def save_manifest() -> None:
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(manifest_path)

    # Exclusive reservation also protects against concurrent invocations.
    with (output_dir / ".comparison-started").open("x"):
        pass
    runs: dict[str, FitRun] = {}
    candidates = (("baseline", config), (f"alternative-{linewidth:g}hz", alternative))
    for name, settings in candidates:
        status: dict[str, Any] = {"status": "started", "directory": name}
        manifest["candidates"][name] = status
        save_manifest()
        started = time.perf_counter()
        try:
            # Preserve shared peak identities inside the copy; never share mutable
            # clusters, corrections or peak parameters between candidates.
            peaks, clusters = deepcopy((data.peaks, data.clusters))
            independent = replace(data, peaks=peaks, clusters=clusters)
            candidate_config = settings.model_copy(deep=True)
            candidate_config.output.directory = output_dir / name
            candidate_config.output.include_timestamp = False
            for output_format in ("json", "csv"):
                if output_format not in candidate_config.output.formats:
                    candidate_config.output.formats.append(output_format)
            status["first_pass_parameters"] = {
                key: {
                    "value": p.value,
                    "min": p.min if math.isfinite(p.min) else None,
                    "max": p.max if math.isfinite(p.max) else None,
                    "vary": p.vary,
                }
                for key, p in initial_parameters(independent, candidate_config).items()
                if not p.computed
            }
            save_manifest()
            run = run_fit(
                independent,
                candidate_config,
                output_dir / name,
                optimizer=optimizer,
                workers=workers,
                reporter=reporter,
            )
            runs[name] = run
            write_fit_run_outputs(run, data.spectra, candidate_config, input_paths, reporter)
            status["status"] = "complete"
        except Exception as exc:  # retain the other candidate even after numerical/output failure
            status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if reporter:
                reporter.warning(f"{name} failed: {exc}")
        status["seconds"] = time.perf_counter() - started
        save_manifest()
    _write_comparison_report(output_dir, runs, manifest, data)
    complete = all(status["status"] == "complete" for status in manifest["candidates"].values())
    manifest["status"] = "complete" if complete else "failed"
    save_manifest()
    return complete


def _write_comparison_report(
    output_dir: Path, runs: dict[str, FitRun], manifest: dict[str, Any], data: LoadedData
) -> None:
    """Project candidate outcomes without inferring an assignment."""
    rows = [
        row
        for name, run in runs.items()
        for row in _candidate_rows(run, name, runs.get("baseline"))
    ]
    for name, status in manifest["candidates"].items():
        if name not in runs:
            for cluster in data.clusters:
                row = dict.fromkeys(
                    rows[0]
                    if rows
                    else (
                        "cluster_id",
                        "peak_identities",
                        "candidate",
                        "classification",
                        "termination",
                        "unusable_reason",
                        "chi_squared",
                        "reduced_chi_squared",
                        "optimality",
                        "uncertainty_warning",
                        "unavailable_parameter_uncertainties",
                        "parameter",
                        "value",
                        "unit",
                        "standard_error",
                        "position_delta_ppm",
                    )
                )
                row.update(
                    cluster_id=cluster.cluster_id,
                    peak_identities="; ".join(sorted(p.name for p in cluster.peaks)),
                    candidate=name,
                    classification="run_failed",
                    unusable_reason=status.get("error", "Run did not complete"),
                )
                rows.append(row)
    for row in rows:
        row["execution_status"] = manifest["candidates"][row["candidate"]]["status"]
    if rows:
        with (output_dir / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Linewidth-start comparison", "", CAUTION, "", "## Candidate execution", ""]
    for name, status in manifest["candidates"].items():
        lines.append(
            f"- {name}: {status['status']} ({status['seconds']:.3f} s). {status.get('error', '')}"
        )
    lines.extend(
        [
            "",
            "Exact first-pass starts and bounds: comparison.json. Positions, linewidths, "
            "uncertainties and signed profiles: candidate outputs. "
            "Position differences: comparison.csv.",
            "",
            "| Cluster / peaks | Candidate | Classification | χ² | Reduced χ² | Optimality | "
            "Pair separations (ppm) | Termination / warnings |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    seen = set()
    for row in rows:
        key = (row["cluster_id"], row["candidate"])
        if key in seen:
            continue
        seen.add(key)
        warning = "; ".join(
            str(row[k]) for k in ("termination", "uncertainty_warning", "unusable_reason") if row[k]
        )
        if row["unavailable_parameter_uncertainties"]:
            warning += "; parameter uncertainties unavailable"
        lines.append(
            f"| {row['cluster_id']} / {row['peak_identities']} | {row['candidate']} | "
            f"{row['classification']} | {row['chi_squared']} | {row['reduced_chi_squared']} | "
            f"{row['optimality']} | {row.get('pair_separations_ppm') or ''} | "
            f"{warning.replace('|', '/')} |"
        )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
