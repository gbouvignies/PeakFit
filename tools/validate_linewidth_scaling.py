"""Investigate intrinsic linewidth starts and VarPro scaling without runtime changes.

Run with one BLAS thread: uv run python tools/validate_linewidth_scaling.py
Results include complete target amplitudes and all per-peak selection decisions.
The temporary SciPy override is process-local, sequential, and always restored.
"""

# Numerical experiment thresholds and sequential experiment driver.
# ruff: noqa: PLR2004, PLR0915
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
from validate_overlap import ROOT, TARGETS, diagnostics

from peakfit.engine.algorithms.common import update_cluster_corrections
from peakfit.engine.algorithms.evaluation import (
    AnalyticalModelEvaluation,
    evaluate_analytical_model,
)
from peakfit.engine.algorithms.varpro import VarProOptimizer, fit_cluster
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.params_vector import FitParameters
from peakfit.fit.fitting import load_data

PAIR = {269: ("36N-H", "93N-H"), 360: ("5N-H", "76N-H"), 424: ("53N-H", "147N-H")}
STARTS = (3.0, 20 / np.pi, 10.0, 20.0, 40.0)


def physical_scale(params, obs):
    """Use 10 Hz center/width changes and 10 degree phase changes, never |ppm|."""
    scales = []
    for name in params.get_vary_names():
        pid = params[name].param_id
        label, axis = pid.label, pid.axis
        if label == "cs":
            scales.append(10.0 / obs[axis])
        elif label in {"lw", "j", "phase"}:
            scales.append(10.0)
        else:
            raise ValueError(f"No physical scale specified for {name}")
    return np.asarray(scales)


def set_widths(params, widths):
    """Change only varied linewidth starts, respecting user bounds."""
    for parameter in params.values():
        pid = parameter.param_id
        if parameter.vary and pid.label == "lw":
            parameter.value = float(np.clip(widths[pid.axis], parameter.min, parameter.max))


def solve(cluster, params, noise, scaling, obs):
    """Call production fit_cluster with only the requested SciPy x_scale override."""
    original = scipy.optimize.least_squares
    captured = {}

    def least_squares(*args, **kwargs):
        if scaling != "standard":
            kwargs["x_scale"] = "jac" if scaling == "jac" else physical_scale(params, obs)
        result = original(*args, **kwargs)
        captured["status"] = int(result.status)
        captured["active_mask"] = result.active_mask.tolist()
        return result

    started = time.perf_counter()
    with patch("scipy.optimize.least_squares", least_squares):
        result = fit_cluster(params, cluster, noise)
    return result, time.perf_counter() - started, captured


def record(cluster, result, initial, noise, seconds, captured):
    """Reuse PR #59 actual-model, conditioning and finite-difference diagnostics."""
    evaluation = evaluate_analytical_model(cluster, result.params, noise)
    assert isinstance(evaluation, AnalyticalModelEvaluation)
    peak_names = [peak.name for peak in cluster.peaks]
    row = {
        "cluster_id": cluster.cluster_id,
        "peaks": [peak.name for peak in cluster.peaks],
        "data_shape": list(cluster.data.shape),
        "mask_sha256": hashlib.sha256(np.asarray(cluster.grid_indices).tobytes()).hexdigest(),
        "correction_norm": float(np.linalg.norm(cluster.corrections)),
        "success": bool(result.success),
        "termination": result.message,
        "optimality": float(result.optimality),
        "nfev": result.nfev,
        "njev": result.njev,
        "seconds": seconds,
        "warning": evaluation.uncertainty_warning,
        "amplitude_standard_errors": [
            float(v) if np.isfinite(v) else None for v in evaluation.amplitude_standard_errors
        ],
        "scaled_amplitude_standard_errors": [
            float(v) if np.isfinite(v) else None
            for v in evaluation.scaled_amplitude_standard_errors
        ],
        "parameters": {
            name: {
                "start": initial[name],
                "value": p.value,
                "error": float(p.stderr) if np.isfinite(p.stderr) else None,
                "bounds": [p.min, p.max],
            }
            for name, p in result.params.items()
            if name in initial
        },
        **captured,
        **diagnostics(cluster, result.params, noise),
    }
    if cluster.cluster_id in PAIR:
        a, b = PAIR[cluster.cluster_id]
        row["pair_separation_ppm"] = {
            axis: abs(result.params[f"{a}.{axis}.cs"].value - result.params[f"{b}.{axis}.cs"].value)
            for axis in ("F2", "F3")
        }
        ai, bi = peak_names.index(a), peak_names.index(b)
        amps = np.asarray(row["amplitudes"])
        row["pair_opposite_sign_planes"] = int(np.sum(amps[ai] * amps[bi] < 0))
    return row


def full_run(data, widths, scaling, obs, *, detailed=True):
    """Fit all groups twice; update neighbor corrections only between passes."""
    parameters = Parameters.from_peaks(data.peaks)
    for cluster in data.clusters:
        cluster.corrections.fill(0)
    rows, elapsed, endpoints = [], [], {}
    for iteration in range(2):
        if iteration:
            update_cluster_corrections(parameters, data.clusters)
        for cluster in data.clusters:
            params = Parameters.from_peaks(cluster.peaks)
            set_widths(params, widths)
            if iteration:
                for name in params:
                    params[name].value = parameters[name].value
            initial = {name: p.value for name, p in params.items()}
            result, seconds, captured = solve(cluster, params, data.noise, scaling, obs)
            elapsed.append(seconds)
            parameters.update(result.params)
            endpoints[cluster.cluster_id] = result
            if cluster.cluster_id in TARGETS and detailed:
                row = record(cluster, result, initial, data.noise, seconds, captured)
                row["pass"] = iteration + 1
                rows.append(row)
    return (
        {"widths": widths, "scaling": scaling, "optimizer_seconds": sum(elapsed), "groups": rows},
        parameters,
        endpoints,
    )


def summarize(values):
    """Describe a distribution without hiding its range or tails."""
    values = np.asarray(values)
    return dict(
        zip(
            ("min", "q1", "median", "q3", "max"),
            np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist(),
            strict=True,
        )
    )


def isolated_widths(data, parameters, endpoints, obs):
    """Screen every singleton with explicit overlap, signal and diagnostic cuts."""
    # Compare shapes over the same union of all fitting masks. Comparing tails
    # normalized on only a tiny singleton mask falsely labels remote peaks as overlapping.
    grid = np.unique(
        np.concatenate([np.asarray(c.grid_indices).T for c in data.clusters]), axis=0
    ).T
    all_shapes = Cluster.evaluate_peaks(data.peaks, parameters, list(grid))
    normalized = all_shapes / np.linalg.norm(all_shapes, axis=1)[:, None]
    overlap = np.abs(normalized @ normalized.T)
    np.fill_diagonal(overlap, 0)
    peak_index = {p.name: i for i, p in enumerate(data.peaks)}
    candidates = []
    for cluster in data.clusters:
        if len(cluster.peaks) != 1:
            continue
        result = endpoints[cluster.cluster_id]
        params = result.params
        peak = cluster.peaks[0]
        shape = cluster.evaluate(params)[0]
        correlations = overlap[peak_index[peak.name]]
        amplitudes = np.linalg.lstsq(shape[:, None], cluster.corrected_data, rcond=None)[0]
        model = shape[:, None] @ amplitudes
        signal = np.max(np.abs(model), axis=0) / data.noise
        residual = (cluster.corrected_data - model) / data.noise
        redchi = float(
            np.sum(residual**2)
            / (cluster.data.size - cluster.n_series - len(params.get_vary_names()))
        )
        widths = {a: params[f"{peak.name}.{a}.lw"].value for a in obs}
        errors = {a: params[f"{peak.name}.{a}.lw"].stderr for a in obs}
        correction_fraction = float(
            np.linalg.norm(cluster.corrections) / np.linalg.norm(cluster.data)
        )
        reasons = []
        if float(correlations.max()) >= 0.25:
            reasons.append("shape overlap >= 0.25")
        if correction_fraction >= 0.1:
            reasons.append("external correction >= 10%")
        if np.sum(signal >= 5) < 10:
            reasons.append("fewer than 10 planes at peak SNR >= 5")
        if redchi >= 2:
            reasons.append("reduced chi squared >= 2")
        if not result.success:
            reasons.append("optimizer unsuccessful")
        if any(not np.isfinite(errors[a]) or errors[a] / widths[a] >= 0.25 for a in obs):
            reasons.append("linewidth error unavailable or >= 25%")
        # Independently restart on exactly the same corrected data: a selection stability test.
        restart_widths = []
        for width in (3.0, 40.0):
            trial = Parameters.from_peaks(cluster.peaks)
            set_widths(trial, dict.fromkeys(obs, width))
            fitted, _, _ = solve(cluster, trial, data.noise, "physical", obs)
            restart_widths.append({a: fitted.params[f"{peak.name}.{a}.lw"].value for a in obs})
        spread = max(abs(w[a] - widths[a]) / widths[a] for w in restart_widths for a in obs)
        if spread >= 0.02:
            reasons.append("restart linewidth disagreement >= 2%")
        candidates.append(
            {
                "cluster_id": cluster.cluster_id,
                "peak": peak.name,
                "widths": widths,
                "errors": errors,
                "max_shape_correlation": float(correlations.max()),
                "correction_fraction": correction_fraction,
                "informative_planes": int(np.sum(signal >= 5)),
                "max_snr": float(signal.max()),
                "reduced_chi_squared": redchi,
                "restart_widths": restart_widths,
                "restart_relative_spread": spread,
                "selected": not reasons,
                "rejections": reasons,
            }
        )
    selected = [c for c in candidates if c["selected"]]
    if not selected:
        raise ValueError("No isolated peaks passed prespecified screening")
    summaries = {a: summarize([c["widths"][a] for c in selected]) for a in obs}
    sensitivity = {}
    for label, subset in {
        "all_singletons": candidates,
        "selected": selected,
        "selected_redchi_below_1.5": [c for c in selected if c["reduced_chi_squared"] < 1.5],
        "selected_overlap_below_0.1": [c for c in selected if c["max_shape_correlation"] < 0.1],
        "selected_snr_above_60": [c for c in selected if c["max_snr"] > 60],
    }.items():
        sensitivity[label] = {
            "n": len(subset),
            "widths": {a: summarize([c["widths"][a] for c in subset]) for a in obs}
            if subset
            else {},
        }
    return {
        "candidates": candidates,
        "n_selected": len(selected),
        "summaries": summaries,
        "sensitivity": sensitivity,
    }


def normalization_probe(cluster, params, noise):
    """Compare residual/Jacobian and column-normalized linear solve at fixed physics."""

    def optimizer(c, p, n):
        return VarProOptimizer(
            c, p.get_vary_names(), p, FitParameters.from_parameters(p, c.peaks), n
        )

    x = params.get_vary_values()
    original = optimizer(cluster, params.copy(deep=True), noise)
    r, j = original.compute_residuals(x), original.compute_jacobian(x)
    scale = 1 / np.max(np.abs(cluster.data))
    scaled = Cluster(cluster.cluster_id, cluster.peaks, cluster.grid_indices, cluster.data * scale)
    scaled.corrections = cluster.corrections * scale
    second = optimizer(scaled, params.copy(deep=True), noise * scale)
    shapes = cluster.evaluate(params)
    norms = np.linalg.norm(shapes, axis=1)
    amps = np.linalg.lstsq((shapes / norms[:, None]).T, cluster.corrected_data, rcond=None)[0]
    restored = amps / norms[:, None]
    original_model = shapes.T @ original.get_optimized_amplitudes()
    return {
        "intensity_scale": float(scale),
        "residual_relative_gap": float(
            np.linalg.norm(r - second.compute_residuals(x)) / np.linalg.norm(r)
        ),
        "jacobian_relative_gap": float(
            np.linalg.norm(j - second.compute_jacobian(x)) / np.linalg.norm(j)
        ),
        "column_scaled_model_relative_gap": float(
            np.linalg.norm(shapes.T @ restored - original_model) / np.linalg.norm(original_model)
        ),
        "column_scaled_amplitude_relative_gap": float(
            np.linalg.norm(restored - original.get_optimized_amplitudes())
            / np.linalg.norm(original.get_optimized_amplitudes())
        ),
        "intensity_scaled_amplitude_relative_gap": float(
            np.linalg.norm(
                second.get_optimized_amplitudes() / scale - original.get_optimized_amplitudes()
            )
            / np.linalg.norm(original.get_optimized_amplitudes())
        ),
        "shape_rank_before": int(np.linalg.matrix_rank(shapes)),
        "shape_rank_after": int(np.linalg.matrix_rank(shapes / norms[:, None])),
        "shape_condition_before": float(np.linalg.cond(shapes)),
        "shape_condition_after": float(np.linalg.cond(shapes / norms[:, None])),
    }


def synthetic_record(cluster, truth, result, obs):
    """Evaluate both named assignments and permutation-invariant peak recovery."""
    names = [p.name for p in cluster.peaks]

    def positions(p):
        return np.array([[p[f"{n}.{a}.cs"].value * obs[a] for a in obs] for n in names])

    true_positions = positions(truth)
    fitted_positions = positions(result.params)
    _, permutation = linear_sum_assignment(
        np.sum((true_positions[:, None] - fitted_positions[None, :]) ** 2, axis=2)
    )
    matched = fitted_positions[permutation] - true_positions
    standardized = []
    matched_width_errors = []
    standardized_widths = []
    for i, index in enumerate(permutation):
        for axis in obs:
            p = result.params[f"{names[index]}.{axis}.cs"]
            if np.isfinite(p.stderr):
                standardized.append(abs(p.value - truth[f"{names[i]}.{axis}.cs"].value) / p.stderr)
            width = result.params[f"{names[index]}.{axis}.lw"]
            width_error = abs(width.value - truth[f"{names[i]}.{axis}.lw"].value)
            matched_width_errors.append(width_error)
            if np.isfinite(width.stderr):
                standardized_widths.append(width_error / width.stderr)
    return {
        "matched_max_width_error_hz": float(max(matched_width_errors)),
        "width_errors_in_standard_errors": standardized_widths,
        "max_position_error_hz": float(np.max(np.abs(fitted_positions - true_positions))),
        "matched_max_position_error_hz": float(np.max(np.abs(matched))),
        "permutation": permutation.tolist(),
        "position_errors_in_standard_errors": standardized,
        "n_positions": int(matched.size),
        "parameters": {
            n: {
                "truth": p.value,
                "value": result.params[n].value,
                "error": float(result.params[n].stderr)
                if np.isfinite(result.params[n].stderr)
                else None,
            }
            for n, p in truth.items()
        },
    }


def synthetic_runs(data, obs, representative):
    """Acquisition-aware spectra spanning linewidths and independent signed profiles."""
    rows = []
    for cluster_id in sorted(TARGETS):
        source = next(c for c in data.clusters if c.cluster_id == cluster_id)
        for truth_factor in (0.65, 1.0, 1.6):
            truth = Parameters.from_peaks(source.peaks)
            for i, peak in enumerate(source.peaks):
                for axis in obs:
                    truth[f"{peak.name}.{axis}.lw"].value = (
                        representative[axis] * truth_factor * (0.85 + 0.15 * i)
                    )
            shapes = source.evaluate(truth)
            t = np.linspace(0, 2 * np.pi, source.n_series)
            amps = np.array(
                [np.cos(t + i) + 0.5 * np.sin((i + 1) * t) for i in range(len(source.peaks))]
            )
            amps *= 30 * data.noise / np.max(np.abs(shapes), axis=1)[:, None]
            clean = shapes.T @ amps
            for seed in range(5):
                synthetic = Cluster(
                    cluster_id,
                    source.peaks,
                    source.grid_indices,
                    clean + data.noise * np.random.default_rng(seed).normal(size=clean.shape),
                )
                for widths in (
                    dict.fromkeys(obs, 20.0),
                    representative,
                    dict.fromkeys(obs, 20 / np.pi),
                ):
                    for scaling in ("standard", "jac", "physical"):
                        params = Parameters.from_peaks(source.peaks)
                        set_widths(params, widths)
                        for n, p in params.items():
                            if n.endswith(".cs"):
                                p.value += 1 / obs[n.rsplit(".", 2)[1]]
                        result, seconds, _ = solve(synthetic, params, data.noise, scaling, obs)
                        rows.append(
                            {
                                "cluster_id": cluster_id,
                                "seed": seed,
                                "truth_factor": truth_factor,
                                "widths": widths,
                                "scaling": scaling,
                                "success": bool(result.success),
                                "chi_squared": float(result.residual @ result.residual),
                                "seconds": seconds,
                                "nfev": result.nfev,
                                "njev": result.njev,
                                "optimality": float(result.optimality),
                                **synthetic_record(synthetic, truth, result, obs),
                            }
                        )
    return rows


def write_results(payload, path):
    """Keep lossless numerical evidence below the repository's per-file size limit."""
    index = dict(payload)
    index["archives"] = {}
    for label, keys in {
        "fits": ("full_runs", "selection_controls"),
        "sweep": ("frozen_correction_sweep",),
        "synthetic": ("synthetic",),
    }.items():
        archive = path.with_name(f"{path.stem}-{label}.json.gz")
        content = {key: index.pop(key) for key in keys}
        encoded = json.dumps(content, separators=(",", ":"), allow_nan=False).encode()
        archive.write_bytes(gzip.compress(encoded, mtime=0))
        index["archives"][label] = archive.name
    path.write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")


def main():
    """Run the prespecified experiments and write all records, including exclusions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/validation/linewidth-scaling-results.json"
    )
    args = parser.parse_args()
    path = ROOT / "examples/02-advanced-fitting/data"
    with threadpool_limits(limits=1):
        data = load_data(
            path / "pseudo3d.ft2", path / "pseudo3d.list", path / "b1_offsets.txt", PeakFitConfig()
        )
        obs = {p.label: p.obs for p in data.spectra.spectral_params}
        baseline, parameters, endpoints = full_run(data, dict.fromkeys(obs, 20.0), "standard", obs)
        isolated_started = time.perf_counter()
        isolated = isolated_widths(data, parameters, endpoints, obs)
        isolated["screen_and_restart_seconds"] = time.perf_counter() - isolated_started
        representative = {a: s["median"] for a, s in isolated["summaries"].items()}
        runs = [baseline]
        for widths in (dict.fromkeys(obs, 20.0), representative, dict.fromkeys(obs, 20 / np.pi)):
            for scaling in ("standard", "jac", "physical"):
                if widths == baseline["widths"] and scaling == "standard":
                    continue
                run, _, _ = full_run(data, widths, scaling, obs)
                runs.append(run)
        selection_controls = {}
        for label, widths in {
            "rounded_per_axis": {"F2": 12.0, "F3": 14.0},
            "single_global_13": dict.fromkeys(obs, 13.0),
            "strict_residual_selection": {
                a: isolated["sensitivity"]["selected_redchi_below_1.5"]["widths"][a]["median"]
                for a in obs
            },
            "all_singletons": {
                a: isolated["sensitivity"]["all_singletons"]["widths"][a]["median"] for a in obs
            },
        }.items():
            run, _, _ = full_run(data, widths, "standard", obs)
            selection_controls[label] = run
        for run in runs:
            timings = [run["optimizer_seconds"]]
            for _ in range(2):
                repeat, _, _ = full_run(data, run["widths"], run["scaling"], obs, detailed=False)
                timings.append(repeat["optimizer_seconds"])
            run["optimizer_seconds_repeats"] = timings
        # Baseline corrections are frozen to isolate optimizer effects on identical data.
        # Full runs above separately measure propagation through external corrections.
        frozen = []
        normalizations = []
        for cluster in data.clusters:
            cluster.corrections.fill(0)
        update_cluster_corrections(parameters, data.clusters)
        for cluster in data.clusters:
            if cluster.cluster_id not in TARGETS:
                continue
            axis_controls = [
                {"F2": representative["F2"], "F3": 20.0},
                {"F2": 20.0, "F3": representative["F3"]},
                {a: w * 0.8 for a, w in representative.items()},
                {a: w * 1.2 for a, w in representative.items()},
            ]
            for width in (*STARTS, representative, *axis_controls):
                widths = width if isinstance(width, dict) else dict.fromkeys(obs, width)
                for scaling in ("standard", "jac", "physical"):
                    params = Parameters.from_peaks(cluster.peaks)
                    set_widths(params, widths)
                    initial = {n: p.value for n, p in params.items()}
                    result, seconds, captured = solve(cluster, params, data.noise, scaling, obs)
                    row = record(cluster, result, initial, data.noise, seconds, captured)
                    row.update(widths=widths, scaling=scaling)
                    frozen.append(row)
                    if width == 20.0:
                        probe = normalization_probe(cluster, result.params, data.noise)
                        probe.update(cluster_id=cluster.cluster_id, scaling=scaling)
                        # Optimize both representations from identical original starts.
                        scaled_cluster = Cluster(
                            cluster.cluster_id,
                            cluster.peaks,
                            cluster.grid_indices,
                            cluster.data * probe["intensity_scale"],
                        )
                        scaled_cluster.corrections = cluster.corrections * probe["intensity_scale"]
                        again = Parameters.from_peaks(cluster.peaks)
                        normalized, timing, _ = solve(
                            scaled_cluster,
                            again,
                            data.noise * probe["intensity_scale"],
                            scaling,
                            obs,
                        )
                        probe["optimized_chi_squared"] = float(
                            normalized.residual @ normalized.residual
                        )
                        probe["original_chi_squared"] = row["chi_squared_actual"]
                        probe["optimized_max_parameter_gap"] = float(
                            np.max(
                                np.abs(
                                    normalized.params.get_vary_values()
                                    - result.params.get_vary_values()
                                )
                            )
                        )
                        probe["normalized_seconds"] = timing
                        probe["original_seconds"] = seconds
                        probe["normalized_nfev"] = normalized.nfev
                        probe["original_nfev"] = result.nfev
                        probe["position_gap_hz"] = max(
                            abs(normalized.params[n].value - result.params[n].value)
                            * obs[n.rsplit(".", 2)[1]]
                            for n in initial
                            if n.endswith(".cs")
                        )
                        probe["linewidth_gap_hz"] = max(
                            abs(normalized.params[n].value - result.params[n].value)
                            for n in initial
                            if n.endswith(".lw")
                        )
                        normalizations.append(probe)
        synthetic = synthetic_runs(data, obs, representative)
    payload = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas_threads": 1,
        "noise": data.noise,
        "n_peaks": len(data.peaks),
        "n_clusters": len(data.clusters),
        "plane_values": data.spectra.z_values.tolist(),
        "input_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.iterdir())
            if p.is_file()
        },
        "axis_metadata": [
            {
                field: getattr(p, field)
                for field in ("label", "obs", "sw", "size", "aq_time", "apodq1", "apodq2", "apodq3")
            }
            for p in data.spectra.spectral_params
        ],
        "isolated": isolated,
        "representative": representative,
        "full_runs": runs,
        "selection_controls": selection_controls,
        "frozen_correction_sweep": frozen,
        "normalization": normalizations,
        "synthetic": synthetic,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_results(payload, args.output)


if __name__ == "__main__":
    main()
