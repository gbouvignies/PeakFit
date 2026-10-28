"""Reproduce overlapping experimental groups without changing their fitting masks.

Run with ``uv run python tools/validate_overlap.py --output /tmp/overlap.json``.
Use --source /path/to/checkout/src to run the same harness against pre-fix main.
--lmfit runs LMFit on current shapes, data, noise and explicit parameter bounds;
it requires LMFit in a separate validation environment, never a runtime dependency.
"""

from __future__ import annotations

# Imports follow --source selection deliberately when comparing source revisions.
# ruff: noqa: PLC0415
import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from threadpoolctl import threadpool_limits

TARGETS = {269, 360, 424}
ROOT = Path(__file__).resolve().parents[1]


def settings(params, scenario):
    """Vary one historical setting at a time, preserving the original mask."""
    for name, parameter in params.items():
        if name.endswith(".lw"):
            if scenario in ("start", "matched"):
                parameter.value = 20 / np.pi
            if scenario == "matched":
                parameter.min, parameter.max = 0.1 / np.pi, 200 / np.pi
        elif name.endswith(".cs") and scenario in ("bounds", "matched"):
            center = parameter.value
            parameter.min = center - (center - parameter.min) / 2
            parameter.max = center + (parameter.max - center) / 2


def solve(cluster, params, noise, use_lmfit):
    """Use the production solver or a controlled LMFit/lstsq reference."""
    from peakfit.engine.algorithms.varpro import fit_cluster

    if not use_lmfit:
        return fit_cluster(params, cluster, noise)
    lmfit = importlib.import_module("lmfit")

    from peakfit.engine.results import FitResult

    names = params.get_vary_names()
    reference = lmfit.Parameters()
    for index, name in enumerate(names):
        parameter = params[name]
        reference.add(f"p{index}", value=parameter.value, min=parameter.min, max=parameter.max)

    def residual(values):
        for index, name in enumerate(names):
            params[name].value = values[f"p{index}"].value
        shapes = cluster.evaluate(params)
        amplitudes = np.linalg.lstsq(shapes.T, cluster.corrected_data, rcond=None)[0]
        return ((cluster.corrected_data - shapes.T @ amplitudes) / noise).ravel()

    result = lmfit.Minimizer(residual, reference, scale_covar=False).least_squares()
    residual(result.params)
    for index, name in enumerate(names):
        error = result.params[f"p{index}"].stderr
        params[name].stderr = np.nan if error is None else error
    return FitResult(
        cluster_id=cluster.cluster_id,
        params=params,
        residual=result.residual,
        cost=result.chisqr / 2,
        success=result.success,
        message=result.message,
        optimality=result.optimality,
        nfev=result.nfev,
        n_amplitude_params=cluster.n_amplitude_params,
    )


def spectrum_metrics(matrix):
    """Record singular values without forming normal equations."""
    singular = np.linalg.svd(matrix, compute_uv=False)
    return {
        "singular_values": singular.tolist(),
        "rank": int(np.linalg.matrix_rank(matrix)),
        "condition": float(singular[0] / singular[-1]) if singular[-1] else None,
    }


def diagnostics(cluster, params, noise):
    """Compare model, residual, Jacobian and a rank-aware least-squares reference."""
    from peakfit.engine.algorithms.varpro import VarProOptimizer
    from peakfit.engine.domain.params_vector import FitParameters

    optimizer = VarProOptimizer(
        cluster,
        params.get_vary_names(),
        params,
        FitParameters.from_parameters(params, cluster.peaks),
        noise,
    )
    x = params.get_vary_values()
    residual = optimizer.compute_residuals(x)
    analytical = optimizer.compute_jacobian(x)
    shapes = cluster.evaluate(params)
    amplitudes = np.linalg.lstsq(shapes.T, cluster.corrected_data, rcond=None)[0]
    model = shapes.T @ amplitudes
    actual = ((cluster.corrected_data - model) / noise).ravel()
    # Sweep physical steps: differences at a collapsed solution otherwise cross
    # its tiny separation and incorrectly suggest an analytical derivative bug.
    errors = []
    for scale in (1e-6, 1e-7, 1e-8):
        steps = [scale if name.endswith(".cs") else scale * 1e3 for name in optimizer.names]
        numerical = approx_derivative(
            optimizer.compute_residuals, x, method="3-point", abs_step=steps
        )
        errors.append(float(np.linalg.norm(analytical - numerical) / np.linalg.norm(numerical)))
    params.set_vary_values(x)
    norms = np.linalg.norm(analytical, axis=0)
    return {
        "shape_matrix": spectrum_metrics(shapes.T),
        "jacobian": spectrum_metrics(analytical),
        "column_scaled_jacobian": spectrum_metrics(analytical / np.where(norms > 0, norms, 1)),
        "jacobian_finite_difference_relative_errors": errors,
        "residual_reference_gap": float(np.linalg.norm(residual - actual)),
        "chi_squared_actual": float(actual @ actual),
        "amplitudes": amplitudes.tolist(),
        "residual_rms_by_plane_in_noise_units": np.sqrt(
            np.mean(((cluster.corrected_data - model) / noise) ** 2, axis=0)
        ).tolist(),
        "relative_residual_norm": float(
            np.linalg.norm(actual * noise) / np.linalg.norm(cluster.corrected_data)
        ),
    }


def main():
    """Run the full dataset for timing/corrections and inspect all target groups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--lmfit", action="store_true")
    parser.add_argument(
        "--scenario", choices=("default", "start", "bounds", "matched"), default="default"
    )
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument(
        "--jitter-seed", type=int, help="Perturb initial shifts by at most 1e-4 ppm"
    )
    args = parser.parse_args()
    if args.source:
        sys.path.insert(0, str(args.source.resolve()))
    from peakfit.engine.algorithms.common import update_cluster_corrections
    from peakfit.engine.domain.config import PeakFitConfig
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.fit.fitting import load_data

    data_path = ROOT / "examples/02-advanced-fitting/data"
    with threadpool_limits(limits=1):
        data = load_data(
            data_path / "pseudo3d.ft2",
            data_path / "pseudo3d.list",
            data_path / "b1_offsets.txt",
            PeakFitConfig(),
        )
        parameters = Parameters.from_peaks(data.peaks)
        records = []
        elapsed = []
        for iteration in range(args.passes):
            if iteration:
                update_cluster_corrections(parameters, data.clusters)
            started = time.perf_counter()
            for cluster in data.clusters:
                params = Parameters.from_peaks(cluster.peaks)
                settings(params, args.scenario)
                if iteration == 0 and args.jitter_seed is not None:
                    rng = np.random.default_rng(args.jitter_seed)
                    for name, parameter in params.items():
                        if name.endswith(".cs"):
                            parameter.value += rng.uniform(-1e-4, 1e-4)
                if iteration:
                    for name in params:
                        params[name].value = parameters[name].value
                initial = {name: parameter.value for name, parameter in params.items()}
                result = solve(cluster, params, data.noise, args.lmfit)
                parameters.update(result.params)
                if cluster.cluster_id in TARGETS:
                    record = {
                        "pass": iteration + 1,
                        "cluster_id": cluster.cluster_id,
                        "peaks": [peak.name for peak in cluster.peaks],
                        "data_shape": list(cluster.data.shape),
                        "mask_sha256": hashlib.sha256(
                            np.asarray(cluster.grid_indices).tobytes()
                        ).hexdigest(),
                        "grid_indices": [indices.tolist() for indices in cluster.grid_indices],
                        "correction_norm": float(np.linalg.norm(cluster.corrections)),
                        "success": bool(result.success),
                        "termination": result.message,
                        "optimality": result.optimality,
                        "nfev": result.nfev,
                        "chi_squared_optimizer": float(result.residual @ result.residual),
                        "parameters": {
                            name: {
                                "start": initial[name],
                                "value": p.value,
                                "error": p.stderr if np.isfinite(p.stderr) else None,
                                "bounds": [p.min, p.max],
                            }
                            for name, p in result.params.items()
                            if name in initial
                        },
                    }
                    # Diagnostics are excluded from the full-dataset timing.
                    elapsed.append(time.perf_counter() - started)
                    record.update(diagnostics(cluster, result.params, data.noise))
                    records.append(record)
                    started = time.perf_counter()
            elapsed.append(time.perf_counter() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": args.scenario,
        "jitter_seed": args.jitter_seed,
        "solver": "lmfit/lstsq" if args.lmfit else "varpro",
        "noise": data.noise,
        "n_peaks": len(data.peaks),
        "n_clusters": len(data.clusters),
        "plane_values": data.spectra.z_values.tolist(),
        "passes": args.passes,
        "optimizer_seconds_full_dataset": sum(elapsed),
        "groups": records,
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
