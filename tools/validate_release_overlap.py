"""Run the actual v0.10.5 source in a separate environment containing LMFit.

Example: python tools/validate_release_overlap.py --source /tmp/v0.10.5
         --output /tmp/release.json --noise 11922.544916060584
Use the current environment's NumPy/SciPy versions to isolate source differences.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Load original data, fit every group twice and record the problematic groups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise", type=float)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    lf = importlib.import_module("lmfit")
    cli = importlib.import_module("peakfit.cli")
    spectra_api = importlib.import_module("peakfit.spectra")
    peaklist = importlib.import_module("peakfit.peaklist")
    clustering = importlib.import_module("peakfit.clustering")
    peak_api = importlib.import_module("peakfit.peak")
    computing = importlib.import_module("peakfit.computing")
    noise_api = importlib.import_module("peakfit.noise")
    data_path = ROOT / "examples/02-advanced-fitting/data"
    config = cli.Arguments(  # ty: ignore[unresolved-attribute]
        path_list=data_path / "pseudo3d.list", noise=args.noise
    )
    with threadpool_limits(limits=1):
        spectra = spectra_api.read_spectra(
            data_path / "pseudo3d.ft2", data_path / "b1_offsets.txt", []
        )
        noise = noise_api.prepare_noise_level(config, spectra)
        peaks = peaklist.read_list(spectra, spectra_api.get_shape_names(config, spectra), config)
        clusters = clustering.create_clusters(spectra, peaks, 5 * noise)
        all_params = lf.Parameters()
        records = []
        elapsed = []
        for iteration in range(2):
            if iteration:
                computing.update_cluster_corrections(all_params, clusters)
            started = time.perf_counter()
            for cluster in clusters:
                params = peak_api.create_params(cluster.peaks, fixed=False)
                for name in params:
                    if name in all_params:
                        params[name] = all_params[name]
                result = lf.Minimizer(
                    computing.residuals, params, fcn_args=(cluster, noise)
                ).least_squares()
                all_params.update(result.params)
                if cluster.cluster_id not in (269, 360, 424):
                    continue
                shapes = computing.calculate_shapes(result.params, cluster)
                amplitudes = computing.calculate_amplitudes(shapes, cluster.corrected_data)
                parameters = {}
                for peak in cluster.peaks:
                    for dim, shape in enumerate(peak.shapes, 2):
                        for suffix, label, scale in (("0", "cs", 1.0), ("_r2", "lw", np.pi)):
                            parameter = result.params[shape.prefix + suffix]
                            parameters[f"{peak.name}.F{dim}.{label}"] = {
                                "value": parameter.value / scale,
                                "error": parameter.stderr / scale
                                if parameter.stderr is not None
                                else None,
                                "bounds": [parameter.min / scale, parameter.max / scale],
                            }
                singular = np.linalg.svd(shapes.T, compute_uv=False)
                records.append(
                    {
                        "pass": iteration + 1,
                        "cluster_id": int(cluster.cluster_id),
                        "peaks": [peak.name for peak in cluster.peaks],
                        "data_shape": list(cluster.data.shape),
                        "mask_sha256": hashlib.sha256(
                            np.asarray(cluster.positions).tobytes()
                        ).hexdigest(),
                        "chi_squared": result.chisqr,
                        "optimality": result.optimality,
                        "success": bool(result.success),
                        "termination": result.message,
                        "nfev": result.nfev,
                        "parameters": parameters,
                        "amplitudes": amplitudes.tolist(),
                        "unit_norm_shapes": (
                            shapes / np.linalg.norm(shapes, axis=1)[:, None]
                        ).tolist(),
                        "singular_values": singular.tolist(),
                        "condition": float(singular[0] / singular[-1]),
                        "correction_norm": float(np.linalg.norm(cluster.corrections)),
                    }
                )
            elapsed.append(time.perf_counter() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "noise": noise,
                "numpy": np.__version__,
                "scipy": importlib.import_module("scipy").__version__,
                "lmfit": lf.__version__,
                "groups": records,
                "seconds_full_dataset": elapsed,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
