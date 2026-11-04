"""Evidence for the σ-only uncertainty policy (docs/validation/sigma-uncertainties.md).

Amplitude errors are σ√diag((ΦΦᵀ)⁻¹), never rescaled by √max(1, χ²ᵣ). This tool
compares them, and the former rescaled errors, with what the amplitudes actually do.

**Synthetic:** a 1D Gaussian cluster over four planes, fitted by the ordinary
pipeline. In each case the observed scatter of the fitted amplitudes over noise
realizations is compared with the reported error. The cases are:
- well-behaved, a single peak and a resolved pair, with MCMC on a subset;
- σ supplied at half its true value;
- Lorentzian data fitted as Gaussian;
- correlated noise with the correct marginal σ.

**Real (optional):** for a fitted CEST-like series, the plane-to-plane scatter of
each peak's intensities is compared with ``intensity_err``. The scatter is a robust
SD of successive differences over the saturation planes, divided by √2. The
spectrum's noise autocorrelation predicts the error inflation that correlated
points cause.

    uv run python tools/validate_sigma_uncertainties.py --output /tmp/sigma.json
    uv run python tools/validate_sigma_uncertainties.py --results RUN --spectrum FT2

With RUN a fit of the bundled CEST example and FT2 its ``pseudo3d.ft2``, the
real-data comparison is added.
"""

# ruff: noqa: T201

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import validate_mcmc_coverage as coverage

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig, LineshapeName, PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.scientific.sampler import SamplerSettings, sample_posterior
from peakfit.engine.scientific.sampling import sampling_problem
from peakfit.fit.pipeline import run_pipeline

SIGMA = 0.05
# name: (separation in points or None, true lineshape, supplied σ / true σ, noise kernel)
CASES: dict[str, tuple[float | None, LineshapeName, float, tuple | None]] = {
    "single peak, well-behaved": (None, "gaussian", 1.0, None),
    "resolved pair (10 points), well-behaved": (10.0, "gaussian", 1.0, None),
    "single peak, σ supplied at half its value": (None, "gaussian", 0.5, None),
    "single peak, Lorentzian data fitted as Gaussian": (None, "lorentzian", 1.0, None),
    "single peak, correlated noise (ρ₁ = 0.67)": (None, "gaussian", 1.0, (0.5, 1.0, 0.5)),
}
MCMC = SamplerSettings(walkers=16, steps=3000)
Z95 = 1.96


def _cluster(case: str, rng: np.random.Generator | None):
    separation, shape, _, kernel = CASES[case]
    spectra = Spectra(
        dic={},
        data=np.zeros((4, coverage.POINTS)),
        z_values=np.arange(4.0),
        params=[coverage._axis(4, False), coverage._axis(coverage.POINTS, True)],
    )
    points = [50.0] if separation is None else [50 - separation / 2, 50 + separation / 2]

    def peaks(lineshape: LineshapeName) -> list[Peak]:
        made = []
        for name, point in zip("AB", points, strict=False):
            center = float(spectra.spectral_params[0].pts2ppm(point))
            config = FitConfig(lineshape=lineshape)
            shapes = create_shapes(spectra, config, name, [center], [lineshape])
            peak = Peak(name, np.array([center]), shapes)
            peak.set_cluster_id(1)
            made.append(peak)
        return made

    truth = peaks(shape)
    amplitudes = coverage.AMPLITUDES[: len(points)]
    empty = Cluster(1, truth, [np.arange(coverage.POINTS)], np.zeros((coverage.POINTS, 4)))
    data = empty.evaluate(Parameters.from_peaks(truth)).T @ amplitudes
    if rng is not None:
        white = rng.standard_normal((coverage.POINTS + 2, 4))
        weights = np.array(kernel or (0.0, 1.0, 0.0))
        weights /= np.sqrt(np.sum(weights**2))  # the marginal SD stays σ
        data = data + SIGMA * sum(w * white[i : i + coverage.POINTS] for i, w in enumerate(weights))
    fitted = peaks("gaussian")
    cluster = Cluster(1, fitted, [np.arange(coverage.POINTS)], data)
    return cluster, Parameters.from_peaks(fitted), amplitudes


def _fit(cluster, params, case):
    config = PeakFitConfig(fitting=FitConfig(lineshape="gaussian"))
    return run_pipeline(config, [cluster], SIGMA * CASES[case][2], params).completed_fit


def realization(case: str, index: int, sample: bool) -> dict | None:
    """One noisy data set: fitted amplitudes, σ-only errors, χ²ᵣ and optional MCMC SDs."""
    rng = np.random.default_rng([5, index, list(CASES).index(case)])
    cluster, params, amplitudes = _cluster(case, rng)
    completed = _fit(cluster, params, case)
    outcome = completed.outcome.cluster(1)
    if not outcome.usable or outcome.analytical_evaluation is None:
        return None
    evaluation = outcome.analytical_evaluation
    row = {
        "amplitudes": evaluation.amplitudes.tolist(),
        "errors": evaluation.amplitude_standard_errors.tolist(),
        "reduced_chi_squared": evaluation.statistics.reduced_chi_squared,
        "truth": amplitudes.tolist(),
    }
    if sample:
        posterior = sample_posterior(
            sampling_problem(completed, 1), MCMC, np.random.SeedSequence([9, index])
        )
        sd = np.full(amplitudes.shape, np.nan)
        names = [p.name for p in cluster.peaks]
        for a in posterior.amplitudes.amplitudes:
            sd[names.index(a.peaks[0]), a.plane] = a.sd
        row.update(mcmc_sd=sd.tolist(), provisional=posterior.provisional)
    return row


def noise_free(case: str) -> list:
    """What the fitted model estimates without noise: the truth unless misspecified."""
    cluster, params, _ = _cluster(case, None)
    evaluation = _fit(cluster, params, case).outcome.cluster(1).analytical_evaluation
    assert evaluation is not None
    return evaluation.amplitudes.tolist()


def _coverage(estimates, reference, errors) -> list[float]:
    z = np.abs(estimates - reference) / errors
    return [float(np.mean(z < 1.0)), float(np.mean(z < Z95))]


def summarize(rows: list[dict], reference: np.ndarray) -> dict:
    """Observed scatter against the σ-only and the former rescaled errors."""
    estimates = np.array([r["amplitudes"] for r in rows])
    errors = np.broadcast_to(np.array([r["errors"] for r in rows])[:, :, None], estimates.shape)
    chi2 = np.array([r["reduced_chi_squared"] for r in rows])
    rescaled = errors * np.sqrt(np.maximum(1.0, chi2))[:, None, None]
    scatter = estimates.std(axis=0)
    summary = {
        "realizations": len(rows),
        "median_reduced_chi_squared": float(np.median(chi2)),
        "scatter_over_sigma_only_error": float(np.median(scatter / errors.mean(axis=0))),
        "scatter_over_rescaled_error": float(np.median(scatter / rescaled.mean(axis=0))),
        "sigma_only_coverage_68_95": _coverage(estimates, reference, errors),
        "rescaled_coverage_68_95": _coverage(estimates, reference, rescaled),
    }
    truth = np.array(rows[0]["truth"])
    if not np.allclose(reference, truth):
        summary["noise_free_bias_relative_to_truth"] = float(np.median((reference - truth) / truth))
        summary["sigma_only_coverage_95_of_truth"] = _coverage(estimates, truth, errors)[1]
        summary["rescaled_coverage_95_of_truth"] = _coverage(estimates, truth, rescaled)[1]
    sampled = [r for r in rows if "mcmc_sd" in r]
    passed = [r for r in sampled if not r["provisional"]]
    if passed:
        sd = np.array([r["mcmc_sd"] for r in passed])
        own = np.array([r["errors"] for r in passed])[:, :, None]
        summary["mcmc_passed_of_sampled"] = [len(passed), len(sampled)]
        summary["mcmc_sd_over_sigma_only_error"] = float(np.nanmedian(sd / own))
        summary["scatter_over_mcmc_sd"] = float(np.median(scatter / np.nanmean(sd, axis=0)))
    return summary


def synthetic(realizations: int, sampled: int, workers: int) -> dict:
    """Every case over ``realizations`` noise draws, MCMC on the first ``sampled``."""
    jobs = [
        (case, i, i < sampled and "well-behaved" in case)
        for case in CASES
        for i in range(realizations)
    ]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        rows = list(pool.map(realization, *zip(*jobs, strict=True), chunksize=4))
        references = dict(zip(CASES, pool.map(noise_free, CASES), strict=True))
    return {
        case: summarize(
            [r for (c, _, _), r in zip(jobs, rows, strict=True) if c == case and r is not None],
            np.array(references[case]),
        )
        for case in CASES
    }


def _noise_autocorrelation(data: np.ndarray, sigma: float, axis: int, lags: int = 12):
    """Mean autocorrelation of noise-only points (|x| < 3σ) along one axis of the planes."""
    curves = []
    for plane in data[1::10]:
        x = np.where(np.abs(plane) < 3 * sigma, plane.astype(float), np.nan)
        power = np.nanmean(x**2)
        curve = [1.0]
        for k in range(1, lags + 1):
            a, b = (x[k:], x[:-k]) if axis == 0 else (x[:, k:], x[:, :-k])
            curve.append(float(np.nanmean(a * b) / power))
        curves.append(curve)
    return np.mean(curves, axis=0)


def real(results: Path, spectrum: Path) -> dict:
    """Plane-to-plane scatter of a fitted series against ``intensity_err``."""
    import nmrglue as ng  # noqa: PLC0415

    from peakfit.engine.algorithms.noise import estimate_noise  # noqa: PLC0415

    peaks = defaultdict(list)
    with (results / "tables" / "intensities.csv").open() as stream:
        for row in csv.DictReader(stream):
            peaks[row["peak_name"]].append(row)
    chi2, sigma_only, rescaled = [], [], []
    for rows in peaks.values():
        error = float(rows[0]["intensity_err"])
        if not np.isfinite(error):
            continue
        intensities = np.array([float(r["intensity"]) for r in rows])[1:]  # saturation planes
        steps = np.diff(intensities)
        scatter = 1.4826 * np.median(np.abs(steps - np.median(steps))) / np.sqrt(2)
        c = float(rows[0]["reduced_chi_squared"])
        chi2.append(c)
        sigma_only.append(scatter / error)
        rescaled.append(scatter / (error * np.sqrt(max(1.0, c))))
    chi2_array, ratio = np.array(chi2), np.array(sigma_only)
    _, data = ng.pipe.read(str(spectrum))
    sigma = float(estimate_noise(data))
    along = [_noise_autocorrelation(data, sigma, axis) for axis in (0, 1)]
    factors = [1.0 + 2.0 * float(np.sum(curve[1:])) for curve in along]
    planes = [float(np.std(p[np.abs(p) < 3 * sigma])) / 0.9866 / sigma for p in data[::13]]
    return {
        "peaks": len(ratio),
        "reduced_chi_squared_median_iqr": np.percentile(chi2_array, [50, 25, 75]).tolist(),
        "fraction_reduced_chi_squared_above_1": float(np.mean(chi2_array > 1.0)),
        "scatter_over_sigma_only_error_median_iqr": np.percentile(ratio, [50, 25, 75]).tolist(),
        "scatter_over_rescaled_error_median": float(np.median(rescaled)),
        "correlation_log_chi2_log_ratio": float(
            np.corrcoef(np.log(chi2_array), np.log(ratio))[0, 1]
        ),
        "noise_lag1_autocorrelation_per_axis": [float(curve[1]) for curve in along],
        "predicted_error_inflation_from_correlation": float(np.sqrt(np.prod(factors))),
        "plane_noise_over_global_sigma_range": [min(planes), max(planes)],
    }


def main() -> None:
    """Run the synthetic study, and the real-data comparison when given a fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--realizations", type=int, default=400)
    parser.add_argument("--sampled", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--results", type=Path, help="a fit run of a CEST-like series")
    parser.add_argument("--spectrum", type=Path, help="its spectrum")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"synthetic": synthetic(args.realizations, args.sampled, args.workers)}
    if args.results and args.spectrum:
        report["real"] = real(args.results, args.spectrum)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
